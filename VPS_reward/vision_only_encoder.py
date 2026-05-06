"""
Generate Qwen2.5-VL vision embeddings for VPS reward model training.

Usage:
    python vision_only_encoder.py --input data/dataset.json --video-dir data/videos --output data/VIDEO_ONLY_EMBEDDINGS.json
"""
import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json
import os
import gc
from pathlib import Path
from typing import List, Optional
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

FPS = 4
NORMALIZE = True
EMPTY_CACHE_EVERY = 20


class QwenVideoEmbedderA1:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        force_cpu: bool = False,
        device: Optional[str] = None,
    ):
        """
        force_cpu: If True, load the full model on CPU (float32). Use when another
        large model already occupies the GPU (e.g. Wan + in-process preference).
        device: If set (and not force_cpu), load the full model on this device via
        device_map={"": device}. If None, uses device_map=\"auto\".
        """
        self.processor = AutoProcessor.from_pretrained(model_name)
        if force_cpu:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            ).to("cpu")
        elif device is not None:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=DTYPE,
                device_map={"": device},
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=DTYPE,
                device_map="auto",
            )
        self.model.eval()

        # Cache special token ids for vision span detection (best effort)
        self._vision_start_ids = []
        self._vision_end_ids = []
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            for s in ["<|vision_start|>", "<vision_start>"]:
                sid = tok.convert_tokens_to_ids(s)
                if isinstance(sid, int) and sid != tok.unk_token_id:
                    self._vision_start_ids.append(sid)
            for s in ["<|vision_end|>", "<vision_end>"]:
                eid = tok.convert_tokens_to_ids(s)
                if isinstance(eid, int) and eid != tok.unk_token_id:
                    self._vision_end_ids.append(eid)

    def _get_vision_span_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Try to find vision token span using vision_start/vision_end markers in input_ids.
        Returns a boolean mask [seq_len] if found, else None.
        """
        if not self._vision_start_ids or not self._vision_end_ids:
            return None

        ids = input_ids.tolist()
        start_pos = None
        for i, tid in enumerate(ids):
            if tid in self._vision_start_ids:
                start_pos = i
                break
        if start_pos is None:
            return None

        end_pos = None
        for j in range(start_pos + 1, len(ids)):
            if ids[j] in self._vision_end_ids:
                end_pos = j
                break
        if end_pos is None or end_pos <= start_pos + 1:
            return None

        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        mask[start_pos + 1 : end_pos] = True
        mask = mask & attention_mask.bool()
        if mask.sum().item() == 0:
            return None
        return mask

    def _embed_common(self, messages) -> torch.Tensor:
        """
        Shared embedding path for both full-video and pre-sampled-frame inputs.
        """
        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        # IMPORTANT: with device_map="auto", safest is moving the whole batch to model.device
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                last_hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
            else:
                outputs = self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = outputs.hidden_states[-1][0]

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        vision_mask = self._get_vision_span_mask(input_ids, attention_mask)
        if vision_mask is None:
            valid_mask = attention_mask.bool()
        else:
            valid_mask = vision_mask

        pooled = last_hidden[valid_mask].mean(dim=0)
        return pooled

    def embed_video_only(self, video_path: str, fps: float = FPS, normalize: bool = NORMALIZE) -> torch.Tensor:
        """
        Video-only embedding (prompt-independent) from a video path: [hidden_dim]
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": " "},
            ],
        }]

        pooled = self._embed_common(messages)

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=0)

        return pooled

    def embed_video_frames_only(
        self,
        frames: List[Image.Image],
        fps: float = FPS,
        normalize: bool = NORMALIZE,
    ) -> torch.Tensor:
        """
        Video-only embedding from an explicit list of pre-sampled frames.
        Frames are treated as a synthetic video sequence.
        """
        if not frames:
            raise ValueError("frames list is empty")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "sample_fps": fps,
                },
                {"type": "text", "text": " "},
            ],
        }]

        pooled = self._embed_common(messages)

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=0)

        return pooled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-VL vision embeddings for VPS reward model."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSON file with 'index' and 'item_filename' fields",
    )
    parser.add_argument(
        "--video-dir", "-v",
        type=str,
        required=True,
        help="Directory containing video files referenced in the input JSON",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSON file for embeddings",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help=f"Frames per second for video sampling (default: {FPS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen VL model name (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of embeddings",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Input: {args.input}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Model: {args.model}")
    
    embedder = QwenVideoEmbedderA1(model_name=args.model)

    with open(args.input, "r") as f:
        data = json.load(f)

    total = len(data)
    results = []
    start_time = time.time()
    normalize = not args.no_normalize

    for i, item in enumerate(data):
        video_path = os.path.join(args.video_dir, item["item_filename"])

        v1 = embedder.embed_video_only(video_path=video_path, fps=args.fps, normalize=normalize)
        results.append({
            "index": item["index"],
            "v1": v1.detach().cpu().tolist(),
        })

        del v1
        gc.collect()
        if torch.cuda.is_available() and (i + 1) % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - (i + 1))
        print(f"Processed {i+1}/{total} | Elapsed: {elapsed:.1f}s | ETA: {remaining/60:.1f}m")

    print(f"Total time taken: {time.time() - start_time:.1f} seconds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Wrote embeddings to: {output_path}")


if __name__ == "__main__":
    main()

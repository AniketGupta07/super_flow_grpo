#!/usr/bin/env python3
"""
Generate Wan t2v reference videos using the same Diffusers stack and hyperparameters as
`config.grpo.general_ocr_wan2_1` / `scripts/train_wan2_1.py` (HF WanPipeline + wan_pipeline_with_logprob).

Run from anywhere; resolves paths relative to the flow_grpo repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

# flow_grpo repo root: .../flow_grpo
_FLOW_GRPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_FLOW_GRPO_ROOT)
if str(_FLOW_GRPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_GRPO_ROOT))

from config.grpo import general_ocr_wan2_1  # noqa: E402
from diffusers import WanPipeline  # noqa: E402

from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob  # noqa: E402
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt  # noqa: E402


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds


def write_manifest(prompt_file: Path, out_dir: Path, manifest_path: Path, line_start: int | None, line_end: int | None):
    rows = []
    for line_num, line in enumerate(
        prompt_file.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if line_start is not None and (line_num < line_start or line_num > line_end):
            continue
        prompt = line.rstrip("\n\r")
        if not prompt.strip():
            rows.append({"index": line_num, "prompt": "", "video_path": None})
            continue
        vid = out_dir / f"ref_{line_num:07d}.mp4"
        rows.append(
            {
                "index": line_num,
                "prompt": prompt,
                "video_path": str(vid.resolve()) if vid.is_file() else None,
            }
        )
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(rows)} lines)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt_file",
        type=Path,
        required=True,
        help="Text file with one prompt per line (e.g., data/extended_prompts.txt).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for ref_XXXXXXX.mp4 files.",
    )
    p.add_argument("--prompt_line_start", type=int, default=None, help="1-based inclusive; requires --prompt_line_end")
    p.add_argument("--prompt_line_end", type=int, default=None, help="1-based inclusive; requires --prompt_line_start")
    p.add_argument(
        "--sampling_mode",
        choices=("eval", "train"),
        default="eval",
        help="eval: eval_num_steps + determistic=True (matches train_wan2_1 eval). "
        "train: num_steps + determistic=False (matches rollout sampling).",
    )
    p.add_argument("--base_seed", type=int, default=None, help="defaults to config.seed")
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override config.pretrained.model (path relative to flow_grpo root or absolute).",
    )
    p.add_argument("--skip_manifest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = general_ocr_wan2_1()

    if args.prompt_line_start is not None or args.prompt_line_end is not None:
        if args.prompt_line_start is None or args.prompt_line_end is None:
            raise SystemExit("--prompt_line_start and --prompt_line_end must be set together.")
        if args.prompt_line_start < 1:
            raise SystemExit("--prompt_line_start must be >= 1")
        if args.prompt_line_end < args.prompt_line_start:
            raise SystemExit("--prompt_line_end must be >= --prompt_line_start")

    base_seed = cfg.seed if args.base_seed is None else args.base_seed

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model_path or cfg.pretrained.model
    if not os.path.isabs(model_id):
        model_id = str(_FLOW_GRPO_ROOT / model_id)

    if args.sampling_mode == "eval":
        num_inference_steps = cfg.sample.eval_num_steps
        determistic = True
    else:
        num_inference_steps = cfg.sample.num_steps
        determistic = False

    print("flow_grpo root:", _FLOW_GRPO_ROOT)
    print("model:", model_id)
    print(
        f"size: {cfg.width}x{cfg.height} | frames: {cfg.frames} | "
        f"guidance: {cfg.sample.guidance_scale} | steps: {num_inference_steps} | "
        f"mode: {args.sampling_mode} | determistic: {determistic}"
    )
    print("out_dir:", out_dir.resolve())

    if not args.prompt_file.is_file():
        raise SystemExit(f"Missing prompt file: {args.prompt_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required.")

    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16

    pipeline = WanPipeline.from_pretrained(model_id)
    pipeline.safety_checker = None
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.transformer.to(device, dtype=dtype)
    pipeline.transformer.eval()

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=512, device=device)

    lines = args.prompt_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_num, line in enumerate(lines, start=1):
        if args.prompt_line_start is not None:
            if line_num < args.prompt_line_start or line_num > args.prompt_line_end:
                continue

        prompt = line.rstrip("\n\r")
        out_path = out_dir / f"ref_{line_num:07d}.mp4"
        if not prompt.strip():
            print(f"[{line_num}] empty line -> skip")
            continue

        prompt_embeds = compute_text_embeddings(
            [prompt], text_encoders, tokenizers, max_sequence_length=512, device=device
        )
        neg = neg_prompt_embed[:1]

        gen = torch.Generator(device=device).manual_seed(base_seed + line_num)

        with torch.autocast(device_type="cuda", dtype=dtype):
            with torch.no_grad():
                videos, _, _, _ = wan_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=cfg.frames,
                    height=cfg.height,
                    width=cfg.width,
                    kl_reward=cfg.sample.kl_reward,
                    determistic=determistic,
                    generator=gen,
                )

        # Match train_wan2_1.eval: tensor (T, C, H, W) -> transpose(0, 2, 3, 1) -> (T, H, W, C)
        v = videos[0]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        v = v.float().cpu().numpy()
        if v.ndim == 5:
            v = v[0]
        video = v.transpose(0, 2, 3, 1)
        frames = [(np.clip(frame, 0, 1) * 255).astype(np.uint8) for frame in video]
        imageio.mimsave(out_path, frames, fps=8, codec="libx264", format="FFMPEG")
        print(f"[{line_num}] -> {out_path}")

    if not args.skip_manifest:
        manifest = out_dir / "manifest.json"
        write_manifest(
            args.prompt_file,
            out_dir,
            manifest,
            args.prompt_line_start,
            args.prompt_line_end,
        )


if __name__ == "__main__":
    main()

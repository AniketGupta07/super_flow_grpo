from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

_PREFERENCE_MODEL_TYPE = "both"


def set_preference_model_type(model_type):
    global _PREFERENCE_MODEL_TYPE
    normalized = str(model_type).strip().lower()
    if normalized not in {"social", "both", "aa"}:
        raise ValueError(
            f"Invalid model_type={model_type!r}. Expected one of: social, both, aa"
        )
    _PREFERENCE_MODEL_TYPE = normalized

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def preference_within_group_remote(device):
    """Scores videos within a group against the first video as reference (via HTTP server)."""
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import pickle

    url = "http://127.0.0.1:18087"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        data = {"videos": [vid for vid in images], "prompts": list(prompts)}
        response = sess.post(url, data=pickle.dumps(data), timeout=300)
        return pickle.loads(response.content)["outputs"], {}

    return _fn

def preference_cached_refs(device):
    """Score generated videos against pre-computed base-model reference embeddings.

    Pre-computed SBERT prompt embeddings and Qwen vision embeddings for reference
    videos are loaded once at init.  Generated clips are written to a short-lived
    MP4 (same fps as reference generation) and embedded with QwenVideoEmbedderA1
    .embed_video_only, matching VPS_reward/embed_wan_ref_grpo_videos.py.

    Environment variables (all required):
        VPS_DATA_ROOT: Root directory containing embeddings and prompts
            - <VPS_DATA_ROOT>/extended_prompts.txt
            - <VPS_DATA_ROOT>/SBERT_PROMPT_EMBEDDINGS.json
            - <VPS_DATA_ROOT>/video_encodings/wan_ref_vision_embeddings*.json
        VPS_MODEL_DIR: Directory containing PairMLP model (best.pth, config.json)
            Download from: huggingface.co/datasets/supervideoVPS/super_video (VPS_models/both)
    """
    import sys, os, json, tempfile

    import imageio

    # Add VPS_reward to path for imports
    _flow_grpo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _vps_reward_dir = os.path.join(_flow_grpo_root, "VPS_reward")
    if _vps_reward_dir not in sys.path:
        sys.path.insert(0, _vps_reward_dir)

    from train_prompt_pair_mlp import PairMLP
    from vision_only_encoder import FPS as _QWEN_REF_EMBED_FPS, NORMALIZE as _QWEN_REF_NORMALIZE
    from vision_only_encoder import QwenVideoEmbedderA1

    # Load paths from environment variables
    vps_data_root = os.environ.get("VPS_DATA_ROOT")
    vps_model_dir = os.environ.get("VPS_MODEL_DIR")

    if not vps_data_root:
        raise ValueError(
            "VPS_DATA_ROOT environment variable not set. "
            "Set it to the directory containing extended_prompts.txt, "
            "SBERT_PROMPT_EMBEDDINGS.json, and video_encodings/"
        )
    if not vps_model_dir:
        raise ValueError(
            "VPS_MODEL_DIR environment variable not set. "
            "Set it to the directory containing best.pth and config.json "
            "(download from huggingface.co/datasets/supervideoVPS/super_video VPS_models/both)"
        )

    emb_dir = os.path.join(vps_data_root, "video_encodings")
    prompt_file = os.path.join(vps_data_root, "extended_prompts.txt")
    sbert_file = os.path.join(vps_data_root, "SBERT_PROMPT_EMBEDDINGS.json")

    with open(prompt_file) as f:
        prompt_lines = [l.strip() for l in f]
    prompt_to_idx = {}
    for i, text in enumerate(prompt_lines):
        if text not in prompt_to_idx:
            prompt_to_idx[text] = i + 1

    with open(sbert_file) as f:
        sbert_data = json.load(f)
    prompt_embs = {e["index"]: torch.tensor(e["prompt_embed"], dtype=torch.float32)
                   for e in sbert_data}

    ref_vision_embs = {}
    for fname in sorted(os.listdir(emb_dir)):
        if fname.startswith("wan_ref_vision_embeddings") and fname.endswith(".json"):
            with open(os.path.join(emb_dir, fname)) as f:
                for entry in json.load(f):
                    ref_vision_embs[entry["index"]] = torch.tensor(entry["v1"], dtype=torch.float32)

    print(f"[preference_cached_refs] {len(prompt_embs)} prompt embs, "
          f"{len(ref_vision_embs)} ref vision embs, "
          f"{len(prompt_to_idx)} prompt->idx mappings")

    with open(os.path.join(vps_model_dir, "config.json")) as f:
        cfg = json.load(f)
    pair_model = PairMLP(cfg["input_dim"], cfg["hidden_dims"], cfg.get("dropout", 0.35))
    pair_model.load_state_dict(
        torch.load(os.path.join(vps_model_dir, "best.pth"), map_location="cpu", weights_only=True)
    )
    pair_model = pair_model.to(device).eval()

    _WAN_REF_MP4_FPS = 8
    embedder = QwenVideoEmbedderA1(device=str(device))

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        scores = []
        for i in range(len(images)):
            idx = prompt_to_idx.get(prompts[i])
            if idx is None or idx not in ref_vision_embs or idx not in prompt_embs:
                scores.append(0.5)
                continue
            frames = list(images[i])
            if len(frames) == 0:
                # Keep training robust if a sample unexpectedly has no frames.
                scores.append(0.5)
                continue
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tmp_path = tf.name
                imageio.mimsave(
                    tmp_path,
                    frames,
                    fps=_WAN_REF_MP4_FPS,
                    codec="libx264",
                    format="FFMPEG",
                )
                gen_emb = embedder.embed_video_only(
                    video_path=tmp_path,
                    fps=_QWEN_REF_EMBED_FPS,
                    normalize=_QWEN_REF_NORMALIZE,
                ).cpu()
            except Exception as exc:
                print(f"[preference_cached_refs] embedding fallback for sample {i}: {exc}")
                scores.append(0.5)
                continue
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            features = torch.cat([gen_emb, ref_vision_embs[idx], prompt_embs[idx]]
                                 ).unsqueeze(0).to(device)
            scores.append(pair_model(features).item())
        return scores, {}

    return _fn


def random_score(device):
    """Uniform random reward in [0, 1), as a control baseline against real preference rewards.

    Environment variables (optional):
        RANDOM_REWARD_SEED: Integer seed for reproducible reward streams.
    """
    import os

    seed = os.environ.get("RANDOM_REWARD_SEED")
    rng = np.random.default_rng(int(seed) if seed else None)

    def _fn(images, prompts, metadata):
        return rng.random(len(prompts)).tolist(), {}

    return _fn


def clip_preference(device):
    """Probability that a generated video beats its reference video, from CLIP scores.

    Both the generated clip and the cached base-model reference clip for the same prompt
    are scored against the prompt with CLIP (mean over sampled frames); a softmax over the
    two scores gives a 0-1 preference probability, matching the output range of
    `preference_cached_refs`.

    Environment variables:
        WAN_REF_VIDEO_DIR (required): Colon-separated directories holding reference videos
            named ref_<7-digit line number>.mp4 (as produced by
            scripts/generate_wan_flow_grpo_reference_videos.py). Searched recursively, so a
            parent of the per-slurm-job subdirectories works.
        CLIP_REF_PROMPT_FILE: Prompt file defining the line-number <-> prompt mapping.
            Defaults to <VPS_DATA_ROOT>/extended_prompts.txt, then <repo>/data/extended_prompts.txt.
        CLIP_PREFERENCE_TEMPERATURE: Softmax temperature on raw CLIP logits (default 1.0).
        CLIP_PREFERENCE_NUM_FRAMES: Frames sampled per video (default 8).
    """
    import os
    import json
    import re

    import imageio

    from flow_grpo.clip_scorer import ClipScorer

    # ClipScorer divides logits_per_image by 30; undo it so the softmax sees raw CLIP logits.
    _CLIP_LOGIT_SCALE = 30.0
    _REF_NAME_RE = re.compile(r"^ref_(\d+)\.mp4$")

    ref_dirs = [p for p in os.environ.get("WAN_REF_VIDEO_DIR", "").split(":") if p]
    if not ref_dirs:
        raise ValueError(
            "WAN_REF_VIDEO_DIR environment variable not set. "
            "Set it to the directory (or colon-separated directories) containing "
            "ref_XXXXXXX.mp4 reference videos."
        )

    idx_to_path = {}
    prompt_to_idx = {}
    for root in ref_dirs:
        if not os.path.isdir(root):
            raise ValueError(f"WAN_REF_VIDEO_DIR entry is not a directory: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                match = _REF_NAME_RE.match(fname)
                if match:
                    idx_to_path.setdefault(int(match.group(1)), os.path.join(dirpath, fname))
            if "manifest.json" in filenames:
                with open(os.path.join(dirpath, "manifest.json")) as f:
                    for entry in json.load(f):
                        if entry.get("prompt"):
                            prompt_to_idx.setdefault(entry["prompt"], entry["index"])

    prompt_file = os.environ.get("CLIP_REF_PROMPT_FILE")
    if not prompt_file:
        vps_data_root = os.environ.get("VPS_DATA_ROOT")
        candidates = []
        if vps_data_root:
            candidates.append(os.path.join(vps_data_root, "extended_prompts.txt"))
        _flow_grpo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(_flow_grpo_root, "data", "extended_prompts.txt"))
        prompt_file = next((p for p in candidates if os.path.isfile(p)), None)
    if prompt_file:
        with open(prompt_file) as f:
            for i, line in enumerate(f):
                text = line.strip()
                if text:
                    prompt_to_idx.setdefault(text, i + 1)
    elif not prompt_to_idx:
        raise ValueError(
            "No prompt -> reference index mapping available. Set CLIP_REF_PROMPT_FILE to the "
            "prompt file used to generate the reference videos, or provide a manifest.json "
            "inside WAN_REF_VIDEO_DIR."
        )

    temperature = float(os.environ.get("CLIP_PREFERENCE_TEMPERATURE", 1.0))
    num_frames = int(os.environ.get("CLIP_PREFERENCE_NUM_FRAMES", 8))

    print(f"[clip_preference] {len(idx_to_path)} reference videos, "
          f"{len(prompt_to_idx)} prompt->idx mappings, "
          f"temperature={temperature}, frames={num_frames}")

    scorer = ClipScorer(device=device)
    ref_score_cache = {}

    def _score_frames(frames, prompt):
        sample_at = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)))
        batch = np.stack([np.asarray(frames[i]) for i in sample_at.round().astype(int)])
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        return scorer(batch, [prompt] * len(batch)).mean().item() * _CLIP_LOGIT_SCALE

    def _reference_score(idx, prompt):
        if idx not in ref_score_cache:
            reader = imageio.get_reader(idx_to_path[idx], format="FFMPEG")
            try:
                frames = [np.asarray(frame) for frame in reader]
            finally:
                reader.close()
            ref_score_cache[idx] = _score_frames(frames, prompt)
        return ref_score_cache[idx]

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        scores = []
        for i in range(len(images)):
            idx = prompt_to_idx.get(prompts[i])
            frames = list(images[i])
            if idx is None or idx not in idx_to_path or len(frames) == 0:
                scores.append(0.5)
                continue
            try:
                gen_score = _score_frames(frames, prompts[i])
                ref_score = _reference_score(idx, prompts[i])
            except Exception as exc:
                print(f"[clip_preference] scoring fallback for sample {i}: {exc}")
                scores.append(0.5)
                continue
            pair = torch.tensor([gen_score, ref_score]) / temperature
            scores.append(torch.softmax(pair, dim=0)[0].item())
        return scores, {}

    return _fn


def clip_video_score(device):
    """CLIP alignment between a generated video and its prompt, with no reference video.

    The video analogue of `clip_score`: sampled frames are scored against the prompt and
    averaged, on the same logits/30 scale (roughly 0-1). Since GRPO normalizes rewards
    within a prompt group and the reference score is constant across a group, this ranks
    samples the same way `clip_preference` does while skipping the reference pass.

    Environment variables (optional):
        CLIP_SCORE_NUM_FRAMES: Frames sampled per video (default 8).
    """
    import os

    from flow_grpo.clip_scorer import ClipScorer

    num_frames = int(os.environ.get("CLIP_SCORE_NUM_FRAMES", 8))
    print(f"[clip_video_score] frames={num_frames}")

    scorer = ClipScorer(device=device)

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        scores = []
        for i in range(len(images)):
            frames = list(images[i])
            if len(frames) == 0:
                # Keep training robust if a sample unexpectedly has no frames.
                scores.append(0.0)
                continue
            sample_at = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)))
            batch = np.stack([np.asarray(frames[j]) for j in sample_at.round().astype(int)])
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            scores.append(scorer(batch, [prompts[i]] * len(batch)).mean().item())
        return scores, {}

    return _fn


_VIDEOSCORE2_CACHE = {}


def videoscore2_score(device):
    """VideoScore2 reward: the model's overall score for a clip, rescaled to 0-1.

    Wraps TIGER-Lab/VideoScore2 (a Qwen2.5-VL-7B fine-tune). The model emits a
    chain-of-thought followed by three 1-5 integer scores for visual quality, text-to-video
    alignment and physical/common-sense consistency; following the official `vs2_inference.py`,
    each is softened by the model's normalized probability over the score tokens 1-5.

    The model does not emit a fourth "overall" score. The official eval code derives one by
    equal-weighted aggregation of the three dimensions -- `(v+t+p)/3` in `get_pairwise_acc.py`
    and `get_acc_corr.py` -- which is what the default VIDEOSCORE2_WEIGHTS of 1,1,1 reproduces.
    Dividing that by 5 puts the reward in 0-1.

    This scores each clip on its own, so unlike the vps/clip rewards it needs no reference
    videos and no WAN_REF_VIDEO_DIR.

    Environment variables (all optional):
        VIDEOSCORE2_MODEL: Model id or local path (default TIGER-Lab/VideoScore2).
        VIDEOSCORE2_DEVICE: Override placement, e.g. "cpu" to keep the training GPU free.
        VIDEOSCORE2_DTYPE: bfloat16 (default), float16 or float32.
        VIDEOSCORE2_FPS: Frame rate fed to the vision tower (default 2.0).
        VIDEOSCORE2_MAX_PIXELS: Per-frame pixel budget (default 200704, i.e. 448x448).
        VIDEOSCORE2_MAX_NEW_TOKENS: Generation budget for the rationale (default 1024).
        VIDEOSCORE2_TEMPERATURE: Sampling temperature (default 0.7, matching the official
            script). Note this makes the reward stochastic: rescoring one clip has ~0.08 std,
            which enters the GRPO group advantages as noise. Set 0 for deterministic greedy.
        VIDEOSCORE2_WEIGHTS: Comma-separated visual,alignment,physical weights. Defaults to
            1,1,1, the equal weighting the official eval code uses for its overall score.
        VIDEOSCORE2_DEBUG: Set to 1 to print the model's raw rationale for every clip.
    """
    import os
    import re
    import tempfile
    from string import Template

    import imageio
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

    # qwen_vl_utils prefers torchcodec, whose shared library is broken against torch 2.6 here.
    # It recovers by falling back to torchvision, but only after re-raising the load error for
    # *every* clip. Pin the working backend up front instead (must precede the import, which
    # reads the variable at module scope).
    if not os.environ.get("FORCE_QWENVL_VIDEO_READER"):
        try:
            import torchcodec.decoders  # noqa: F401
        except Exception:
            os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
    from qwen_vl_utils import process_vision_info

    _VS2_QUERY_TEMPLATE = Template("""
    You are an expert for evaluating AI-generated videos from three dimensions:
    (1) visual quality – clarity, smoothness, artifacts;
    (2) text-to-video alignment – fidelity to the prompt;
    (3) physical/common-sense consistency – naturalness and physics plausibility.

    Video prompt: $t2v_prompt

    Please output in this format:
    visual quality: <v_score>; 
    text-to-video alignment: <t_score>, 
    physical/common-sense consistency: <p_score>
    """)
    # The model answers either in the terse requested form ("visual quality: 4") or by echoing
    # the dimension's description first ("(1) visual quality - clarity, ...: 4"), so allow any
    # same-line filler between a dimension name and its digit. Staying on one line and stopping
    # at the first digit keeps each pattern anchored to its own dimension.
    _DIMENSIONS = ("visual quality", "text-to-video alignment",
                   "physical/common-sense consistency")
    _DIM_RES = tuple(re.compile(re.escape(name) + r"[^\n\d]*?(\d)", re.IGNORECASE)
                     for name in _DIMENSIONS)
    _THINK_END = "</think>"
    _MAX_SCORE = 5
    # Match the rate reference clips are written at, so playback speed matches training videos.
    _MP4_FPS = 8

    model_name = os.environ.get("VIDEOSCORE2_MODEL", "TIGER-Lab/VideoScore2")
    dtype = getattr(torch, os.environ.get("VIDEOSCORE2_DTYPE", "bfloat16"))
    vs2_device = torch.device(os.environ.get("VIDEOSCORE2_DEVICE") or device)
    infer_fps = float(os.environ.get("VIDEOSCORE2_FPS", 2.0))
    max_pixels = int(os.environ.get("VIDEOSCORE2_MAX_PIXELS", 448 * 448))
    max_new_tokens = int(os.environ.get("VIDEOSCORE2_MAX_NEW_TOKENS", 1024))
    temperature = float(os.environ.get("VIDEOSCORE2_TEMPERATURE", 0.7))
    debug = os.environ.get("VIDEOSCORE2_DEBUG", "") not in ("", "0")
    weights = [float(w) for w in os.environ.get("VIDEOSCORE2_WEIGHTS", "1,1,1").split(",")]
    if len(weights) != 3 or sum(weights) <= 0:
        raise ValueError(f"VIDEOSCORE2_WEIGHTS must be 3 positive-summing floats, got {weights!r}")

    print(f"[videoscore2] loading {model_name} on {vs2_device} ({dtype}), fps={infer_fps}, "
          f"max_pixels={max_pixels}, max_new_tokens={max_new_tokens}, temperature={temperature}, "
          f"weights={weights}")

    # train_wan2_1.py builds a separate multi_score for training and eval, which would otherwise
    # put two 15.5 GiB copies of this 7B scorer on the same GPU as the Wan training run.
    cache_key = (model_name, str(vs2_device), str(dtype))
    if cache_key in _VIDEOSCORE2_CACHE:
        print(f"[videoscore2] reusing already-loaded {model_name} on {vs2_device}")
        model, processor, tokenizer = _VIDEOSCORE2_CACHE[cache_key]
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(vs2_device).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        _VIDEOSCORE2_CACHE[cache_key] = (model, processor, tokenizer)

    # Token ids for "1".."5"; the soft score is a probability-weighted version of the hard one.
    score_token_ids = {}
    for s in range(1, _MAX_SCORE + 1):
        ids = tokenizer.encode(str(s), add_special_tokens=False)
        if len(ids) == 1:
            score_token_ids[s] = ids[0]
    if not score_token_ids:
        raise ValueError("Tokenizer does not encode digits 1-5 as single tokens; "
                         "VideoScore2 soft scoring cannot work with this checkpoint.")

    def _soft_score(hard_val, token_idx, scores):
        """Rescale a 1-5 score by how confident the model was at that token position."""
        if hard_val is None or token_idx < 0:
            return None
        logprobs = torch.log_softmax(scores[token_idx][0].float(), dim=-1)
        probs = {s: float(logprobs[tid].exp()) for s, tid in score_token_ids.items()}
        total = sum(probs.values())
        if total <= 0:
            return None
        best = max(probs, key=probs.get)
        return best * (probs[best] / total)

    def _find_scores(gen_str):
        """Locate each dimension's score digit as (char offset, value), or None per dimension."""
        # The rationale restates the scores before the final answer, so search the answer block
        # first and take the last hit within whichever block is used.
        think_end = gen_str.rfind(_THINK_END)
        starts = ([think_end + len(_THINK_END)] if think_end >= 0 else []) + [0]
        for start in starts:
            found = []
            for regex in _DIM_RES:
                last = None
                for last in regex.finditer(gen_str, start):
                    pass
                found.append(last)
            if all(m is not None for m in found):
                return [(m.start(1), int(m.group(1))) for m in found]
        return [None] * len(_DIM_RES)

    def _token_indices_at(gen_ids, char_offsets):
        """Map char offsets in the decoded generation back to the tokens that produced them."""
        pending = sorted((off, k) for k, off in enumerate(char_offsets) if off is not None)
        indices = [-1] * len(char_offsets)
        cursor = 0
        for i in range(len(gen_ids)):
            decoded_len = len(tokenizer.decode(gen_ids[:i + 1], skip_special_tokens=False))
            while cursor < len(pending) and pending[cursor][0] < decoded_len:
                indices[pending[cursor][1]] = i
                cursor += 1
            if cursor == len(pending):
                break
        return indices

    @torch.no_grad()
    def _score_video(video_path, prompt):
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": infer_fps, "max_pixels": max_pixels},
            {"type": "text", "text": _VS2_QUERY_TEMPLATE.substitute(t2v_prompt=prompt)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            fps=infer_fps, padding=True, return_tensors="pt",
        ).to(vs2_device)

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        gen_ids = gen_out.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        # Offsets must index the same string the tokens decode to, so keep special tokens.
        gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)

        hits = _find_scores(gen_str)
        token_indices = _token_indices_at(gen_ids, [h[0] if h else None for h in hits])
        soft = [_soft_score(h[1] if h else None, idx, gen_out.scores)
                for h, idx in zip(hits, token_indices)]

        if debug or any(s is None for s in soft):
            truncated = len(gen_ids) >= max_new_tokens
            print(f"[videoscore2] {'unparseable' if any(s is None for s in soft) else 'raw'} "
                  f"output ({len(gen_ids)} tokens"
                  f"{', hit max_new_tokens' if truncated else ''}): "
                  f"{tokenizer.decode(gen_ids, skip_special_tokens=True)!r}")
        return soft

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        scores, visual, alignment, physical = [], [], [], []
        for i in range(len(images)):
            frames = list(images[i])
            soft = [None] * 3
            if frames:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                        tmp_path = tf.name
                    imageio.mimsave(tmp_path, frames, fps=_MP4_FPS,
                                    codec="libx264", format="FFMPEG")
                    soft = _score_video(tmp_path, prompts[i])
                except Exception as exc:
                    print(f"[videoscore2] scoring fallback for sample {i}: {exc}")
                finally:
                    if tmp_path is not None:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            # A dimension the model failed to emit falls back to the neutral mid-score.
            filled = [s if s is not None else _MAX_SCORE / 2 for s in soft]
            visual.append(filled[0])
            alignment.append(filled[1])
            physical.append(filled[2])
            weighted = sum(w * s for w, s in zip(weights, filled)) / sum(weights)
            scores.append(weighted / _MAX_SCORE)
        return scores, {
            "videoscore2_visual": visual,
            "videoscore2_alignment": alignment,
            "videoscore2_physical": physical,
        }

    return _fn


def preference_within_group_local(device):
    """Scores videos within a group against the first video as reference (in-process).

    Qwen2.5-VL-7B runs on CPU so it does not contend with the Wan pipeline on the
    training GPU. Only the small PairMLP and a single feature vector batch use GPU.

    Environment variable (required):
        VPS_MODEL_DIR: Directory containing PairMLP model (best.pth, config.json)
            Download from: huggingface.co/datasets/supervideoVPS/super_video (VPS_models/both)
    """
    import sys, os, json

    # Add VPS_reward to path for imports
    _flow_grpo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _vps_reward_dir = os.path.join(_flow_grpo_root, "VPS_reward")
    if _vps_reward_dir not in sys.path:
        sys.path.insert(0, _vps_reward_dir)

    from vision_only_encoder import QwenVideoEmbedderA1
    from prompt_encoder import SbertPromptEmbedder
    from train_prompt_pair_mlp import PairMLP

    vps_model_dir = os.environ.get("VPS_MODEL_DIR")
    if not vps_model_dir:
        raise ValueError(
            "VPS_MODEL_DIR environment variable not set. "
            "Set it to the directory containing best.pth and config.json "
            "(download from huggingface.co/datasets/supervideoVPS/super_video VPS_models/both)"
        )

    with open(os.path.join(vps_model_dir, "config.json")) as f:
        cfg = json.load(f)

    vision_embedder = QwenVideoEmbedderA1(force_cpu=True)
    sbert_embedder = SbertPromptEmbedder(device="cpu")
    pair_model = PairMLP(cfg["input_dim"], cfg["hidden_dims"], cfg.get("dropout", 0.3)).to(device)
    pair_model.load_state_dict(
        torch.load(os.path.join(vps_model_dir, "best.pth"), map_location=device, weights_only=True)
    )
    pair_model.eval()

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        prompt = prompts[0]
        prompt_emb = sbert_embedder.embed_prompts([prompt]).float().view(-1)

        ref_frames = [Image.fromarray(f) for f in images[0]]
        ref_emb = vision_embedder.embed_video_frames_only(ref_frames, fps=4).float().view(-1)

        scores = [0.5]
        for vid_np in images[1:]:
            cand_frames = [Image.fromarray(f) for f in vid_np]
            cand_emb = vision_embedder.embed_video_frames_only(cand_frames, fps=4).float().view(-1)
            features = torch.cat([cand_emb, ref_emb, prompt_emb], dim=0).unsqueeze(0).to(
                device, dtype=pair_model.network[0].weight.dtype
            )
            scores.append(pair_model(features).item())
        return scores, {}

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "preference_within_group": preference_within_group_remote,
        "preference_within_group_local": preference_within_group_local,
        "preference_cached_refs": preference_cached_refs,
        "random": random_score,
        "clip_preference": clip_preference,
        "clip_video": clip_video_score,
        "videoscore2": videoscore2_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            # Reward fns can return extra per-sample breakdowns (e.g. videoscore2's individual
            # dimensions) alongside the score they are weighted on; surface them for logging.
            if isinstance(rewards, dict):
                score_details.update(rewards)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()

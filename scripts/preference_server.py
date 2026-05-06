"""
Preference scoring server for within-group GRPO training.

Loads QwenVideoEmbedderA1 (GPU), SbertPromptEmbedder (CPU), and PairMLP (GPU).
Receives G videos + prompt via pickled HTTP POST, returns scores where
video[0] is the reference (score=0.5) and video[i>0] gets P(video_i > ref).

Runs on a dedicated GPU (default port 18087).

Environment variable (required):
    VPS_MODEL_DIR: Directory containing PairMLP model (best.pth, config.json)
        Download from: huggingface.co/datasets/supervideoVPS/super_video (VPS_models/both)
"""

import os
import sys
import pickle
import json

import numpy as np
import torch
from flask import Flask, request, Response
from PIL import Image

# Add VPS_reward to path for imports
_FLOW_GRPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VPS_REWARD_DIR = os.path.join(_FLOW_GRPO_ROOT, "VPS_reward")
if _VPS_REWARD_DIR not in sys.path:
    sys.path.insert(0, _VPS_REWARD_DIR)

from vision_only_encoder import QwenVideoEmbedderA1
from prompt_encoder import SbertPromptEmbedder
from train_prompt_pair_mlp import PairMLP

DEFAULT_CHECKPOINT = "best.pth"
FPS = 4

app = Flask(__name__)

vision_embedder: QwenVideoEmbedderA1 = None  # type: ignore[assignment]
sbert_embedder: SbertPromptEmbedder = None   # type: ignore[assignment]
pair_model: PairMLP = None                    # type: ignore[assignment]
device: torch.device = None                   # type: ignore[assignment]


def load_models():
    global vision_embedder, sbert_embedder, pair_model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model directory from environment variable
    vps_model_dir = os.environ.get("VPS_MODEL_DIR")
    if not vps_model_dir:
        raise ValueError(
            "VPS_MODEL_DIR environment variable not set. "
            "Set it to the directory containing best.pth and config.json "
            "(download from huggingface.co/datasets/supervideoVPS/super_video VPS_models/both)"
        )

    print("[preference_server] Loading QwenVideoEmbedderA1 ...")
    vision_embedder = QwenVideoEmbedderA1()

    print("[preference_server] Loading SbertPromptEmbedder (CPU) ...")
    sbert_embedder = SbertPromptEmbedder(device="cpu")

    print("[preference_server] Loading PairMLP ...")
    cfg_path = os.path.join(vps_model_dir, "config.json")
    ckpt_path = os.path.join(vps_model_dir, DEFAULT_CHECKPOINT)
    with open(cfg_path) as f:
        config = json.load(f)
    pair_model = PairMLP(
        config["input_dim"],
        config["hidden_dims"],
        config.get("dropout", 0.3),
    ).to(device)
    pair_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    pair_model.eval()
    print("[preference_server] All models loaded.")


def frames_from_numpy(video_np: np.ndarray) -> list[Image.Image]:
    """Convert a (T, H, W, 3) uint8 numpy video to a list of PIL Images."""
    return [Image.fromarray(frame) for frame in video_np]


@torch.no_grad()
def score_group(videos: list[np.ndarray], prompt: str) -> list[float]:
    """
    Score G videos within a group.
    video[0] is reference (score 0.5).
    video[i>0] gets P(video_i > reference) from the PairMLP.
    """
    ref_frames = frames_from_numpy(videos[0])
    ref_emb = vision_embedder.embed_video_frames_only(ref_frames, fps=FPS).to(device).float().view(-1)

    prompt_emb = sbert_embedder.embed_prompts([prompt]).to(device).float().view(-1)

    scores = [0.5]
    for vid_np in videos[1:]:
        cand_frames = frames_from_numpy(vid_np)
        cand_emb = vision_embedder.embed_video_frames_only(cand_frames, fps=FPS).to(device).float().view(-1)
        features = torch.cat([cand_emb, ref_emb, prompt_emb], dim=0).unsqueeze(0)
        prob = pair_model(features).item()
        scores.append(prob)
    return scores


@app.route("/", methods=["POST"])
def handle_request():
    data = pickle.loads(request.data)
    videos = data["videos"]
    prompts = data["prompts"]
    prompt = prompts[0]

    scores = score_group(videos, prompt)

    response_data = pickle.dumps({"outputs": scores})
    return Response(response_data, mimetype="application/octet-stream")


@app.route("/health", methods=["GET"])
def health():
    return "ok"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18087)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_models()
    app.run(host=args.host, port=args.port, threaded=False)

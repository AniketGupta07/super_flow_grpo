#!/usr/bin/env python3
"""
Compute OpenAI CLIP text-video similarity for every video in a directory.

Per video (same pipeline as api_test/scripts/CLIPs/openai_clip.py):
- Sample `--frames` frames uniformly with PyAV.
- Encode frames with the CLIP image encoder, L2-normalize, mean-pool into one
  video embedding, re-normalize.
- Cosine similarity against the L2-normalized prompt embedding.

Prompts are matched per video index (the trailing digits of `ckpt_0000042.mp4`)
using `manifest.json` in the video directory when present, otherwise a prompt
file where line N corresponds to index N.

Results are written next to the videos:
    clip_scores.json          per-video scores
    clip_scores.csv           same, flat
    clip_scores_summary.json  aggregate statistics + failures

Typical usage (from anywhere):

    conda activate clip
    python3 flow_grpo/scripts/score_videos_clip.py \\
        /simurgh2/projects/super_video/flow_grpo/test_videos_rebuttal/baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import av
import torch
from PIL import Image

try:
    import clip
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'clip' package is required. Install it with "
        "`pip install git+https://github.com/openai/CLIP.git`."
    ) from exc

_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")
_INDEX_RE = re.compile(r"(\d+)")


def _video_stream(container: av.container.input.InputContainer) -> av.video.stream.VideoStream:
    for stream in container.streams:
        if stream.type == "video":
            stream.thread_type = "AUTO"
            return stream
    raise RuntimeError("No video stream found in container")


def _count_frames(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        stream = _video_stream(container)
        return sum(1 for _ in container.decode(stream))


def sample_video_frames(video_path: Path, num_frames: int) -> tuple[list[Image.Image], int]:
    """Uniformly sample up to `num_frames` frames; also returns the total frame count."""
    try:
        with av.open(str(video_path)) as container:
            total_frames = _video_stream(container).frames or 0
    except (av.error.FFmpegError, av.error.InvalidDataError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to open video {video_path}: {exc}") from exc

    if total_frames <= 0:
        total_frames = _count_frames(video_path)
    if total_frames == 0:
        raise ValueError(f"No frames could be read from {video_path}.")

    steps = min(num_frames, total_frames)
    if total_frames > steps:
        sample_indices = torch.linspace(0, total_frames - 1, steps=steps).long().tolist()
    else:
        sample_indices = list(range(total_frames))

    frames: list[Image.Image] = []
    with av.open(str(video_path)) as container:
        stream = _video_stream(container)
        index_iter = iter(sample_indices)
        next_index = next(index_iter, None)
        for frame_idx, frame in enumerate(container.decode(stream)):
            if next_index is None:
                break
            if frame_idx == next_index:
                frames.append(Image.fromarray(frame.to_rgb().to_ndarray()))
                next_index = next(index_iter, None)

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}.")
    return frames, total_frames


@torch.no_grad()
def clip_similarity(
    frames: list[Image.Image],
    prompt: str,
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> float:
    step = batch_size if batch_size > 0 else len(frames)
    frame_feats = []
    for start in range(0, len(frames), step):
        images = torch.stack([preprocess(f) for f in frames[start : start + step]]).to(device)
        feats = model.encode_image(images).float()
        frame_feats.append(feats / feats.norm(dim=-1, keepdim=True))
    frame_feats = torch.cat(frame_feats)

    tokens = clip.tokenize(prompt, truncate=True).to(device)
    text_feats = model.encode_text(tokens).float()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    video_feat = frame_feats.mean(dim=0, keepdim=True)
    video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
    return float(torch.matmul(video_feat, text_feats.T).squeeze().item())


def video_index(path: Path) -> int | None:
    """Trailing integer in the file stem, e.g. ckpt_0000042 -> 42."""
    matches = _INDEX_RE.findall(path.stem)
    return int(matches[-1]) if matches else None


def find_videos(video_dir: Path, pattern: str | None) -> list[Path]:
    if pattern:
        return sorted(p for p in video_dir.glob(pattern) if p.is_file())
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def load_prompts(video_dir: Path, prompt_file: Path | None) -> tuple[dict[int, str], str]:
    """Map video index -> prompt, preferring the manifest written at generation time."""
    manifest = video_dir / "manifest.json"
    if prompt_file is None and manifest.is_file():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        prompts = {
            int(entry["index"]): entry["prompt"]
            for entry in entries
            if entry.get("prompt", "").strip()
        }
        if prompts:
            return prompts, str(manifest)

    if prompt_file is not None:
        candidates = [prompt_file]
    else:
        candidates = [video_dir / "test_prompts.txt", video_dir.parent / "test_prompts.txt"]
    resolved = next((p for p in candidates if p.is_file()), None)
    if resolved is None:
        raise SystemExit(
            "Could not resolve prompts: no usable manifest.json in the video directory and "
            f"none of {[str(c) for c in candidates]} exist. Pass --prompt_file explicitly."
        )

    prompts = {}
    for line_num, line in enumerate(
        resolved.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        text = line.strip()
        if text:
            prompts[line_num] = text
    return prompts, str(resolved)


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "video_dir",
        type=Path,
        help="Directory of videos to score; results are written here too.",
    )
    p.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help=(
            "Text file with one prompt per line, line N matching video index N. Defaults to "
            "manifest.json in the video directory, then test_prompts.txt there or in its parent."
        ),
    )
    p.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Number of uniformly sampled frames per video (default 32).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="CLIP model variant to load (default ViT-B/32).",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional glob for video files (default: every video-suffixed file in the directory).",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="clip_scores",
        help="Basename for the result files (default clip_scores).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Frames per image-encoder pass; 0 encodes all sampled frames at once.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default cuda when available, else cpu).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    video_dir = args.video_dir
    if not video_dir.is_dir():
        raise SystemExit(f"Not a directory: {video_dir}")
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")

    videos = find_videos(video_dir, args.pattern)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")
    prompts, prompt_source = load_prompts(video_dir, args.prompt_file)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"video_dir: {video_dir.resolve()}")
    print(f"videos: {len(videos)} | prompts: {len(prompts)} from {prompt_source}")
    print(f"model: {args.model} | device: {device} | frames per video: {args.frames}")

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    rows = []
    failures = []
    for n, path in enumerate(videos, start=1):
        idx = video_index(path)
        prompt = prompts.get(idx) if idx is not None else None
        if prompt is None:
            print(f"[{n}/{len(videos)}] {path.name}: no prompt for index {idx} -> skip")
            failures.append({"video": path.name, "index": idx, "error": "no prompt for index"})
            continue
        try:
            frames, total_frames = sample_video_frames(path, args.frames)
            score = clip_similarity(
                frames, prompt, model, preprocess, device, args.batch_size
            )
        except Exception as exc:  # keep going so one bad file doesn't lose the whole run
            print(f"[{n}/{len(videos)}] {path.name}: FAILED ({exc})")
            failures.append({"video": path.name, "index": idx, "error": str(exc)})
            continue
        rows.append(
            {
                "index": idx,
                "video": path.name,
                "video_path": str(path.resolve()),
                "prompt": prompt,
                "clip_score": score,
                "num_frames_total": total_frames,
                "num_frames_scored": len(frames),
            }
        )
        print(f"[{n}/{len(videos)}] {path.name}: clip_score={score:.4f}")

    rows.sort(key=lambda r: (r["index"] is None, r["index"] or -1, r["video"]))
    scores = [r["clip_score"] for r in rows]

    per_video_json = video_dir / f"{args.out_prefix}.json"
    per_video_csv = video_dir / f"{args.out_prefix}.csv"
    summary_json = video_dir / f"{args.out_prefix}_summary.json"

    per_video_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with per_video_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "video", "clip_score", "num_frames_scored"])
        for r in rows:
            writer.writerow([r["index"], r["video"], f"{r['clip_score']:.6f}", r["num_frames_scored"]])

    summary = {
        "video_dir": str(video_dir.resolve()),
        "name": video_dir.resolve().name,
        "prompt_source": prompt_source,
        "clip_model": args.model,
        "frames_requested": args.frames,
        "num_videos_found": len(videos),
        "num_videos_scored": len(rows),
        "num_videos_failed": len(failures),
        "clip_score": summarize(scores),
        "failures": failures,
        "command": " ".join(sys.argv),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {per_video_json}")
    print(f"Wrote {per_video_csv}")
    print(f"Wrote {summary_json}")
    if scores:
        s = summary["clip_score"]
        print(
            f"clip_score over {s['count']} videos: mean={s['mean']:.4f} std={s['std']:.4f} "
            f"min={s['min']:.4f} max={s['max']:.4f}"
        )
    if failures:
        print(f"{len(failures)} video(s) failed; see {summary_json}")


if __name__ == "__main__":
    main()

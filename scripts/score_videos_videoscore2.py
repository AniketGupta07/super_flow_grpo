#!/usr/bin/env python3
"""
Compute VideoScore2 metrics for every video in a directory.

Per video (same pipeline as api_test/scripts/metrics calc/videoscore.py):
- Load TIGER-Lab/VideoScore2 (Qwen2-VL) once.
- Sample the video at `--fps` and score visual quality, text-to-video alignment,
  and physical/common-sense consistency on a 1-5 scale (parse failures -> -1).
- Overall is the mean of the three dimensions.

Prompts are matched per video index (the trailing digits of `ckpt_0000042.mp4`)
using `manifest.json` in the video directory when present, otherwise a prompt
file where line N corresponds to index N.

Results are written next to the videos:
    videoscore2_scores.json          per-video scores
    videoscore2_scores.csv           same, flat
    videoscore2_scores_summary.json  aggregate statistics + failures

Typical usage (from anywhere):

    conda activate vs2_met   # or: source .../conda/bin/activate vs2
    python3 flow_grpo/scripts/score_videos_videoscore2.py \\
        /simurgh2/projects/super_video/flow_grpo/test_videos_rebuttal/baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import warnings
from pathlib import Path
from string import Template
from typing import Optional

import torch

warnings.filterwarnings("ignore")

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "VideoScore2 requires transformers and qwen-vl-utils. "
        "Activate the vs2 conda env, or: pip install transformers accelerate qwen-vl-utils"
    ) from exc

_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")
_INDEX_RE = re.compile(r"(\d+)")
_SCORE_KEYS = ("visual_quality", "text_alignment", "physical_consistency", "overall")

_QUERY_TEMPLATE = Template(
    """
    You are an expert for evaluating AI-generated videos from three dimensions:
    (1) visual quality – clarity, smoothness, artifacts;
    (2) text-to-video alignment – fidelity to the prompt;
    (3) physical/common-sense consistency – naturalness and physics plausibility.

    Video prompt: $t2v_prompt

    Please output in this format:
    visual quality: <v_score>; 
    text-to-video alignment: <t_score>, 
    physical/common-sense consistency: <p_score>
    """
)


def extract_score_from_text(text: str, label: str) -> Optional[int]:
    """Extract a 1-5 score after a numbered dimension label in the model output."""
    pattern = r"\(\d\)\s*" + re.escape(label) + r"[^:\n]*:\s*(\d)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    return None


def _get_video_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


class VideoScoreModel:
    """Cached VideoScore2 scorer (same logic as api_test videoscore.py)."""

    def __init__(
        self,
        model_name: str = "TIGER-Lab/VideoScore2",
        device: Optional[str] = None,
        infer_fps: float | str = 2.0,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.infer_fps = infer_fps
        self.max_tokens = max_tokens
        self.temperature = temperature

        print(f"Loading VideoScore2 model '{model_name}' on {self.device}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("VideoScore2 model loaded successfully.")

    def compute_scores(self, video_path: Path, prompt: str) -> dict:
        if not video_path.is_file():
            raise ValueError(f"Video not found: {video_path}")

        infer_fps = self.infer_fps
        if infer_fps == "raw":
            infer_fps = _get_video_fps(video_path)

        user_prompt = _QUERY_TEMPLATE.substitute(t2v_prompt=prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "fps": infer_fps},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=infer_fps,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.batch_decode(
            gen_out[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        v_score = extract_score_from_text(output_text, "visual quality")
        t_score = extract_score_from_text(output_text, "text-to-video alignment")
        p_score = extract_score_from_text(
            output_text, "physical/common-sense consistency"
        )
        if any(s is None for s in (v_score, t_score, p_score)):
            print(
                f"[warn] parse miss for {video_path.name}: "
                f"V={v_score}, T={t_score}, P={p_score}"
            )
            print(f"[raw]\n{output_text}")

        v_final = float(v_score) if v_score is not None else -1.0
        t_final = float(t_score) if t_score is not None else -1.0
        p_final = float(p_score) if p_score is not None else -1.0
        return {
            "visual_quality": v_final,
            "text_alignment": t_final,
            "physical_consistency": p_final,
            "overall": (v_final + t_final + p_final) / 3.0,
            "raw_output": output_text,
        }


def video_index(path: Path) -> int | None:
    matches = _INDEX_RE.findall(path.stem)
    return int(matches[-1]) if matches else None


def find_videos(video_dir: Path, pattern: str | None) -> list[Path]:
    if pattern:
        return sorted(p for p in video_dir.glob(pattern) if p.is_file())
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def load_prompts(video_dir: Path, prompt_file: Path | None) -> tuple[dict[int, str], str]:
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
    # Exclude parse failures (-1) so they don't drag the mean.
    valid = [s for s in scores if s >= 0]
    if not valid:
        return {"count": 0, "valid_count": 0, "parse_failures": len(scores)}
    return {
        "count": len(scores),
        "valid_count": len(valid),
        "parse_failures": len(scores) - len(valid),
        "mean": statistics.fmean(valid),
        "std": statistics.stdev(valid) if len(valid) > 1 else 0.0,
        "median": statistics.median(valid),
        "min": min(valid),
        "max": max(valid),
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
        "--model",
        type=str,
        default="TIGER-Lab/VideoScore2",
        help="VideoScore2 model id or local path (default TIGER-Lab/VideoScore2).",
    )
    p.add_argument(
        "--fps",
        type=str,
        default="2.0",
        help="Inference FPS (default 2.0), or 'raw' to use each video's native FPS.",
    )
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional glob for video files (default: every video-suffixed file in the directory).",
    )
    p.add_argument(
        "--out_prefix",
        type=str,
        default="videoscore2_scores",
        help="Basename for the result files (default videoscore2_scores).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default cuda when available, else cpu).",
    )
    p.add_argument(
        "--save_raw",
        action="store_true",
        help="Include raw model outputs in the per-video JSON (can be large).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    video_dir = args.video_dir
    if not video_dir.is_dir():
        raise SystemExit(f"Not a directory: {video_dir}")

    if args.fps == "raw":
        infer_fps: float | str = "raw"
    else:
        try:
            infer_fps = float(args.fps)
        except ValueError as exc:
            raise SystemExit("--fps must be a float or 'raw'") from exc

    videos = find_videos(video_dir, args.pattern)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")
    prompts, prompt_source = load_prompts(video_dir, args.prompt_file)

    print(f"video_dir: {video_dir.resolve()}")
    print(f"videos: {len(videos)} | prompts: {len(prompts)} from {prompt_source}")
    print(f"model: {args.model} | fps: {infer_fps}")

    scorer = VideoScoreModel(
        model_name=args.model,
        device=args.device,
        infer_fps=infer_fps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    rows = []
    failures = []
    start = time.time()
    for n, path in enumerate(videos, start=1):
        idx = video_index(path)
        prompt = prompts.get(idx) if idx is not None else None
        if prompt is None:
            print(f"[{n}/{len(videos)}] {path.name}: no prompt for index {idx} -> skip")
            failures.append({"video": path.name, "index": idx, "error": "no prompt for index"})
            continue
        try:
            scores = scorer.compute_scores(path, prompt)
        except Exception as exc:
            print(f"[{n}/{len(videos)}] {path.name}: FAILED ({exc})")
            failures.append({"video": path.name, "index": idx, "error": str(exc)})
            continue

        row = {
            "index": idx,
            "video": path.name,
            "video_path": str(path.resolve()),
            "prompt": prompt,
            "visual_quality": scores["visual_quality"],
            "text_alignment": scores["text_alignment"],
            "physical_consistency": scores["physical_consistency"],
            "overall": scores["overall"],
        }
        if args.save_raw:
            row["raw_output"] = scores["raw_output"]
        rows.append(row)

        elapsed = time.time() - start
        eta = (elapsed / n) * (len(videos) - n)
        print(
            f"[{n}/{len(videos)}] {path.name}: "
            f"V={scores['visual_quality']:.1f} T={scores['text_alignment']:.1f} "
            f"P={scores['physical_consistency']:.1f} overall={scores['overall']:.2f} "
            f"(ETA {eta / 60:.1f}m)"
        )

    rows.sort(key=lambda r: (r["index"] is None, r["index"] or -1, r["video"]))

    per_video_json = video_dir / f"{args.out_prefix}.json"
    per_video_csv = video_dir / f"{args.out_prefix}.csv"
    summary_json = video_dir / f"{args.out_prefix}_summary.json"

    # Drop raw_output from the on-disk JSON unless asked; keep rows as-is when --save_raw.
    dump_rows = rows
    per_video_json.write_text(
        json.dumps(dump_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with per_video_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "video", "visual_quality", "text_alignment", "physical_consistency", "overall"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["index"],
                    r["video"],
                    f"{r['visual_quality']:.1f}",
                    f"{r['text_alignment']:.1f}",
                    f"{r['physical_consistency']:.1f}",
                    f"{r['overall']:.4f}",
                ]
            )

    summary = {
        "video_dir": str(video_dir.resolve()),
        "name": video_dir.resolve().name,
        "prompt_source": prompt_source,
        "model": args.model,
        "fps": infer_fps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_videos_found": len(videos),
        "num_videos_scored": len(rows),
        "num_videos_failed": len(failures),
        "scores": {k: summarize([r[k] for r in rows]) for k in _SCORE_KEYS},
        "failures": failures,
        "elapsed_sec": time.time() - start,
        "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {per_video_json}")
    print(f"Wrote {per_video_csv}")
    print(f"Wrote {summary_json}")
    if rows:
        for k in _SCORE_KEYS:
            s = summary["scores"][k]
            if s.get("valid_count", 0):
                print(
                    f"{k}: mean={s['mean']:.3f} std={s['std']:.3f} "
                    f"(n={s['valid_count']}, parse_fail={s['parse_failures']})"
                )
    if failures:
        print(f"{len(failures)} video(s) failed; see {summary_json}")


if __name__ == "__main__":
    main()

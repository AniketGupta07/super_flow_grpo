#!/usr/bin/env python3
"""
Embed Wan reference videos (from generate_wan_flow_grpo_reference_videos.py / wan_ref_grpo.sh)
using Qwen2.5-VL, same format as vision_only_encoder.main(): list of
{"index": <prompt line number>, "v1": <list[float]>}.

Reads manifest.json in --video_dir (written by the generator) or --manifest path.
Skips rows with missing video_path or nonexistent files unless --fail_on_missing.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

# Defaults must match api_test/EMBEDDING_MODEL/vision_only_encoder.py (FPS, NORMALIZE).
_DEFAULT_FPS = 4
_DEFAULT_NORMALIZE = True

EMPTY_CACHE_EVERY = 20

REF_NAME_RE = re.compile(r"^ref_(\d+)\.mp4$")


def load_jobs(manifest_path: Path) -> list[dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {manifest_path}")
    return data


def discover_from_glob(video_dir: Path) -> list[dict[str, Any]]:
    """Fallback: scan ref_XXXXXXX.mp4 and build pseudo-manifest rows."""
    rows: list[dict[str, Any]] = []
    for p in sorted(video_dir.glob("ref_*.mp4")):
        m = REF_NAME_RE.match(p.name)
        if not m:
            continue
        line_num = int(m.group(1))
        rows.append(
            {
                "index": line_num,
                "prompt": "",
                "video_path": str(p.resolve()) if p.is_file() else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video_dir",
        type=Path,
        required=True,
        help="Directory with ref_XXXXXXX.mp4 and usually manifest.json (WAN_FG_REF_OUT).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override manifest path (default: <video_dir>/manifest.json).",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output path (default: <video_dir>/wan_ref_vision_embeddings_fps<FPS>.json).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=_DEFAULT_FPS,
        help=f"Video sampling FPS for Qwen (default: {_DEFAULT_FPS}, same as vision_only_encoder).",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_NORMALIZE,
        help=f"L2-normalize embeddings (default: {_DEFAULT_NORMALIZE}, same as vision_only_encoder).",
    )
    p.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Exit with error if any non-empty prompt row has no video file.",
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--force_cpu", action="store_true", help="Load Qwen on CPU (float32).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from vision_only_encoder import QwenVideoEmbedderA1

    video_dir = args.video_dir.resolve()
    if not video_dir.is_dir():
        raise SystemExit(f"Not a directory: {video_dir}")

    manifest_path = args.manifest or (video_dir / "manifest.json")
    if manifest_path.is_file():
        rows = load_jobs(manifest_path)
    else:
        print(f"No {manifest_path}; scanning ref_*.mp4 under {video_dir}")
        rows = discover_from_glob(video_dir)
        if not rows:
            raise SystemExit(f"No manifest and no ref_*.mp4 in {video_dir}")

    out_path = args.output_json or (video_dir / f"wan_ref_vision_embeddings_fps{args.fps:g}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = QwenVideoEmbedderA1(model_name=args.model_name, force_cpu=args.force_cpu)

    results: list[dict[str, Any]] = []
    start_time = time.time()
    missing = 0
    skipped_empty = 0

    for i, item in enumerate(rows):
        idx = item.get("index")
        vp = item.get("video_path")
        prompt = (item.get("prompt") or "").strip()

        if vp is None or not str(vp).strip():
            if prompt:
                missing += 1
                if args.fail_on_missing:
                    raise SystemExit(f"Missing video_path for index={idx!r} (prompt non-empty)")
            else:
                skipped_empty += 1
            continue

        video_path = Path(vp)
        if not video_path.is_file():
            missing += 1
            if args.fail_on_missing:
                raise SystemExit(f"Video missing for index={idx}: {video_path}")
            continue

        v1 = embedder.embed_video_only(
            video_path=str(video_path),
            fps=args.fps,
            normalize=args.normalize,
        )
        results.append({"index": idx, "v1": v1.detach().cpu().tolist()})
        del v1
        gc.collect()
        if torch.cuda.is_available() and (i + 1) % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        n_done = len(results)
        avg_time = elapsed / n_done if n_done else 0.0
        n_left = len(rows) - (i + 1)
        print(
            f"Embedded {n_done} | row {i + 1}/{len(rows)} | "
            f"Elapsed: {elapsed:.1f}s | ETA: {avg_time * max(n_left, 0) / 60:.1f}m"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    total_time = time.time() - start_time
    print(
        f"Wrote {len(results)} embeddings to {out_path} "
        f"(skipped_empty={skipped_empty}, missing_or_bad_path={missing}) in {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()

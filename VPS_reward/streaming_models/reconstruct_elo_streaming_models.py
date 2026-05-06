#!/usr/bin/env python3
"""
Reconstruct per-model ELO from streaming_model PairMLP predictions (AA×AA pairs only).

Reads:
  ./predictions-streaming-{aa,both}/train_preds.csv (and val, test)
  ./encoding_splits-{aa,both}-streaming_model/model_tiers.json  (for scatter highlights)

Writes:
  ./predictions-streaming-{dataset}/elo_{split}.csv
  ./predictions-streaming-{dataset}/elo_correlations.csv, .json
  ./predictions-streaming-{dataset}/plots/scatter_{split}_{ref}.png

Scatter plots: points whose model name is in the streaming tier that matches the evaluation
split (train preds → train_tier highlight, val → val_tier, test → test_tier) use a distinct
marker and appear in the legend.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMBED_DIR = os.path.dirname(_SCRIPT_DIR)
if _EMBED_DIR not in sys.path:
    sys.path.insert(0, _EMBED_DIR)

import reconstruct_elo_from_probs as rep  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


I2V_REF = rep.I2V_REF
T2V_REF = rep.T2V_REF


def parse_args():
    p = argparse.ArgumentParser(description="ELO from streaming_model MLP preds (AA×AA).")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aa", "both"],
        help="Matches predictions-streaming-{dataset} and encoding_splits-{dataset}-streaming_model.",
    )
    return p.parse_args()


def predictions_dir(dataset: str) -> str:
    return os.path.join(_SCRIPT_DIR, f"predictions-streaming-{dataset}")


def splits_dir(dataset: str) -> str:
    return os.path.join(_SCRIPT_DIR, f"encoding_splits-{dataset}-streaming_model")


def filter_aa_aa(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["dataset1"] == "aa") & (df["dataset2"] == "aa")].copy()


def df_to_prob_rows(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    return list(
        zip(
            df["index1"].astype(np.int64),
            df["index2"].astype(np.int64),
            df["y_pred"].astype(np.float64),
        )
    )


def load_model_tiers(dataset: str) -> dict | None:
    path = os.path.join(splits_dir(dataset), "model_tiers.json")
    if not os.path.isfile(path):
        print(f"Warning: missing {path}; scatter plots will not highlight tier models.")
        print(
            f"  Re-run: python create_prob_splits.py --dataset {dataset} "
            "--mode streaming_model"
        )
        return None
    with open(path) as f:
        return json.load(f)


def tier_highlight_for_split(split_name: str, tiers: dict | None) -> set[str]:
    if tiers is None:
        return set()
    key = {"train": "train_tier", "val": "val_tier", "test": "test_tier"}[split_name]
    return set(tiers.get(key, []))


def write_elo_csv(model_to_elo: dict, ref_original: dict, out_path: str) -> None:
    sorted_models = sorted(model_to_elo.keys(), key=lambda m: model_to_elo[m], reverse=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "elo", "elo_original"])
        for model in sorted_models:
            elo_pred = model_to_elo[model]
            eo = ref_original.get(model, "")
            w.writerow([model, f"{elo_pred:.2f}", f"{eo:.2f}" if eo != "" else ""])
    print(f"Wrote {out_path} ({len(model_to_elo)} models)")


def make_scatter_plot_streaming(
    our_elo: dict,
    ref_elo: dict,
    split_name: str,
    ref_name: str,
    output_path: str,
    highlight_models: set[str],
) -> None:
    """
    Scatter original vs reconstructed ELO; highlight models in `highlight_models`
    (streaming tier aligned with this split: train/val/test tier).
    """
    if plt is None:
        print(f"  Skipping scatter ({output_path}): matplotlib not installed")
        return
    common = sorted(set(our_elo.keys()) & set(ref_elo.keys()))
    if len(common) < 2:
        print(f"  Skipping scatter: < 2 common models for {ref_name}/{split_name}")
        return

    hi_models = sorted(m for m in common if m in highlight_models)
    base = sorted(m for m in common if m not in highlight_models)

    fig, ax = plt.subplots(figsize=(9, 9))

    if base:
        rx = np.array([ref_elo[m] for m in base])
        oy = np.array([our_elo[m] for m in base])
        ax.scatter(
            rx,
            oy,
            alpha=0.55,
            s=36,
            c="0.45",
            edgecolors="k",
            linewidths=0.25,
            label=f"Other models ({len(base)})",
        )
    if hi_models:
        rxh = np.array([ref_elo[m] for m in hi_models])
        oyh = np.array([our_elo[m] for m in hi_models])
        tier_label = {
            "train": "train-tier",
            "val": "val-tier",
            "test": "test-tier",
        }.get(split_name, "tier")
        ax.scatter(
            rxh,
            oyh,
            alpha=0.9,
            s=95,
            c="tab:orange",
            edgecolors="darkred",
            linewidths=0.7,
            marker="s",
            label=f"{tier_label} ({len(hi_models)})",
            zorder=5,
        )

    ref_vals = np.array([ref_elo[m] for m in common])
    our_vals = np.array([our_elo[m] for m in common])
    lo = min(ref_vals.min(), our_vals.min())
    hi_lim = max(ref_vals.max(), our_vals.max())
    margin = (hi_lim - lo) * 0.05
    ax.plot(
        [lo - margin, hi_lim + margin],
        [lo - margin, hi_lim + margin],
        "r--",
        linewidth=1,
        label="y = x",
    )

    ax.set_xlabel(f"Original ELO ({ref_name})")
    ax.set_ylabel("Reconstructed ELO")
    ax.set_title(
        f"{split_name}: Original vs Reconstructed ({ref_name})\n"
        f"{len(common)} models — orange squares = streaming {split_name}-tier models"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plot: {output_path}")


def main():
    args = parse_args()
    kind = args.dataset
    pred_root = predictions_dir(kind)
    os.makedirs(pred_root, exist_ok=True)
    plots_dir = os.path.join(pred_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tiers_meta = load_model_tiers(kind)

    index_to_model = rep.load_index_to_model()
    ref_i2v = rep.load_reference_elo(I2V_REF) if I2V_REF.exists() else {}
    ref_t2v = rep.load_reference_elo(T2V_REF) if T2V_REF.exists() else {}
    refs = [r for r in (ref_i2v, ref_t2v) if r]
    ref_combined = rep.combine_references(*refs) if refs else {}

    elos_per_split: dict[str, dict] = {}
    k_scale = rep.K

    for split in ["train", "val", "test"]:
        p = os.path.join(pred_root, f"{split}_preds.csv")
        if not os.path.isfile(p):
            print(f"Missing {p}; skip {split}")
            continue
        df = pd.read_csv(p)
        if kind == "both":
            df = filter_aa_aa(df)
        if df is None or len(df) == 0:
            print(f"Skip {split}: no AA×AA rows")
            continue

        rows = df_to_prob_rows(df)
        model_to_elo = rep.build_elo_from_prob_rows(rows, index_to_model, ref_combined, k=k_scale)
        out_csv = os.path.join(pred_root, f"elo_{split}.csv")
        write_elo_csv(model_to_elo, ref_combined, out_csv)
        elos_per_split[split] = model_to_elo

    corr_rows = rep.compute_correlations(elos_per_split, ref_i2v, ref_t2v)
    rep.print_correlations(elos_per_split, ref_i2v, ref_t2v)

    corr_csv_path = os.path.join(pred_root, "elo_correlations.csv")
    fieldnames = [
        "split",
        "reference",
        "n_intersection",
        "spearman",
        "kendall",
        "top5_overlap",
        "top10_overlap",
        "median_abs_rank_err",
    ]
    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in corr_rows:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames}
            writer.writerow(row)
    corr_json_path = os.path.join(pred_root, "elo_correlations.json")
    with open(corr_json_path, "w") as f:
        json.dump(corr_rows, f, indent=2)
    print(f"Saved correlation metrics: {corr_csv_path}")

    for split_name, our_elo in elos_per_split.items():
        hi = tier_highlight_for_split(split_name, tiers_meta)
        for ref_label, ref_elo in [("i2v", ref_i2v), ("t2v", ref_t2v)]:
            if not ref_elo:
                continue
            make_scatter_plot_streaming(
                our_elo,
                ref_elo,
                split_name,
                ref_label,
                os.path.join(plots_dir, f"scatter_{split_name}_{ref_label}.png"),
                hi,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

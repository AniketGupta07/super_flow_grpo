#!/usr/bin/env python3
"""
Reconstruct scaled per-model ELO from prompt PairMLP predictions on AA×AA pairs.

- aa / both: use existing predictions-prompt-{kind}/{train,val,test}_preds.csv.
  For "both", only rows with dataset1==aa and dataset2==aa are used.
- cross / social / social_fixed: the default training predictions are not AA×AA (cross is
  aa×social; social and social_fixed are social×social). This script runs the saved
  checkpoint on prompt_splits-aa and writes aa_eval_{split}_preds.csv, then reconstructs
  ELO from those.

Outputs (under predictions-prompt-{kind}/): elo_{split}.csv, plots/scatter_* .png
Same scaling as reconstruct_elo_from_probs.py (k * logit(p̂), affine-aligned to reference).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import train_prompt_pair_mlp as tpm  # noqa: E402
import reconstruct_elo_from_probs as rep  # noqa: E402

# Re-use reference paths from reconstruct_elo_from_probs
I2V_REF = rep.I2V_REF
T2V_REF = rep.T2V_REF


def parse_args():
    p = argparse.ArgumentParser(description="ELO from prompt MLP preds on AA×AA pairs.")
    p.add_argument(
        "--model-kind",
        type=str,
        required=True,
        choices=["aa", "both", "cross", "social", "social_fixed"],
        help="Which trained models-prompt-* checkpoint to use for inference when needed.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="Checkpoint file under models-prompt-{kind}/.",
    )
    p.add_argument(
        "--force-infer",
        action="store_true",
        help="For cross/social/social_fixed: re-run inference on AA splits even if aa_eval_* CSVs exist.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size when running inference for cross/social/social_fixed.",
    )
    return p.parse_args()


def predictions_dir_for(kind: str) -> str:
    return os.path.join(_SCRIPT_DIR, f"predictions-prompt-{kind}")


def filter_aa_aa(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["dataset1"] == "aa") & (df["dataset2"] == "aa")].copy()


def df_to_prob_rows(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    """(index1, index2, y_pred) for ELO reconstruction."""
    return list(
        zip(
            df["index1"].astype(np.int64),
            df["index2"].astype(np.int64),
            df["y_pred"].astype(np.float64),
        )
    )


def _load_model_impl(kind: str, checkpoint_name: str) -> tuple[tpm.PairMLP, dict]:
    model_dir = os.path.join(_SCRIPT_DIR, f"models-prompt-{kind}")
    cfg_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, checkpoint_name)
    with open(cfg_path) as f:
        config = json.load(f)
    model = tpm.PairMLP(config["input_dim"], config["hidden_dims"], config.get("dropout", tpm.DROPOUT))
    model = model.to(tpm.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=tpm.DEVICE, weights_only=True))
    return model, config


def load_checkpoint(kind: str, checkpoint_name: str) -> tuple[tpm.PairMLP, dict]:
    return _load_model_impl(kind, checkpoint_name)


def build_aa_dataset(split_name: str, aa_vision, social_vision, prompt_encodings, aa_idx_to_prompt, social_idx_to_prompt):
    splits_dir = os.path.join(_SCRIPT_DIR, "prompt_splits-aa")
    x_path = os.path.join(splits_dir, f"prob_X_{split_name}.csv")
    y_path = os.path.join(splits_dir, f"prob_Y_{split_name}.csv")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        return None
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    ds_kwargs = dict(
        aa_vision=aa_vision,
        social_vision=social_vision,
        prompt_encodings=prompt_encodings,
        aa_idx_to_prompt=aa_idx_to_prompt,
        social_idx_to_prompt=social_idx_to_prompt,
    )
    return tpm.PromptPairDataset(X, Y, **ds_kwargs)


def run_inference_on_aa_splits(
    model_kind: str,
    checkpoint_name: str,
    out_dir: str,
    batch_size: int,
    force: bool,
) -> None:
    """Write aa_eval_{train,val,test}_preds.csv with same columns as train_prompt_pair_mlp."""
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    pending = []
    for sp in splits:
        out_path = os.path.join(out_dir, f"aa_eval_{sp}_preds.csv")
        if force or not os.path.isfile(out_path):
            pending.append(sp)
    if not pending:
        print("AA eval prediction files already exist; use --force-infer to overwrite.")
        return

    data_dir = os.path.join(_SCRIPT_DIR, "data")
    mapping_csv = os.path.join(data_dir, "mapping.csv")
    print("Loading encodings for inference on prompt_splits-aa ...", flush=True)
    aa_vision = tpm.load_aa_vision_encodings(data_dir)
    social_vision = tpm.load_social_vision_encodings(data_dir)
    prompt_encodings = tpm.load_prompt_encodings(data_dir, mapping_csv)
    aa_idx_to_prompt, social_idx_to_prompt = tpm.build_video_to_prompt(data_dir)

    model, _ = load_checkpoint(model_kind, checkpoint_name)
    model.eval()
    criterion = nn.BCELoss()

    for sp in pending:
        dataset = build_aa_dataset(
            sp, aa_vision, social_vision, prompt_encodings, aa_idx_to_prompt, social_idx_to_prompt
        )
        if dataset is None or len(dataset) == 0:
            print(f"  Skip aa_eval_{sp}: no data", flush=True)
            continue
        n = len(dataset)
        bs = max(1, min(batch_size, n))
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        _, _, _, _, preds = tpm.evaluate(model, loader, criterion)
        df = pd.DataFrame({
            "dataset1": dataset.ds1,
            "index1": dataset.idx1,
            "dataset2": dataset.ds2,
            "index2": dataset.idx2,
            "y_true": dataset.probs,
            "y_pred": preds,
        })
        out_path = os.path.join(out_dir, f"aa_eval_{sp}_preds.csv")
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} ({len(df)} rows)", flush=True)


def write_elo_csv(model_to_elo: dict, ref_original: dict, out_path: str) -> None:
    sorted_models = sorted(model_to_elo.keys(), key=lambda m: model_to_elo[m], reverse=True)
    import csv

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "elo", "elo_original"])
        for model in sorted_models:
            elo_pred = model_to_elo[model]
            eo = ref_original.get(model, "")
            w.writerow([model, f"{elo_pred:.2f}", f"{eo:.2f}" if eo != "" else ""])
    print(f"Wrote {out_path} ({len(model_to_elo)} models)")


def main():
    args = parse_args()
    tpm.set_seed(tpm.SEED)
    kind = args.model_kind
    pred_root = predictions_dir_for(kind)
    os.makedirs(pred_root, exist_ok=True)
    plots_dir = os.path.join(pred_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # cross/social/social_fixed: ensure checkpoint exists before infer
    if kind in ("cross", "social", "social_fixed"):
        ckpt = os.path.join(_SCRIPT_DIR, f"models-prompt-{kind}", args.checkpoint)
        if not os.path.isfile(ckpt):
            print(f"Missing checkpoint {ckpt}; train {kind} first or fix --checkpoint.")
            sys.exit(1)
        run_inference_on_aa_splits(
            kind, args.checkpoint, pred_root, args.batch_size, force=args.force_infer
        )

    index_to_model = rep.load_index_to_model()
    ref_i2v = rep.load_reference_elo(I2V_REF) if I2V_REF.exists() else {}
    ref_t2v = rep.load_reference_elo(T2V_REF) if T2V_REF.exists() else {}
    refs = [r for r in (ref_i2v, ref_t2v) if r]
    ref_combined = rep.combine_references(*refs) if refs else {}

    elos_per_split: dict[str, dict] = {}
    k_scale = rep.K

    for split in ["train", "val", "test"]:
        if kind in ("cross", "social", "social_fixed"):
            aa_eval_path = os.path.join(pred_root, f"aa_eval_{split}_preds.csv")
            if not os.path.isfile(aa_eval_path):
                print(f"Missing {aa_eval_path}; skip {split}")
                continue
            df = pd.read_csv(aa_eval_path)
        else:
            p = os.path.join(pred_root, f"{split}_preds.csv")
            if not os.path.isfile(p):
                print(f"Missing {p}; skip {split}")
                continue
            df = pd.read_csv(p)
            if kind == "both":
                df = filter_aa_aa(df)
        if df is None or len(df) == 0:
            print(f"Skip {split}: no rows")
            continue

        rows = df_to_prob_rows(df)
        model_to_elo = rep.build_elo_from_prob_rows(rows, index_to_model, ref_combined, k=k_scale)
        out_csv = os.path.join(pred_root, f"elo_{split}.csv")
        write_elo_csv(model_to_elo, ref_combined, out_csv)
        elos_per_split[split] = model_to_elo

    corr_rows = rep.compute_correlations(elos_per_split, ref_i2v, ref_t2v)
    rep.print_correlations(elos_per_split, ref_i2v, ref_t2v)

    # Save correlation metrics (Spearman/Kendall/top-k overlap) to predictions directory.
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
    import csv

    with open(corr_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in corr_rows:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames}
            writer.writerow(row)
    corr_json_path = os.path.join(pred_root, "elo_correlations.json")
    with open(corr_json_path, "w") as f:
        json.dump(corr_rows, f, indent=2)
    print(f"Saved correlation metrics: {corr_csv_path} and {corr_json_path}")

    for split_name, our_elo in elos_per_split.items():
        for ref_label, ref_elo in [("i2v", ref_i2v), ("t2v", ref_t2v)]:
            if not ref_elo:
                continue
            rep.make_scatter_plot(
                our_elo,
                ref_elo,
                split_name,
                ref_label,
                os.path.join(plots_dir, f"scatter_{split_name}_{ref_label}.png"),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

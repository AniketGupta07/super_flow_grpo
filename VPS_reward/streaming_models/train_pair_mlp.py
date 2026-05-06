"""
Train the prompt PairMLP on streaming_model splits (encoding_splits-{aa,both}-streaming_model).

Mirrors ../train_prompt_pair_mlp.py (architecture, loss weighting for both) but:
  --dataset aa | both
  Data under ../data/
  Splits under ./encoding_splits-{dataset}-streaming_model/
  Checkpoints ./models-streaming-{dataset}/
  Predictions ./predictions-streaming-{dataset}/  (train_preds.csv, val_preds.csv, test_preds.csv)

For aa splits (prob_X has index1/index2 only), rows are wrapped as AA×AA before loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_EMBED_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EMBED_DIR not in sys.path:
    sys.path.insert(0, _EMBED_DIR)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_prompt_pair_mlp as tpm


def parse_args():
    p = argparse.ArgumentParser(description="Train PairMLP on streaming_model splits.")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aa", "both"],
        help="Must match encoding_splits-{dataset}-streaming_model from create_prob_splits.py.",
    )
    return p.parse_args()


def ensure_aa_dual_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Streaming aa splits store index1,index2 only; PromptPairDataset needs dataset tags."""
    cols = set(X.columns.str.lower())
    if "dataset1" in cols and "dataset2" in cols:
        return X
    return pd.DataFrame(
        {
            "dataset1": "aa",
            "index1": X["index1"].astype(int),
            "dataset2": "aa",
            "index2": X["index2"].astype(int),
        }
    )


def main():
    args = parse_args()
    tpm.set_seed(tpm.SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_EMBED_DIR, "data")
    splits_dir = os.path.join(script_dir, f"encoding_splits-{args.dataset}-streaming_model")
    output_dir = os.path.join(script_dir, f"models-streaming-{args.dataset}")
    predictions_dir = os.path.join(script_dir, f"predictions-streaming-{args.dataset}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"Using device: {tpm.DEVICE} (seed={tpm.SEED})")
    print(f"Streaming dataset: {args.dataset}")
    print(f"Splits dir: {splits_dir}")

    print("\nLoading encodings...")
    aa_vision: dict = {}
    social_vision: dict = {}

    if args.dataset in ("aa", "both"):
        aa_vision = tpm.load_aa_vision_encodings(data_dir)
    if args.dataset == "both":
        social_vision = tpm.load_social_vision_encodings(data_dir)

    prompt_encodings = tpm.load_prompt_encodings(
        data_dir, os.path.join(data_dir, "mapping.csv")
    )
    aa_idx_to_prompt, social_idx_to_prompt = tpm.build_video_to_prompt(data_dir)

    sample_vision = next(iter(aa_vision.values())) if aa_vision else next(iter(social_vision.values()))
    vision_dim = len(sample_vision)
    prompt_dim = len(next(iter(prompt_encodings.values())))
    input_dim = vision_dim * 2 + prompt_dim
    print(f"\nInput dim: {input_dim} (vision1={vision_dim} + vision2={vision_dim} + prompt={prompt_dim})")

    print("\nLoading splits...")
    X_train = ensure_aa_dual_columns(pd.read_csv(os.path.join(splits_dir, "prob_X_train.csv")))
    X_val = ensure_aa_dual_columns(pd.read_csv(os.path.join(splits_dir, "prob_X_val.csv")))
    X_test = ensure_aa_dual_columns(pd.read_csv(os.path.join(splits_dir, "prob_X_test.csv")))
    Y_train = pd.read_csv(os.path.join(splits_dir, "prob_Y_train.csv"))
    Y_val = pd.read_csv(os.path.join(splits_dir, "prob_Y_val.csv"))
    Y_test = pd.read_csv(os.path.join(splits_dir, "prob_Y_test.csv"))
    print(f"  Train: {len(X_train)} pairs")
    print(f"  Val:   {len(X_val)} pairs")
    print(f"  Test:  {len(X_test)} pairs")

    print("\nCreating datasets...")
    ds_kwargs_base = dict(
        aa_vision=aa_vision,
        social_vision=social_vision,
        prompt_encodings=prompt_encodings,
        aa_idx_to_prompt=aa_idx_to_prompt,
        social_idx_to_prompt=social_idx_to_prompt,
    )
    train_aa_w = tpm.AA_PAIR_LOSS_WEIGHT if args.dataset == "both" else 1.0
    if train_aa_w != 1.0:
        print(f"  AA×AA pair loss weight (train only): {train_aa_w}x vs other pairs")

    train_dataset = tpm.PromptPairDataset(
        X_train, Y_train, **ds_kwargs_base, aa_pair_weight=train_aa_w
    )
    val_dataset = tpm.PromptPairDataset(X_val, Y_val, **ds_kwargs_base)
    test_dataset = tpm.PromptPairDataset(X_test, Y_test, **ds_kwargs_base)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("\nError: No training samples.")
        return

    g = torch.Generator()
    g.manual_seed(tpm.SEED)
    bs = min(tpm.BATCH_SIZE, len(train_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=0, generator=g
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0)
        if len(val_dataset) > 0
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
        if len(test_dataset) > 0
        else None
    )

    model = tpm.PairMLP(input_dim, tpm.HIDDEN_DIMS, tpm.DROPOUT).to(tpm.DEVICE)
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tpm.LEARNING_RATE, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    weighted_train = args.dataset == "both"
    print("\nTraining...")
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(tpm.EPOCHS):
        train_loss = tpm.train_epoch(
            model, train_loader, optimizer, criterion, weighted=weighted_train
        )

        if val_loader is not None:
            val_loss, val_mae, val_corr, val_acc, _ = tpm.evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            print(
                f"Epoch {epoch+1:3d}/{tpm.EPOCHS}: Train Loss={train_loss:.6f}, "
                f"Val Loss={val_loss:.6f}, Val MAE={val_mae:.4f}, Val Corr={val_corr:.4f}, Val Acc={val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
            else:
                patience_counter += 1
                if patience_counter >= tpm.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        else:
            scheduler.step(train_loss)
            print(f"Epoch {epoch+1:3d}/{tpm.EPOCHS}: Train Loss={train_loss:.6f}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))

    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best.pth"), weights_only=True)
    )

    print("\n" + "=" * 50)
    print("Final Evaluation (Best Model)")
    print("=" * 50)

    train_eval_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0)
    train_loss, train_mae, train_corr, train_acc, train_preds = tpm.evaluate(
        model, train_eval_loader, criterion
    )
    print(f"  Train: Loss={train_loss:.6f}, MAE={train_mae:.4f}, Corr={train_corr:.4f}, Acc={train_acc:.4f}")
    if args.dataset == "both":
        print(f"    Per-category Acc: {tpm.per_category_accuracy(train_dataset, train_preds)}")

    val_preds, test_preds = None, None
    if val_loader is not None:
        val_loss, val_mae, val_corr, val_acc, val_preds = tpm.evaluate(model, val_loader, criterion)
        print(f"  Val:   Loss={val_loss:.6f}, MAE={val_mae:.4f}, Corr={val_corr:.4f}, Acc={val_acc:.4f}")
        if args.dataset == "both":
            print(f"    Per-category Acc: {tpm.per_category_accuracy(val_dataset, val_preds)}")

    if test_loader is not None:
        test_loss, test_mae, test_corr, test_acc, test_preds = tpm.evaluate(
            model, test_loader, criterion
        )
        print(f"  Test:  Loss={test_loss:.6f}, MAE={test_mae:.4f}, Corr={test_corr:.4f}, Acc={test_acc:.4f}")
        if args.dataset == "both":
            print(f"    Per-category Acc: {tpm.per_category_accuracy(test_dataset, test_preds)}")

    print("\nSaving predictions...")

    def save_preds(dataset, preds, name):
        df = pd.DataFrame(
            {
                "dataset1": dataset.ds1,
                "index1": dataset.idx1,
                "dataset2": dataset.ds2,
                "index2": dataset.idx2,
                "y_true": dataset.probs,
                "y_pred": preds,
            }
        )
        path = os.path.join(predictions_dir, f"{name}_preds.csv")
        df.to_csv(path, index=False)
        print(f"  {path}")

    save_preds(train_dataset, train_preds, "train")
    if val_preds is not None and len(val_dataset) > 0:
        save_preds(val_dataset, val_preds, "val")
    if test_preds is not None and len(test_dataset) > 0:
        save_preds(test_dataset, test_preds, "test")

    torch.save(model.state_dict(), os.path.join(output_dir, "final.pth"))
    config = {
        "dataset": args.dataset,
        "split_source": "streaming_model",
        "input_dim": input_dim,
        "hidden_dims": tpm.HIDDEN_DIMS,
        "dropout": tpm.DROPOUT,
        "vision_dim": vision_dim,
        "prompt_dim": prompt_dim,
        "aa_pair_loss_weight_train": train_aa_w,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved model to {output_dir}/ (best.pth, final.pth, config.json)")


if __name__ == "__main__":
    main()

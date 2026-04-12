"""
Script to train an MLP to predict win probability for video pairs,
using prompt-based splits from create_prompt_splits.py.

Handles four dataset modes:
- aa:     Both videos are from the AA benchmark.
          Vision encodings from VIDEO_ONLY_EMBEDDINGS-4.json (keyed by dataset_index).
- social: Both videos are from social media.
          Vision encodings from AA_VISION_EMBEDDINGS_fps4_maxf32_part{1..5}.json (keyed by idx).
- both:   Videos can come from either dataset. Each pair's (dataset, index) tags
          determine which vision encoding dictionary to use. Training uses 10× loss
          weight on AA×AA pairs vs social×social and cross pairs.
- cross:  Only AA×Social cross-dataset pairs. Loads both vision encoding sets.

Input features per pair:
  [vision1 (3584) | vision2 (3584) | prompt (384)] = 7552 dim

Output: predicted probability (single value, P(video1 wins)).
"""

import os
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS = [8192, 4096, 2048, 1024, 512, 256, 64, 4]
DROPOUT = 0.35
LEARNING_RATE = 1e-5
BATCH_SIZE = 256
EPOCHS = 1000
PATIENCE = 25
# When --dataset both, multiply loss weight for AA×AA pairs (vs social×social and cross).
AA_PAIR_LOSS_WEIGHT = 10.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True


def load_aa_vision_encodings(data_dir: str) -> dict[int, np.ndarray]:
    """Load AA benchmark vision encodings. Keyed by dataset_index (1-based)."""
    path = os.path.join(data_dir, "VIDEO_ONLY_EMBEDDINGS-4.json")
    print(f"  Loading AA vision encodings from {os.path.basename(path)}...")
    with open(path) as f:
        data = json.load(f)
    encodings = {}
    for item in data:
        v = item.get("v1")
        if v is not None:
            encodings[item["index"]] = np.array(v, dtype=np.float32)
    print(f"    {len(encodings)} AA vision encodings (dim={len(next(iter(encodings.values())))})")
    return encodings


def load_social_vision_encodings(data_dir: str) -> dict[int, np.ndarray]:
    """Load social media vision encodings from all part files. Keyed by idx (0-based)."""
    encodings = {}
    for part in range(1, 6):
        path = os.path.join(data_dir, f"AA_VISION_EMBEDDINGS_fps4_maxf32_part{part}.json")
        if not os.path.exists(path):
            continue
        print(f"  Loading social vision encodings from part {part}...")
        with open(path) as f:
            data = json.load(f)
        count = 0
        for item in data:
            v = item.get("v1")
            if v is not None:
                encodings[item["idx"]] = np.array(v, dtype=np.float32)
                count += 1
        print(f"    Part {part}: {count} valid encodings out of {len(data)}")
    print(f"    Total social vision encodings: {len(encodings)}")
    return encodings


def load_prompt_encodings(data_dir: str, mapping_path: str) -> dict[int, np.ndarray]:
    """
    Load prompt encodings and build a prompt_line → embedding mapping.
    SBERT_PROMPT_EMBEDDINGS.json is keyed by AA dataset_index; we use mapping.csv
    to group by prompt_line and take the first available embedding per prompt.
    """
    emb_path = os.path.join(data_dir, "SBERT_PROMPT_EMBEDDINGS.json")
    print(f"  Loading prompt encodings from {os.path.basename(emb_path)}...")
    with open(emb_path) as f:
        data = json.load(f)
    index_to_embed = {item["index"]: np.array(item["prompt_embed"], dtype=np.float32) for item in data}

    mapping = pd.read_csv(mapping_path)
    prompt_encodings: dict[int, np.ndarray] = {}
    for _, row in mapping.iterrows():
        pl = int(row.prompt_line)
        if pl not in prompt_encodings and row.dataset_index in index_to_embed:
            prompt_encodings[pl] = index_to_embed[row.dataset_index]

    dim = len(next(iter(prompt_encodings.values())))
    print(f"    {len(prompt_encodings)} prompt encodings (dim={dim})")
    return prompt_encodings


def build_video_to_prompt(data_dir: str) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build index→prompt_line mappings for both datasets.
    Returns (aa_idx_to_prompt, social_idx_to_prompt).
    """
    mapping = pd.read_csv(os.path.join(data_dir, "mapping.csv"))
    aa_map = dict(zip(mapping.dataset_index.astype(int), mapping.prompt_line.astype(int)))

    seg_path = os.path.join(data_dir, "aa_final_video_segments.json")
    with open(seg_path) as f:
        segments = json.load(f)
    social_map = {int(item["idx"]): int(item["prompt_idx"]) for item in segments}

    return aa_map, social_map


class PromptPairDataset(Dataset):
    """Dataset that handles mixed aa/social pairs via (dataset, index) tags."""

    def __init__(
        self,
        X_df: pd.DataFrame,
        Y_df: pd.DataFrame,
        aa_vision: dict[int, np.ndarray],
        social_vision: dict[int, np.ndarray],
        prompt_encodings: dict[int, np.ndarray],
        aa_idx_to_prompt: dict[int, int],
        social_idx_to_prompt: dict[int, int],
        aa_pair_weight: float = 1.0,
    ):
        self.aa_vision = aa_vision
        self.social_vision = social_vision
        self.prompt_encodings = prompt_encodings
        self.aa_idx_to_prompt = aa_idx_to_prompt
        self.social_idx_to_prompt = social_idx_to_prompt

        ds1 = X_df["dataset1"].values
        idx1 = X_df["index1"].values.astype(int)
        ds2 = X_df["dataset2"].values
        idx2 = X_df["index2"].values.astype(int)
        probs = Y_df["probability"].values.astype(np.float32)

        valid = []
        for i in range(len(ds1)):
            v1 = self._get_vision(ds1[i], idx1[i])
            v2 = self._get_vision(ds2[i], idx2[i])
            pl = self._get_prompt_line(ds1[i], idx1[i])
            if v1 is not None and v2 is not None and pl is not None and pl in prompt_encodings:
                valid.append(i)

        if len(valid) < len(ds1):
            print(f"  Warning: Filtered {len(ds1) - len(valid)} pairs (missing encodings)")

        self.ds1 = ds1[valid]
        self.idx1 = idx1[valid]
        self.ds2 = ds2[valid]
        self.idx2 = idx2[valid]
        self.probs = probs[valid]

        if aa_pair_weight != 1.0:
            aa_mask = (self.ds1 == "aa") & (self.ds2 == "aa")
            self.weights = np.where(aa_mask, aa_pair_weight, 1.0).astype(np.float32)
        else:
            self.weights = np.ones(len(self.probs), dtype=np.float32)

    def _get_vision(self, dataset: str, index: int):
        if dataset == "aa":
            return self.aa_vision.get(index)
        return self.social_vision.get(index)

    def _get_prompt_line(self, dataset: str, index: int):
        if dataset == "aa":
            return self.aa_idx_to_prompt.get(index)
        return self.social_idx_to_prompt.get(index)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, i):
        v1 = self._get_vision(self.ds1[i], self.idx1[i])
        v2 = self._get_vision(self.ds2[i], self.idx2[i])
        pl = self._get_prompt_line(self.ds1[i], self.idx1[i])
        prompt = self.prompt_encodings[pl]

        features = np.concatenate([v1, v2, prompt])
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(self.probs[i], dtype=torch.float32),
            torch.tensor(self.weights[i], dtype=torch.float32),
        )


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, weighted: bool):
    model.train()
    total_loss = 0.0
    total_w = 0.0
    for features, targets, weights in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        weights = weights.to(DEVICE)
        optimizer.zero_grad()
        out = model(features)
        if weighted:
            loss_per = torch.nn.functional.binary_cross_entropy(out, targets, reduction="none")
            wsum = weights.sum().clamp_min(1e-8)
            loss = (loss_per * weights).sum() / wsum
            total_loss += (loss_per.detach() * weights).sum().item()
            total_w += wsum.item()
        else:
            loss = criterion(out, targets)
            total_loss += loss.item() * len(features)
        loss.backward()
        optimizer.step()
    if weighted:
        return total_loss / max(total_w, 1e-8)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets, _weights in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            out = model(features)
            loss = criterion(out, targets)
            total_loss += loss.item() * len(features)
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.array(all_preds)
    tgts = np.array(all_targets)
    mae = np.mean(np.abs(preds - tgts))
    corr = np.corrcoef(preds, tgts)[0, 1] if len(preds) > 1 else 0.0
    acc = np.mean((preds >= 0.5) == (tgts >= 0.5)) if len(preds) > 0 else 0.0
    return avg_loss, mae, corr, acc, preds


def per_category_accuracy(dataset, preds):
    """Compute accuracy broken down by pair type: aa, social, cross."""
    ds1 = dataset.ds1
    ds2 = dataset.ds2
    tgts = dataset.probs
    correct = (preds >= 0.5) == (tgts >= 0.5)

    aa_mask = (ds1 == "aa") & (ds2 == "aa")
    social_mask = (ds1 == "social") & (ds2 == "social")
    cross_mask = ((ds1 == "aa") & (ds2 == "social")) | ((ds1 == "social") & (ds2 == "aa"))

    parts = []
    for name, mask in [("AA", aa_mask), ("Social", social_mask), ("Cross", cross_mask)]:
        n = mask.sum()
        if n > 0:
            parts.append(f"{name}={correct[mask].mean():.4f} ({n})")
    return ", ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP on prompt-based video pair splits."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aa", "social", "both", "cross"],
        help="Which dataset splits to train on (must match create_prompt_splits.py output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    splits_dir = os.path.join(script_dir, f"prompt_splits-{args.dataset}")
    output_dir = os.path.join(script_dir, f"models-prompt-{args.dataset}")
    predictions_dir = os.path.join(script_dir, f"predictions-prompt-{args.dataset}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"Using device: {DEVICE} (seed={SEED})")
    print(f"Dataset mode: {args.dataset}")
    print(f"Splits dir: {splits_dir}")

    # --- Load encodings ---
    print("\nLoading encodings...")
    aa_vision: dict[int, np.ndarray] = {}
    social_vision: dict[int, np.ndarray] = {}

    if args.dataset in ("aa", "both", "cross"):
        aa_vision = load_aa_vision_encodings(data_dir)
    if args.dataset in ("social", "both", "cross"):
        social_vision = load_social_vision_encodings(data_dir)

    prompt_encodings = load_prompt_encodings(
        data_dir, os.path.join(data_dir, "mapping.csv")
    )
    aa_idx_to_prompt, social_idx_to_prompt = build_video_to_prompt(data_dir)

    # Determine dimensions from whichever dict is populated
    sample_vision = next(iter(aa_vision.values())) if aa_vision else next(iter(social_vision.values()))
    vision_dim = len(sample_vision)
    prompt_dim = len(next(iter(prompt_encodings.values())))
    input_dim = vision_dim * 2 + prompt_dim
    print(f"\nInput dim: {input_dim} (vision1={vision_dim} + vision2={vision_dim} + prompt={prompt_dim})")

    # --- Load splits ---
    print("\nLoading splits...")
    X_train = pd.read_csv(os.path.join(splits_dir, "prob_X_train.csv"))
    X_val = pd.read_csv(os.path.join(splits_dir, "prob_X_val.csv"))
    X_test = pd.read_csv(os.path.join(splits_dir, "prob_X_test.csv"))
    Y_train = pd.read_csv(os.path.join(splits_dir, "prob_Y_train.csv"))
    Y_val = pd.read_csv(os.path.join(splits_dir, "prob_Y_val.csv"))
    Y_test = pd.read_csv(os.path.join(splits_dir, "prob_Y_test.csv"))
    print(f"  Train: {len(X_train)} pairs")
    print(f"  Val:   {len(X_val)} pairs")
    print(f"  Test:  {len(X_test)} pairs")

    # --- Create datasets ---
    print("\nCreating datasets...")
    ds_kwargs_base = dict(
        aa_vision=aa_vision,
        social_vision=social_vision,
        prompt_encodings=prompt_encodings,
        aa_idx_to_prompt=aa_idx_to_prompt,
        social_idx_to_prompt=social_idx_to_prompt,
    )
    train_aa_w = AA_PAIR_LOSS_WEIGHT if args.dataset == "both" else 1.0
    if train_aa_w != 1.0:
        print(f"  AA×AA pair loss weight (train only): {train_aa_w}x vs other pairs")
    train_dataset = PromptPairDataset(
        X_train, Y_train, **ds_kwargs_base, aa_pair_weight=train_aa_w
    )
    val_dataset = PromptPairDataset(X_val, Y_val, **ds_kwargs_base)
    test_dataset = PromptPairDataset(X_test, Y_test, **ds_kwargs_base)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("\nError: No training samples! Check encoding files.")
        return

    # --- Dataloaders ---
    g = torch.Generator()
    g.manual_seed(SEED)
    bs = min(BATCH_SIZE, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0) if len(test_dataset) > 0 else None

    # --- Model ---
    model = PairMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # --- Training loop ---
    print("\nTraining...")
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, weighted=(args.dataset == "both")
        )

        if val_loader is not None:
            val_loss, val_mae, val_corr, val_acc, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS}: Train Loss={train_loss:.6f}, "
                f"Val Loss={val_loss:.6f}, Val MAE={val_mae:.4f}, Val Corr={val_corr:.4f}, Val Acc={val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        else:
            scheduler.step(train_loss)
            print(f"Epoch {epoch+1:3d}/{EPOCHS}: Train Loss={train_loss:.6f}")
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))

    # --- Final evaluation ---
    model.load_state_dict(torch.load(os.path.join(output_dir, "best.pth"), weights_only=True))

    print("\n" + "=" * 50)
    print("Final Evaluation (Best Model)")
    print("=" * 50)

    train_eval_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0)
    train_loss, train_mae, train_corr, train_acc, train_preds = evaluate(model, train_eval_loader, criterion)
    print(f"  Train: Loss={train_loss:.6f}, MAE={train_mae:.4f}, Corr={train_corr:.4f}, Acc={train_acc:.4f}")
    if args.dataset == "both":
        print(f"    Per-category Acc: {per_category_accuracy(train_dataset, train_preds)}")

    val_preds, test_preds = None, None
    if val_loader is not None:
        val_loss, val_mae, val_corr, val_acc, val_preds = evaluate(model, val_loader, criterion)
        print(f"  Val:   Loss={val_loss:.6f}, MAE={val_mae:.4f}, Corr={val_corr:.4f}, Acc={val_acc:.4f}")
        if args.dataset == "both":
            print(f"    Per-category Acc: {per_category_accuracy(val_dataset, val_preds)}")

    if test_loader is not None:
        test_loss, test_mae, test_corr, test_acc, test_preds = evaluate(model, test_loader, criterion)
        print(f"  Test:  Loss={test_loss:.6f}, MAE={test_mae:.4f}, Corr={test_corr:.4f}, Acc={test_acc:.4f}")
        if args.dataset == "both":
            print(f"    Per-category Acc: {per_category_accuracy(test_dataset, test_preds)}")

    # --- Save predictions ---
    print("\nSaving predictions...")

    def save_preds(dataset, preds, name):
        df = pd.DataFrame({
            "dataset1": dataset.ds1,
            "index1": dataset.idx1,
            "dataset2": dataset.ds2,
            "index2": dataset.idx2,
            "y_true": dataset.probs,
            "y_pred": preds,
        })
        path = os.path.join(predictions_dir, f"{name}_preds.csv")
        df.to_csv(path, index=False)
        print(f"  {path}")

    save_preds(train_dataset, train_preds, "train")
    if val_preds is not None and len(val_dataset) > 0:
        save_preds(val_dataset, val_preds, "val")
    if test_preds is not None and len(test_dataset) > 0:
        save_preds(test_dataset, test_preds, "test")

    # --- Save model + config ---
    torch.save(model.state_dict(), os.path.join(output_dir, "final.pth"))
    config = {
        "dataset": args.dataset,
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "vision_dim": vision_dim,
        "prompt_dim": prompt_dim,
        "aa_pair_loss_weight_train": train_aa_w,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved model to {output_dir}/ (best.pth, final.pth, config.json)")


if __name__ == "__main__":
    main()

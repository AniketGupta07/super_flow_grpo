#!/usr/bin/env python3
"""
Train two regressors on metric combinations from X.csv using prompt_splits-both labels:
1) Linear regression
2) MLP regression (PyTorch)

Outputs:
- linear/: model, metrics, prob_{train,val,test}.csv
- mlp/: model, scaler, metrics, prob_{train,val,test}.csv
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def _find_api_test_root() -> Path:
    here = Path(__file__).resolve().parent
    d = here
    for _ in range(8):
        if (d / "EMBEDDING_MODEL").is_dir():
            return d
        if d.parent == d:
            break
        d = d.parent
    raise RuntimeError(f"Could not find api_test root from {here}")


_API_TEST_ROOT = _find_api_test_root()
_EMB_MODEL = _API_TEST_ROOT / "EMBEDDING_MODEL"
if str(_EMB_MODEL) not in sys.path:
    sys.path.insert(0, str(_EMB_MODEL))

import reconstruct_elo_from_probs as rep  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1.0 - 1e-6)


class RegressionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPRegression(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (64, 128, 256, 128, 64), dropout_rate: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return torch.sigmoid(self.head(h))


def load_pair_probabilities(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep same setup as vs2_exp: only AA x AA prompt split rows
    return df[(df["dataset1"] == "aa") & (df["dataset2"] == "aa")][["index1", "index2", "probability"]].copy()


def build_pair_prob_lookup(df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    lookup: Dict[Tuple[int, int], float] = {}
    for row in df.itertuples(index=False):
        i1 = int(row.index1)
        i2 = int(row.index2)
        p = float(row.probability)
        lookup[(i1, i2)] = p
        # Also support reversed ordering if needed
        lookup[(i2, i1)] = 1.0 - p
    return lookup


def build_split_dataframe(X_df: pd.DataFrame, split_probs_df: pd.DataFrame) -> pd.DataFrame:
    lookup = build_pair_prob_lookup(split_probs_df)
    y = []
    keep = []
    for row in X_df.itertuples(index=False):
        key = (int(row.index1), int(row.index2))
        p = lookup.get(key, None)
        if p is None:
            keep.append(False)
            y.append(np.nan)
        else:
            keep.append(True)
            y.append(float(p))
    out = X_df.loc[keep].copy()
    out["y"] = np.array([v for v, k in zip(y, keep) if k], dtype=np.float64)
    return out


def write_prob_csv(df_split: pd.DataFrame, y_pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index1", "index2", "y_true", "prob_pred"])
        for row, yp in zip(df_split.itertuples(index=False), y_pred):
            writer.writerow([int(row.index1), int(row.index2), float(row.y), float(yp)])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run_linear(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["y"].to_numpy(dtype=np.float64)
    y_val = val_df["y"].to_numpy(dtype=np.float64)
    y_test = test_df["y"].to_numpy(dtype=np.float64)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_train_pred = clip_prob(reg.predict(X_train))
    y_val_pred = clip_prob(reg.predict(X_val))
    y_test_pred = clip_prob(reg.predict(X_test))

    write_prob_csv(train_df, y_train_pred, out_dir / "prob_train.csv")
    write_prob_csv(val_df, y_val_pred, out_dir / "prob_val.csv")
    write_prob_csv(test_df, y_test_pred, out_dir / "prob_test.csv")

    metrics = {
        "model": "linear_regression",
        "n_features": len(feature_cols),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train": evaluate_predictions(y_train, y_train_pred),
        "val": evaluate_predictions(y_val, y_val_pred),
        "test": evaluate_predictions(y_test, y_test_pred),
        "coefficients": {name: float(v) for name, v in zip(feature_cols, reg.coef_)},
        "intercept": float(reg.intercept_),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(reg, f)


def predict_mlp_all(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.cpu().numpy())
    return clip_prob(np.concatenate(preds, axis=0).flatten())


def run_mlp(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["y"].to_numpy(dtype=np.float64)
    y_val = val_df["y"].to_numpy(dtype=np.float64)
    y_test = test_df["y"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    train_dataset = RegressionDataset(X_train_scaled, y_train)
    val_dataset = RegressionDataset(X_val_scaled, y_val)
    test_dataset = RegressionDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # Keep deterministic ordering for evaluation/output writing.
    train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = (64, 128, 256, 128, 64)
    model = MLPRegression(
        input_size=len(feature_cols),
        hidden_sizes=hidden_sizes,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    num_epochs = 500
    early_stop_patience = 40
    best_val_mse = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        y_val_pred = predict_mlp_all(model, val_loader, device)
        val_mse = mean_squared_error(y_val, y_val_pred)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_train_pred = predict_mlp_all(model, train_eval_loader, device)
    y_val_pred = predict_mlp_all(model, val_loader, device)
    y_test_pred = predict_mlp_all(model, test_loader, device)

    write_prob_csv(train_df, y_train_pred, out_dir / "prob_train.csv")
    write_prob_csv(val_df, y_val_pred, out_dir / "prob_val.csv")
    write_prob_csv(test_df, y_test_pred, out_dir / "prob_test.csv")

    metrics = {
        "model": "mlp_regression",
        "n_features": len(feature_cols),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "architecture": {
            "hidden_sizes": list(hidden_sizes),
            "dropout_rate": 0.2,
        },
        "training": {
            "num_epochs_max": num_epochs,
            "early_stop_patience": early_stop_patience,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
        },
        "train": evaluate_predictions(y_train, y_train_pred),
        "val": evaluate_predictions(y_val, y_val_pred),
        "test": evaluate_predictions(y_test, y_test_pred),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_cols),
            "hidden_sizes": hidden_sizes,
        },
        out_dir / "model.pth",
    )


def write_elo_csv(model_to_elo: Dict[str, float], ref_original: Dict[str, float], out_path: Path) -> None:
    sorted_models = sorted(model_to_elo.keys(), key=lambda m: model_to_elo[m], reverse=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "elo", "elo_original"])
        for model in sorted_models:
            elo_pred = model_to_elo[model]
            eo = ref_original.get(model, "")
            w.writerow([model, f"{elo_pred:.2f}", f"{eo:.2f}" if eo != "" else ""])


def run_elo_pipeline_for_dir(model_out_dir: Path, model_label: str) -> Dict[str, object]:
    index_to_model = rep.load_index_to_model()
    ref_i2v = rep.load_reference_elo(rep.I2V_REF) if rep.I2V_REF.exists() else {}
    ref_t2v = rep.load_reference_elo(rep.T2V_REF) if rep.T2V_REF.exists() else {}
    refs = [r for r in (ref_i2v, ref_t2v) if r]
    ref_combined = rep.combine_references(*refs) if refs else {}

    if not ref_combined:
        raise RuntimeError(
            f"Missing reference CSVs: {rep.I2V_REF} and/or {rep.T2V_REF}; cannot run ELO correlations."
        )

    elos_per_split: Dict[str, Dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        prob_path = model_out_dir / f"prob_{split}.csv"
        if not prob_path.exists():
            continue
        df = pd.read_csv(prob_path)
        rows = [
            (int(r.index1), int(r.index2), float(r.prob_pred))
            for r in df.itertuples(index=False)
        ]
        if not rows:
            continue
        model_to_elo = rep.build_elo_from_prob_rows(rows, index_to_model, ref_combined, k=rep.K)
        write_elo_csv(model_to_elo, ref_combined, model_out_dir / f"elo_{split}.csv")
        elos_per_split[split] = model_to_elo

    corr_rows = rep.compute_correlations(elos_per_split, ref_i2v, ref_t2v)
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
    with open(model_out_dir / "elo_correlations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in corr_rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    with open(model_out_dir / "elo_correlations.json", "w") as f:
        json.dump(corr_rows, f, indent=2)

    plots_dir = model_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
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

    metrics_path = model_out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics["elo"] = {
        "k_scale": rep.K,
        "reference_paths": {
            "i2v_final": str(rep.I2V_REF),
            "t2v_final": str(rep.T2V_REF),
        },
        "n_models_by_split": {k: len(v) for k, v in elos_per_split.items()},
        "correlations_file_csv": str(model_out_dir / "elo_correlations.csv"),
        "correlations_file_json": str(model_out_dir / "elo_correlations.json"),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "correlation_rows": corr_rows,
        "n_models_by_split": {k: len(v) for k, v in elos_per_split.items()},
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    x_path = here / "X.csv"
    splits_dir = here.parent.parent / "EMBEDDING_MODEL" / "prompt_splits-both"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing features file: {x_path}")
    if not splits_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {splits_dir}")

    X_df = pd.read_csv(x_path)
    required = {"index1", "index2"}
    if not required.issubset(set(X_df.columns)):
        raise ValueError("X.csv must include index1 and index2 columns.")

    feature_cols = [c for c in X_df.columns if c not in ("index1", "index2")]
    if not feature_cols:
        raise ValueError("No feature columns found in X.csv.")

    train_probs = load_pair_probabilities(splits_dir / "prob_train.csv")
    val_probs = load_pair_probabilities(splits_dir / "prob_val.csv")
    test_probs = load_pair_probabilities(splits_dir / "prob_test.csv")

    train_df = build_split_dataframe(X_df, train_probs)
    val_df = build_split_dataframe(X_df, val_probs)
    test_df = build_split_dataframe(X_df, test_probs)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Empty split after joining X.csv with prompt_splits-both labels: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    run_linear(train_df, val_df, test_df, feature_cols, here / "linear")
    run_mlp(train_df, val_df, test_df, feature_cols, here / "mlp")
    linear_elo_summary = run_elo_pipeline_for_dir(here / "linear", "linear")
    mlp_elo_summary = run_elo_pipeline_for_dir(here / "mlp", "mlp")

    summary = {
        "features_path": str(x_path),
        "splits_dir": str(splits_dir),
        "n_features": len(feature_cols),
        "n_rows_used": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "elo": {
            "linear": linear_elo_summary,
            "mlp": mlp_elo_summary,
        },
    }
    with open(here / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Linear outputs: {here / 'linear'}")
    print(f"MLP outputs: {here / 'mlp'}")


if __name__ == "__main__":
    main()

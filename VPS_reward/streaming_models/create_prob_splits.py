"""
Train/val/test splits for probability prediction on video pairs.

Extends api_test/EMBEDDING_MODEL/create_prob_splits.py with:
- `--dataset both`: same pair construction as create_prompt_splits.py (AA + social + cross).
- `--mode streaming_model`: rank models by mean ELO (best first). Partition models into
  top `--test-model-frac` (test tier), next `--val-model-frac` (val tier), remainder (train tier).
  Defaults: 7%% test tier, 10%% val tier (`--test-model-frac` / `--val-model-frac`).
  **AA×AA pairs** are labeled hierarchically (no pairs dropped):
  - **test**: any pair involving at least one test-tier model (test vs test, test vs val,
    test vs train).
  - **val**: any remaining pair involving at least one val-tier model (val vs val,
    val vs train).
  - **train**: both models are train-tier only (train vs train).
  Social×social pairs use the prompt split from create_prompt_splits.py.
  **Cross (AA×social)** pairs follow the AA model's tier (test / val / train) only.

`--mode model` uses the same AA×AA / cross rule but assigns models to tiers by random shuffle
with the usual 60/20/20 counts (first block train tier, …, last block test tier).

Also supports random / video modes on the combined (or AA-only) dataframe.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CROSS_DATASET_SOCIAL_WIN_PROB = 0.75

MODEL_TRAIN_RATIO = 0.60
MODEL_VAL_RATIO = 0.20
MODEL_TEST_RATIO = 0.20

# streaming_model tier sizes (distinct models); remainder → train tier
DEFAULT_TEST_MODEL_FRAC = 0.10
DEFAULT_VAL_MODEL_FRAC = 0.10


def compute_elo_probability(elo1: float, elo2: float, scale: float = 100.0) -> float:
    return 1.0 / (1.0 + np.exp((elo2 - elo1) / scale))


def compute_respond_probability(rc1: float, rc2: float) -> float:
    total = rc1 + rc2
    if total == 0:
        return 0.5
    return rc1 / total


def compute_probability(elo1: float, elo2: float) -> float:
    """Win prob for index1 from ELO pair columns (same as parent create_prob_splits)."""
    return compute_elo_probability(elo1, elo2, scale=100.0)


def split_prompts(n_prompts: int):
    indices = np.arange(1, n_prompts + 1)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
    )
    return set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())


def load_index_to_model(dataset_path: str) -> dict[int, str]:
    with open(dataset_path) as f:
        data = json.load(f)
    return {item["index"]: item["model"] for item in data}


def load_aa_data(data_dir: str):
    mapping = pd.read_csv(os.path.join(data_dir, "mapping.csv"))
    pairs_df = pd.read_csv(os.path.join(data_dir, "pairs_with_elo.csv"))
    index_to_prompt = dict(zip(mapping.dataset_index, mapping.prompt_line))
    prompt_to_aa_indices: dict[int, list[int]] = {}
    for _, row in mapping.iterrows():
        prompt_to_aa_indices.setdefault(row.prompt_line, []).append(row.dataset_index)
    return prompt_to_aa_indices, pairs_df, index_to_prompt


def load_social_data(data_dir: str):
    with open(os.path.join(data_dir, "aa_final_video_segments.json")) as f:
        data = json.load(f)
    prompt_to_social_indices: dict[int, list[int]] = {}
    idx_to_respond_count: dict[int, int] = {}
    for item in data:
        prompt_idx = item["prompt_idx"]
        idx = item["idx"]
        rc = item["engagement"].get("respond_count", 0)
        prompt_to_social_indices.setdefault(prompt_idx, []).append(idx)
        idx_to_respond_count[idx] = rc
    return prompt_to_social_indices, idx_to_respond_count


def build_aa_pairs(pairs_df: pd.DataFrame, index_to_prompt: dict[int, int]):
    rows = []
    for _, row in pairs_df.iterrows():
        prompt = index_to_prompt.get(row.index1)
        if prompt is None:
            continue
        rows.append(
            {
                "dataset1": "aa",
                "index1": int(row.index1),
                "dataset2": "aa",
                "index2": int(row.index2),
                "probability": compute_probability(row.elo1, row.elo2),
                "prompt": int(prompt),
            }
        )
    return rows


def build_social_pairs(
    prompt_to_social_indices: dict[int, list[int]],
    idx_to_respond_count: dict[int, int],
):
    rows = []
    for prompt_idx, indices in prompt_to_social_indices.items():
        for i, j in combinations(sorted(indices), 2):
            rc1 = idx_to_respond_count[i]
            rc2 = idx_to_respond_count[j]
            probability = compute_respond_probability(rc1, rc2)
            rows.append(
                {
                    "dataset1": "social",
                    "index1": i,
                    "dataset2": "social",
                    "index2": j,
                    "probability": probability,
                    "prompt": prompt_idx,
                }
            )
    return rows


def build_cross_pairs(
    prompt_to_aa_indices: dict[int, list[int]],
    prompt_to_social_indices: dict[int, list[int]],
):
    rows = []
    all_prompts = set(prompt_to_aa_indices.keys()) | set(prompt_to_social_indices.keys())
    for prompt_idx in sorted(all_prompts):
        aa_indices = prompt_to_aa_indices.get(prompt_idx, [])
        social_indices = prompt_to_social_indices.get(prompt_idx, [])
        for aa_idx in sorted(aa_indices):
            for social_idx in sorted(social_indices):
                rows.append(
                    {
                        "dataset1": "aa",
                        "index1": aa_idx,
                        "dataset2": "social",
                        "index2": social_idx,
                        "probability": 1.0 - CROSS_DATASET_SOCIAL_WIN_PROB,
                        "prompt": prompt_idx,
                    }
                )
    return rows


def rows_to_xy(rows: list[dict]):
    if not rows:
        return pd.DataFrame(
            columns=["dataset1", "index1", "dataset2", "index2"]
        ), pd.DataFrame(columns=["probability"])
    df = pd.DataFrame(rows)
    X = df[["dataset1", "index1", "dataset2", "index2"]].reset_index(drop=True)
    if "prompt" in df.columns:
        X["prompt"] = df["prompt"].astype(int).values
    Y = df[["probability"]].reset_index(drop=True)
    return X, Y


def drop_prompt_column(X: pd.DataFrame) -> pd.DataFrame:
    if "prompt" in X.columns:
        return X.drop(columns=["prompt"]).reset_index(drop=True)
    return X


def build_both_rows(data_dir: str):
    prompt_to_aa_indices, pairs_df, index_to_prompt = load_aa_data(data_dir)
    prompt_to_social_indices, idx_to_respond_count = load_social_data(data_dir)
    rows: list[dict] = []
    rows.extend(build_aa_pairs(pairs_df, index_to_prompt))
    rows.extend(build_social_pairs(prompt_to_social_indices, idx_to_respond_count))
    rows.extend(build_cross_pairs(prompt_to_aa_indices, prompt_to_social_indices))
    return rows


def build_aa_only_rows(data_dir: str):
    _, pairs_df, index_to_prompt = load_aa_data(data_dir)
    return build_aa_pairs(pairs_df, index_to_prompt)


def _find_connected_components(nodes_edges: list[tuple[tuple, tuple]]) -> list[set]:
    adj: dict[tuple, set[tuple]] = {}
    for u, v in nodes_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    visited = set()
    components = []
    for start in adj:
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for neighbor in adj.get(n, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(comp)
    return components


def _video_edges_from_xy(X: pd.DataFrame) -> list[tuple[tuple, tuple]]:
    edges = []
    for _, row in X.iterrows():
        u = (row["dataset1"], int(row["index1"]))
        v = (row["dataset2"], int(row["index2"]))
        edges.append((u, v))
    return edges


def split_random(X: pd.DataFrame, Y: pd.DataFrame):
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def split_video_based(X: pd.DataFrame, Y: pd.DataFrame):
    edges = _video_edges_from_xy(X)
    components = _find_connected_components(edges)
    print(f"  Found {len(components)} video groups (connected components)")
    components = sorted(components, key=len, reverse=True)
    rng = np.random.default_rng(RANDOM_STATE + 1)
    rng.shuffle(components)
    n = len(components)
    n_train = max(1, int(n * TRAIN_RATIO)) if n >= 3 else min(1, n)
    n_val = max(1, int(n * VAL_RATIO)) if n >= 3 else min(1, max(0, n - 1))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    groups_train = components[:n_train]
    groups_val = components[n_train : n_train + n_val]
    groups_test = components[n_train + n_val :]
    video_to_split: dict[tuple, str] = {}
    for g in groups_train:
        for v in g:
            video_to_split[v] = "train"
    for g in groups_val:
        for v in g:
            video_to_split[v] = "val"
    for g in groups_test:
        for v in g:
            video_to_split[v] = "test"

    def row_split(row):
        u = (row["dataset1"], int(row["index1"]))
        return video_to_split.get(u, "train")

    splits = X.apply(row_split, axis=1)
    X_train = X.loc[splits == "train"].reset_index(drop=True)
    X_val = X.loc[splits == "val"].reset_index(drop=True)
    X_test = X.loc[splits == "test"].reset_index(drop=True)
    Y_train = Y.loc[splits == "train"].reset_index(drop=True)
    Y_val = Y.loc[splits == "val"].reset_index(drop=True)
    Y_test = Y.loc[splits == "test"].reset_index(drop=True)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def _bucket_counts(n: int, train_frac: float, val_frac: float, test_frac: float):
    """Match create_prob_splits.model_based sizing."""
    n_train = max(1, int(n * train_frac)) if n >= 3 else min(1, n)
    n_val = max(1, int(n * val_frac)) if n >= 3 else min(1, max(0, n - 1))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    return n_train, n_val, n_test


def ranked_model_tier_counts(
    n: int, test_model_frac: float, val_model_frac: float
) -> tuple[int, int, int]:
    """Returns (n_test, n_val, n_train) for ranked models; best models occupy test tier."""
    if n == 0:
        return 0, 0, 0
    n_test = max(1, int(round(n * test_model_frac))) if n >= 3 else min(1, n)
    n_val = (
        max(1, int(round(n * val_model_frac)))
        if n >= 3
        else min(1, max(0, n - n_test - 1))
    )
    n_train = n - n_test - n_val
    if n_train < 1:
        while n_train < 1 and n_val > 1:
            n_val -= 1
            n_train += 1
        while n_train < 1 and n_test > 1:
            n_test -= 1
            n_train += 1
        if n_train < 1:
            n_train = 1
            n_test = min(n_test, max(0, n - n_train - n_val))
            n_val = n - n_train - n_test
    return n_test, n_val, n_train


def partition_models_ranked(
    models_ranked_best_first: list[str],
    test_model_frac: float,
    val_model_frac: float,
) -> tuple[set[str], set[str], set[str]]:
    """
    Best-ELO models → test tier, next → val tier, remainder → train tier.
    Returns (M_test, M_val, M_train).
    """
    r = list(models_ranked_best_first)
    n = len(r)
    n_test, n_val, _n_train = ranked_model_tier_counts(n, test_model_frac, val_model_frac)
    M_test = set(r[:n_test])
    M_val = set(r[n_test : n_test + n_val])
    M_train = set(r[n_test + n_val :])
    return M_test, M_val, M_train


def partition_models_shuffled(models: list[str]) -> tuple[set[str], set[str], set[str]]:
    """
    Random tiers with same counts as model_based (60/20/20).
    First chunk → train tier, middle → val, last → test tier (for hierarchical pair labeling).
    Returns (M_test, M_val, M_train).
    """
    rng = np.random.default_rng(RANDOM_STATE * 2)
    shuffled = list(models)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train_b, n_val_b, n_test_b = _bucket_counts(
        n, MODEL_TRAIN_RATIO, MODEL_VAL_RATIO, MODEL_TEST_RATIO
    )
    M_train = set(shuffled[:n_train_b])
    M_val = set(shuffled[n_train_b : n_train_b + n_val_b])
    M_test = set(shuffled[n_train_b + n_val_b :])
    return M_test, M_val, M_train


def aa_aa_pair_split_label(m1: str | None, m2: str | None, M_test: set[str], M_val: set[str]) -> str | None:
    """Hierarchical split: test if either side is test-tier; else val if either is val-tier; else train."""
    if m1 is None or m2 is None:
        return None
    if m1 in M_test or m2 in M_test:
        return "test"
    if m1 in M_val or m2 in M_val:
        return "val"
    return "train"


def cross_pair_split_label(aa_model: str | None, M_test: set[str], M_val: set[str]) -> str | None:
    """Cross pair follows AA model tier only."""
    if aa_model is None:
        return None
    if aa_model in M_test:
        return "test"
    if aa_model in M_val:
        return "val"
    return "train"


def compute_mean_elo_per_model(
    pairs_df: pd.DataFrame, index_to_model: dict[int, str]
) -> dict[str, float]:
    index_elos: dict[int, list[float]] = {}
    for _, row in pairs_df.iterrows():
        i1, i2 = int(row.index1), int(row.index2)
        index_elos.setdefault(i1, []).append(float(row.elo1))
        index_elos.setdefault(i2, []).append(float(row.elo2))
    model_vals: dict[str, list[float]] = {}
    for idx, elos in index_elos.items():
        m = index_to_model.get(idx)
        if m is None:
            continue
        model_vals.setdefault(m, []).extend(elos)
    return {m: float(np.mean(vals)) for m, vals in model_vals.items()}


def models_ranked_by_mean_elo_desc(data_dir: str, index_to_model: dict[int, str]) -> list[str]:
    _, pairs_df, _ = load_aa_data(data_dir)
    mean_elo = compute_mean_elo_per_model(pairs_df, index_to_model)
    models = sorted(mean_elo.keys(), key=lambda mm: (-mean_elo[mm], mm))
    return models


def _models_touching_aa_pairs(X: pd.DataFrame, index_to_model: dict[int, str]) -> list[str]:
    aa_indices: set[int] = set()
    for _, row in X.iterrows():
        if row["dataset1"] == "aa":
            aa_indices.add(int(row["index1"]))
        if row["dataset2"] == "aa":
            aa_indices.add(int(row["index2"]))
    return sorted({index_to_model[i] for i in aa_indices if i in index_to_model})


def split_model_or_streaming(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    index_to_model: dict[int, str],
    train_prompts: set[int],
    val_prompts: set[int],
    test_prompts: set[int],
    streaming: bool,
    data_dir: str,
    test_model_frac: float,
    val_model_frac: float,
):
    models_in_pairs = _models_touching_aa_pairs(X, index_to_model)
    model_set = set(models_in_pairs)
    if streaming:
        ranked = [
            m
            for m in models_ranked_by_mean_elo_desc(data_dir, index_to_model)
            if m in model_set
        ]
        M_test, M_val, M_train_tier = partition_models_ranked(
            ranked, test_model_frac=test_model_frac, val_model_frac=val_model_frac
        )
        print(
            f"  streaming_model: ranked {len(ranked)} models by mean ELO → "
            f"test tier={len(M_test)}, val tier={len(M_val)}, train tier={len(M_train_tier)}"
        )
    else:
        M_test, M_val, M_train_tier = partition_models_shuffled(models_in_pairs)
        print(
            f"  model: shuffled {len(models_in_pairs)} models → "
            f"test tier={len(M_test)}, val tier={len(M_val)}, train tier={len(M_train_tier)}"
        )

    def pair_split(row) -> str | None:
        d1, d2 = row["dataset1"], row["dataset2"]
        if d1 == "social" and d2 == "social":
            p = int(row["prompt"])
            if p in train_prompts:
                return "train"
            if p in val_prompts:
                return "val"
            if p in test_prompts:
                return "test"
            return None
        if d1 == "aa" and d2 == "aa":
            m1 = index_to_model.get(int(row["index1"]))
            m2 = index_to_model.get(int(row["index2"]))
            return aa_aa_pair_split_label(m1, m2, M_test, M_val)
        if d1 == "aa" and d2 == "social":
            m = index_to_model.get(int(row["index1"]))
            return cross_pair_split_label(m, M_test, M_val)
        if d1 == "social" and d2 == "aa":
            m = index_to_model.get(int(row["index2"]))
            return cross_pair_split_label(m, M_test, M_val)
        return None

    splits = X.apply(pair_split, axis=1)
    mask_train = splits == "train"
    mask_val = splits == "val"
    mask_test = splits == "test"
    X_train = X.loc[mask_train].reset_index(drop=True)
    X_val = X.loc[mask_val].reset_index(drop=True)
    X_test = X.loc[mask_test].reset_index(drop=True)
    Y_train = Y.loc[mask_train].reset_index(drop=True)
    Y_val = Y.loc[mask_val].reset_index(drop=True)
    Y_test = Y.loc[mask_test].reset_index(drop=True)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def split_model_based_aa_indices_only(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    index_to_model: dict[int, str],
    *,
    streaming: bool,
    data_dir: str | None,
    test_model_frac: float,
    val_model_frac: float,
):
    """Original prob_splits logic: X columns index1, index2 only."""
    indices_in_pairs = set(X["index1"].unique()) | set(X["index2"].unique())
    all_models = sorted({index_to_model[i] for i in indices_in_pairs if i in index_to_model})
    if streaming:
        assert data_dir is not None
        ranked = [
            m
            for m in models_ranked_by_mean_elo_desc(data_dir, index_to_model)
            if m in set(all_models)
        ]
        M_test, M_val, M_train_tier = partition_models_ranked(
            ranked, test_model_frac=test_model_frac, val_model_frac=val_model_frac
        )
        print(
            f"  streaming_model: ranked {len(ranked)} models → "
            f"test={len(M_test)}, val={len(M_val)}, train={len(M_train_tier)}"
        )
    else:
        M_test, M_val, M_train_tier = partition_models_shuffled(all_models)
        print(
            f"  model: shuffled {len(all_models)} models → "
            f"test={len(M_test)}, val={len(M_val)}, train={len(M_train_tier)}"
        )

    def pair_split(row):
        m1 = index_to_model[row["index1"]]
        m2 = index_to_model[row["index2"]]
        return aa_aa_pair_split_label(m1, m2, M_test, M_val)

    splits = X.apply(pair_split, axis=1)
    mask_train = splits == "train"
    mask_val = splits == "val"
    mask_test = splits == "test"
    return (
        X.loc[mask_train].reset_index(drop=True),
        X.loc[mask_val].reset_index(drop=True),
        X.loc[mask_test].reset_index(drop=True),
        Y.loc[mask_train].reset_index(drop=True),
        Y.loc[mask_val].reset_index(drop=True),
        Y.loc[mask_test].reset_index(drop=True),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Probability pair splits with streaming_model rank-based model partitioning."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["aa", "both"],
        help="aa: pairs_with_elo pairs only; both: AA + social + cross (same as create_prompt_splits both).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="streaming_model",
        choices=["random", "video", "model", "streaming_model"],
        help="streaming_model: ELO-ranked tiers; AA×AA pairs → test if any test-tier model, "
        "else val if any val-tier, else train-only.",
    )
    p.add_argument(
        "--test-model-frac",
        type=float,
        default=DEFAULT_TEST_MODEL_FRAC,
        help="Fraction of distinct models in the test tier (streaming_model). Default: 0.07.",
    )
    p.add_argument(
        "--val-model-frac",
        type=float,
        default=DEFAULT_VAL_MODEL_FRAC,
        help="Fraction of distinct models in the val tier after test (streaming_model). Default: 0.10.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embed_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(embed_dir, "data")
    dataset_path = os.path.join(data_dir, "dataset.json")
    prompts_path = os.path.join(data_dir, "prompts.txt")

    mode = args.mode
    streaming = mode == "streaming_model"
    model_like = mode in ("model", "streaming_model")

    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    n_prompts = len(prompts)
    train_prompts, val_prompts, test_prompts = split_prompts(n_prompts)
    print(f"Prompt split — train: {len(train_prompts)}, val: {len(val_prompts)}, test: {len(test_prompts)}")

    if args.dataset == "both":
        print("Building combined rows (AA + social + cross)...")
        all_rows = build_both_rows(data_dir)
        print(f"  Total pairs: {len(all_rows)}")
        X, Y = rows_to_xy(all_rows)
    else:
        print("Building AA-only rows from pairs_with_elo...")
        all_rows = build_aa_only_rows(data_dir)
        print(f"  Total pairs: {len(all_rows)}")
        df_xy = pd.DataFrame(all_rows)
        X = pd.DataFrame(
            {
                "dataset1": "aa",
                "index1": df_xy["index1"].astype(int),
                "dataset2": "aa",
                "index2": df_xy["index2"].astype(int),
            }
        )
        Y = df_xy[["probability"]].copy()

    print(f"\nSplitting dataset={args.dataset} mode={mode}...")
    if mode == "random":
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_random(X, Y)
    elif mode == "video":
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_video_based(X, Y)
    elif model_like:
        index_to_model = load_index_to_model(dataset_path)
        if args.dataset == "both":
            X_train, X_val, X_test, Y_train, Y_val, Y_test = split_model_or_streaming(
                X,
                Y,
                index_to_model=index_to_model,
                train_prompts=train_prompts,
                val_prompts=val_prompts,
                test_prompts=test_prompts,
                streaming=streaming,
                data_dir=data_dir,
                test_model_frac=args.test_model_frac,
                val_model_frac=args.val_model_frac,
            )
        else:
            X_train, X_val, X_test, Y_train, Y_val, Y_test = split_model_based_aa_indices_only(
                X,
                Y,
                index_to_model,
                streaming=streaming,
                data_dir=data_dir if streaming else None,
                test_model_frac=args.test_model_frac,
                val_model_frac=args.val_model_frac,
            )
    else:
        raise RuntimeError(f"Unhandled mode {mode}")

    n_total = len(X_train) + len(X_val) + len(X_test)
    print(f"\nSplit sizes (retained pairs): {n_total}")
    if n_total > 0:
        print(f"  Train: {len(X_train)} ({100 * len(X_train) / n_total:.1f}%)")
        print(f"  Val: {len(X_val)} ({100 * len(X_val) / n_total:.1f}%)")
        print(f"  Test: {len(X_test)} ({100 * len(X_test) / n_total:.1f}%)")

    X_train = drop_prompt_column(X_train)
    X_val = drop_prompt_column(X_val)
    X_test = drop_prompt_column(X_test)

    if args.dataset == "aa":
        X_train = X_train[["index1", "index2"]].reset_index(drop=True)
        X_val = X_val[["index1", "index2"]].reset_index(drop=True)
        X_test = X_test[["index1", "index2"]].reset_index(drop=True)

    out_name = f"encoding_splits-{args.dataset}-{mode}"
    output_dir = os.path.join(script_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "prob_X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "prob_X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "prob_X_test.csv"), index=False)
    Y_train.to_csv(os.path.join(output_dir, "prob_Y_train.csv"), index=False)
    Y_val.to_csv(os.path.join(output_dir, "prob_Y_val.csv"), index=False)
    Y_test.to_csv(os.path.join(output_dir, "prob_Y_test.csv"), index=False)

    for split_name, xs, ys in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        pd.concat([xs.reset_index(drop=True), ys.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(output_dir, f"prob_{split_name}.csv"), index=False
        )

    print(f"\nSaved splits to {output_dir}/")

    if streaming:
        index_tm = load_index_to_model(dataset_path)
        touching = _models_touching_aa_pairs(X, index_tm)
        ranked_tm = [
            m
            for m in models_ranked_by_mean_elo_desc(data_dir, index_tm)
            if m in set(touching)
        ]
        Mt_test, Mt_val, Mt_train = partition_models_ranked(
            ranked_tm,
            test_model_frac=args.test_model_frac,
            val_model_frac=args.val_model_frac,
        )
        tier_meta = {
            "dataset": args.dataset,
            "mode": mode,
            "test_model_frac": args.test_model_frac,
            "val_model_frac": args.val_model_frac,
            "n_models_tiered": len(ranked_tm),
            "test_tier": sorted(Mt_test),
            "val_tier": sorted(Mt_val),
            "train_tier": sorted(Mt_train),
        }
        tier_path = os.path.join(output_dir, "model_tiers.json")
        with open(tier_path, "w") as f:
            json.dump(tier_meta, f, indent=2)
        print(f"Wrote tier labels for plots/training metadata: {tier_path}")


if __name__ == "__main__":
    main()

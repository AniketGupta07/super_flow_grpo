"""
Script to create prompt-based train/val/test splits for video pair comparison.

X: (dataset1, index1, dataset2, index2)
Y: probability (that video1 wins over video2)

Dataset modes:
- aa: AA benchmark videos (dataset.json / pairs_with_elo.csv).
  Probability based on ELO scores: 1 / (1 + exp((elo2 - elo1) / 100))
- social: Social media videos (aa_final_video_segments.json).
  Probability based on respond_count: rc1 / (rc1 + rc2)
- social_fixed: Social media videos (aa_final_video_segments.json).
  Same logistic as AA ELO, but ratings are log(1 + respond_count):
  1 / (1 + exp((log(1 + rc2) - log(1 + rc1)) / 100))
- both: Videos from both datasets combined.
  AA×AA → ELO-based, Social×Social → respond_count-based,
  AA×Social → social wins with probability 0.75.
- cross: Only AA×Social cross-dataset pairs.
  Social wins with probability 0.75.

The 694 prompts are split into train/val/test first, then all pairs
belonging to a prompt are assigned to that prompt's split.
"""

import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.model_selection import train_test_split
import os
import argparse

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CROSS_DATASET_SOCIAL_WIN_PROB = 0.75


def compute_elo_probability(elo1: float, elo2: float, scale: float = 100.0) -> float:
    return 1.0 / (1.0 + np.exp((elo2 - elo1) / scale))


def compute_respond_probability(rc1: float, rc2: float) -> float:
    total = rc1 + rc2
    if total == 0:
        return 0.5
    return rc1 / total


def compute_log_respond_elo_probability(rc1: float, rc2: float) -> float:
    """ELO-style win prob using log(1 + respond_count) as the rating (scale 100, same as AA)."""
    return compute_elo_probability(np.log1p(rc1), np.log1p(rc2), scale=10.0)
    # return compute_respond_probability(np.log1p(rc1), np.log1p(rc2))


def split_prompts(n_prompts: int):
    """Split prompt indices (1..n_prompts) into train/val/test sets."""
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


def load_aa_data(data_dir: str):
    """
    Returns:
        prompt_to_aa_indices: {prompt_line: [dataset_index, ...]}
        pairs_df: DataFrame with columns index1, index2, elo1, elo2
        index_to_prompt: {dataset_index: prompt_line}
    """
    mapping = pd.read_csv(os.path.join(data_dir, "mapping.csv"))
    pairs_df = pd.read_csv(os.path.join(data_dir, "pairs_with_elo.csv"))

    index_to_prompt = dict(zip(mapping.dataset_index, mapping.prompt_line))

    prompt_to_aa_indices: dict[int, list[int]] = {}
    for _, row in mapping.iterrows():
        prompt_to_aa_indices.setdefault(row.prompt_line, []).append(row.dataset_index)

    return prompt_to_aa_indices, pairs_df, index_to_prompt


def load_social_data(data_dir: str):
    """
    Returns:
        prompt_to_social_indices: {prompt_idx: [idx, ...]}
        idx_to_respond_count: {idx: respond_count}
    """
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
    """Build pair rows from pairs_with_elo.csv."""
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
                "probability": compute_elo_probability(row.elo1, row.elo2),
                "prompt": int(prompt),
            }
        )
    return rows


def build_social_pairs(
    prompt_to_social_indices: dict[int, list[int]],
    idx_to_respond_count: dict[int, int],
    use_log_respond_elo: bool = False,
):
    """Build all C(n,2) pairs per prompt from social videos."""
    rows = []
    for prompt_idx, indices in prompt_to_social_indices.items():
        for i, j in combinations(sorted(indices), 2):
            rc1 = idx_to_respond_count[i]
            rc2 = idx_to_respond_count[j]
            if use_log_respond_elo:
                probability = compute_log_respond_elo_probability(rc1, rc2)
            else:
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
    """Build AA×Social cross-dataset pairs. Social wins with 0.75."""
    rows = []
    all_prompts = set(prompt_to_aa_indices.keys()) | set(
        prompt_to_social_indices.keys()
    )
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


def assign_splits(
    all_rows: list[dict],
    train_prompts: set[int],
    val_prompts: set[int],
    test_prompts: set[int],
):
    """Partition rows into train/val/test based on prompt membership."""
    train, val, test = [], [], []
    for row in all_rows:
        p = row["prompt"]
        if p in train_prompts:
            train.append(row)
        elif p in val_prompts:
            val.append(row)
        elif p in test_prompts:
            test.append(row)
    return train, val, test


def rows_to_xy(rows: list[dict]):
    if not rows:
        return pd.DataFrame(columns=["dataset1", "index1", "dataset2", "index2"]), pd.DataFrame(columns=["probability"])
    df = pd.DataFrame(rows)
    X = df[["dataset1", "index1", "dataset2", "index2"]].reset_index(drop=True)
    Y = df[["probability"]].reset_index(drop=True)
    return X, Y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create prompt-based train/val/test splits for video pairs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aa", "social", "social_fixed", "both", "cross"],
        help="Which dataset(s) to use: aa, social, social_fixed, both, or cross (AA×Social only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    prompts_path = os.path.join(data_dir, "prompts.txt")
    output_dir = os.path.join(script_dir, f"prompt_splits-{args.dataset}")

    os.makedirs(output_dir, exist_ok=True)

    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    n_prompts = len(prompts)
    print(f"Loaded {n_prompts} prompts")

    print("Splitting prompts into train/val/test...")
    train_prompts, val_prompts, test_prompts = split_prompts(n_prompts)
    print(
        f"  Prompts — train: {len(train_prompts)}, "
        f"val: {len(val_prompts)}, test: {len(test_prompts)}"
    )

    all_rows: list[dict] = []

    if args.dataset in ("aa", "both"):
        print("Loading AA data...")
        prompt_to_aa_indices, pairs_df, index_to_prompt = load_aa_data(data_dir)
        aa_rows = build_aa_pairs(pairs_df, index_to_prompt)
        print(f"  AA pairs: {len(aa_rows)}")
        all_rows.extend(aa_rows)

    if args.dataset in ("social", "social_fixed", "both"):
        print("Loading social data...")
        prompt_to_social_indices, idx_to_respond_count = load_social_data(data_dir)
        social_rows = build_social_pairs(
            prompt_to_social_indices,
            idx_to_respond_count,
            use_log_respond_elo=(args.dataset == "social_fixed"),
        )
        print(f"  Social pairs: {len(social_rows)}")
        all_rows.extend(social_rows)

    if args.dataset in ("both", "cross"):
        if args.dataset == "cross":
            print("Loading AA data...")
            prompt_to_aa_indices, pairs_df, index_to_prompt = load_aa_data(data_dir)
            print("Loading social data...")
            prompt_to_social_indices, idx_to_respond_count = load_social_data(data_dir)
        print("Building cross-dataset pairs (AA × Social)...")
        cross_rows = build_cross_pairs(prompt_to_aa_indices, prompt_to_social_indices)
        print(f"  Cross pairs: {len(cross_rows)}")
        all_rows.extend(cross_rows)

    print(f"\nTotal pairs: {len(all_rows)}")

    train_rows, val_rows, test_rows = assign_splits(
        all_rows, train_prompts, val_prompts, test_prompts
    )

    X_train, Y_train = rows_to_xy(train_rows)
    X_val, Y_val = rows_to_xy(val_rows)
    X_test, Y_test = rows_to_xy(test_rows)

    for label, X, Y in [
        ("Train", X_train, Y_train),
        ("Val", X_val, Y_val),
        ("Test", X_test, Y_test),
    ]:
        print(f"  {label}: {len(X)} pairs")
        if len(Y) > 0:
            print(
                f"    Prob — min: {Y['probability'].min():.4f}, "
                f"max: {Y['probability'].max():.4f}, "
                f"mean: {Y['probability'].mean():.4f}, "
                f"std: {Y['probability'].std():.4f}"
            )

    X_train.to_csv(os.path.join(output_dir, "prob_X_train.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, "prob_X_val.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "prob_X_test.csv"), index=False)
    Y_train.to_csv(os.path.join(output_dir, "prob_Y_train.csv"), index=False)
    Y_val.to_csv(os.path.join(output_dir, "prob_Y_val.csv"), index=False)
    Y_test.to_csv(os.path.join(output_dir, "prob_Y_test.csv"), index=False)

    for split_name, X_split, Y_split in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        combined = pd.concat(
            [X_split.reset_index(drop=True), Y_split.reset_index(drop=True)], axis=1
        )
        combined.to_csv(os.path.join(output_dir, f"prob_{split_name}.csv"), index=False)

    print(f"\nSaved splits to {output_dir}/")
    print("  prob_X_train.csv, prob_X_val.csv, prob_X_test.csv")
    print("  prob_Y_train.csv, prob_Y_val.csv, prob_Y_test.csv")
    print("  prob_train.csv, prob_val.csv, prob_test.csv")


if __name__ == "__main__":
    main()

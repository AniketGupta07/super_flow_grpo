#!/usr/bin/env python3
"""
Read predicted probability CSVs (prob_train, prob_val, prob_test), reconstruct ELO
scores per MODEL (not per index) by minimizing:
  min  sum_{(a,b) in E} w_ab * [(r_a - r_b) - d_ab]^2,
where r_a, r_b are model ratings, d_ab = k * logit(p̂_ab), and p̂_ab is the predicted
probability that model a is preferred over b. Each row (index1, index2) is mapped to
(model1, model2) via dataset_2.json. In create_Y_and_splits (and thus prob_pred),
prob_pred = P(index1 beats index2) = P(a beats b), so d_ab = k * logit(prob_pred).

Output: one CSV per split with columns model, elo (one row per distinct model).
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

try:
    from scipy.stats import spearmanr, kendalltau
except ImportError:
    spearmanr = kendalltau = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SPLITS_DIR = Path(__file__).parent / "encoding_splits-video"
DATASET_JSON = Path(__file__).parent / "data" / "dataset.json"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"
OUTPUT_DIR = SPLITS_DIR
I2V_REF = Path(__file__).parent / "data" / "i2v_final.csv"
T2V_REF = Path(__file__).parent / "data" / "t2v_final.csv"

# Scale for ELO: d_ab = k * logit(...). Matches create_Y_and_splits (elo_diff/100).
K = 100.0


def logit(p):
    """logit(p) = log(p/(1-p)). p must be in (0, 1)."""
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return np.log(p) - np.log(1.0 - p)


def load_prob_csv(path):
    """Load prob_*.csv; return list of (index1, index2, prob_pred)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a, b = int(row['index1']), int(row['index2'])
            prob_pred = float(row['y_pred'])
            rows.append((a, b, prob_pred))
    return rows


def load_index_to_model():
    """Load dataset_2.json and return index -> model name."""
    with open(DATASET_JSON) as f:
        data = json.load(f)
    return {item['index']: item['model'] for item in data}


def solve_elos_per_model(edges, weights=None):
    """
    edges: list of (model_a, model_b, d_ab) where d_ab = k * logit(P(a beats b)).
    We want r_a - r_b = d_ab for each (model_a, model_b). Vertices are model names.
    weights: optional list of w_ab (same length as edges). Default 1.
    Returns: dict model_name -> elo (float).
    """
    if weights is None:
        weights = [1.0] * len(edges)
    models = sorted(set(m_a for m_a, m_b, _ in edges) | set(m_b for m_a, m_b, _ in edges))
    model_to_pos = {m: i for i, m in enumerate(models)}
    n = len(models)

    A_list = []
    d_list = []
    for (m_a, m_b, d_ab), w in zip(edges, weights):
        row = np.zeros(n)
        row[model_to_pos[m_a]] = np.sqrt(w)
        row[model_to_pos[m_b]] = -np.sqrt(w)
        A_list.append(row)
        d_list.append(d_ab * np.sqrt(w))

    A = np.array(A_list)
    d = np.array(d_list)

    # Pin one model's rating to 0 to fix the free constant.
    pin_row = np.zeros(n)
    pin_row[0] = 1.0
    A = np.vstack([A, pin_row])
    d = np.append(d, 0.0)

    r_vec, residuals, rank, s = np.linalg.lstsq(A, d, rcond=None)
    return {model: float(r_vec[i]) for i, model in enumerate(models)}


def load_reference_elo(path):
    """Load reference CSV (model, elo); aggregate to one elo per model (mean if duplicates)."""
    from collections import defaultdict
    elos = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"].strip()
            elos[model].append(float(row["elo"]))
    return {m: sum(v) / len(v) for m, v in elos.items()}


def combine_references(*ref_dicts):
    """Combine multiple reference dicts into one by averaging ELOs per model."""
    from collections import defaultdict
    agg = defaultdict(list)
    for ref in ref_dicts:
        for m, v in ref.items():
            agg[m].append(v)
    return {m: sum(vs) / len(vs) for m, vs in agg.items()}


def align_to_reference(our_elo, ref_elo):
    """Affine-align reconstructed ELOs to reference scale via least-squares
    over the intersection of models: ref ≈ a * ours + b."""
    common = sorted(set(our_elo.keys()) & set(ref_elo.keys()))
    if len(common) < 2:
        return our_elo
    our_vals = np.array([our_elo[m] for m in common])
    ref_vals = np.array([ref_elo[m] for m in common])
    A = np.column_stack([our_vals, np.ones_like(our_vals)])
    params, _, _, _ = np.linalg.lstsq(A, ref_vals, rcond=None)
    a, b = params
    return {m: float(a * v + b) for m, v in our_elo.items()}


def make_scatter_plot(
    our_elo, ref_elo, split_name, ref_name, output_path, title_override=None
):
    """Scatter plot: original ELO (x) vs reconstructed ELO (y) for common models."""
    if plt is None:
        print(f"  Skipping scatter plot ({output_path}): matplotlib not installed")
        return
    common = sorted(set(our_elo.keys()) & set(ref_elo.keys()))
    if len(common) < 2:
        print(f"  Skipping scatter plot: < 2 common models for {ref_name}/{split_name}")
        return
    ref_vals = np.array([ref_elo[m] for m in common])
    our_vals = np.array([our_elo[m] for m in common])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_vals, our_vals, alpha=0.6, s=30, edgecolors="k", linewidths=0.3)

    lo = min(ref_vals.min(), our_vals.min())
    hi = max(ref_vals.max(), our_vals.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1, label="y = x")

    ax.set_xlabel(f"Original ELO ({ref_name})")
    ax.set_ylabel("Reconstructed ELO")
    if title_override:
        ax.set_title(title_override)
    else:
        ax.set_title(
            f"{split_name}: Original vs Reconstructed ELO ({ref_name})\n"
            f"({len(common)} common models)"
        )
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plot: {output_path}")


def correlation_over_intersection(our_elo, ref_elo):
    """Return (spearman, kendall) over intersection of models; (None, None) if < 2 models."""
    common = sorted(set(our_elo.keys()) & set(ref_elo.keys()))
    if len(common) < 2:
        return None, None
    our_vals = np.array([our_elo[m] for m in common])
    ref_vals = np.array([ref_elo[m] for m in common])
    if spearmanr is None or kendalltau is None:
        return None, None
    r_spearman, _ = spearmanr(our_vals, ref_vals)
    r_kendall, _ = kendalltau(our_vals, ref_vals)
    return float(r_spearman), float(r_kendall)


def top_k_overlap(our_elo, ref_elo, k):
    """
    Overlap between our top-k and reference top-k (by ELO).
    Returns |top_k_our ∩ top_k_ref| / k; None if either side has fewer than k models.
    """
    if len(our_elo) < k or len(ref_elo) < k:
        return None
    our_sorted = sorted(our_elo.keys(), key=lambda m: our_elo[m], reverse=True)
    ref_sorted = sorted(ref_elo.keys(), key=lambda m: ref_elo[m], reverse=True)
    top_k_our = set(our_sorted[:k])
    top_k_ref = set(ref_sorted[:k])
    return len(top_k_our & top_k_ref) / k


def median_absolute_rank_error(our_elo, ref_elo):
    """
    For models in the intersection, compute rank in our ordering and in reference
    ordering (1 = best). Return median of |rank_our - rank_ref|; None if no common models.
    """
    common = sorted(set(our_elo.keys()) & set(ref_elo.keys()))
    if not common:
        return None
    our_sorted = sorted(our_elo.keys(), key=lambda m: our_elo[m], reverse=True)
    ref_sorted = sorted(ref_elo.keys(), key=lambda m: ref_elo[m], reverse=True)
    rank_our = {m: i + 1 for i, m in enumerate(our_sorted)}
    rank_ref = {m: i + 1 for i, m in enumerate(ref_sorted)}
    errors = [abs(rank_our[m] - rank_ref[m]) for m in common]
    return float(np.median(errors))


def compute_correlations(elos_per_split, ref_i2v, ref_t2v):
    """Compute Spearman, Kendall tau, top-k overlap, and median abs rank error.

    Returns a list of dicts with keys:
      split, reference, n_intersection, spearman, kendall, top5_overlap, top10_overlap,
      median_abs_rank_err
    """
    results = []
    for split_name in ["train", "val", "test"]:
        our_elo = elos_per_split.get(split_name)
        if our_elo is None:
            continue

        for label, ref_elo in [("i2v", ref_i2v), ("t2v", ref_t2v)]:
            n_common = len(set(our_elo.keys()) & set(ref_elo.keys()))
            s, k = correlation_over_intersection(our_elo, ref_elo)
            top5 = top_k_overlap(our_elo, ref_elo, 5)
            top10 = top_k_overlap(our_elo, ref_elo, 10)
            mare = median_absolute_rank_error(our_elo, ref_elo)
            results.append(
                {
                    "split": split_name,
                    "reference": label,
                    "n_intersection": n_common,
                    "spearman": s,
                    "kendall": k,
                    "top5_overlap": top5,
                    "top10_overlap": top10,
                    "median_abs_rank_err": mare,
                }
            )
    return results


def build_elo_from_prob_rows(rows, index_to_model, ref_original, k=K):
    """
    Reconstruct per-model ELO from rows of (index1, index2, prob_pred).
    prob_pred = P(index1 beats index2) = P(model_a beats model_b).
    Averages d_ab = k*logit(p) over index pairs that map to the same (model_a, model_b).
    """
    from collections import defaultdict

    pair_d_values = defaultdict(list)
    for a, b, p in rows:
        m_a = index_to_model.get(a)
        m_b = index_to_model.get(b)
        if m_a is None or m_b is None:
            continue
        d_ab = k * logit(p)
        pair_d_values[(m_a, m_b)].append(d_ab)
    edges = [(m_a, m_b, np.mean(ds)) for (m_a, m_b), ds in pair_d_values.items()]

    model_to_elo = solve_elos_per_model(edges, weights=None)
    if ref_original:
        model_to_elo = align_to_reference(model_to_elo, ref_original)
    return model_to_elo


def run_split(split_name, index_to_model, ref_original, k=K):
    """Run ELO reconstruction for one split; solve for per-model ELOs, align to
    the reference scale, and write CSV including original ELO."""
    prob_path = PREDICTIONS_DIR / f"pair_mlp_{split_name}_preds.csv"
    if not prob_path.exists():
        print(f"Missing {prob_path}; run train_linear_regression.py first.")
        return None

    rows = load_prob_csv(prob_path)
    model_to_elo = build_elo_from_prob_rows(rows, index_to_model, ref_original, k=k)

    out_path = OUTPUT_DIR / f"elo_{split_name}.csv"
    # Sort by ELO descending (highest first)
    sorted_models = sorted(model_to_elo.keys(), key=lambda m: model_to_elo[m], reverse=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # `elo` = reconstructed / predicted ELO, `elo_original` = reference value on same scale
        writer.writerow(['model', 'elo', 'elo_original'])
        for model in sorted_models:
            elo_pred = model_to_elo[model]
            elo_orig = ref_original.get(model, "")
            writer.writerow([model, f"{elo_pred:.2f}", f"{elo_orig:.2f}" if elo_orig != "" else ""])
    print(f"Wrote {out_path} ({len(model_to_elo)} models)")
    return model_to_elo


def print_correlations(elos_per_split, ref_i2v, ref_t2v):
    """Print Spearman, Kendall tau, top-k overlap, and median absolute rank error for each split."""
    if spearmanr is None or kendalltau is None:
        print("Skipping correlations: scipy not installed (pip install scipy)")
        return
    rows = compute_correlations(elos_per_split, ref_i2v, ref_t2v)
    print("\n" + "=" * 60)
    print("Rank correlations (our ELO vs reference, over model intersection)")
    print("=" * 60)

    # Preserve original print ordering.
    for split_name in ["train", "val", "test"]:
        print(f"\n--- {split_name} ---")
        for label in ["i2v", "t2v"]:
            match = [r for r in rows if r["split"] == split_name and r["reference"] == label]
            if not match:
                continue
            r = match[0]
            n_common = r["n_intersection"]
            s = r["spearman"]
            k = r["kendall"]
            top5 = r["top5_overlap"]
            top10 = r["top10_overlap"]
            mare = r["median_abs_rank_err"]

            parts = [f"n_intersection={n_common}"]
            if s is not None:
                parts.append(f"Spearman={s:.4f}")
                parts.append(f"Kendall={k:.4f}")
            if top5 is not None:
                parts.append(f"Top-5_overlap={top5:.4f}")
            if top10 is not None:
                parts.append(f"Top-10_overlap={top10:.4f}")
            if mare is not None:
                parts.append(f"Median_abs_rank_err={mare:.2f}")
            if s is None and top5 is None and top10 is None and mare is None:
                parts.append("(need >= 2 for correlation)")
            print(f"  {label}: " + "  ".join(parts))


def main():
    print("Reconstructing per-MODEL ELO from predicted probabilities (d_ab = k * logit(p̂), p̂ = P(a beats b))...")
    print(f"k = {K}")
    index_to_model = load_index_to_model()
    ref_i2v = load_reference_elo(I2V_REF) if I2V_REF.exists() else {}
    ref_t2v = load_reference_elo(T2V_REF) if T2V_REF.exists() else {}
    if not ref_i2v:
        print(f"Warning: {I2V_REF} not found, skipping i2v correlations.")
    if not ref_t2v:
        print(f"Warning: {T2V_REF} not found, skipping t2v correlations.")

    refs = [r for r in (ref_i2v, ref_t2v) if r]
    ref_combined = combine_references(*refs) if refs else {}
    if ref_combined:
        print(f"Combined reference: {len(ref_combined)} models "
              f"(ELO range {min(ref_combined.values()):.0f}–{max(ref_combined.values()):.0f})")

    elos_per_split = {}
    for split in ["train", "val", "test"]:
        elo = run_split(split, index_to_model, ref_combined, k=K)
        if elo is not None:
            elos_per_split[split] = elo

    print_correlations(elos_per_split, ref_i2v, ref_t2v)

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for split_name, our_elo in elos_per_split.items():
        for ref_label, ref_elo in [("i2v", ref_i2v), ("t2v", ref_t2v)]:
            if not ref_elo:
                continue
            make_scatter_plot(
                our_elo, ref_elo, split_name, ref_label,
                plots_dir / f"scatter_{split_name}_{ref_label}.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()

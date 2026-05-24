"""
Generate the simulated (Gaussian-mixture) tier and evaluate diversity measures.

Usage:
    python simulated_gmm.py [--output_dir OUTPUT_DIR] [--save_datasets] [--run_measures]

No GPU needed.

Dependencies (already pulled in by ``uv sync`` / ``pip install -e .``):
    numpy pandas scipy scikit-learn

Output:
    - .npz files (with --save_datasets): compressed (X, y) per dataset
    - CSV files (with --run_measures):
        measures_dim{d}_seed{s}.csv  -- per-dataset measure values
        measures_all.csv             -- combined across all (dim, seed)
        spearman_dim{d}_seed{s}.csv -- per-seed Spearman rho
        spearman_per_seed.csv       -- all seeds combined
        spearman_summary.csv        -- aggregated mean/std/pass_rate per (axis, dim, measure)

Balance conditions (5 levels, ordered from most to least uniform):
    uniform                 - Datapoints evenly distributed across all topics
    slight_head20_40        - Top 20% of topics account for 40% of datapoints
    mild_head20_60          - Top 20% of topics account for 60% of datapoints
    zipf                    - Topic frequencies follow a Zipfian distribution
    strong_top1_50_next4_30 - Top 1 topic: 50%, next 4: 30%, rest: 20%
"""

import argparse
import json
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np

# ──────────────────────────────────────────────────────────
# Optional imports (measures, pandas, scipy)
# These are only needed when --run_measures is passed.
# ──────────────────────────────────────────────────────────

try:
    import pandas as pd
    from scipy.stats import spearmanr
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# The EmbDivBench source tree at src/EmbDivBench/ must be importable
# (e.g. after `uv sync` or `pip install -e .` from the repo root). The
# import is attempted lazily so the script can still be used for plotting
# only (without --run_measures).
MEASURES_AVAILABLE = False
_measure_import_error = None


def _try_import_measures():
    global MEASURES_AVAILABLE, _measure_import_error
    try:
        from EmbDivBench import (  # noqa: F401
            mean_pw_dist, sum_pw_dist, cluster_inertia,
            radius, graph_entropy, chamfer_dist,
            span_centroid, span_medoid, diameter, bottleneck, energy,
            vendi_score, dcscore, log_determinant, sum_diameter,
            mst_dispersion, bins_entropy, renyi_entropy, hamdiv,
        )
        MEASURES_AVAILABLE = True
    except ImportError as e:
        _measure_import_error = str(e)
        MEASURES_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# 0. Balance condition definitions
# ──────────────────────────────────────────────────────────

# Ordered from most uniform to most skewed.
BALANCE_CONDITIONS = [
    "uniform",
    "slight_head20_40",
    "mild_head20_60",
    "zipf",
    "strong_top1_50_next4_30",
]

# Numeric ordinal level used as x-axis for Spearman rho.
# Higher level = more balanced = expected higher diversity (rho > 0).
BALANCE_CONDITION_LEVEL = {
    "strong_top1_50_next4_30": 1,
    "zipf":                    2,
    "mild_head20_60":          3,
    "slight_head20_40":        4,
    "uniform":                 5,
}


def _make_balance_counts(n_points, k, condition):
    """
    Generate deterministic class counts for a given balance condition.
    All conditions produce exactly n_points total across k classes.

    Parameters
    ----------
    n_points : int
    k : int
    condition : str  -- one of BALANCE_CONDITIONS

    Returns
    -------
    counts : np.ndarray of shape (k,), dtype int32
    """
    if condition == "uniform":
        base = n_points // k
        rem = n_points % k
        counts = np.full(k, base, dtype=np.int32)
        counts[:rem] += 1

    elif condition == "slight_head20_40":
        n_head = max(1, int(round(0.2 * k)))
        n_tail = k - n_head
        head_total = int(round(0.4 * n_points))
        tail_total = n_points - head_total
        counts = np.zeros(k, dtype=np.int32)
        base_h, rem_h = divmod(head_total, n_head)
        counts[:n_head] = base_h
        counts[:rem_h] += 1
        if n_tail > 0:
            base_t, rem_t = divmod(tail_total, n_tail)
            counts[n_head:] = base_t
            counts[n_head : n_head + rem_t] += 1

    elif condition == "mild_head20_60":
        n_head = max(1, int(round(0.2 * k)))
        n_tail = k - n_head
        head_total = int(round(0.6 * n_points))
        tail_total = n_points - head_total
        counts = np.zeros(k, dtype=np.int32)
        base_h, rem_h = divmod(head_total, n_head)
        counts[:n_head] = base_h
        counts[:rem_h] += 1
        if n_tail > 0:
            base_t, rem_t = divmod(tail_total, n_tail)
            counts[n_head:] = base_t
            counts[n_head : n_head + rem_t] += 1

    elif condition == "zipf":
        weights = np.array([1.0 / (i + 1) for i in range(k)])
        weights /= weights.sum()
        raw = (weights * n_points).astype(np.int32)
        remainder = n_points - raw.sum()
        fracs = weights * n_points - raw
        top_idx = np.argsort(fracs)[::-1][:remainder]
        raw[top_idx] += 1
        counts = raw

    elif condition == "strong_top1_50_next4_30":
        counts = np.zeros(k, dtype=np.int32)
        top1_total = int(round(0.5 * n_points))
        n_next = min(4, k - 1)
        next_total = int(round(0.3 * n_points))
        rest_total = n_points - top1_total - next_total
        counts[0] = top1_total
        base_n, rem_n = divmod(next_total, n_next)
        counts[1 : 1 + n_next] = base_n
        counts[1 : 1 + rem_n] += 1
        n_rest = k - 1 - n_next
        if n_rest > 0:
            base_r, rem_r = divmod(rest_total, n_rest)
            counts[1 + n_next :] = base_r
            counts[1 + n_next : 1 + n_next + rem_r] += 1

    else:
        raise ValueError(
            f"Unknown balance condition: '{condition}'. "
            f"Valid options: {BALANCE_CONDITIONS}"
        )

    assert counts.sum() == n_points, (
        f"[_make_balance_counts] counts.sum()={counts.sum()} != n_points={n_points}"
    )
    return counts


# ──────────────────────────────────────────────────────────
# 1. Dataset generation (Hierarchical GMM)
# ──────────────────────────────────────────────────────────

@dataclass
class SimulatedConfig:
    n_points: int = 1000
    dim: int = 768
    super_scale_coef: float = 5.0   # super-center spread ~ coef * sqrt(dim)
    class_scale_coef: float = 1.0   # class-center around super ~ coef * sqrt(dim)
    intra_scale: float = 0.5        # within-class noise (constant)


def _make_centers(seed, dim, n_classes, disparity_m, super_scale_coef, class_scale_coef,
                  *, ref_m=None):
    """
    Generate centers using per-element sub-RNGs for incrementality.

    Parameters
    ----------
    ref_m : int, optional
        Reference m for **positioning** class centers.  Class i is placed near
        super_center[i % ref_m], so its coordinates stay fixed even when
        disparity_m changes.  ``class_to_super`` still uses disparity_m for
        group labeling.  Default (None) ⇒ ref_m = disparity_m (original
        behaviour where positions depend on disparity_m).

    Incrementality guarantees
    ------------------------
    - super_center[j] depends only on (seed, j) — stable across n_classes/m.
    - class_center[i] depends only on (seed, i, i % ref_m) — stable across
      disparity_m changes when ref_m is fixed.
    - class_to_super = i % disparity_m — deterministic labeling.
    """
    pos_m = ref_m if ref_m is not None else disparity_m
    super_scale = super_scale_coef * np.sqrt(dim)
    class_scale = class_scale_coef * np.sqrt(dim)

    # Generate pos_m super-centers for positioning (stable across disparity_m)
    pos_super = np.empty((pos_m, dim), dtype=np.float32)
    for i in range(pos_m):
        rng_i = np.random.default_rng([seed, 0, i])
        pos_super[i] = rng_i.normal(0.0, super_scale, size=dim)

    # Class centers positioned relative to pos_m super-centers
    class_centers = np.empty((n_classes, dim), dtype=np.float32)
    for i in range(n_classes):
        rng_i = np.random.default_rng([seed, 1, i])
        class_centers[i] = rng_i.normal(
            loc=pos_super[i % pos_m],
            scale=class_scale, size=dim,
        )

    # Group labeling based on disparity_m (may differ from positioning)
    class_to_super = np.array([i % disparity_m for i in range(n_classes)], dtype=np.int32)

    return pos_super[:disparity_m], class_centers, class_to_super


def _sample_points(seed, class_centers, class_counts, intra_scale):
    """
    Sample points using per-class sub-RNGs for incrementality:
    - Points for class i depend only on (seed, i) and class_centers[i]
    - If class i gets cnt1 points in one setting and cnt2<cnt1 in another,
      the cnt2 points are the first cnt2 of the cnt1 points (subset).
    """
    X_parts, y_parts = [], []
    for cls, cnt in enumerate(class_counts):
        if cnt <= 0:
            continue
        rng_cls = np.random.default_rng([seed, 2, cls])
        pts = rng_cls.normal(
            loc=class_centers[cls], scale=intra_scale,
            size=(cnt, class_centers.shape[1]),
        ).astype(np.float32)
        X_parts.append(pts)
        y_parts.append(np.full(cnt, cls, dtype=np.int32))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    rng_shuffle = np.random.default_rng([seed, 3])
    idx = rng_shuffle.permutation(len(y))
    return X[idx], y[idx]


def make_variety_dataset(cfg, k, seed, disparity_m_for_variety=10):
    """
    Variety axis: vary number of classes k; balance=uniform.

    Incremental: class_centers[0:k1] is a subset of class_centers[0:k2] for k1<k2,
    and points for class i are a prefix-subset when count decreases.
    """
    m = min(disparity_m_for_variety, k)
    _, class_centers, _ = _make_centers(
        seed, cfg.dim, k, m, cfg.super_scale_coef, cfg.class_scale_coef
    )
    base = cfg.n_points // k
    rem = cfg.n_points % k
    class_counts = np.full(k, base, dtype=np.int32)
    class_counts[:rem] += 1
    X, y = _sample_points(seed, class_centers, class_counts, cfg.intra_scale)
    meta = dict(axis="variety", k=k, disparity_m=m, balance="uniform",
                level=k, seed=seed, dim=cfg.dim)
    return X, y, meta


def make_balance_dataset(cfg, k_fixed, balance_condition, seed, disparity_m_fixed=10):
    """
    Balance axis: fix k, vary topic distribution skewness.

    Incremental: all balance conditions share the same class centers (same k, same m).
    Points for class i are a prefix-subset: if class i gets 20 points under "uniform"
    and 500 under "strong", the 20 are the first 20 of the 500 (same sub-RNG).
    """
    m = min(disparity_m_fixed, k_fixed)
    _, class_centers, _ = _make_centers(
        seed, cfg.dim, k_fixed, m, cfg.super_scale_coef, cfg.class_scale_coef
    )
    class_counts = _make_balance_counts(cfg.n_points, k_fixed, balance_condition)
    X, y = _sample_points(seed, class_centers, class_counts, cfg.intra_scale)
    meta = dict(
        axis="balance",
        k=k_fixed,
        balance_condition=balance_condition,
        level=BALANCE_CONDITION_LEVEL[balance_condition],
        disparity_m=m,
        seed=seed,
        dim=cfg.dim,
    )
    return X, y, meta


def make_disparity_dataset(cfg, k_fixed, disparity_m, seed):
    """
    Disparity axis: fix k, vary number of super-groups m; balance=uniform.

    Incremental: class i is assigned to super-group (i % m).  When m increases,
    some classes stay in the same super-group (same center, same points),
    while others move to new super-groups (new center, new points).
    ~60% of datapoints are shared between consecutive m levels.
    """
    m = min(disparity_m, k_fixed)
    _, class_centers, _ = _make_centers(
        seed, cfg.dim, k_fixed, m, cfg.super_scale_coef, cfg.class_scale_coef,
    )
    base = cfg.n_points // k_fixed
    rem = cfg.n_points % k_fixed
    class_counts = np.full(k_fixed, base, dtype=np.int32)
    class_counts[:rem] += 1
    X, y = _sample_points(seed, class_centers, class_counts, cfg.intra_scale)
    meta = dict(axis="disparity", k=k_fixed, disparity_m=m, balance="uniform",
                level=m, seed=seed, dim=cfg.dim)
    return X, y, meta


# ──────────────────────────────────────────────────────────
# 2. Measure computation
# ──────────────────────────────────────────────────────────

def reduce_to_2d(embs):
    """PCA projection to 2D (used by convex_hull_volume_2d)."""
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(embs)
    except Exception:
        if embs.shape[1] >= 2:
            return embs[:, :2]
        pad = np.zeros((embs.shape[0], 2 - embs.shape[1]))
        return np.concatenate([embs, pad], axis=1)


def convex_hull_volume_2d(two_d):
    """Convex hull area in 2D PCA space. Returns None for degenerate inputs."""
    try:
        from scipy.spatial import ConvexHull
        if two_d.shape[0] < 3:
            return None
        return float(ConvexHull(two_d).volume)
    except Exception:
        return None


def compute_all_measures(X, two_d):
    """
    Compute all 22 diversity measures on one dataset.

    Parameters
    ----------
    X     : np.ndarray (n, d)
    two_d : np.ndarray (n, 2)  -- PCA projection for convex hull

    Returns
    -------
    measure_values      : dict[str, float | None]
    measure_costs : dict[str, float]  -- wall-clock seconds per measure
    """
    from EmbDivBench import (
        mean_pw_dist, sum_pw_dist, cluster_inertia,
        radius, graph_entropy, chamfer_dist,
        span_centroid, span_medoid, diameter, bottleneck, energy,
        vendi_score, dcscore, log_determinant, sum_diameter,
        mst_dispersion, bins_entropy, renyi_entropy, hamdiv,
    )

    data = X.tolist()
    measure_costs = {}
    measure_values = {}

    measure_jobs = [
        ("mean_pw_dist",     lambda: float(mean_pw_dist(data))),
        ("sum_pw_dist",        lambda: float(sum_pw_dist(data))),
        ("cluster_inertia",  lambda: float(cluster_inertia(data))),
        ("radius",           lambda: float(radius(data))),
        ("chamfer_dist", lambda: float(chamfer_dist(data))),
        ("convex_hull_volume_2d",      lambda: convex_hull_volume_2d(two_d)),
        ("span_centroid",         lambda: float(span_centroid(data))),
        ("span_medoid",           lambda: float(span_medoid(data))),
        ("diameter",                   lambda: float(diameter(data))),
        ("sum_diameter",               lambda: float(sum_diameter(data))),
        ("bottleneck",                 lambda: float(bottleneck(data))),
        ("energy",                     lambda: float(energy(data))),
        ("vendi_score",      lambda: float(vendi_score(data))),
        ("vendi_score_diversity_05",   lambda: float(vendi_score(data, q=0.5))),
        ("vendi_score_diversity_15",   lambda: float(vendi_score(data, q=1.5))),
        ("dcscore",                    lambda: float(dcscore(data))),
        ("log_determinant",  lambda: float(log_determinant(data))),
        ("mst_dispersion",             lambda: float(mst_dispersion(data))),
        ("bins_entropy",     lambda: float(bins_entropy(data))),
        ("renyi_entropy",       lambda: float(renyi_entropy(data))),
        ("graph_entropy",              lambda: float(graph_entropy(data))),
        ("hamdiv",                     lambda: float(hamdiv(data))),
    ]

    for measure_name, fn in measure_jobs:
        t0 = perf_counter()
        try:
            value = fn()
        except Exception as e:
            measure_values[f"{measure_name}_error"] = str(e)
            value = None
        measure_costs[measure_name] = float(perf_counter() - t0)
        measure_values[measure_name] = value

    measure_values["n_samples"] = int(X.shape[0])
    measure_values["dim"] = int(X.shape[1])
    measure_values["measure_runtime_total_seconds"] = float(sum(measure_costs.values()))
    return measure_values, measure_costs


MEASURE_COLS = [
    "mean_pw_dist", "sum_pw_dist", "cluster_inertia",
    "radius", "chamfer_dist", "convex_hull_volume_2d",
    "span_centroid", "span_medoid", "diameter", "sum_diameter",
    "bottleneck", "energy", "vendi_score", "vendi_score_diversity_05",
    "vendi_score_diversity_15", "dcscore", "log_determinant",
    "mst_dispersion", "bins_entropy", "renyi_entropy",
    "graph_entropy", "hamdiv",
]


def run_measures_on_group(datasets):
    """
    Run all measures on a list of (X, y, meta) tuples.
    Returns a list of row dicts suitable for pd.DataFrame.
    """
    rows = []
    for i, (X, y, meta) in enumerate(datasets):
        two_d = reduce_to_2d(X)
        measure_values, measure_costs = compute_all_measures(X, two_d)
        row = {}
        row.update(meta)
        row.update(measure_values)
        row["metric_costs_json"] = json.dumps(measure_costs)
        rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"    measures progress: {i + 1}/{len(datasets)}")
    return rows


# ──────────────────────────────────────────────────────────
# 4. Spearman correlation analysis
# ──────────────────────────────────────────────────────────

# For all three axes: higher level = more diverse => expected rho > 0
#   variety:   level = k  (more classes)
#   balance:   level = 1-5 (5=uniform, most balanced)
#   disparity: level = m  (more super-groups, more spread)
EXPECT_POSITIVE = {"variety": True, "balance": True, "disparity": True}


def spearman_by_seed(df_wide):
    """Compute per-(axis, dim, seed, measure) Spearman rho between level and measure value."""
    id_cols = ["axis", "level", "seed", "dim"]
    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=[c for c in MEASURE_COLS if c in df_wide.columns],
        var_name="measure",
        value_name="value",
    ).dropna(subset=["value"])

    out = []
    for (axis, dim, seed, measure), g in df_long.groupby(["axis", "dim", "seed", "measure"]):
        g = g.sort_values("level")
        rho, p = spearmanr(g["level"].to_numpy(), g["value"].to_numpy())
        out.append({"axis": axis, "dim": dim, "seed": seed,
                    "measure": measure, "rho": float(rho), "p": float(p)})
    return pd.DataFrame(out)


def make_spearman_summary(df_rho):
    """Aggregate Spearman rho across seeds: mean, std, pass_rate per (axis, dim, measure)."""
    agg_rows = []
    for (axis, dim, measure), g in df_rho.groupby(["axis", "dim", "measure"]):
        mean_rho = g["rho"].mean()
        std_rho = g["rho"].std(ddof=1) if len(g) > 1 else 0.0
        if EXPECT_POSITIVE.get(axis) is True:
            pass_rate = (g["rho"] > 0).mean()
        elif EXPECT_POSITIVE.get(axis) is False:
            pass_rate = (g["rho"] < 0).mean()
        else:
            pass_rate = float("nan")
        agg_rows.append({
            "axis": axis, "dim": dim, "measure": measure,
            "rho_mean": float(mean_rho), "rho_std": float(std_rho),
            "pass_rate": float(pass_rate), "n_seeds": int(len(g)),
        })
    return pd.DataFrame(agg_rows).sort_values(
        ["axis", "dim", "rho_mean"], ascending=[True, True, False]
    )


# ──────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build the simulated (Gaussian-mixture) tier and evaluate "
                    "diversity measures."
    )
    parser.add_argument("--output_dir", type=str, default="./simulated_output",
                        help="Directory for .npz files and measure CSVs")
    parser.add_argument("--measures_dir", type=str, default=None,
                        help="Directory for measure CSVs (default: output_dir/measures)")
    parser.add_argument("--save_datasets", action="store_true",
                        help="Save .npz files for each dataset")
    parser.add_argument("--run_measures", action="store_true",
                        help="Compute all diversity measures and save CSVs")

    parser.add_argument("--n_points", type=int, default=10000,
                        help="Points per dataset (default: 10000, matches the "
                             "per-file count in datasets/natural_text_data/).")
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[256, 384, 1024, 2048, 4096],
                        help="Embedding dimensions to sweep over. Default is the "
                             "full paper sweep [256, 384, 1024, 2048, 4096]; "
                             "384 specifically matches sentence-transformers/"
                             "all-MiniLM-L6-v2 (the default semantic embedding "
                             "model registered in src/EmbDivBench/axes_registry.py).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--variety_ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument(
        "--balance_conditions", type=str, nargs="+", default=BALANCE_CONDITIONS,
        help="Balance conditions (default: all 5). "
             "uniform slight_head20_40 mild_head20_60 zipf strong_top1_50_next4_30",
    )
    parser.add_argument("--disparity_ms", type=int, nargs="+", default=[10, 20, 30, 40, 50])

    args = parser.parse_args()

    # Validate balance conditions
    for cond in args.balance_conditions:
        if cond not in BALANCE_CONDITIONS:
            raise ValueError(f"Invalid balance condition: '{cond}'. Valid: {BALANCE_CONDITIONS}")

    # Validate measure dependencies
    if args.run_measures:
        if not PANDAS_AVAILABLE:
            raise ImportError("--run_measures requires: pip install pandas scipy")
        _try_import_measures()
        if not MEASURES_AVAILABLE:
            raise ImportError(
                f"--run_measures requires the EmbDivBench source tree to be "
                f"importable. Run `uv sync` (or `pip install -e .`) from the "
                f"repo root first. Import failed: {_measure_import_error}"
            )

    # Set up directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.save_datasets:
        os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)
    measures_dir = args.measures_dir or os.path.join(output_dir, "measures")
    if args.run_measures:
        os.makedirs(measures_dir, exist_ok=True)

    print("Config:")
    print(f"  dims:               {args.dims}")
    print(f"  seeds:              {args.seeds}")
    print(f"  variety_ks:         {args.variety_ks}")
    print(f"  balance_conditions: {args.balance_conditions}")
    print(f"  disparity_ms:       {args.disparity_ms}")
    print(f"  save_datasets:      {args.save_datasets}")
    print(f"  run_measures:        {args.run_measures}")
    print()

    all_metric_rows = []
    all_rho_rows = []

    for dim in args.dims:
        cfg = SimulatedConfig(n_points=args.n_points, dim=dim)
        for seed in args.seeds:
            print(f"{'='*60}")
            print(f"dim={dim}, seed={seed}")
            print(f"{'='*60}")

            group_datasets = []  # collect all (X, y, meta) for this (dim, seed)

            # ── Variety ──
            for k in args.variety_ks:
                X, y, meta = make_variety_dataset(cfg, k=k, seed=seed)
                print(f"[variety]   k={k:<3}  X={X.shape}")
                group_datasets.append((X, y, meta))
                if args.save_datasets:
                    np.savez_compressed(
                        os.path.join(output_dir, "datasets",
                                     f"variety_k{k}_dim{dim}_seed{seed}.npz"),
                        X=X, y=y,
                    )

            # ── Balance ──
            for cond in args.balance_conditions:
                X, y, meta = make_balance_dataset(
                    cfg, k_fixed=50, balance_condition=cond, seed=seed
                )
                print(f"[balance]   cond={cond:<28}  X={X.shape}")
                group_datasets.append((X, y, meta))
                if args.save_datasets:
                    np.savez_compressed(
                        os.path.join(output_dir, "datasets",
                                     f"balance_{cond}_dim{dim}_seed{seed}.npz"),
                        X=X, y=y,
                    )

            # ── Disparity ──
            for m in args.disparity_ms:
                X, y, meta = make_disparity_dataset(cfg, k_fixed=50, disparity_m=m, seed=seed)
                print(f"[disparity] m={m:<3}  X={X.shape}")
                group_datasets.append((X, y, meta))
                if args.save_datasets:
                    np.savez_compressed(
                        os.path.join(output_dir, "datasets",
                                     f"disparity_m{m}_dim{dim}_seed{seed}.npz"),
                        X=X, y=y,
                    )

            # ── Measures for this (dim, seed) group ──
            if args.run_measures:
                print(f"\nComputing measures for dim={dim}, seed={seed} "
                      f"({len(group_datasets)} datasets)...")
                rows = run_measures_on_group(group_datasets)

                # Save per-(dim, seed) CSV immediately (incremental, crash-safe)
                df_chunk = pd.DataFrame(rows)
                chunk_path = os.path.join(measures_dir, f"measures_dim{dim}_seed{seed}.csv")
                df_chunk.to_csv(chunk_path, index=False)
                print(f"  -> saved measures_dim{dim}_seed{seed}.csv  shape={df_chunk.shape}")

                df_rho_chunk = spearman_by_seed(df_chunk)
                rho_path = os.path.join(measures_dir, f"spearman_dim{dim}_seed{seed}.csv")
                df_rho_chunk.to_csv(rho_path, index=False)
                print(f"  -> saved spearman_dim{dim}_seed{seed}.csv  shape={df_rho_chunk.shape}")

                all_metric_rows.extend(rows)
                all_rho_rows.append(df_rho_chunk)

    # ── Final combined outputs ──
    if args.run_measures and all_metric_rows:
        df_all = pd.DataFrame(all_metric_rows)
        df_all.to_csv(os.path.join(measures_dir, "measures_all.csv"), index=False)
        print(f"\nSaved measures_all.csv  shape={df_all.shape}")

        df_rho_all = pd.concat(all_rho_rows, ignore_index=True)
        df_rho_all.to_csv(os.path.join(measures_dir, "spearman_per_seed.csv"), index=False)

        df_summary = make_spearman_summary(df_rho_all)
        df_summary.to_csv(os.path.join(measures_dir, "spearman_summary.csv"), index=False)
        print(f"Saved spearman_per_seed.csv  shape={df_rho_all.shape}")
        print(f"Saved spearman_summary.csv   shape={df_summary.shape}")

        # Print top-5 measures per axis
        print("\n--- Spearman summary (top 5 per axis by rho_mean) ---")
        for axis in ["variety", "balance", "disparity"]:
            sub = df_summary[df_summary["axis"] == axis].head(5)
            print(f"\n[{axis}]")
            print(sub[["dim", "measure", "rho_mean", "rho_std", "pass_rate"]].to_string(index=False))

    print(f"\nDone. All outputs in: {output_dir}")
    if args.run_measures:
        print(f"Measure CSVs in:      {measures_dir}")


if __name__ == "__main__":
    main()

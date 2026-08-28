# EmbDivBench

This repository accompanies the paper **Measuring Data Diversity with
Embeddings: A Taxonomy and Benchmark for Diversity Measures**. It contains:

1. **`src/EmbDivBench/`** — reference implementations of the 21 diversity
   measures (distance-, geometry-, graph-, and distribution-based)
   evaluated in the paper, operating on arbitrary sentence-embedding models.
2. **`datasets/`** — the natural-text tier (5 seeds × {variety,
   balance, disparity} Wikipedia datasets) used in the paper. See
   [`datasets/README.md`](datasets/README.md) for the file layout.
3. **`data_creation/`** — the scripts that build the two evaluation tiers
   used in the paper (the natural-text tier in
   [`data_creation/natural_text/`](data_creation/natural_text/), and the
   simulated tier in [`data_creation/simulated/`](data_creation/simulated/)).

> The simulated data `.npz` arrays would total ~25 GB and are
> therefore **not** shipped in this repository. They are deterministic
> outputs of `data_creation/simulated/simulated_gmm.py`, which is
> included so reviewers can regenerate them. See
> [`datasets/README.md`](datasets/README.md#simulated-tier) for the command.

## Table of Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
- [Available measures](#available-measures)
- [Reproducing the benchmarks](#reproducing-the-benchmarks)
- [Repository layout](#repository-layout)

## Quickstart

```python
from EmbDivBench import measure_diversity

texts = [
    "The cat sat on the mat.",
    "Dogs love to play fetch.",
    "It was a sunny afternoon.",
]

# Default: graph_entropy on the semantic axis (all-MiniLM-L6-v2)
measure_diversity(texts)
# Use a different diversity axis
measure_diversity(texts, diversity_axis="style")
# Use a stronger / different embedding model
measure_diversity(texts, embedding_model="Qwen/Qwen3-Embedding-8B")
# Run the core set of measures
measure_diversity(texts, measure="core")
# Run specific measures
measure_diversity(texts, measure=["mean_pw_dist", "diameter"])

# You can also call individual measures directly
from EmbDivBench import log_determinant
log_determinant(texts)
log_determinant(texts, diversity_axis="style")
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the
repo root:

```bash
uv sync
source .venv/bin/activate
```

(Or `pip install -e .` if you prefer pip; `pyproject.toml` lists all
runtime dependencies and `requires-python = ">=3.11"`.)

### Optional: `mag_areas`

The `mag_areas` measure relies on
[`aidos-lab/magnipy`](https://github.com/aidos-lab/magnipy), which pins
`scipy==1.13.0` and therefore conflicts with the `scipy>=1.16.0` used by
the rest of the code. Add it manually after `uv sync`:

```bash
pip install --no-deps "magnipy @ git+https://github.com/aidos-lab/magnipy.git@54cb6a2c64f442b339118d6922339231cdb60a82"
pip install numexpr seaborn krypy
```

At runtime `mag_areas` shims `scipy.integrate.trapz = trapezoid` (removed
in scipy 1.14) so magnipy imports cleanly on the project's scipy version.
All other measures work without this step.

## Available measures

All 21 measures live under `src/EmbDivBench/measures/` and are grouped into
four families:

| Family       | Measures |
|--------------|----------|
| Distance     | `mean_pw_dist`, `sum_pw_dist`, `energy`, `chamfer_dist`, `diameter`, `bottleneck`, `sum_diameter`, `sum_bottleneck` |
| Geometry     | `convex_hull_volume_2d`, `span_centroid`, `geo_mean_of_std`, `span_medoid`, `cluster_inertia`, `log_determinant` |
| Graph        | `graph_entropy`, `mst_dispersion` |
| Distribution | `vendi_score`, `dcscore`, `renyi_entropy`, `bins_entropy`, `mag_areas` |

Each single-dataset measure is decorated with `@accepts_text` and
registered in `src/EmbDivBench/measures_registry.py`, so it can be invoked
uniformly through `measure_diversity(...)` or called directly with raw
embeddings. `mag_areas` takes a list of datasets rather than a single
dataset and is exposed as a top-level function instead of through the
registry.

**Defaults** (when no keyword is passed):

| | Value |
|---|---|
| `diversity_axis` | `"semantic"` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| Single measure | `graph_entropy` |
| `measure="core"` set | `graph_entropy`, `log_determinant`, `mean_pw_dist`, `vendi_score`, `convex_hull_volume_2d`, `energy` (one representative per family) |

Other embedding models can be picked via `embedding_model=...`. The
`semantic` axis registers `sentence-transformers/all-mpnet-base-v2` and
`Qwen/Qwen3-Embedding-8B` as alternatives; the `style` axis uses
`AnnaWegmann/Style-Embedding` by default. Both axis registrations are in
`src/EmbDivBench/axes_registry.py`.

### Per-measure hyperparameters

All measures share the same underlying cosine geometry, but differ in the
matrix they consume: distance-based measures take the cosine *distance*
matrix, whereas Vendi Score, Rényi Kernel Entropy, DCScore and
Log-Determinant take the cosine *similarity* (Gram) matrix; for Vendi
Score, Rényi Kernel Entropy and Log-Determinant this matrix must be
positive (semi-)definite for their eigenvalue- or determinant-based
formulations to be well defined. Cluster Inertia inherits squared
Euclidean distance from K-means, and Geo. Mean of Std., Convex Hull
Volume and Bins Entropy involve no distance metric at all. "—" means the
measure takes no further parameters beyond the shared cosine metric. The
ε terms are numerical stabilizers rather than substantive choices.

| Measure | Hyperparameters |
|---|---|
| ***Distance-based*** | |
| Mean Pairwise Dist. (`mean_pw_dist`) | — |
| Sum Pairwise Dist. (`sum_pw_dist`) | — |
| Energy (`energy`) | γ=1.0, ε=1e-12 |
| Diameter (`diameter`) | — |
| Bottleneck (`bottleneck`) | — |
| Sum Diameter (`sum_diameter`) | no division by n |
| Sum Bottleneck (`sum_bottleneck`) | no division by n |
| Chamfer Dist. (`chamfer_dist`) | — |
| ***Geometry-based*** | |
| Convex Hull Vol. (2D) (`convex_hull_volume_2d`) | 2D UMAP projection, `random_state=42` |
| Span (Centroid) (`span_centroid`) | 90th percentile |
| Span (Medoid) (`span_medoid`) | — |
| Cluster Inertia (`cluster_inertia`) | K=200 |
| Geo. Mean of Std. (`geo_mean_of_std`) | no distance metric |
| Log-Determinant (`log_determinant`) | cosine kernel, τ=1.0, normalized, ε=1e-6, Cholesky |
| ***Distribution-based*** | |
| Vendi Score (`vendi_score`) | q=1.0, normalized, dual form |
| Rényi Kernel Ent. (`renyi_entropy`) | α=2.0, cosine kernel, τ=1.0, normalized, ε=1e-12 |
| DCScore (`dcscore`) | cosine kernel, τ=1.0, normalized |
| Bins Entropy (UMAP) (`bins_entropy`) | 5×5 grid, UMAP projection, effective normalization |
| MagArea (`mag_areas`) | n_ts=30 scales |
| ***Graph-based*** | |
| MST Dispersion (`mst_dispersion`) | — |
| Graph Entropy (`graph_entropy`) | — |

## Reproducing the benchmarks

The two evaluation tiers from the paper are constructed by the scripts
under `data_creation/`:

- **Natural-text tier** (Wikipedia data) — `data_creation/natural_text/`
- **Simulated tier** (Gaussian-mixture data) — `data_creation/simulated/`

See [`data_creation/README.md`](data_creation/README.md) for the exact
ordering of scripts, required external inputs (e.g. the L2 Wikipedia category
metadata), and expected outputs.

The end-to-end measure evaluation used in the paper lives at
`src/EmbDivBench/evaluate_measures.py`. The same functionality is reachable
from a command-line wrapper (`EmbDivBench --help` after `uv sync`).

## Repository layout

```
.
├── src/EmbDivBench/        # measure implementations + embedding / eval / CLI code
│   ├── measures/           # one module per diversity measure
│   ├── embeddings/         # SBERT / SimCSE helpers
│   ├── eval/               # STEL-style style evaluation data + loaders
│   ├── plot/               # plotting helpers used in the paper
│   ├── utility/            # caching and project_root helpers
│   ├── convenience.py      # `measure_diversity(...)` entry point
│   ├── evaluate_measures.py
│   └── cli.py              # command-line wrapper
├── datasets/
│   └── natural_text_data/  # natural-text tier (5 seeds × 3 axes, Wikipedia)
├── data_creation/
│   ├── natural_text/       # natural-text tier construction scripts + L2 metadata
│   └── simulated/          # simulated tier construction scripts (data regenerable)
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

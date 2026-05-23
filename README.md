# EmbDivBench — Code for EMNLP Submission

This repository accompanies an EMNLP submission on **embedding-based diversity
measurement for text datasets**. It contains:

1. **`src/EmbDivBench/`** — a Python package that implements 22 diversity
   measures (distance-, geometry-, graph-, and distribution-based)
   on top of arbitrary sentence embedding models.
2. **`data_creation/`** — the scripts that generate the two evaluation
   tiers used in the paper (synthetic simulated data and natural text data from Wikipedia).

> Reviewer note: the code is provided so that the measures, the benchmark
> construction pipeline, and the end-to-end evaluation can be reproduced.
> Some scripts under `data_creation/wiki/` contain absolute paths that
> reflect the authors' compute environment; see the notes in
> [`data_creation/README.md`](data_creation/README.md) for which arguments
> to override.

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

# Default measure (log_determinant), semantic embeddings
measure_diversity(texts)
# Use a different diversity axis
measure_diversity(texts, diversity_axis="style")
# Use a specific embedding model
measure_diversity(texts, embedding_model="Qwen/Qwen3-8B")
# Run the core set of measures
measure_diversity(texts, measure="core")
# Run specific measures
measure_diversity(texts, measure=["mean_pw_dist", "diameter"])

# You can also call individual measures directly
from EmbDivBench import log_determinant
log_determinant(texts)
log_determinant(texts, diversity_axis="style")
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone, then from the repo root:
uv sync
source .venv/bin/activate
```

A plain `pip install -e .` also works if you prefer not to use `uv`; all
runtime dependencies are listed in `pyproject.toml`.

### Optional extras

The `mag_areas` measure depends on
[`aidos-lab/magnipy`](https://github.com/aidos-lab/magnipy), which pins an
older scipy. Install it separately:

```bash
pip install -e ".[magarea]" --no-deps
```

See the comment block in `pyproject.toml` for details.

## Available measures

All 22 measures live under `src/EmbDivBench/measures/` and are grouped into
four families:

| Family       | Measures |
|--------------|----------|
| Distance     | `mean_pw_dist`, `dist_dispersion`, `energy`, `chamfer_dist` |
| Geometry     | `convex_hull_volume_2d`, `span_centroid`, `radius`, `diameter`, `bottleneck`, `span_medoid`, `sum_diameter`, `sum_bottleneck`, `cluster_inertia` |
| Graph        | `graph_entropy`, `mst_dispersion`, `hamdiv` |
| Distribution | `vendi_score`, `dcscore`, `renyi_entropy`, `log_determinant`, `bins_entropy`, `mag_areas` |

Each single-dataset measure is registered via `@accepts_text` in
`src/EmbDivBench/measures_registry.py`, so they can be invoked uniformly through
`measure_diversity(...)` or called directly with raw embeddings. `mag_areas`
has a multi-dataset API and is exposed as a module-level function rather than
through the registry.

## Reproducing the benchmarks

The two evaluation benchmarks from the paper are constructed by the scripts
under `data_creation/`:

- **Wikipedia semantic-diversity benchmark** — `data_creation/wiki/`
- **Synthetic GMM benchmark** — `data_creation/synthetic/`

See [`data_creation/README.md`](data_creation/README.md) for the exact
ordering of scripts, required external inputs (e.g. the L2 Wikipedia category
metadata), and expected outputs.

The end-to-end measure evaluation entry point used in the paper is
`src/EmbDivBench/evaluate_measures.py`; it can also be driven through the
`EmbDivBench` CLI installed by `pyproject.toml`.

## Repository layout

```
.
├── src/EmbDivBench/        # measures, embedding helpers, CLI, evaluation
│   ├── measures/           # each diversity measure as a single module
│   ├── embeddings/         # SBERT / SimCSE helpers
│   ├── eval/               # STEL-style style evaluation data + loaders
│   ├── plot/               # plotting helpers used in the paper
│   ├── utility/            # caching and project_root helpers
│   ├── convenience.py      # `measure_diversity(...)` entry point
│   ├── evaluate_measures.py
│   └── cli.py              # `EmbDivBench` command line interface
├── data_creation/
│   ├── wiki/               # Wikipedia semantic-diversity benchmark scripts
│   └── synthetic/          # Synthetic GMM benchmark scripts
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

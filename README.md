# EmbDivBench — Code for EMNLP Submission

This repository accompanies an EMNLP submission on **embedding-based diversity
measurement for text datasets**. It contains:

1. **`src/EmbDivBench/`** — reference implementations of the 22 diversity
   measures (distance-, geometry-, graph-, and distribution-based)
   evaluated in the paper, operating on arbitrary sentence-embedding models.
2. **`datasets/`** — the natural-text benchmarks (5 seeds × {variety,
   balance, disparity} Wikipedia datasets) used in the paper. See
   [`datasets/README.md`](datasets/README.md) for the file layout.
3. **`data_creation/`** — the scripts that build the two evaluation tiers
   used in the paper (the natural-text Wikipedia benchmark in
   [`data_creation/wiki/`](data_creation/wiki/), and the synthetic
   Gaussian-mixture benchmark in [`data_creation/synthetic/`](data_creation/synthetic/)).

> The synthetic Gaussian-mixture `.npz` arrays would total ~25 GB and are
> therefore **not** shipped in this repository. They are deterministic
> outputs of `data_creation/synthetic/synthetic_umap_vis.py`, which is
> included so reviewers can regenerate them. See
> [`datasets/README.md`](datasets/README.md#simulated-data) for the command.

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

All 22 measures live under `src/EmbDivBench/measures/` and are grouped into
four families:

| Family       | Measures |
|--------------|----------|
| Distance     | `mean_pw_dist`, `sum_pw_dist`, `energy`, `chamfer_dist` |
| Geometry     | `convex_hull_volume_2d`, `span_centroid`, `radius`, `diameter`, `bottleneck`, `span_medoid`, `sum_diameter`, `sum_bottleneck`, `cluster_inertia` |
| Graph        | `graph_entropy`, `mst_dispersion`, `hamdiv` |
| Distribution | `vendi_score`, `dcscore`, `renyi_entropy`, `log_determinant`, `bins_entropy`, `mag_areas` |

Each single-dataset measure is decorated with `@accepts_text` and
registered in `src/EmbDivBench/measures_registry.py`, so it can be invoked
uniformly through `measure_diversity(...)` or called directly with raw
embeddings. `mag_areas` takes a list of datasets rather than a single
dataset and is exposed as a top-level function instead of through the
registry.

## Reproducing the benchmarks

The two evaluation benchmarks from the paper are constructed by the scripts
under `data_creation/`:

- **Wikipedia semantic-diversity benchmark** — `data_creation/wiki/`
- **Synthetic GMM benchmark** — `data_creation/synthetic/`

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
│   └── natural_text_data/  # bundled Wikipedia benchmarks (5 seeds × 3 axes)
├── data_creation/
│   ├── wiki/               # Wikipedia benchmark construction scripts + L2 metadata
│   └── synthetic/          # Synthetic GMM benchmark scripts (data regenerable)
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

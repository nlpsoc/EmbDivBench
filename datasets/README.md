# Datasets

This directory ships the **natural-text tier** used in the paper. Each
file contains 10 000 Wikipedia sentence extracts drawn from a controlled
distribution of topics. The simulated tier is not shipped (see below);
its `.npz` arrays are regenerable from a script.

## Natural-text tier (`natural_text_data/`)

Built by `data_creation/natural_text/4_build_natural_text_bench.py` — see
[`../data_creation/README.md`](../data_creation/README.md) for the
construction pipeline.

```
natural_text_data/
├── clean/        # plain text, one sentence per line (10 000 lines per file)
│   └── seed_{42..46}/
│       ├── variety/    variety_k{10,20,30,40,50}_clean.txt
│       ├── balance/    balance_{uniform, slight_head20_40, mild_head20_60,
│       │                        zipf, strong_top1_50_next4_30}_clean.txt
│       └── disparity/  disparity_m{10,20,30,40,50}_clean.txt
└── labelled/     # TSV with columns: text, topic, parent_L1
    └── seed_{42..46}/
        ├── variety/    variety_k{10,20,30,40,50}.tsv
        ├── balance/    balance_<condition>.tsv
        └── disparity/  disparity_m{10,20,30,40,50}.tsv
```

Three diversity axes are evaluated with aligned constructions:

- **variety** — `k` ∈ {10, 20, 30, 40, 50} controls the number of distinct
  L2 topics (more topics → more semantic variety).
- **balance** — 50 topics held fixed; five count distributions ranging from
  uniform to a strong top-1 50% / next-4 30% skew.
- **disparity** — 50 topics drawn from `m` ∈ {10, 20, 30, 40, 50} supporting
  L1 categories (fewer supporting L1s → more concentrated coverage, lower
  disparity).

Each axis × condition combination is provided across 5 random seeds
(42–46) for variance estimation.

## Simulated tier

The simulated Gaussian-mixture datasets used in the paper are **not
shipped** because the `.npz` arrays total ~25 GB across all `(dim, seed,
condition)` combinations and exceed practical repository size limits.

They can be regenerated deterministically by running

```bash
bash data_creation/simulated/simulated_gmm.sh
```

or, with explicit options,

```bash
python data_creation/simulated/simulated_gmm.py \
    --output_dir ./datasets/simulated_data \
    --save_datasets \
    --run_metrics
```

The script writes one `.npz` per `(dim, seed, axis, level)` cell, plus the
Spearman-ρ summary CSVs that go into the paper's tables.

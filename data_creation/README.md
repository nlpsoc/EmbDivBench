# Benchmark construction

This directory contains the two pipelines used to build the evaluation
tiers reported in the paper:

- **`natural_text/`** — natural-text tier (Wikipedia data). Builds aligned
  variety / balance / disparity datasets by running farthest-point sampling
  (FPS) over Qwen3-Embedding-8B representations of Wikipedia L1/L2 category
  labels.
- **`simulated/`** — simulated tier (Gaussian-mixture data) plus diversity
  metric evaluation on a controlled set of variety, balance, and disparity
  factors.

Both pipelines are runnable on a single machine; the natural-text pipeline
requires GPU only for step 2 (embedding precomputation). Once the embedding
cache exists, every later step runs on CPU.

---

## Natural-text tier (Wikipedia data)

Pipeline (run in numerical order):

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `natural_text/1_inspect_l1_pool.py` | Inspect the viable L1 candidate pool from `L2_all.json`. Optional sweep mode prints how pool size changes across thresholds. |
| 2 | `natural_text/2_precompute_fps_embeddings.py` (`.sh`) | Embed every L1 and L2 label with Qwen3-Embedding-8B and write the cache used by step 4. **GPU recommended.** |
| 3 | `natural_text/3_fps_l1_selection_check.py` (`.sh`) | Diagnostic: verifies that seed-randomised FPS over the L1 embeddings actually yields seed-varied selections. Outputs `fps_l1_selections.json`. |
| 4 | `natural_text/4_build_natural_text_bench.py` (`.sh`) | Build the aligned variety / balance / disparity datasets across 5 seeds. CPU only after step 2. |

`natural_text/fps_l1_selections.json` is the diagnostic output captured
from the authors' run of step 3, kept here as a reference for what step 3
should produce.

### Inputs

The Wikipedia category metadata needed by the pipeline ships with the repo:

- `natural_text/metadata/L2/L2_all.json` (3.0 MB, 11 554 L2 records) — the
  catalog of L1/L2 Wikipedia categories with `topic_label`,
  `category_title`, `direct_pages`, `parent_L1`, etc. produced by a
  separate scraping pass.
- `natural_text/metadata/L2/viable_l1_min30_min5.txt` — pre-filtered list
  of L1 categories used by step 3.

The only external resources you need are:

1. `Qwen/Qwen3-Embedding-8B` from Hugging Face (fetched automatically on
   first use; pass a local model path via `--model_path` if you have one
   already downloaded).
2. Network access for the live Wikipedia article fetch in step 4.

The `.sh` wrappers resolve every path relative to the script location, so
you can run them from any working directory:

```bash
bash natural_text/2_precompute_fps_embeddings.sh    # GPU step
bash natural_text/3_fps_l1_selection_check.sh       # diagnostic
bash natural_text/4_build_natural_text_bench.sh     # main build, CPU
```

You can override `EMB_MODEL`, `EMB_CACHE`, `OUT_BASE` via environment
variables, or invoke the underlying `.py` files directly with the
appropriate `--*` flags.

### Output

`4_build_natural_text_bench.py` writes JSON datasets for variety
(`k = 10/20/30/40/50`), balance (50 topics, varying skew), and disparity
(`m = 10/20/30/40/50` supporting L1 categories) — 5 seeds per condition.
The default output root is `natural_text/output/datasets/natural_text/`.
The same 5-seed output is also pre-built and bundled at
[`../datasets/natural_text_data/`](../datasets/natural_text_data/), so
reviewers can skip this step entirely.

---

## Simulated tier (Gaussian-mixture data)

```bash
bash simulated/simulated_gmm.sh
```

or, with finer control:

```bash
python simulated/simulated_gmm.py \
    --output_dir ./simulated_output \
    --save_datasets \
    --run_metrics
```

No GPU required. The script:

- generates Gaussian-mixture datasets across five balance regimes (uniform,
  slight head 20/40, mild head 20/60, Zipfian, strong top-1 50%);
- writes the raw datasets as `.npz` when `--save_datasets` is passed;
- evaluates every registered diversity measure (when `--run_metrics` is
  passed) and emits per-seed Spearman-ρ summary CSVs.

Like the natural-text wrappers, `simulated/simulated_gmm.sh` resolves all
paths relative to its own location (`simulated/output/` by default) and
can also be submitted via `sbatch`.

---

## Reproducibility notes

- Both pipelines are seeded; the natural-text pipeline reports across 5
  seeds and the simulated pipeline sweeps `(dim, seed)` combinations.
- The `.sh` wrappers resolve all paths relative to their own location and
  pull configurable values from environment variables (`EMB_MODEL`,
  `EMB_CACHE`, `OUT_BASE`, …). The original SLURM headers are kept so the
  wrappers can also be submitted via `sbatch`; under plain `bash` they are
  ignored.
- The pipelines do **not** ship the raw Wikipedia article corpus; only the
  category metadata + label embeddings are needed at build time, and step 4
  fetches article text on demand from Wikipedia via `wikipediaapi`.

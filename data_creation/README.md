# Benchmark construction

This directory contains the two pipelines used to build the evaluation
benchmarks reported in the paper:

- **`wiki/`** — Wikipedia semantic-diversity benchmark. Builds aligned
  variety / balance / disparity datasets by running farthest-point sampling
  (FPS) over Qwen3-Embedding-8B representations of Wikipedia L1/L2 category
  labels.
- **`synthetic/`** — Synthetic Gaussian-mixture benchmark plus 3D UMAP
  visualisation and metric evaluation on a controlled set of variety,
  balance, and disparity factors.

Both pipelines are runnable on a single machine; the Wikipedia pipeline
requires GPU only for step 2 (embedding precomputation). Once the embedding
cache exists, every later step runs on CPU.

---

## Wikipedia semantic-diversity benchmark

Pipeline (run in numerical order):

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `wiki/1_inspect_l1_pool.py` | Inspect the viable L1 candidate pool from `L2_all.json`. Optional sweep mode prints how pool size changes across thresholds. |
| 2 | `wiki/2_precompute_fps_embeddings.py` (`.sh`) | Embed every L1 and L2 label with Qwen3-Embedding-8B and write the cache used by step 4. **GPU recommended.** |
| 3 | `wiki/3_fps_l1_selection_check.py` (`.sh`) | Diagnostic: verifies that seed-randomised FPS over the L1 embeddings actually yields seed-varied selections. Outputs `fps_l1_selections.json`. |
| 4 | `wiki/4_build_wiki_semdiv_shuffle.py` (`.sh`) | Build the aligned variety / balance / disparity datasets across 5 seeds. CPU only after step 2. |

`wiki/fps_l1_selections.json` is the diagnostic output captured from the
authors' run of step 3, kept here as a reference for what step 3 should
produce.

### External inputs required

Step 2 expects a Wikipedia category metadata file (`L2_all.json`) and a local
copy of the Qwen3-Embedding-8B model. Both are passed as CLI arguments — the
sample `.sh` scripts contain the absolute paths used by the authors:

```bash
python3 wiki/2_precompute_fps_embeddings.py \
    --l2_json /path/to/L2_all.json \
    --min_direct_pages 30 \
    --model_path /path/to/Qwen3-Embedding-8B \
    --cache_dir /path/to/cache/fps_embeddings \
    --device cuda --batch_size 32
```

Reviewers reproducing the pipeline should:

1. Obtain `L2_all.json` (the scraped Wikipedia category metadata). The
   scraper used to produce this file is not included in this repository.
2. Download `Qwen3-Embedding-8B` from Hugging Face.
3. Replace the `/hpc/...` paths in the `.sh` wrappers with their own paths,
   or invoke the `.py` scripts directly with the appropriate `--*` flags.

### Output

`4_build_wiki_semdiv_shuffle.py` writes JSON datasets for variety
(`k = 10/20/30/40/50`), balance (50 topics, varying skew), and disparity
(`m = 10/20/30/40/50` supporting L1 categories) — 5 seeds per condition.

---

## Synthetic GMM benchmark

Single script:

```bash
python synthetic/synthetic_umap_vis.py \
    --output_dir ./synthetic_output \
    --save_datasets \
    --run_metrics
```

No GPU required — UMAP on 1 000 points runs in seconds on CPU. The script:

- generates Gaussian-mixture datasets across five balance regimes (uniform,
  slight head 20/40, mild head 20/60, Zipfian, strong top-1 50%);
- produces interactive 3D UMAP HTML plots (one per dataset × UMAP-parameter
  combination);
- optionally writes the raw datasets as `.npz`;
- optionally evaluates every registered diversity measure and emits per-seed
  Spearman-ρ summary CSVs.

The Slurm wrapper at `synthetic/synthetic_umap_vis.sh` shows the CPU-only
configuration used in the paper (and again contains the authors' absolute
paths — feel free to ignore it if you are running locally).

---

## Reproducibility notes

- Both pipelines are seeded; the Wikipedia pipeline reports across 5 seeds
  and the synthetic pipeline sweeps `(dim, seed)` combinations.
- Hardcoded `/hpc/...` paths in the `.sh` wrappers reflect the authors'
  compute environment. Replace them or call the underlying `.py` scripts
  directly with your own paths.
- The pipelines do **not** ship the raw Wikipedia article corpus; only the
  category metadata + label embeddings are needed at build time, and the
  scripts fetch article text on demand from the configured Wikipedia source.

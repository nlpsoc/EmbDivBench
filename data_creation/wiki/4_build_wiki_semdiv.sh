#!/bin/bash
#SBATCH --job-name=build_wiki_semdiv
#SBATCH --partition=cpu
#SBATCH --time=30:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=/PATH/TO/wiki_scraper/logs/%x_%j.out
#SBATCH --error=/PATH/TO/wiki_scraper/logs/%x_%j.err

set -euo pipefail

cd /PATH/TO/wiki_scraper
source .venv/bin/activate

export HF_HOME=/PATH/TO/.cache/huggingface
export TRANSFORMERS_CACHE=/PATH/TO/.cache/huggingface

SCRIPT=/PATH/TO/embediver/data_creation/wiki/build_wiki_semdiv_shuffle.py
L2_JSON=/PATH/TO/wiki_scraper/metadata/L2/L2_all.json
OUT_BASE=/PATH/TO/wiki_scraper/output/datasets/labelled/wiki
EMB_MODEL=/PATH/TO/models/embedding/Qwen3-Embedding-8B
EMB_CACHE=/PATH/TO/wiki_scraper/cache/fps_embeddings

# NOTE: If you haven't precomputed the FPS embeddings on GPU yet, the first
# seed below will compute them on CPU (~30-60 min for ~2000 labels). To avoid
# this, run `precompute_fps_embeddings.sbatch` once on GPU first; it populates
# ${EMB_CACHE}, after which every seed below skips embedding entirely.

for SEED in 42 43 44 45 46; do
  echo "========== seed=${SEED} =========="
  python3 "${SCRIPT}" \
    --l2_json "${L2_JSON}" \
    --out_root "${OUT_BASE}/seed_${SEED}_" \
    --seed ${SEED} \
    --total 10000 \
    --min_direct_pages 30 \
    --max_per_topic 5000 \
    --cost_json "${OUT_BASE}/seed_${SEED}_/cost_log_wiki_semdiv.json" \
    --min_chars 100 \
    --max_chars 300 \
    --base_depth 1 \
    --max_depth_limit 3 \
    --base_articles 4000 \
    --max_articles_limit 20000 \
    --base_cap 5 \
    --max_cap_limit 40 \
    --expand_rounds 5 \
    --sleep 0.01 \
    --allow_parent_L1_fallback \
    --embedding_model_path "${EMB_MODEL}" \
    --embedding_cache_dir "${EMB_CACHE}" \
    --embedding_batch_size 8

done

#!/bin/bash
# Step 4 of the Wikipedia semantic-diversity pipeline.
# Builds variety / balance / disparity datasets across 5 seeds. CPU only
# after step 2 has populated the FPS embedding cache; if that cache is
# empty, the first seed below recomputes ~2k label embeddings on CPU
# (~30-60 min). Article text itself is fetched live from Wikipedia.
#
# Run from any cwd:
#   bash data_creation/natural_text/4_build_natural_text_bench.sh
# or under SLURM:
#   sbatch data_creation/natural_text/4_build_natural_text_bench.sh

#SBATCH --job-name=build_natural_text_bench
#SBATCH --partition=cpu
#SBATCH --time=30:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT="${SCRIPT_DIR}/4_build_natural_text_bench.py"
L2_JSON="${SCRIPT_DIR}/metadata/L2/L2_all.json"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/output/datasets/natural_text}"
EMB_MODEL="${EMB_MODEL:-Qwen/Qwen3-Embedding-8B}"
EMB_CACHE="${EMB_CACHE:-${SCRIPT_DIR}/cache/fps_embeddings}"

mkdir -p "${OUT_BASE}" "${EMB_CACHE}" "${SCRIPT_DIR}/logs"

for SEED in 42 43 44 45 46; do
  echo "========== seed=${SEED} =========="
  python3 "${SCRIPT}" \
    --l2_json "${L2_JSON}" \
    --out_root "${OUT_BASE}/seed_${SEED}_" \
    --seed ${SEED} \
    --total 10000 \
    --min_direct_pages 30 \
    --max_per_topic 5000 \
    --cost_json "${OUT_BASE}/seed_${SEED}_/cost_log_natural_text.json" \
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

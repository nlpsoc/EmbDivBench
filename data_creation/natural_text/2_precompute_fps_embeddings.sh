#!/bin/bash
# Step 2 of the Wikipedia semantic-diversity pipeline.
# Precomputes Qwen3-Embedding-8B vectors for every L1 / L2 label so that
# the seeded FPS in step 4 reuses cached embeddings instead of re-encoding
# ~2k labels per seed. GPU recommended; ~1-2 minutes on a single GPU.
#
# Run from any cwd:
#   bash data_creation/natural_text/2_precompute_fps_embeddings.sh
# or under SLURM:
#   sbatch data_creation/natural_text/2_precompute_fps_embeddings.sh

#SBATCH --job-name=precompute_fps_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT="${SCRIPT_DIR}/2_precompute_fps_embeddings.py"
L2_JSON="${SCRIPT_DIR}/metadata/L2/L2_all.json"
EMB_MODEL="${EMB_MODEL:-Qwen/Qwen3-Embedding-8B}"   # HF id; override with a local path if desired
EMB_CACHE="${EMB_CACHE:-${SCRIPT_DIR}/cache/fps_embeddings}"

mkdir -p "${EMB_CACHE}" "${SCRIPT_DIR}/logs"

python3 "${SCRIPT}" \
  --l2_json "${L2_JSON}" \
  --min_direct_pages 30 \
  --model_path "${EMB_MODEL}" \
  --cache_dir "${EMB_CACHE}" \
  --device cuda \
  --batch_size 32

echo "[done] FPS embedding cache ready at ${EMB_CACHE}"

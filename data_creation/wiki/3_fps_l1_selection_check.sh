#!/bin/bash
# Step 3 of the Wikipedia semantic-diversity pipeline (diagnostic).
# Verifies that seed-randomised FPS over Qwen3-embedded L1 labels actually
# produces seed-varied selections (and not nearly the same set each time).
# Output: fps_l1_selections.json in this directory.
#
# Run from any cwd:
#   bash data_creation/wiki/3_fps_l1_selection_check.sh
# or under SLURM:
#   sbatch data_creation/wiki/3_fps_l1_selection_check.sh

#SBATCH --job-name=fps_l1_selection_check
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EMB_MODEL="${EMB_MODEL:-Qwen/Qwen3-Embedding-8B}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cache}"

mkdir -p "${CACHE_DIR}" "${SCRIPT_DIR}/logs"

python3 "${SCRIPT_DIR}/3_fps_l1_selection_check.py" \
  --l1_list_txt "${SCRIPT_DIR}/metadata/L2/viable_l1_min30_min5.txt" \
  --model_path "${EMB_MODEL}" \
  --seeds 42 43 44 45 46 \
  --k_max 50 \
  --k_levels 10 20 30 40 50 \
  --cache_embeddings "${CACHE_DIR}/l1_emb_qwen3_8b.npy" \
  --out_json "${SCRIPT_DIR}/fps_l1_selections.json"

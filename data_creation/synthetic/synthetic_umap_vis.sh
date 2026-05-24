#!/bin/bash
# Synthetic GMM benchmark + UMAP visualisation + Spearman-rho metric eval.
# No GPU required.
#
# Run from any cwd:
#   bash data_creation/synthetic/synthetic_umap_vis.sh
# or under SLURM:
#   sbatch data_creation/synthetic/synthetic_umap_vis.sh

#SBATCH --job-name=umap-vis-metrics
#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/output/umap_vis_output}"

mkdir -p "${OUT_DIR}" "${SCRIPT_DIR}/logs"

# Disable tokenizer parallelism / GPU; this script is CPU-only.
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

START_TIME=$(date +%s)

python -u "${SCRIPT_DIR}/synthetic_umap_vis.py" \
  --output_dir "${OUT_DIR}" \
  --no_plots \
  --run_metrics

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo
echo "[done] Total runtime: ${RUNTIME}s ($(awk "BEGIN {printf \"%.3f\", ${RUNTIME}/3600}") h)"
echo "Outputs written to: ${OUT_DIR}"

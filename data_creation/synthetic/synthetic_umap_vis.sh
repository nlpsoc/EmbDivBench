#!/bin/bash
#SBATCH --job-name=umap-vis-metrics
#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/PATH/TO/EmbDivBench/local_misc/other_tests/logs/%x_%j.out
#SBATCH --error=/PATH/TO/EmbDivBench/local_misc/other_tests/logs/%x_%j.err


set -euo pipefail

# -------------------------
# Basic job info
# -------------------------

echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Working dir (before cd): $(pwd)"
echo

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

cd /PATH/TO/EmbDivBench
source .venv/bin/activate

echo "Working dir (after cd): $(pwd)"
echo "Python: $(which python)"
python --version
echo

# -------------------------
# Runtime tracking
# -------------------------
START_TIME=$(date +%s)

python -u local_misc/other_tests/synthetic_umap_vis.py \
  --output_dir ./local_misc/other_tests/output/umap_vis_output \
  --no_plots \
  --run_metrics

EXIT_CODE=$?

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo
echo "=== Job finished at $(date) ==="
echo "Exit code: ${EXIT_CODE}"
echo "Total runtime (seconds): ${RUNTIME}"
echo "Total runtime (hours): $(awk "BEGIN {printf \"%.3f\", ${RUNTIME}/3600}")"

# -------------------------
# Memory & CPU usage summary
# -------------------------
echo
echo "=== Slurm accounting (sacct) ==="
sacct -j "${SLURM_JOB_ID}" \
  --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State

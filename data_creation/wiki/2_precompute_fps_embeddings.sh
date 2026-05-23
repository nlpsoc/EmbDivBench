#!/bin/bash
#SBATCH --job-name=precompute_fps_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/PATH/TO/wiki_scraper/logs/%x_%j.out
#SBATCH --error=/PATH/TO/wiki_scraper/logs/%x_%j.err

set -euo pipefail

cd /PATH/TO/wiki_scraper
source .venv/bin/activate

export HF_HOME=/PATH/TO/.cache/huggingface
export TRANSFORMERS_CACHE=/PATH/TO/.cache/huggingface

SCRIPT=/PATH/TO/embediver/data_creation/wiki/precompute_fps_embeddings.py
L2_JSON=/PATH/TO/wiki_scraper/metadata/L2/L2_all.json
EMB_MODEL=/PATH/TO/models/embedding/Qwen3-Embedding-8B
EMB_CACHE=/PATH/TO/wiki_scraper/cache/fps_embeddings

python3 "${SCRIPT}" \
  --l2_json "${L2_JSON}" \
  --min_direct_pages 30 \
  --model_path "${EMB_MODEL}" \
  --cache_dir "${EMB_CACHE}" \
  --device cuda \
  --batch_size 32

echo "[done] FPS embedding cache ready at ${EMB_CACHE}"

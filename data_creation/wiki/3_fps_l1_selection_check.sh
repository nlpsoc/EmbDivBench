#!/bin/bash
#SBATCH --job-name=fps_l1_selection_check
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=/PATH/TO/wiki_scraper/logs/%x_%j.out
#SBATCH --error=/PATH/TO/wiki_scraper/logs/%x_%j.err

set -euo pipefail
cd /PATH/TO/wiki_scraper
source .venv/bin/activate

export HF_HOME=/PATH/TO/.cache/huggingface
export TRANSFORMERS_CACHE=/PATH/TO/.cache/huggingface

python3 /PATH/TO/embediver/data_creation/wiki/fps_l1_selection_check.py \
  --l1_list_txt /PATH/TO/wiki_scraper/metadata/L2/viable_l1_min30_min5.txt \
  --model_path /PATH/TO/models/embedding/Qwen3-Embedding-8B \
  --seeds 42 43 44 45 46 \
  --k_max 50 \
  --k_levels 10 20 30 40 50 \
  --cache_embeddings /PATH/TO/wiki_scraper/cache/l1_emb_qwen3_8b.npy \
  --out_json /PATH/TO/embediver/data_creation/wiki/fps_l1_selections.json
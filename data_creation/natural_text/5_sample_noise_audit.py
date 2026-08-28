#!/usr/bin/env python3
"""Sample 10 sentences per (component, level, seed) for a blind per-level manual noise audit."""
import csv, hashlib, random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "datasets" / "natural_text_data" / "labelled"
SEEDS = ["seed_42", "seed_43", "seed_44", "seed_45", "seed_46"]
N_PER_DATASET = 10

# level_idx 0..4 ordering follows the BALANCE_ORDER convention in
# sample_sentences_with_candidates.py: from least diverse to most diverse.
#   variety    - ascending k (fewer topics -> more topics)
#   balance    - strongest skew -> uniform
#   disparity  - ascending m (fewer supporting L1s -> more supporting L1s)
LEVEL_FILES = {
    "variety": [
        "variety_k10.tsv",
        "variety_k20.tsv",
        "variety_k30.tsv",
        "variety_k40.tsv",
        "variety_k50.tsv",
    ],
    "balance": [
        "balance_strong_top1_50_next4_30.tsv",
        "balance_zipf.tsv",
        "balance_mild_head20_60.tsv",
        "balance_slight_head20_40.tsv",
        "balance_uniform.tsv",
    ],
    "disparity": [
        "disparity_m10.tsv",
        "disparity_m20.tsv",
        "disparity_m30.tsv",
        "disparity_m40.tsv",
        "disparity_m50.tsv",
    ],
}

def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

def stable_seed(component, level_idx, seed):
    key = f"{component}|{level_idx}|{seed}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

def main():
    key_rows, uid = [], 0
    for comp, files in LEVEL_FILES.items():
        for level_idx, fname in enumerate(files):
            for seed in SEEDS:
                path = ROOT / seed / comp / fname
                rows = read_tsv(path)
                pool = sorted({r["topic"] for r in rows})
                rng = random.Random(stable_seed(comp, level_idx, seed))
                for r in rng.sample(rows, min(N_PER_DATASET, len(rows))):
                    key_rows.append({
                        "uid": uid, "component": comp, "level_idx": level_idx,
                        "level_file": fname, "seed": seed, "text": r["text"],
                        "topic_pool": " | ".join(pool), "wikipedia_topic": r["topic"],
                    })
                    uid += 1

    random.Random(20260807).shuffle(key_rows)   # blind audit: shuffle

    with open("manual_noise_audit_KEY.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=[
            "uid","component","level_idx","level_file","seed","text","topic_pool","wikipedia_topic"])
        w.writeheader(); w.writerows(key_rows)

    with open("manual_noise_audit_BLIND.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=["uid","text","topic_pool","your_topic"])
        w.writeheader()
        for r in key_rows:
            w.writerow({"uid": r["uid"], "text": r["text"],
                        "topic_pool": r["topic_pool"], "your_topic": ""})

    print(f"Exported {len(key_rows)} sentences for manual audit "
          f"({len(LEVEL_FILES)} components x 5 levels x {len(SEEDS)} seeds x {N_PER_DATASET}).")

if __name__ == "__main__":
    main()

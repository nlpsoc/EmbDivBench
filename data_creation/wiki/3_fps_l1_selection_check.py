#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fps_l1_selection_check.py

Goal: check whether seed-randomized FPS over Qwen3-embedded L1 labels
actually produces seed-varied L1 selections, or whether greedy FPS
converges to nearly the same set regardless of starting point.

Pipeline:
  1. Load viable L1 names (from a text file or from L2_all.json).
  2. Embed each L1 label with Qwen3-Embedding-8B (local path).
  3. For each seed, pick a random start index (seeded RNG),
     then run greedy FPS to select k_max items (incremental ordering).
  4. Report pairwise Jaccard overlap at k in {10, 20, 30, 40, 50} and
     the "stable core" (intersection across all seeds) at each k.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# I/O: get the L1 candidate pool
# -----------------------------------------------------------------------------
def load_l1_names_from_text(path: str) -> List[str]:
    """Parse lines of the form '  49  Government by country' or plain names."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^(\d+)\s+(.+)$", s)
        names.append(m.group(2).strip() if m else s)
    # Deduplicate, preserve order.
    seen = set()
    unique: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def load_l1_names_from_json(
    l2_json: str, min_direct_pages: int, min_l2_per_l1: int
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Returns (ordered viable L1 list, {L1 -> [L2 labels]})."""
    with open(l2_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dedup by topic_label, keep richer record.
    best: Dict[str, dict] = {}
    for x in data:
        if x.get("level") != "L2":
            continue
        dp = int(x.get("direct_pages", 0))
        if dp < min_direct_pages:
            continue
        label = (x.get("topic_label") or "").strip()
        if not label:
            continue
        old = best.get(label)
        rich = (dp, int(x.get("est_pages", 0)))
        rich_old = (
            (int(old.get("direct_pages", 0)), int(old.get("est_pages", 0)))
            if old
            else (-1, -1)
        )
        if old is None or rich > rich_old:
            best[label] = x

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for x in best.values():
        l1 = (x.get("parent_L1") or "UNKNOWN_L1").strip() or "UNKNOWN_L1"
        grouped[l1].append(x)

    viable = [(l1, ts) for l1, ts in grouped.items() if len(ts) >= min_l2_per_l1]
    viable.sort(key=lambda p: (-len(p[1]), p[0]))
    names = [l1 for l1, _ in viable]
    l2_labels = {
        l1: sorted(
            (t.get("topic_label") or "").strip() for t in ts if (t.get("topic_label") or "").strip()
        )
        for l1, ts in viable
    }
    return names, l2_labels


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
def build_encode_inputs(
    names: List[str],
    l2_labels: Optional[Dict[str, List[str]]],
    enrich_with_l2: bool,
    max_l2_per_l1: int,
) -> List[str]:
    if not enrich_with_l2 or not l2_labels:
        return list(names)
    enriched: List[str] = []
    for n in names:
        children = (l2_labels.get(n) or [])[:max_l2_per_l1]
        if children:
            enriched.append(f"Topic: {n}. Subtopics: {', '.join(children)}.")
        else:
            enriched.append(f"Topic: {n}.")
    return enriched


def embed_names(
    texts: List[str],
    model_path: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Embed with sentence-transformers. L2-normalized."""
    # Lazy import so the script still lists its args quickly without torch installed.
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading model from: {model_path}")
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    print(f"[embed] encoding {len(texts)} strings (bs={batch_size}) on {device}")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"[embed] shape={emb.shape}, dtype={emb.dtype}")
    return emb.astype(np.float32)


# -----------------------------------------------------------------------------
# FPS
# -----------------------------------------------------------------------------
def fps_selection(emb: np.ndarray, k: int, start_idx: int) -> List[int]:
    """Greedy farthest-point sampling on L2-normalized embeddings.

    Because inputs are unit-normalized, Euclidean distance is a monotonic
    function of cosine distance, so FPS on L2-norm is FPS on cosine.
    Returns indices in selection order (selected[0] = start_idx).
    """
    N = emb.shape[0]
    if k > N:
        raise ValueError(f"k={k} > pool size {N}")
    selected: List[int] = [start_idx]
    dists = np.linalg.norm(emb - emb[start_idx], axis=1)  # 1-D
    for _ in range(k - 1):
        nxt = int(np.argmax(dists))
        selected.append(nxt)
        new_d = np.linalg.norm(emb - emb[nxt], axis=1)
        dists = np.minimum(dists, new_d)
    return selected


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def print_per_seed_topk(names: List[str], seed_to_order: Dict[int, List[int]], k: int) -> None:
    print("\n" + "=" * 78)
    print(f"Top-{k} FPS selections per seed")
    print("=" * 78)
    for seed, order in seed_to_order.items():
        print(f"\nseed={seed}  start={names[order[0]]!r}")
        for rank, idx in enumerate(order[:k], 1):
            print(f"  {rank:2d}. {names[idx]}")


def print_pairwise_overlap(
    names: List[str], seed_to_order: Dict[int, List[int]], k_levels: List[int]
) -> None:
    seeds = list(seed_to_order.keys())
    for k in k_levels:
        print("\n" + "=" * 78)
        print(f"Pairwise overlap at k={k}")
        print("=" * 78)
        header = "        " + "  ".join(f"{s:>6d}" for s in seeds)
        print(header)
        for s1 in seeds:
            row = [f"seed {s1:>3d}:"]
            set1 = set(seed_to_order[s1][:k])
            for s2 in seeds:
                if s1 == s2:
                    row.append(f"{'-':>6}")
                else:
                    set2 = set(seed_to_order[s2][:k])
                    row.append(f"{jaccard(set1, set2):.3f}")
            print("  ".join(row))
        # Stable core
        core = set(seed_to_order[seeds[0]][:k])
        for s in seeds[1:]:
            core &= set(seed_to_order[s][:k])
        print(f"\n  Stable core at k={k} (items present in ALL {len(seeds)} seeds): "
              f"{len(core)}/{k}")


def print_stable_and_variable(
    names: List[str], seed_to_order: Dict[int, List[int]], k: int
) -> None:
    seeds = list(seed_to_order.keys())
    sets = {s: set(seed_to_order[s][:k]) for s in seeds}
    # Count per-item frequency across seeds.
    freq: Dict[int, int] = defaultdict(int)
    for s in seeds:
        for idx in sets[s]:
            freq[idx] += 1
    print("\n" + "=" * 78)
    print(f"Item frequency across seeds at k={k}")
    print("=" * 78)
    by_freq: Dict[int, List[int]] = defaultdict(list)
    for idx, c in freq.items():
        by_freq[c].append(idx)
    for c in sorted(by_freq.keys(), reverse=True):
        items = sorted(by_freq[c], key=lambda i: names[i])
        print(f"\n  appeared in {c}/{len(seeds)} seeds ({len(items)} items):")
        for idx in items:
            print(f"    - {names[idx]}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--l1_list_txt",
        type=str,
        default=None,
        help="Text file listing L1 candidates (count-prefixed or plain). Mutually exclusive with --l2_json.",
    )
    ap.add_argument("--l2_json", type=str, default=None)
    ap.add_argument("--min_direct_pages", type=int, default=30)
    ap.add_argument("--min_l2_per_l1", type=int, default=5)

    ap.add_argument(
        "--model_path",
        type=str,
        default="/PATH/TO/models/embedding/Qwen3-Embedding-8B",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu; auto-detected if None")
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument(
        "--enrich_with_l2",
        action="store_true",
        help="Augment each L1 label with its L2 children when embedding. Requires --l2_json.",
    )
    ap.add_argument("--max_l2_per_l1", type=int, default=10)

    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--k_max", type=int, default=50)
    ap.add_argument("--k_levels", type=int, nargs="+", default=[10, 20, 30, 40, 50])

    ap.add_argument("--out_json", type=str, default="fps_l1_selections.json")
    ap.add_argument("--cache_embeddings", type=str, default=None,
                    help="Optional .npy path to cache the embedding matrix for reruns.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Respect the HPC cache convention.
    os.environ.setdefault("HF_HOME", "/PATH/TO/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/PATH/TO/.cache/huggingface")

    # Auto device.
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    # Load L1 names (+ optional L2 labels for enrichment).
    if args.l1_list_txt and args.l2_json:
        raise ValueError("Specify only one of --l1_list_txt or --l2_json.")
    l2_labels: Optional[Dict[str, List[str]]] = None
    if args.l1_list_txt:
        names = load_l1_names_from_text(args.l1_list_txt)
    elif args.l2_json:
        names, l2_labels = load_l1_names_from_json(
            args.l2_json, args.min_direct_pages, args.min_l2_per_l1
        )
    else:
        raise ValueError("Provide either --l1_list_txt or --l2_json.")

    print(f"[pool] {len(names)} L1 candidates loaded")
    if len(names) < args.k_max:
        raise ValueError(f"Pool too small: {len(names)} < k_max={args.k_max}")

    if args.enrich_with_l2 and l2_labels is None:
        print("[warn] --enrich_with_l2 requires --l2_json; falling back to plain labels.")
        args.enrich_with_l2 = False

    texts = build_encode_inputs(
        names, l2_labels, args.enrich_with_l2, args.max_l2_per_l1
    )
    print(f"[pool] example encode inputs:")
    for t in texts[:3]:
        print(f"   {t!r}")

    # Embed (with optional cache).
    if args.cache_embeddings and Path(args.cache_embeddings).exists():
        emb = np.load(args.cache_embeddings)
        print(f"[embed] loaded cached embeddings: {emb.shape}")
        if emb.shape[0] != len(texts):
            raise ValueError(
                f"Cached embeddings size {emb.shape[0]} != current pool size {len(texts)}."
                " Delete the cache file."
            )
    else:
        emb = embed_names(texts, args.model_path, args.device, args.batch_size)
        if args.cache_embeddings:
            Path(args.cache_embeddings).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.cache_embeddings, emb)
            print(f"[embed] saved cache -> {args.cache_embeddings}")

    # FPS per seed.
    seed_to_order: Dict[int, List[int]] = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, len(names)))
        order = fps_selection(emb, args.k_max, start_idx)
        seed_to_order[seed] = order

    # Reports.
    print_per_seed_topk(names, seed_to_order, k=min(10, args.k_max))
    print_pairwise_overlap(names, seed_to_order, k_levels=args.k_levels)
    print_stable_and_variable(names, seed_to_order, k=min(10, args.k_max))

    # Save full ordering + overlap summary.
    summary = {
        "pool_size": len(names),
        "pool_names": names,
        "enrich_with_l2": bool(args.enrich_with_l2),
        "model_path": args.model_path,
        "k_max": args.k_max,
        "seeds": {
            str(s): {
                "start_idx": int(order[0]),
                "start_name": names[order[0]],
                "ordering_indices": [int(i) for i in order],
                "ordering_names": [names[i] for i in order],
            }
            for s, order in seed_to_order.items()
        },
        "pairwise_jaccard": {
            str(k): {
                f"{s1}_vs_{s2}": jaccard(
                    set(seed_to_order[s1][:k]), set(seed_to_order[s2][:k])
                )
                for i, s1 in enumerate(args.seeds)
                for s2 in args.seeds[i + 1 :]
            }
            for k in args.k_levels
        },
        "stable_core_sizes": {
            str(k): len(
                set.intersection(*[set(seed_to_order[s][:k]) for s in args.seeds])
            )
            for k in args.k_levels
        },
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[done] saved -> {args.out_json}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_fps_embeddings.py

Precompute L1 and L2 label embeddings with Qwen3-Embedding-8B and save to the
FPS cache directory. Running this once on GPU is ~1-2 minutes, after which
the main build script reads from cache on CPU across all 5 seeds.

Usage (single GPU job, run once before launching the 5-seed CPU job):

    python3 precompute_fps_embeddings.py \
        --l2_json /PATH/TO/metadata/L2/L2_all.json \
        --min_direct_pages 30 \
        --model_path /PATH/TO/models/embedding/Qwen3-Embedding-8B \
        --cache_dir /PATH/TO/cache/fps_embeddings \
        --device cuda --batch_size 32

Idempotent: re-running only computes embeddings for any new names, so it is
safe to re-invoke after e.g. lowering min_direct_pages.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_l1_l2(l2_json: str, min_direct_pages: int) -> (List[str], List[str]):
    with open(l2_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dedup by topic_label (keep richer one), then filter by min_direct_pages.
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
            if old else (-1, -1)
        )
        if old is None or rich > rich_old:
            best[label] = x

    l1_names = sorted({(x.get("parent_L1") or "UNKNOWN_L1").strip() or "UNKNOWN_L1"
                       for x in best.values()})
    l2_labels = sorted(best.keys())
    return l1_names, l2_labels


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        names = [str(n) for n in data["names"]]
        emb = data["emb"]
        return {n: emb[i] for i, n in enumerate(names)}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}; starting fresh")
        return {}


def save_cache(d: Dict[str, np.ndarray], path: Path) -> None:
    """Atomic write: np.savez auto-appends '.npz' if the filename doesn't already
    end in it, which breaks naive tmp-rename patterns. Pass a file handle instead
    so numpy writes to exactly the path we give it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(d.keys())
    emb = np.stack([d[n] for n in names]).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp")  # e.g. l1_embeddings.npz.tmp
    with open(tmp, "wb") as f:
        np.savez(f, names=np.asarray(names), emb=emb)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2_json", type=str, required=True)
    ap.add_argument("--min_direct_pages", type=int, default=30)
    ap.add_argument(
        "--model_path",
        type=str,
        default="/PATH/TO/models/embedding/Qwen3-Embedding-8B",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default="/PATH/TO/cache/fps_embeddings",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda/cpu; auto-detect if None")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/PATH/TO/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/PATH/TO/.cache/huggingface")

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    cache_dir = Path(args.cache_dir)
    l1_cache = cache_dir / "l1_embeddings.npz"
    l2_cache = cache_dir / "l2_embeddings.npz"

    l1_names, l2_labels = load_l1_l2(args.l2_json, args.min_direct_pages)
    print(f"[pool] {len(l1_names)} L1 names | {len(l2_labels)} L2 labels "
          f"(min_direct_pages={args.min_direct_pages})")

    l1_map = load_cache(l1_cache)
    l2_map = load_cache(l2_cache)
    l1_missing = [n for n in l1_names if n not in l1_map]
    l2_missing = [n for n in l2_labels if n not in l2_map]

    print(f"[cache] L1 hit: {len(l1_names) - len(l1_missing)}/{len(l1_names)}; "
          f"L2 hit: {len(l2_labels) - len(l2_missing)}/{len(l2_labels)}")

    if not l1_missing and not l2_missing:
        print("[done] cache already complete; nothing to compute.")
        return

    print(f"[model] loading {args.model_path} on {args.device}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_path, device=args.device, trust_remote_code=True)

    if l1_missing:
        print(f"[embed] encoding {len(l1_missing)} L1 labels (bs={args.batch_size})")
        new = model.encode(
            l1_missing,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for n, e in zip(l1_missing, new.astype(np.float32)):
            l1_map[n] = e
        save_cache(l1_map, l1_cache)
        print(f"[save] L1 cache -> {l1_cache}")

    if l2_missing:
        print(f"[embed] encoding {len(l2_missing)} L2 labels (bs={args.batch_size})")
        new = model.encode(
            l2_missing,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for n, e in zip(l2_missing, new.astype(np.float32)):
            l2_map[n] = e
        save_cache(l2_map, l2_cache)
        print(f"[save] L2 cache -> {l2_cache}")

    print("[done]")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect the L1 candidate pool from L2_all.json.

A viable L1 is one that has at least `min_l2_per_l1` child L2 topics
each with at least `min_direct_pages` direct pages.

Usage:
    python inspect_l1_pool.py --l2_json /path/to/L2_all.json
    python inspect_l1_pool.py --l2_json ... --min_direct_pages 30 --min_l2_per_l1 5
    python inspect_l1_pool.py --l2_json ... --sweep

The --sweep flag prints a small table across common threshold combinations
so you can see how the pool size changes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_l2(l2_json_path: str) -> List[dict]:
    with open(l2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if x.get("level") == "L2"]


def group_by_l1(
    l2_records: List[dict],
    min_direct_pages: int,
) -> Dict[str, List[dict]]:
    """Return {L1 -> [L2 records with direct_pages >= min_direct_pages]},
    deduplicated by topic_label (keep richer one)."""
    # Dedup by topic_label globally, keep the richer one.
    best: Dict[str, dict] = {}
    for x in l2_records:
        label = (x.get("topic_label") or "").strip()
        if not label:
            continue
        if int(x.get("direct_pages", 0)) < min_direct_pages:
            continue
        key = label
        old = best.get(key)
        if old is None or (
            int(x.get("direct_pages", 0)),
            int(x.get("est_pages", 0)),
        ) > (
            int(old.get("direct_pages", 0)),
            int(old.get("est_pages", 0)),
        ):
            best[key] = x

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for x in best.values():
        l1 = (x.get("parent_L1") or "UNKNOWN_L1").strip() or "UNKNOWN_L1"
        grouped[l1].append(x)
    for l1 in grouped:
        grouped[l1].sort(
            key=lambda t: (
                -int(t.get("direct_pages", 0)),
                -int(t.get("est_pages", 0)),
                t.get("topic_label", ""),
            )
        )
    return grouped


def filter_viable_l1s(
    grouped: Dict[str, List[dict]],
    min_l2_per_l1: int,
) -> List[Tuple[str, int]]:
    """Return list of (L1, n_viable_L2) sorted by n_viable_L2 desc, for L1s with >= min_l2_per_l1."""
    out = [(l1, len(topics)) for l1, topics in grouped.items() if len(topics) >= min_l2_per_l1]
    out.sort(key=lambda p: (-p[1], p[0]))
    return out


def print_pool_summary(
    l2_records: List[dict],
    min_direct_pages: int,
    min_l2_per_l1: int,
    show_top: int = 50,
    show_all_names: bool = False,
) -> None:
    grouped = group_by_l1(l2_records, min_direct_pages=min_direct_pages)
    viable = filter_viable_l1s(grouped, min_l2_per_l1=min_l2_per_l1)

    print("=" * 72)
    print(f"min_direct_pages = {min_direct_pages}")
    print(f"min_l2_per_l1    = {min_l2_per_l1}")
    print("-" * 72)
    print(f"Total L1 groups (any size): {len(grouped)}")
    print(f"Viable L1 groups          : {len(viable)}")
    if viable:
        sizes = [n for _, n in viable]
        print(
            f"  viable-L2 count per L1: min={min(sizes)}, median={sorted(sizes)[len(sizes)//2]}, "
            f"max={max(sizes)}, total_viable_L2={sum(sizes)}"
        )
    print("-" * 72)

    limit = len(viable) if show_all_names else min(show_top, len(viable))
    print(f"Top {limit} viable L1 groups by #viable L2:")
    for l1, n in viable[:limit]:
        print(f"  {n:4d}  {l1}")
    if not show_all_names and len(viable) > show_top:
        print(f"  ... and {len(viable) - show_top} more")
    print()


def sweep(l2_records: List[dict]) -> None:
    print("Sweep over (min_direct_pages, min_l2_per_l1):\n")
    header = f"{'min_pages':>10} | " + " | ".join(f"ml2>={m:>2}" for m in [3, 5, 8, 10, 15])
    print(header)
    print("-" * len(header))
    for mdp in [10, 20, 30, 50, 100]:
        grouped = group_by_l1(l2_records, min_direct_pages=mdp)
        cells = []
        for ml2 in [3, 5, 8, 10, 15]:
            n = len(filter_viable_l1s(grouped, min_l2_per_l1=ml2))
            cells.append(f"{n:>6}")
        print(f"{mdp:>10} | " + " | ".join(cells))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--l2_json", type=str, required=True)
    p.add_argument("--min_direct_pages", type=int, default=30)
    p.add_argument("--min_l2_per_l1", type=int, default=5)
    p.add_argument("--show_top", type=int, default=50, help="How many viable L1s to print by name")
    p.add_argument("--show_all_names", action="store_true", help="Print all viable L1 names, not just top N")
    p.add_argument("--sweep", action="store_true", help="Print a threshold sweep table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    l2_records = load_l2(args.l2_json)
    print(f"Loaded {len(l2_records)} raw L2 records from {args.l2_json}\n")

    if args.sweep:
        sweep(l2_records)

    print_pool_summary(
        l2_records,
        min_direct_pages=args.min_direct_pages,
        min_l2_per_l1=args.min_l2_per_l1,
        show_top=args.show_top,
        show_all_names=args.show_all_names,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Wikipedia datasets for a semantic-diversity benchmark with aligned,
incremental constructions for variety / balance / disparity.

Compared with the previous count-based selection, this version picks L1
categories and L2 subtopics via farthest-point sampling (FPS) in the
Qwen3-Embedding-8B space, seeded per run so different seeds yield different
topic coverage while still giving a well-spread set each time.

Design summary
--------------
1) Variety + Balance share one master world:
   - exactly 10 L1 groups, 5 L2 topics per L1, 50 topics total
   - L1s are chosen by FPS over all L1s with >=5 viable L2s in embedding space
   - L2s within each L1 are chosen by FPS over that L1's viable L2s
   - topic order is round-robin across the 10 L1s
   - variety k = 10/20/30/40/50 uses prefixes of that round-robin order
   - balance uses the full 50-topic master set

2) Disparity uses a second FPS world (different seed offset):
   - fixed total topics k = 50
   - support levels m = 10/20/30/40/50
   - the first m accepted L1s (in FPS order) supply the topics at each m

3) Sentence pools are built adaptively and cached per topic.
4) The seed controls (i) the FPS starting index for L1 and L2 selection,
   (ii) intra-category article/sentence sampling. Variety/balance and
   disparity use independent seed offsets to keep the two worlds decoupled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import wikipediaapi
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# Text utils
# -----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def clean_sentence(s: str) -> Optional[str]:
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def is_ok_length(s: str, min_chars: int, max_chars: Optional[int]) -> bool:
    if min_chars and len(s) < min_chars:
        return False
    if max_chars is not None and len(s) > max_chars:
        return False
    return True


# -----------------------------------------------------------------------------
# Wikipedia client
# -----------------------------------------------------------------------------
def make_wiki(lang: str, user_agent: str, connect_timeout: int, read_timeout: int, retries: int):
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent=user_agent,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )
    try:
        wiki._session.timeout = (connect_timeout, read_timeout)
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        wiki._session.mount("https://", adapter)
        wiki._session.mount("http://", adapter)
    except Exception:
        pass
    return wiki


def safe_page(wiki, title: str):
    try:
        p = wiki.page(title)
        if not p.exists():
            return None
        return p
    except (ReadTimeout, ReqConnectionError, TimeoutError, ReadTimeoutError):
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Topic universe
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class L2Topic:
    parent_L0: str
    parent_L1: str
    topic_label: str
    category_title: str
    direct_pages: int
    direct_subcats: int
    est_pages: int


def load_L2_all_json(path: str, min_direct_pages: int) -> List[L2Topic]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[L2Topic] = []
    for x in data:
        if x.get("level") != "L2":
            continue
        dp = int(x.get("direct_pages", 0))
        if dp < min_direct_pages:
            continue

        label = (x.get("topic_label") or "").strip()
        cat = (x.get("category_title") or "").strip()
        if not label or not cat:
            continue

        out.append(
            L2Topic(
                parent_L0=(x.get("parent_L0") or "").strip(),
                parent_L1=(x.get("parent_L1") or "UNKNOWN_L1").strip() or "UNKNOWN_L1",
                topic_label=label,
                category_title=cat,
                direct_pages=dp,
                direct_subcats=int(x.get("direct_subcats", 0)),
                est_pages=int(x.get("est_pages", dp)),
            )
        )

    # Deduplicate by topic label: keep richer one.
    best: Dict[str, L2Topic] = {}
    for t in out:
        old = best.get(t.topic_label)
        if old is None or (t.direct_pages, t.est_pages) > (old.direct_pages, old.est_pages):
            best[t.topic_label] = t

    return list(best.values())


def group_by_L1(topics: Sequence[L2Topic]) -> Dict[str, List[L2Topic]]:
    g: Dict[str, List[L2Topic]] = defaultdict(list)
    for t in topics:
        g[t.parent_L1].append(t)
    for l1 in g:
        g[l1].sort(key=lambda t: (-t.direct_pages, -t.est_pages, t.topic_label))
    return g


# -----------------------------------------------------------------------------
# Embedding + FPS
# -----------------------------------------------------------------------------
def _load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        names = [str(n) for n in data["names"]]
        emb = data["emb"]
        return {n: emb[i] for i, n in enumerate(names)}
    except Exception as e:
        print(f"[embed] WARN: failed to load cache {path}: {e}; starting fresh")
        return {}


def _save_npz_dict(d: Dict[str, np.ndarray], path: Path) -> None:
    """Atomic write. np.savez auto-appends '.npz' if the filename doesn't
    already end in it, so we pass a file handle (no name mangling).
    Names are stored as a Unicode array (not object/pickle) so load works
    with allow_pickle=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(d.keys())
    emb = np.stack([d[n] for n in names]).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, names=np.asarray(names), emb=emb)
    tmp.replace(path)


def prepare_embeddings(
    byL1: Dict[str, List[L2Topic]],
    cache_dir: Path,
    model_path: str,
    device: str,
    batch_size: int,
    allow_compute: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return (l1_emb_map, l2_emb_map). Loads cache if present; otherwise
    computes missing entries with sentence-transformers (if allow_compute)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    l1_cache = cache_dir / "l1_embeddings.npz"
    l2_cache = cache_dir / "l2_embeddings.npz"

    all_l1_names = sorted(byL1.keys())
    all_l2_labels = sorted({t.topic_label for topics in byL1.values() for t in topics})

    l1_map = _load_npz_dict(l1_cache)
    l2_map = _load_npz_dict(l2_cache)

    l1_missing = [n for n in all_l1_names if n not in l1_map]
    l2_missing = [n for n in all_l2_labels if n not in l2_map]

    print(
        f"[embed] pool: {len(all_l1_names)} L1 | {len(all_l2_labels)} L2; "
        f"cache has {len(all_l1_names) - len(l1_missing)}/{len(all_l1_names)} L1 and "
        f"{len(all_l2_labels) - len(l2_missing)}/{len(all_l2_labels)} L2"
    )

    if l1_missing or l2_missing:
        if not allow_compute:
            raise RuntimeError(
                f"Embedding cache is incomplete (L1 missing {len(l1_missing)}, "
                f"L2 missing {len(l2_missing)}) and --no_embedding_compute was set. "
                f"Run precompute_fps_embeddings.py first."
            )

        print(f"[embed] loading model: {model_path} on device={device}")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

        if l1_missing:
            print(f"[embed] encoding {len(l1_missing)} new L1 labels (bs={batch_size})")
            new = model.encode(
                l1_missing,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for n, e in zip(l1_missing, new.astype(np.float32)):
                l1_map[n] = e
            _save_npz_dict(l1_map, l1_cache)
            print(f"[embed] saved L1 cache -> {l1_cache}")

        if l2_missing:
            print(f"[embed] encoding {len(l2_missing)} new L2 labels (bs={batch_size})")
            new = model.encode(
                l2_missing,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for n, e in zip(l2_missing, new.astype(np.float32)):
                l2_map[n] = e
            _save_npz_dict(l2_map, l2_cache)
            print(f"[embed] saved L2 cache -> {l2_cache}")

    return l1_map, l2_map


def derive_l2_seed(main_seed: int, l1_name: str) -> int:
    """Stable per-(seed, L1) seed for L2 FPS start, independent of PYTHONHASHSEED."""
    h = hashlib.sha256(f"{main_seed}|{l1_name}".encode("utf-8")).hexdigest()[:8]
    return int(h, 16) & 0x7FFFFFFF


def fps_order(
    candidate_names: List[str],
    emb_map: Dict[str, np.ndarray],
    seed: int,
) -> List[str]:
    """Return full FPS ordering of candidate_names using L2-normalized embeddings.

    Start index is drawn from a seeded permutation (method 1): different seeds
    almost always map to different start items, and the greedy FPS proceeds
    from there.
    """
    if not candidate_names:
        return []

    missing = [n for n in candidate_names if n not in emb_map]
    if missing:
        raise KeyError(
            f"Missing embeddings for {len(missing)} candidates (e.g. {missing[:3]!r})"
        )

    emb = np.stack([emb_map[n] for n in candidate_names])  # (N, D), assumed unit-norm
    N = emb.shape[0]

    rng = np.random.default_rng(seed)
    start_idx = int(rng.permutation(N)[0])

    selected: List[int] = [start_idx]
    dists = np.linalg.norm(emb - emb[start_idx], axis=1)
    for _ in range(N - 1):
        nxt = int(np.argmax(dists))
        selected.append(nxt)
        new_d = np.linalg.norm(emb - emb[nxt], axis=1)
        dists = np.minimum(dists, new_d)

    return [candidate_names[i] for i in selected]


# -----------------------------------------------------------------------------
# Balance allocations (mirrors the simulated tier)
# -----------------------------------------------------------------------------
BALANCE_CONDITIONS = [
    "uniform",
    "slight_head20_40",
    "mild_head20_60",
    "zipf",
    "strong_top1_50_next4_30",
]

VARIETY_KS = [10, 20, 30, 40, 50]
DISPARITY_MS = [10, 20, 30, 40, 50]


def alloc_uniform(n_points: int, k: int) -> List[int]:
    base, rem = divmod(n_points, k)
    out = [base] * k
    for i in range(rem):
        out[i] += 1
    return out


def alloc_balance_condition(n_points: int, k: int, condition: str) -> List[int]:
    if condition == "uniform":
        return alloc_uniform(n_points, k)

    if condition in {"slight_head20_40", "mild_head20_60"}:
        frac = 0.4 if condition == "slight_head20_40" else 0.6
        n_head = max(1, int(round(0.2 * k)))
        n_tail = k - n_head
        head_total = int(round(frac * n_points))
        tail_total = n_points - head_total
        out = [0] * k
        hb, hr = divmod(head_total, n_head)
        tb, tr = divmod(tail_total, max(1, n_tail))
        for i in range(n_head):
            out[i] = hb + (1 if i < hr else 0)
        if n_tail > 0:
            for i in range(n_tail):
                out[n_head + i] = tb + (1 if i < tr else 0)
        return out

    if condition == "zipf":
        weights = [1.0 / (i + 1) for i in range(k)]
        z = sum(weights)
        weights = [w / z for w in weights]
        raw = [int(w * n_points) for w in weights]
        rem = n_points - sum(raw)
        fracs = [(weights[i] * n_points - raw[i], i) for i in range(k)]
        fracs.sort(reverse=True)
        for _, idx in fracs[:rem]:
            raw[idx] += 1
        return raw

    if condition == "strong_top1_50_next4_30":
        out = [0] * k
        top1 = int(round(0.5 * n_points))
        n_next = min(4, k - 1)
        next_total = int(round(0.3 * n_points))
        rest_total = n_points - top1 - next_total
        out[0] = top1
        if n_next > 0:
            b, r = divmod(next_total, n_next)
            for i in range(n_next):
                out[1 + i] = b + (1 if i < r else 0)
        n_rest = k - 1 - n_next
        if n_rest > 0:
            b, r = divmod(rest_total, n_rest)
            for i in range(n_rest):
                out[1 + n_next + i] = b + (1 if i < r else 0)
        return out

    raise ValueError(f"Unknown balance condition: {condition}")


# -----------------------------------------------------------------------------
# Category crawling + adaptive pool building  (unchanged)
# -----------------------------------------------------------------------------
def collect_articles_shallow(
    wiki,
    category_title: str,
    max_depth: int,
    max_articles: int,
    sleep_s: float,
    rng: random.Random,
) -> List[str]:
    root = category_title if category_title.startswith("Category:") else f"Category:{category_title}"
    stack = [(root, 0)]
    seen_cats = set()
    seen_articles = set()
    articles: List[str] = []

    while stack and len(articles) < max_articles:
        cat, depth = stack.pop()
        if cat in seen_cats or depth > max_depth:
            continue
        seen_cats.add(cat)

        page = safe_page(wiki, cat)
        if page is None:
            continue

        try:
            members = list(page.categorymembers.values())
        except Exception:
            continue
        finally:
            if sleep_s:
                time.sleep(sleep_s)

        rng.shuffle(members)
        for m in members:
            if m.ns == wikipediaapi.Namespace.MAIN:
                if m.title not in seen_articles:
                    seen_articles.add(m.title)
                    articles.append(m.title)
                    if len(articles) >= max_articles:
                        break
            elif m.ns == wikipediaapi.Namespace.CATEGORY and depth < max_depth:
                stack.append((m.title, depth + 1))

    rng.shuffle(articles)
    return articles


def extend_pool_once(
    wiki,
    topic: L2Topic,
    pool: Set[str],
    *,
    target_size: int,
    cat_max_depth: int,
    cat_max_articles: int,
    min_chars: int,
    max_chars: Optional[int],
    per_article_sentence_cap: int,
    sleep_s: float,
    page_text_cache: Dict[str, str],
    seed: int,
) -> int:
    seed_text = f"{seed}|{topic.topic_label}|{cat_max_depth}|{cat_max_articles}"
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    articles = collect_articles_shallow(
        wiki=wiki,
        category_title=topic.category_title,
        max_depth=cat_max_depth,
        max_articles=cat_max_articles,
        sleep_s=sleep_s,
        rng=rng,
    )

    added_total = 0
    for title in articles:
        if len(pool) >= target_size:
            break

        if title in page_text_cache:
            text = page_text_cache[title]
        else:
            p = safe_page(wiki, title)
            if p is None:
                continue
            try:
                text = p.text or ""
            except Exception:
                continue
            finally:
                if sleep_s:
                    time.sleep(sleep_s)
            page_text_cache[title] = text

        if not text.strip():
            continue

        sents = split_sentences(text)
        rng.shuffle(sents)
        added_here = 0
        for s in sents:
            s = clean_sentence(s)
            if not s or not is_ok_length(s, min_chars, max_chars):
                continue
            if s in pool:
                continue
            pool.add(s)
            added_total += 1
            added_here += 1
            if len(pool) >= target_size or added_here >= per_article_sentence_cap:
                break

    return added_total


def ensure_topic_pool(
    wiki,
    topic: L2Topic,
    required: int,
    *,
    pool_dir: Path,
    cat_depth_schedule: Sequence[int],
    cat_articles_schedule: Sequence[int],
    min_chars: int,
    max_chars: Optional[int],
    per_article_sentence_cap_schedule: Sequence[int],
    sleep_s: float,
    page_text_cache: Dict[str, str],
    seed: int,
    verbose: bool,
) -> List[str]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_path = pool_dir / f"{safe_filename(topic.topic_label)}.json"
    pool: List[str] = []
    if pool_path.exists():
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)

    pool_set = set(pool)

    for round_idx, (depth, max_articles, per_cap) in enumerate(
        zip(cat_depth_schedule, cat_articles_schedule, per_article_sentence_cap_schedule), start=1
    ):
        if len(pool_set) >= required:
            break
        added = extend_pool_once(
            wiki,
            topic,
            pool_set,
            target_size=required,
            cat_max_depth=depth,
            cat_max_articles=max_articles,
            min_chars=min_chars,
            max_chars=max_chars,
            per_article_sentence_cap=per_cap,
            sleep_s=sleep_s,
            page_text_cache=page_text_cache,
            seed=seed + round_idx,
        )
        if verbose:
            print(
                f"    pool round {round_idx}: topic={topic.topic_label!r} depth={depth} "
                f"max_articles={max_articles} per_cap={per_cap} added={added} size={len(pool_set)}/{required}"
            )

    pool = list(pool_set)
    random.Random(seed).shuffle(pool)
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False)
    return pool


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")[:180] or "topic"


def write_dataset(rows: List[Dict[str, str]], out_tsv: Path, out_txt: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("text\ttopic\tparent_L1\n")
        for r in rows:
            text = r["text"].replace("\t", " ")
            f.write(f"{text}\t{r['topic']}\t{r['parent_L1']}\n")
    with open(out_txt, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["text"] + "\n")


# -----------------------------------------------------------------------------
# Topic planning
# -----------------------------------------------------------------------------
def required_variety_counts(total: int) -> Dict[Tuple[int, int], int]:
    req: Dict[Tuple[int, int], int] = {}
    max_per_column = {
        0: max(alloc_uniform(total, 10)),
        1: max(alloc_uniform(total, 20)),
        2: max(alloc_uniform(total, 30)),
        3: max(alloc_uniform(total, 40)),
        4: max(alloc_uniform(total, 50)),
    }
    for col in range(5):
        for l1_slot in range(10):
            req[(col, l1_slot)] = max_per_column[col]
    return req


def required_balance_counts(total: int) -> Dict[int, int]:
    req = [0] * 50
    for cond in BALANCE_CONDITIONS:
        alloc = alloc_balance_condition(total, 50, cond)
        for i in range(50):
            req[i] = max(req[i], alloc[i])
    return {i: req[i] for i in range(50)}


def choose_variety_balance_master(
    wiki,
    byL1: Dict[str, List[L2Topic]],
    *,
    l1_emb_map: Dict[str, np.ndarray],
    l2_emb_map: Dict[str, np.ndarray],
    total: int,
    out_dir: Path,
    min_chars: int,
    max_chars: Optional[int],
    cat_depth_schedule: Sequence[int],
    cat_articles_schedule: Sequence[int],
    per_article_sentence_cap_schedule: Sequence[int],
    sleep_s: float,
    seed: int,
    verbose: bool,
) -> Tuple[List[str], List[List[L2Topic]], Dict[str, List[str]], Dict[str, object]]:
    """FPS-ordered L1 + FPS-ordered L2 selection, with viability checks unchanged."""
    variety_req = required_variety_counts(total)
    balance_req = required_balance_counts(total)
    rr_max_req_by_slot = {(col, l1_slot): balance_req[col * 10 + l1_slot] for col in range(5) for l1_slot in range(10)}
    for key, val in variety_req.items():
        rr_max_req_by_slot[key] = max(rr_max_req_by_slot[key], val)

    # Candidate pool: L1s with >=5 a priori viable L2s (already filtered by min_direct_pages).
    viable_l1_pool = sorted(l1 for l1, topics in byL1.items() if len(topics) >= 5)
    if len(viable_l1_pool) < 10:
        raise RuntimeError(f"Only {len(viable_l1_pool)} L1 candidates have >=5 viable L2s; need at least 10.")
    l1_fps_order = fps_order(viable_l1_pool, l1_emb_map, seed=seed)
    print(f"[var/bal master] L1 pool size={len(viable_l1_pool)}, FPS start={l1_fps_order[0]!r}")

    page_text_cache: Dict[str, str] = {}
    pools: Dict[str, List[str]] = {}
    chosen_l1s: List[str] = []
    chosen_topics_by_l1: List[List[L2Topic]] = []
    l1_fps_trace: List[Dict[str, object]] = []

    pool_dir = out_dir / "topic_pools_variety_balance"

    for l1 in l1_fps_order:
        if len(chosen_l1s) >= 10:
            break

        # FPS-order L2s within this L1.
        candidates_raw = byL1[l1]
        l2_labels_here = [t.topic_label for t in candidates_raw]
        l2_seed = derive_l2_seed(seed, l1)
        ordered_l2_labels = fps_order(l2_labels_here, l2_emb_map, seed=l2_seed)
        name_to_topic = {t.topic_label: t for t in candidates_raw}
        candidates = [name_to_topic[lb] for lb in ordered_l2_labels]

        accepted: List[L2Topic] = []
        rejected_here: List[Tuple[str, int, int]] = []
        for cand in candidates:
            if len(accepted) >= 5:
                break
            col = len(accepted)
            l1_slot = len(chosen_l1s)
            need = rr_max_req_by_slot[(col, l1_slot)]
            pool = ensure_topic_pool(
                wiki,
                cand,
                need,
                pool_dir=pool_dir,
                cat_depth_schedule=cat_depth_schedule,
                cat_articles_schedule=cat_articles_schedule,
                per_article_sentence_cap_schedule=per_article_sentence_cap_schedule,
                min_chars=min_chars,
                max_chars=max_chars,
                sleep_s=sleep_s,
                page_text_cache=page_text_cache,
                seed=seed,
                verbose=verbose,
            )
            if len(pool) < need:
                rejected_here.append((cand.topic_label, len(pool), need))
                if verbose:
                    print(
                        f"  reject L2 {cand.topic_label!r} in {l1!r} "
                        f"(have {len(pool)}, need {need})"
                    )
                continue
            pools[cand.topic_label] = pool
            accepted.append(cand)

        if len(accepted) == 5:
            chosen_l1s.append(l1)
            chosen_topics_by_l1.append(accepted)
            l1_fps_trace.append({
                "l1": l1,
                "status": "accepted",
                "slot": len(chosen_l1s) - 1,
                "accepted_l2s": [t.topic_label for t in accepted],
                "n_rejected_l2s": len(rejected_here),
            })
            print(f"[var/bal master] accepted L1={l1!r} at slot {len(chosen_l1s) - 1}")
        else:
            l1_fps_trace.append({
                "l1": l1,
                "status": "skipped",
                "n_accepted_l2s": len(accepted),
                "n_rejected_l2s": len(rejected_here),
            })
            if verbose:
                print(f"[var/bal master] skip L1={l1!r} (got only {len(accepted)}/5 viable L2s)")

    if len(chosen_l1s) < 10:
        raise RuntimeError(f"Could not find 10 viable L1s (found {len(chosen_l1s)}).")

    meta = {
        "seed": seed,
        "l1_pool_size": len(viable_l1_pool),
        "l1_fps_start": l1_fps_order[0],
        "l1_fps_ordering": l1_fps_order,
        "l1_selection_trace": l1_fps_trace,
    }
    return chosen_l1s, chosen_topics_by_l1, pools, meta


def round_robin_topics(chosen_l1s: List[str], chosen_topics_by_l1: List[List[L2Topic]]) -> List[L2Topic]:
    ordered: List[L2Topic] = []
    for col in range(5):
        for l1_slot in range(10):
            ordered.append(chosen_topics_by_l1[l1_slot][col])
    return ordered


def choose_disparity_master(
    wiki,
    byL1: Dict[str, List[L2Topic]],
    *,
    l1_emb_map: Dict[str, np.ndarray],
    l2_emb_map: Dict[str, np.ndarray],
    total: int,
    out_dir: Path,
    min_chars: int,
    max_chars: Optional[int],
    cat_depth_schedule: Sequence[int],
    cat_articles_schedule: Sequence[int],
    per_article_sentence_cap_schedule: Sequence[int],
    sleep_s: float,
    seed: int,
    verbose: bool,
) -> Tuple[List[str], Dict[str, List[L2Topic]], Dict[str, List[str]], Dict[str, object]]:
    """Accept L1s in FPS order; each accepted L1 provides max_topics_per_slot[slot] L2s.
    If an L1 cannot provide the required L2 count at its slot, it is skipped and the
    slot is filled by the next FPS-ordered L1 (preserving incremental prefix semantics
    across disparity levels).
    """
    # Maximum topics ever needed from the j-th L1 (0-based) across m levels.
    max_topics_per_slot = [0] * 50
    for m in DISPARITY_MS:
        alloc = alloc_uniform(50, m)
        for j in range(m):
            max_topics_per_slot[j] = max(max_topics_per_slot[j], alloc[j])
    max_count_per_topic = max(alloc_uniform(total, 10))

    # Candidate L1 pool: any L1 with >=1 viable L2 could in principle serve some slot,
    # but for simplicity we require >=5 (matching variety/balance), which is generous
    # since later slots need only 1-2 L2s.
    viable_l1_pool = sorted(l1 for l1, topics in byL1.items() if len(topics) >= 5)
    if len(viable_l1_pool) < 50:
        raise RuntimeError(f"Only {len(viable_l1_pool)} L1 candidates have >=5 viable L2s; need at least 50.")
    l1_fps_order = fps_order(viable_l1_pool, l1_emb_map, seed=seed)
    print(f"[disparity master] L1 pool size={len(viable_l1_pool)}, FPS start={l1_fps_order[0]!r}")

    page_text_cache: Dict[str, str] = {}
    pools: Dict[str, List[str]] = {}
    accepted_l1s: List[str] = []
    chosen_by_l1: Dict[str, List[L2Topic]] = {}
    l1_fps_trace: List[Dict[str, object]] = []
    pool_dir = out_dir / "topic_pools_disparity"

    for l1 in l1_fps_order:
        if len(accepted_l1s) >= 50:
            break
        slot = len(accepted_l1s)
        need_topics = max_topics_per_slot[slot]
        if need_topics == 0:
            # Shouldn't happen given max_topics_per_slot[:50] is >=1, but be defensive.
            accepted_l1s.append(l1)
            chosen_by_l1[l1] = []
            continue

        candidates_raw = byL1[l1]
        l2_labels_here = [t.topic_label for t in candidates_raw]
        l2_seed = derive_l2_seed(seed, l1)
        ordered_l2_labels = fps_order(l2_labels_here, l2_emb_map, seed=l2_seed)
        name_to_topic = {t.topic_label: t for t in candidates_raw}
        candidates = [name_to_topic[lb] for lb in ordered_l2_labels]

        accepted: List[L2Topic] = []
        for cand in candidates:
            if len(accepted) >= need_topics:
                break
            pool = ensure_topic_pool(
                wiki,
                cand,
                max_count_per_topic,
                pool_dir=pool_dir,
                cat_depth_schedule=cat_depth_schedule,
                cat_articles_schedule=cat_articles_schedule,
                per_article_sentence_cap_schedule=per_article_sentence_cap_schedule,
                min_chars=min_chars,
                max_chars=max_chars,
                sleep_s=sleep_s,
                page_text_cache=page_text_cache,
                seed=seed,
                verbose=verbose,
            )
            if len(pool) < max_count_per_topic:
                if verbose:
                    print(f"  reject disparity L2 {cand.topic_label!r} in {l1!r} "
                          f"(have {len(pool)}, need {max_count_per_topic})")
                continue
            pools[cand.topic_label] = pool
            accepted.append(cand)

        if len(accepted) >= need_topics:
            accepted_l1s.append(l1)
            chosen_by_l1[l1] = accepted
            l1_fps_trace.append({
                "l1": l1,
                "status": "accepted",
                "slot": slot,
                "need_topics": need_topics,
                "accepted_l2s": [t.topic_label for t in accepted],
            })
            print(f"[disparity master] accepted L1={l1!r} at slot {slot} "
                  f"(need {need_topics} L2s, got {len(accepted)})")
        else:
            l1_fps_trace.append({
                "l1": l1,
                "status": "skipped",
                "slot_attempted": slot,
                "need_topics": need_topics,
                "got": len(accepted),
            })
            if verbose:
                print(f"[disparity master] skip L1={l1!r} at slot {slot} "
                      f"(need {need_topics}, got only {len(accepted)})")

    if len(accepted_l1s) < 50:
        raise RuntimeError(
            f"Disparity: accepted only {len(accepted_l1s)}/50 L1s from pool of {len(viable_l1_pool)}."
        )

    meta = {
        "seed": seed,
        "l1_pool_size": len(viable_l1_pool),
        "l1_fps_start": l1_fps_order[0],
        "l1_fps_ordering": l1_fps_order,
        "l1_selection_trace": l1_fps_trace,
    }
    return accepted_l1s, chosen_by_l1, pools, meta


# -----------------------------------------------------------------------------
# Assembly
# -----------------------------------------------------------------------------
def assemble_uniform_dataset(topics: Sequence[L2Topic], pools: Dict[str, List[str]], total: int) -> List[Dict[str, str]]:
    counts = alloc_uniform(total, len(topics))
    rows: List[Dict[str, str]] = []
    for topic, need in zip(topics, counts):
        pool = pools[topic.topic_label]
        if len(pool) < need:
            raise RuntimeError(f"Topic {topic.topic_label!r} has only {len(pool)} sentences; need {need}")
        for sent in pool[:need]:
            rows.append({"text": sent, "topic": topic.topic_label, "parent_L1": topic.parent_L1})
    return rows


def assemble_balance_dataset(topics: Sequence[L2Topic], pools: Dict[str, List[str]], total: int, cond: str) -> List[Dict[str, str]]:
    counts = alloc_balance_condition(total, len(topics), cond)
    rows: List[Dict[str, str]] = []
    for topic, need in zip(topics, counts):
        pool = pools[topic.topic_label]
        if len(pool) < need:
            raise RuntimeError(f"Topic {topic.topic_label!r} has only {len(pool)} sentences; need {need}")
        for sent in pool[:need]:
            rows.append({"text": sent, "topic": topic.topic_label, "parent_L1": topic.parent_L1})
    return rows


# -----------------------------------------------------------------------------
# Schedule helper + CLI
# -----------------------------------------------------------------------------
def _linear_schedule(base: int, max_limit: int, rounds: int) -> List[int]:
    if rounds <= 1:
        return [max_limit]
    if base is None:
        base = max_limit
    if max_limit is None:
        max_limit = base
    if base == max_limit:
        return [base] * rounds
    vals = []
    for i in range(rounds):
        x = base + (max_limit - base) * i / (rounds - 1)
        vals.append(int(round(x)))
    vals[0] = base
    vals[-1] = max_limit
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build aligned Wikipedia SemDiv datasets (FPS selection)")
    p.add_argument("--l2_json", type=str, required=True, help="Path to L2_all.json")
    p.add_argument("--out_root", type=str, required=True, help="Output root directory")
    p.add_argument("--lang", type=str, default="en", help="Wikipedia language code")
    p.add_argument("--user_agent", type=str, default="SemDivBenchmark/1.0 (research use)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_sentences", "--total", dest="total_sentences", type=int, default=1000,
                   help="Dataset size per condition")
    p.add_argument("--min_direct_pages", type=int, default=30)
    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--max_chars", type=int, default=300)
    p.add_argument("--connect_timeout", type=int, default=10)
    p.add_argument("--read_timeout", type=int, default=20)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--max_per_topic", type=int, default=5000,
                   help="Compatibility arg; currently used as a soft sanity cap only")
    p.add_argument("--cost_json", type=str, default=None,
                   help="Optional path to write a small build-cost summary JSON")

    # Compat schedule args (from the old script).
    p.add_argument("--base_depth", type=int, default=None)
    p.add_argument("--max_depth_limit", type=int, default=None)
    p.add_argument("--base_articles", type=int, default=None)
    p.add_argument("--max_articles_limit", type=int, default=None)
    p.add_argument("--base_cap", type=int, default=None)
    p.add_argument("--max_cap_limit", type=int, default=None)
    p.add_argument("--expand_rounds", type=int, default=None)

    p.add_argument("--sleep", dest="sleep_s", type=float, default=0.0)
    p.add_argument("--allow_parent_L1_fallback", action="store_true",
                   help="Compatibility flag; ignored in FPS mode since FPS order naturally falls through")
    p.add_argument("--cat_depth_schedule", type=int, nargs="+", default=[1, 2, 2, 3, 3])
    p.add_argument("--cat_articles_schedule", type=int, nargs="+", default=[300, 800, 1500, 3000, 6000])
    p.add_argument("--per_article_sentence_cap_schedule", type=int, nargs="+", default=[5, 8, 10, 12, 15])
    p.add_argument("--verbose", action="store_true")

    # Embedding / FPS args.
    p.add_argument("--embedding_model_path", type=str,
                   default="/PATH/TO/models/embedding/Qwen3-Embedding-8B")
    p.add_argument("--embedding_cache_dir", type=str,
                   default="/PATH/TO/cache/fps_embeddings")
    p.add_argument("--embedding_device", type=str, default=None,
                   help="cuda / cpu; auto-detected if None")
    p.add_argument("--embedding_batch_size", type=int, default=8)
    p.add_argument("--no_embedding_compute", action="store_true",
                   help="Require a complete embedding cache; fail instead of computing missing entries.")
    p.add_argument("--disparity_seed_offset", type=int, default=1000,
                   help="Offset added to --seed for the disparity world (keeps L1/L2 FPS independent between worlds).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.expand_rounds is not None:
        rounds = args.expand_rounds
        if args.base_depth is not None or args.max_depth_limit is not None:
            args.cat_depth_schedule = _linear_schedule(
                args.base_depth or args.cat_depth_schedule[0],
                args.max_depth_limit or args.cat_depth_schedule[-1], rounds)
        if args.base_articles is not None or args.max_articles_limit is not None:
            args.cat_articles_schedule = _linear_schedule(
                args.base_articles or args.cat_articles_schedule[0],
                args.max_articles_limit or args.cat_articles_schedule[-1], rounds)
        if args.base_cap is not None or args.max_cap_limit is not None:
            args.per_article_sentence_cap_schedule = _linear_schedule(
                args.base_cap or args.per_article_sentence_cap_schedule[0],
                args.max_cap_limit or args.per_article_sentence_cap_schedule[-1], rounds)

    if not (
        len(args.cat_depth_schedule)
        == len(args.cat_articles_schedule)
        == len(args.per_article_sentence_cap_schedule)
    ):
        raise ValueError("The three schedule arguments must have the same length.")

    # Respect the HPC cache convention.
    os.environ.setdefault("HF_HOME", "/PATH/TO/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/PATH/TO/.cache/huggingface")

    # Auto device for the embedding model.
    if args.embedding_device is None:
        try:
            import torch
            args.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.embedding_device = "cpu"

    random.seed(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading topic universe...")
    topics = load_L2_all_json(args.l2_json, args.min_direct_pages)
    byL1 = group_by_L1(topics)
    print(f"Loaded {len(topics)} viable L2 topics across {len(byL1)} L1 groups")

    # -- FPS embeddings (shared cache across seeds) -----------------------------
    print("\n=== Preparing L1/L2 embeddings (cached) ===")
    l1_emb_map, l2_emb_map = prepare_embeddings(
        byL1,
        cache_dir=Path(args.embedding_cache_dir),
        model_path=args.embedding_model_path,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        allow_compute=not args.no_embedding_compute,
    )

    wiki = make_wiki(
        lang=args.lang,
        user_agent=args.user_agent,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        retries=args.retries,
    )

    # -- Variety / Balance world ------------------------------------------------
    print("\n=== Building variety/balance master ===")
    vb_l1s, vb_topics_by_l1, vb_pools, vb_meta = choose_variety_balance_master(
        wiki,
        byL1,
        l1_emb_map=l1_emb_map,
        l2_emb_map=l2_emb_map,
        total=args.total_sentences,
        out_dir=out_root,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        cat_depth_schedule=args.cat_depth_schedule,
        cat_articles_schedule=args.cat_articles_schedule,
        per_article_sentence_cap_schedule=args.per_article_sentence_cap_schedule,
        sleep_s=args.sleep_s,
        seed=args.seed,
        verbose=args.verbose,
    )
    vb_master = round_robin_topics(vb_l1s, vb_topics_by_l1)

    master_meta = {
        "selection_method": "FPS in Qwen3-Embedding-8B space (method 1: rng.permutation start)",
        "embedding_model": args.embedding_model_path,
        "variety_balance_l1s": vb_l1s,
        "variety_balance_topics_round_robin": [
            {"topic": t.topic_label, "parent_L1": t.parent_L1, "category_title": t.category_title}
            for t in vb_master
        ],
        "variety_balance_fps_meta": vb_meta,
    }
    with open(out_root / "master_variety_balance.json", "w", encoding="utf-8") as f:
        json.dump(master_meta, f, ensure_ascii=False, indent=2)

    print("\n=== Writing variety datasets ===")
    for k in VARIETY_KS:
        subset = vb_master[:k]
        rows = assemble_uniform_dataset(subset, vb_pools, args.total_sentences)
        write_dataset(
            rows,
            out_root / "variety" / f"variety_k{k}.tsv",
            out_root / "variety" / f"variety_k{k}_clean.txt",
        )
        print(f"  variety k={k}: wrote {len(rows)} rows")

    print("\n=== Writing balance datasets ===")
    for cond in BALANCE_CONDITIONS:
        rows = assemble_balance_dataset(vb_master, vb_pools, args.total_sentences, cond)
        write_dataset(
            rows,
            out_root / "balance" / f"balance_{cond}.tsv",
            out_root / "balance" / f"balance_{cond}_clean.txt",
        )
        print(f"  balance {cond}: wrote {len(rows)} rows")

    # -- Disparity world --------------------------------------------------------
    print("\n=== Building disparity master ===")
    disp_l1s, disp_topics_by_l1, disp_pools, disp_meta_fps = choose_disparity_master(
        wiki,
        byL1,
        l1_emb_map=l1_emb_map,
        l2_emb_map=l2_emb_map,
        total=args.total_sentences,
        out_dir=out_root,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        cat_depth_schedule=args.cat_depth_schedule,
        cat_articles_schedule=args.cat_articles_schedule,
        per_article_sentence_cap_schedule=args.per_article_sentence_cap_schedule,
        sleep_s=args.sleep_s,
        seed=args.seed + args.disparity_seed_offset,
        verbose=args.verbose,
    )
    disp_meta = {
        "selection_method": "FPS in Qwen3-Embedding-8B space (method 1: rng.permutation start)",
        "embedding_model": args.embedding_model_path,
        "disparity_l1_order": disp_l1s,
        "topics_by_l1": {
            l1: [
                {"topic": t.topic_label, "parent_L1": t.parent_L1, "category_title": t.category_title}
                for t in disp_topics_by_l1[l1]
            ]
            for l1 in disp_l1s
        },
        "disparity_fps_meta": disp_meta_fps,
    }
    with open(out_root / "master_disparity.json", "w", encoding="utf-8") as f:
        json.dump(disp_meta, f, ensure_ascii=False, indent=2)

    print("\n=== Writing disparity datasets ===")
    for m in DISPARITY_MS:
        used_l1s = disp_l1s[:m]
        alloc_topics = alloc_uniform(50, m)
        selected_topics: List[L2Topic] = []
        for slot, l1 in enumerate(used_l1s):
            take = alloc_topics[slot]
            selected_topics.extend(disp_topics_by_l1[l1][:take])
        if len(selected_topics) != 50:
            raise RuntimeError(f"Disparity m={m}: selected {len(selected_topics)} topics, expected 50")
        rows = assemble_uniform_dataset(selected_topics, disp_pools, args.total_sentences)
        write_dataset(
            rows,
            out_root / "disparity" / f"disparity_m{m}.tsv",
            out_root / "disparity" / f"disparity_m{m}_clean.txt",
        )
        print(f"  disparity m={m}: wrote {len(rows)} rows")

    print(f"\nDone. Outputs in: {out_root}")


if __name__ == "__main__":
    main()
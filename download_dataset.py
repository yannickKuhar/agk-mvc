"""
download_dataset.py
-------------------
Download PKU-ML/Erdos from HuggingFace, filter to min_vertex_cover examples,
and save to data/erdos/{split}.json.

Usage:
    python download_dataset.py                      # train split, all examples
    python download_dataset.py --split test
    python download_dataset.py --all-splits
    python download_dataset.py --max-examples 500 --min-nodes 5 --max-nodes 100
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

_OUT_DIR = Path("data/erdos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Erdos MVC dataset from HuggingFace")
    p.add_argument("--split", default="train", help="Dataset split (default: train)")
    p.add_argument("--all-splits", action="store_true", help="Download train and test splits")
    p.add_argument("--max-examples", type=int, default=None, help="Max examples to save")
    p.add_argument("--min-nodes", type=int, default=0, help="Skip graphs with fewer nodes")
    p.add_argument("--max-nodes", type=int, default=10**9, help="Skip graphs with more nodes")
    return p.parse_args()


def _parse_tuple_str(s: str):
    """Parse a string like '(1, 21)' -> (1, 21)."""
    try:
        result = ast.literal_eval(s.strip())
        if isinstance(result, tuple):
            return result
        if isinstance(result, int):
            return (result, result)
    except Exception:
        pass
    # regex fallback: extract all integers
    nums = list(map(int, re.findall(r"-?\d+", s)))
    if len(nums) >= 2:
        return (nums[0], nums[-1])
    if len(nums) == 1:
        return (nums[0], nums[0])
    return None


def _parse_list_str(s: str):
    """Parse a string like '[(1,2),(3,4)]' or '[1, 3, 5]' -> Python list."""
    try:
        return ast.literal_eval(s.strip())
    except Exception:
        pass
    # regex fallback: extract all integers
    nums = list(map(int, re.findall(r"-?\d+", s)))
    return nums


def _convert_example(ex: dict, min_nodes: int, max_nodes: int):
    """
    Convert one HuggingFace record to a {n_nodes, edges, mvc, source} dict.
    Returns None if the example should be skipped.
    """
    nodes_parsed = _parse_tuple_str(ex["nodes"])
    if nodes_parsed is None:
        return None
    # nodes field is (min_id, max_id), both 1-indexed
    n_nodes = nodes_parsed[1]  # max node id equals node count (1-indexed → count)

    if n_nodes < min_nodes or n_nodes > max_nodes:
        return None

    raw_edges = _parse_list_str(ex["edges"])
    if not raw_edges:
        edges_0 = []
    elif isinstance(raw_edges[0], (list, tuple)):
        # list of pairs
        edges_0 = [[int(u) - 1, int(v) - 1] for u, v in raw_edges]
    else:
        # flat list of ints — shouldn't happen for edges but handle gracefully
        it = iter(raw_edges)
        edges_0 = [[int(u) - 1, int(v) - 1] for u, v in zip(it, it)]

    raw_mvc = _parse_list_str(ex["answer"])
    if isinstance(raw_mvc, list) and raw_mvc and isinstance(raw_mvc[0], (list, tuple)):
        raw_mvc = [item for sub in raw_mvc for item in sub]
    mvc_0 = [int(v) - 1 for v in raw_mvc]

    # Validate
    if any(v < 0 or v >= n_nodes for v in mvc_0):
        return None

    return {
        "n_nodes": n_nodes,
        "edges": edges_0,
        "mvc": mvc_0,
        "source": ex.get("source_file", ""),
    }


def download_split(split: str, max_examples, min_nodes: int, max_nodes: int):
    print(f"[download] Loading PKU-ML/Erdos split='{split}' from HuggingFace ...")
    ds = load_dataset("PKU-ML/Erdos", split=split)

    records = []
    n_skipped = 0
    all_sizes = []
    for ex in ds:
        if ex.get("task") != "min_vertex_cover":
            continue

        rec = _convert_example(ex, min_nodes=0, max_nodes=10**9)
        if rec is not None:
            all_sizes.append(rec["n_nodes"])

        if max_examples is not None and len(records) >= max_examples:
            continue

        rec = _convert_example(ex, min_nodes, max_nodes)
        if rec is None:
            n_skipped += 1
            continue

        records.append(rec)

    if all_sizes:
        print(f"[download] Dataset node-count range: "
              f"min={min(all_sizes)}, max={max(all_sizes)}, "
              f"mean={sum(all_sizes)/len(all_sizes):.1f}  "
              f"(total MVC examples: {len(all_sizes)})")

    if len(records) == 0 and all_sizes:
        print(f"[download] ERROR: --min-nodes {min_nodes} / --max-nodes {max_nodes} "
              f"filtered out ALL examples. "
              f"The dataset node range is [{min(all_sizes)}, {max(all_sizes)}].")
        sys.exit(1)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(
        f"[download] Saved {len(records)} records to {out_path} "
        f"(skipped {n_skipped})."
    )
    return len(records)


def main():
    args = parse_args()
    splits = ["train", "test"] if args.all_splits else [args.split]

    for split in splits:
        n = download_split(
            split,
            max_examples=args.max_examples,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
        )
        if n == 0:
            print(f"[download] WARNING: No records saved for split '{split}'.")
            sys.exit(1)


if __name__ == "__main__":
    main()

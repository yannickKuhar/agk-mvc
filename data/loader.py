"""
data/loader.py
--------------
Load pre-downloaded Erdos MVC graphs from local JSON files.

Expected files:
    data/erdos/train.json
    data/erdos/test.json

Each file is a JSON array of records with fields:
    n_nodes : int            — number of nodes (0-indexed: 0..n_nodes-1)
    edges   : [[u, v], ...]  — undirected edges, 0-indexed
    mvc     : [v, ...]       — optimal MVC node ids, 0-indexed
    source  : str            — original source file (informational)

To create these files, run:
    python download_dataset.py --all-splits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np

_DATA_DIR = Path(__file__).parent / "erdos"


def load_erdos(
    split: str = "train",
    max_graphs: Optional[int] = None,
    min_nodes: int = 5,
    max_nodes: int = 500,
) -> List[nx.Graph]:
    """
    Load the Erdos MVC dataset from a local JSON file.

    Each returned graph has node attribute 'label': 1 if in optimal MVC, 0 otherwise.

    Parameters
    ----------
    split      : dataset split to load ('train' or 'test')
    max_graphs : cap on the number of graphs returned (None = all)
    min_nodes  : skip graphs with fewer nodes than this
    max_nodes  : skip graphs with more nodes than this
    """
    json_path = _DATA_DIR / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"[loader] Dataset file not found: {json_path}\n"
            f"  Run:  python download_dataset.py --split {split}\n"
            f"  or:   python download_dataset.py --all-splits\n"
            f"  to download and save the dataset locally."
        )

    print(f"[loader] Loading from {json_path} ...")
    with open(json_path) as f:
        records = json.load(f)

    graphs: List[nx.Graph] = []
    n_skipped = 0
    for rec in records:
        if max_graphs is not None and len(graphs) >= max_graphs:
            break

        n = rec["n_nodes"]
        if n < min_nodes or n > max_nodes:
            n_skipped += 1
            continue

        G = _record_to_nx(rec)
        if G is None:
            n_skipped += 1
            continue

        graphs.append(G)

    print(
        f"[loader] Loaded {len(graphs)} graphs "
        f"(skipped {n_skipped} — too small/large or parse errors)."
    )
    return graphs


def _record_to_nx(rec: dict) -> Optional[nx.Graph]:
    try:
        n = int(rec["n_nodes"])
        mvc_set = set(int(v) for v in rec["mvc"])

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for node_id in range(n):
            G.nodes[node_id]["label"] = 1 if node_id in mvc_set else 0

        for u, v in rec["edges"]:
            u, v = int(u), int(v)
            if u != v and 0 <= u < n and 0 <= v < n:
                G.add_edge(u, v)

        return G
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilities for downstream use
# ---------------------------------------------------------------------------

def get_node_labels(G: nx.Graph) -> np.ndarray:
    """Return node labels as a numpy array, ordered by node id."""
    n = G.number_of_nodes()
    return np.array([G.nodes[i].get("label", -1) for i in range(n)], dtype=int)


def graph_summary(graphs: List[nx.Graph]) -> dict:
    """Return basic statistics about a list of graphs. Safe on empty lists."""
    if not graphs:
        return {
            "num_graphs"       : 0,
            "avg_nodes"        : float("nan"),
            "avg_edges"        : float("nan"),
            "avg_pos_fraction" : float("nan"),
            "min_nodes"        : 0,
            "max_nodes"        : 0,
        }

    sizes = [G.number_of_nodes() for G in graphs]
    edges = [G.number_of_edges() for G in graphs]
    pos_fracs = []
    for G in graphs:
        lbls = get_node_labels(G)
        valid = lbls[lbls >= 0]
        if len(valid) > 0:
            pos_fracs.append(float(valid.mean()))

    return {
        "num_graphs"       : len(graphs),
        "avg_nodes"        : float(np.mean(sizes)),
        "avg_edges"        : float(np.mean(edges)),
        "avg_pos_fraction" : float(np.mean(pos_fracs)) if pos_fracs else float("nan"),
        "min_nodes"        : int(np.min(sizes)),
        "max_nodes"        : int(np.max(sizes)),
    }

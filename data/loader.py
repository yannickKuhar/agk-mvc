"""
data/loader.py
--------------
Load the PKU-ML/Erdos dataset from HuggingFace and convert each example
into a NetworkX graph with binary node labels (1 = in optimal MVC, 0 = not).

Dataset schema (inferred from HuggingFace card):
  Each example is a graph with:
    - edge_index : List[List[int]]  — pairs of node indices
    - num_nodes  : int
    - y          : List[int]        — per-node binary MVC label
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from typing import Iterator, Tuple, List, Optional
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_erdos(
    split: str = "train",
    max_graphs: Optional[int] = None,
    min_nodes: int = 5,
    max_nodes: int = 500,
) -> List[nx.Graph]:
    """
    Load the Erdos dataset and return a list of NetworkX graphs.

    Each graph has:
      - node attribute 'label' : int  (1 if in optimal MVC, 0 otherwise)

    Parameters
    ----------
    split       : HuggingFace dataset split ('train', 'test', etc.)
    max_graphs  : cap on number of graphs to load (None = all)
    min_nodes   : skip graphs smaller than this
    max_nodes   : skip graphs larger than this (ORCA is expensive on dense graphs)
    """
    print(f"[loader] Loading PKU-ML/Erdos ({split}) from HuggingFace ...")
    ds = load_dataset("PKU-ML/Erdos", split=split, trust_remote_code=True)

    graphs: List[nx.Graph] = []
    for i, example in enumerate(ds):
        if max_graphs is not None and len(graphs) >= max_graphs:
            break

        G = _example_to_nx(example)
        if G is None:
            continue
        n = G.number_of_nodes()
        if n < min_nodes or n > max_nodes:
            continue

        graphs.append(G)

    print(f"[loader] Loaded {len(graphs)} graphs.")
    return graphs


def iter_erdos(
    split: str = "train",
    min_nodes: int = 5,
    max_nodes: int = 500,
) -> Iterator[nx.Graph]:
    """Streaming version — yields one graph at a time (memory efficient)."""
    ds = load_dataset("PKU-ML/Erdos", split=split, trust_remote_code=True, streaming=True)
    for example in ds:
        G = _example_to_nx(example)
        if G is None:
            continue
        n = G.number_of_nodes()
        if min_nodes <= n <= max_nodes:
            yield G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _example_to_nx(example: dict) -> Optional[nx.Graph]:
    """Convert one HuggingFace Erdos example to a NetworkX graph."""
    try:
        edge_index = example.get("edge_index", [])
        num_nodes  = example.get("num_nodes", 0)
        labels     = example.get("y", [])

        if num_nodes == 0:
            return None

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # Labels — some datasets store flat list, some nested
        if labels:
            flat_labels = _flatten(labels)
            for node_id, lbl in enumerate(flat_labels):
                G.nodes[node_id]["label"] = int(lbl)
        else:
            for node_id in range(num_nodes):
                G.nodes[node_id]["label"] = -1  # unknown

        # Edges — edge_index can be [[src,dst], ...] or [[srcs],[dsts]]
        edges = _parse_edge_index(edge_index, num_nodes)
        G.add_edges_from(edges)

        # Remove self-loops (kernel algorithms assume simple graphs)
        G.remove_edges_from(nx.selfloop_edges(G))

        return G

    except Exception as e:
        print(f"[loader] Warning: skipping example due to error: {e}")
        return None


def _flatten(lst) -> List:
    """Flatten a possibly nested list one level deep."""
    if lst and isinstance(lst[0], (list, tuple)):
        return [x for sub in lst for x in sub]
    return list(lst)


def _parse_edge_index(edge_index, num_nodes: int):
    """
    Accept edge_index in two common formats:
      1. [[src, dst], [src, dst], ...]   — list of pairs
      2. [[src1, src2, ...], [dst1, dst2, ...]]  — two parallel lists (PyG style)
    """
    if not edge_index:
        return []

    # Check format
    if len(edge_index) == 2 and isinstance(edge_index[0], (list, np.ndarray)):
        srcs, dsts = edge_index[0], edge_index[1]
        edges = list(zip(srcs, dsts))
    else:
        edges = [tuple(e) for e in edge_index]

    # Filter out-of-range
    valid = [(u, v) for u, v in edges if u < num_nodes and v < num_nodes and u != v]
    return valid


# ---------------------------------------------------------------------------
# Utilities for downstream use
# ---------------------------------------------------------------------------

def get_node_labels(G: nx.Graph) -> np.ndarray:
    """Return node labels as a numpy array, ordered by node id."""
    n = G.number_of_nodes()
    labels = np.array([G.nodes[i].get("label", -1) for i in range(n)], dtype=int)
    return labels


def graph_summary(graphs: List[nx.Graph]) -> dict:
    """Print basic statistics about a list of graphs."""
    sizes = [G.number_of_nodes() for G in graphs]
    edges = [G.number_of_edges() for G in graphs]
    pos_fracs = []
    for G in graphs:
        lbls = get_node_labels(G)
        valid = lbls[lbls >= 0]
        if len(valid) > 0:
            pos_fracs.append(valid.mean())

    return {
        "num_graphs"       : len(graphs),
        "avg_nodes"        : float(np.mean(sizes)),
        "avg_edges"        : float(np.mean(edges)),
        "avg_pos_fraction" : float(np.mean(pos_fracs)) if pos_fracs else float("nan"),
        "min_nodes"        : int(np.min(sizes)),
        "max_nodes"        : int(np.max(sizes)),
    }

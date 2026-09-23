"""
data/realworld.py
-----------------
Real-world graph dataset loader.

Graphs are stored as data/realworld/{name}.json.
Run download_realworld.py to generate these files.

Available datasets (after running download_realworld.py):
  karate      - Zachary karate club (34 nodes, 78 edges)
  lesmis      - Les Misérables character co-occurrence (77 nodes, 254 edges)
  florentine  - Florentine families (15 nodes, 20 edges)
  davis       - Davis southern women (18 nodes, bipartite)
  petersen    - Petersen graph (10 nodes, 15 edges)
  dodecahedron - Dodecahedral graph (20 nodes, 30 edges)
  icosahedron - Icosahedral graph (12 nodes, 30 edges)
  complete_bipartite_5_5 - Complete bipartite K_{5,5} (10 nodes)
  grid_5x5    - 5x5 grid (25 nodes, 40 edges)
  grid_7x7    - 7x7 grid (49 nodes, 84 edges)
  grid_10x10  - 10x10 grid (100 nodes, 180 edges)
  cycle_20    - Cycle C_20 (20 nodes, 20 edges)
  wheel_30    - Wheel W_30 (30 nodes, 59 edges)
  barbell_15  - Barbell graph (35 nodes)
  builtin     - All of the above in one file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import networkx as nx

_DATA_DIR = Path(__file__).parent / "realworld"


def _load_json_records(
    path: Path,
    max_graphs: Optional[int],
    min_nodes: int,
    max_nodes: int,
    dataset_tag: str,
) -> List[nx.Graph]:
    """Load graphs from a {n_nodes, edges, mvc, source} JSON file."""
    with open(path) as f:
        records = json.load(f)
    graphs: List[nx.Graph] = []
    for rec in records:
        if max_graphs is not None and len(graphs) >= max_graphs:
            break
        n = rec["n_nodes"]
        if n < min_nodes or n > max_nodes:
            continue
        mvc_set = set(rec["mvc"])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in rec["edges"]:
            G.add_edge(u, v)
        for node in G.nodes():
            G.nodes[node]["label"] = 1 if node in mvc_set else 0
        G.graph["dataset"] = dataset_tag
        graphs.append(G)
    return graphs


def load_realworld(
    name: str,
    max_graphs: Optional[int] = None,
    min_nodes: int = 5,
    max_nodes: int = 500,
    solver_timeout: int = 120,
    recompute: bool = False,
) -> List[nx.Graph]:
    """
    Load real-world graphs from data/realworld/{name}.json.

    Parameters
    ----------
    name           : dataset name (e.g. 'karate', 'builtin')
    max_graphs     : cap on graphs returned (None = all)
    min_nodes      : skip graphs with fewer nodes than this
    max_nodes      : skip graphs with more nodes than this
    solver_timeout : unused (labels are pre-computed); kept for API symmetry
    recompute      : unused (labels are pre-computed); kept for API symmetry

    Returns
    -------
    List of nx.Graph with node attribute 'label' ∈ {0, 1}
    and graph attribute 'dataset' = 'realworld:{name}'.

    Raises
    ------
    FileNotFoundError
        If data/realworld/{name}.json does not exist. Run download_realworld.py
        to generate it.
    """
    path = _DATA_DIR / f"{name}.json"
    if not path.exists():
        available = list_available()
        hint = (
            f"\nAvailable datasets: {available}"
            if available
            else "\nNo datasets found yet."
        )
        raise FileNotFoundError(
            f"[realworld] '{name}.json' not found at {path}.\n"
            f"  Run:  python download_realworld.py\n"
            f"  or:   python download_realworld.py --dataset {name}\n"
            f"  to generate real-world graph files with ILP labels.{hint}"
        )

    dataset_tag = f"realworld:{name}"
    print(f"[realworld] Loading '{name}' from {path} ...")
    graphs = _load_json_records(path, max_graphs, min_nodes, max_nodes, dataset_tag)
    print(f"[realworld] Loaded {len(graphs)} graph(s) from '{name}'.")
    return graphs


def list_available() -> List[str]:
    """Return names (without .json extension) of available dataset files."""
    if not _DATA_DIR.exists():
        return []
    return sorted(p.stem for p in _DATA_DIR.glob("*.json"))

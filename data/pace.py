"""
data/pace.py
------------
Load PACE 2019 Vertex Cover track graphs.

Download:
    wget https://pacechallenge.org/files/pace2019-vc-exact-public.tar.gz
    tar -xzf pace2019-vc-exact-public.tar.gz -C data/pace/

Graph format (.gr — DIMACS-like):
    p td <n_nodes> <n_edges>
    <u> <v>
    ...

Solution format (.sol):
    s vc <n_nodes> <mvc_size>
    <node_in_cover>
    ...

Nodes are 1-indexed in .gr/.sol files; we convert to 0-indexed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import networkx as nx

_DATA_DIR = Path(__file__).parent / "pace"


def _parse_gr(path: Path) -> Optional[nx.Graph]:
    G = nx.Graph()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                if line.startswith("p"):
                    n = int(line.split()[2])
                    G.add_nodes_from(range(n))  # 0-indexed
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]) - 1, int(parts[1]) - 1
                        if 0 <= u < G.number_of_nodes() and 0 <= v < G.number_of_nodes() and u != v:
                            G.add_edge(u, v)
    except Exception as e:
        print(f"[pace] Parse error {path.name}: {e}")
        return None
    return G if G.number_of_nodes() > 0 else None


def _parse_sol(path: Path, n_nodes: int) -> Optional[List[int]]:
    cover: List[int] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c") or line.startswith("s"):
                    continue
                v = int(line) - 1  # 1 → 0 indexed
                if 0 <= v < n_nodes:
                    cover.append(v)
    except Exception as e:
        print(f"[pace] Parse error {path.name}: {e}")
        return None
    return cover


def load_pace(
    data_dir: Path = _DATA_DIR,
    max_graphs: Optional[int] = None,
    min_nodes: int = 5,
    max_nodes: int = 2000,
) -> List[nx.Graph]:
    """
    Load PACE 2019 Vertex Cover graphs with their optimal solutions.

    Parameters
    ----------
    data_dir   : directory containing .gr and (optionally) .sol files
    max_graphs : cap on graphs returned (None = all)
    min_nodes  : skip graphs smaller than this
    max_nodes  : skip graphs larger than this

    Returns
    -------
    List of nx.Graph with node attribute 'label' ∈ {0, 1}
    and graph attribute 'dataset' = 'pace'.
    Skips graphs without a matching .sol file.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[pace] Directory not found: {data_dir}")
        print("[pace] Download instructions:")
        print("  wget https://pacechallenge.org/files/pace2019-vc-exact-public.tar.gz")
        print("  tar -xzf pace2019-vc-exact-public.tar.gz -C data/pace/")
        return []

    gr_files = sorted(data_dir.rglob("*.gr"))
    print(f"[pace] Found {len(gr_files)} .gr files in {data_dir}")
    if not gr_files:
        return []

    graphs: List[nx.Graph] = []
    n_no_sol = n_size_skip = n_parse_err = 0

    for gr_path in gr_files:
        if max_graphs is not None and len(graphs) >= max_graphs:
            break

        sol_path = gr_path.with_suffix(".sol")
        if not sol_path.exists():
            n_no_sol += 1
            continue

        G = _parse_gr(gr_path)
        if G is None:
            n_parse_err += 1
            continue

        n = G.number_of_nodes()
        if n < min_nodes or n > max_nodes:
            n_size_skip += 1
            continue

        mvc = _parse_sol(sol_path, n)
        if mvc is None:
            n_parse_err += 1
            continue

        mvc_set = set(mvc)
        for v in G.nodes():
            G.nodes[v]["label"] = 1 if v in mvc_set else 0
        G.graph["dataset"] = "pace"
        graphs.append(G)

    print(f"[pace] Loaded {len(graphs)} graphs  "
          f"(no_sol={n_no_sol}, size_skip={n_size_skip}, parse_err={n_parse_err})")
    return graphs

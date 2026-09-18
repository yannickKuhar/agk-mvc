"""
data/tudataset.py
-----------------
Load TUDataset graphs and compute MVC labels with our ILP solver.

Supported datasets (structurally interesting, connected to the symmetry
kernel literature of Kuhar & Cibej 2026):
  MUTAG, NCI1, NCI109, PROTEINS, DD, IMDB-BINARY, REDDIT-BINARY

Raw data is downloaded from:
  https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip

MVC labels are computed once and cached to:
  data/tudatasets/{name}_mvc.json

The cache format is identical to data/erdos/train.json so the unified
loader can read both without distinction.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

_DATA_DIR = Path(__file__).parent / "tudatasets"
_BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"

KNOWN_DATASETS = [
    # Small molecular graphs (avg ~18–30 nodes) — easy for ILP
    "MUTAG", "NCI1", "NCI109", "PROTEINS",
    # Hard social/collaboration graphs (avg ~74 nodes, dense) — ILP takes 10–30 s
    "COLLAB",
    # Large graphs — ILP times out; included for completeness
    "DD", "IMDB-BINARY", "REDDIT-BINARY",
]


def _download_tudataset(name: str, data_dir: Path) -> bool:
    target_dir = data_dir / name
    # Check if already extracted
    if target_dir.exists() and any(target_dir.glob(f"{name}_A.txt")):
        return True

    url = f"{_BASE_URL}/{name}.zip"
    print(f"[tudataset] Downloading {name} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(data_dir)
        print(f"[tudataset] Extracted {name} → {data_dir}")
        return True
    except Exception as e:
        print(f"[tudataset] Download failed: {e}")
        print(f"[tudataset] Manual download:")
        print(f"  wget {url}")
        print(f"  unzip {name}.zip -d data/tudatasets/")
        return False


def _find_base(name: str, data_dir: Path) -> Optional[Path]:
    """Locate the base path for {name}_A.txt, handling varied zip layouts."""
    # Most common: data_dir/name/name_A.txt
    candidate = data_dir / name / f"{name}_A.txt"
    if candidate.exists():
        return data_dir / name / name
    # Flat layout: data_dir/name_A.txt
    candidate2 = data_dir / f"{name}_A.txt"
    if candidate2.exists():
        return data_dir / name
    return None


def _parse_tudataset(name: str, data_dir: Path) -> List[nx.Graph]:
    """Parse raw TUDataset files into NetworkX graphs (no labels)."""
    base = _find_base(name, data_dir)
    if base is None:
        print(f"[tudataset] Could not locate {name}_A.txt under {data_dir}")
        return []

    edge_file      = Path(f"{base}_A.txt")
    indicator_file = Path(f"{base}_graph_indicator.txt")

    if not edge_file.exists() or not indicator_file.exists():
        print(f"[tudataset] Missing required files at {base}")
        return []

    # node_id (1-indexed) → graph_id (1-indexed)
    node_to_graph: Dict[int, int] = {}
    with open(indicator_file) as f:
        for node_id, line in enumerate(f, start=1):
            node_to_graph[node_id] = int(line.strip())

    n_graphs = max(node_to_graph.values())
    graph_nodes: Dict[int, List[int]] = {g: [] for g in range(1, n_graphs + 1)}
    for nid, gid in node_to_graph.items():
        graph_nodes[gid].append(nid)

    graph_edges: Dict[int, List] = {g: [] for g in range(1, n_graphs + 1)}
    with open(edge_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            u, v = int(parts[0].strip()), int(parts[1].strip())
            gid = node_to_graph.get(u)
            if gid is not None and u != v:
                graph_edges[gid].append((u, v))

    graphs: List[nx.Graph] = []
    for gid in range(1, n_graphs + 1):
        nodes = sorted(graph_nodes[gid])
        if not nodes:
            continue
        node_map = {old: new for new, old in enumerate(nodes)}
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        for u, v in graph_edges[gid]:
            nu, nv = node_map[u], node_map[v]
            G.add_edge(nu, nv)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.graph["dataset"] = f"tudataset:{name}"
        graphs.append(G)

    return graphs


def load_tudataset(
    name: str,
    data_dir: Path = _DATA_DIR,
    max_graphs: Optional[int] = None,
    min_nodes: int = 5,
    max_nodes: int = 500,
    solver_timeout: int = 60,
    recompute: bool = False,
) -> List[nx.Graph]:
    """
    Load a TUDataset with MVC labels (cached after first solve).

    Parameters
    ----------
    name           : dataset name (e.g. 'MUTAG')
    data_dir       : root directory for raw data and cache
    max_graphs     : cap on graphs returned (None = all)
    min_nodes      : skip graphs smaller than this
    max_nodes      : skip graphs larger than this
    solver_timeout : ILP timeout per graph; graph is skipped on timeout
    recompute      : re-solve even if cache exists

    Returns
    -------
    List of nx.Graph with node attribute 'label' ∈ {0, 1}
    and graph attribute 'dataset' = 'tudataset:{name}'.
    """
    from solver.mvc_solver import MVCSolver

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{name}_mvc.json"

    # --- Serve from cache ---
    if cache_path.exists() and not recompute:
        print(f"[tudataset] Loading {name} from cache {cache_path} ...")
        with open(cache_path) as f:
            records = json.load(f)
        graphs: List[nx.Graph] = []
        for rec in records:
            n = rec["n_nodes"]
            if n < min_nodes or n > max_nodes:
                continue
            if max_graphs is not None and len(graphs) >= max_graphs:
                break
            mvc_set = set(rec["mvc"])
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for u, v in rec["edges"]:
                G.add_edge(u, v)
            for node in G.nodes():
                G.nodes[node]["label"] = 1 if node in mvc_set else 0
            G.graph["dataset"] = f"tudataset:{name}"
            graphs.append(G)
        print(f"[tudataset] Loaded {len(graphs)} graphs from cache.")
        return graphs

    # --- Download and parse ---
    if not _download_tudataset(name, data_dir):
        return []

    raw_graphs = _parse_tudataset(name, data_dir)
    print(f"[tudataset] Parsed {len(raw_graphs)} raw graphs for {name}")

    # --- Solve MVC for each graph ---
    solver = MVCSolver(backend="ilp", timeout=solver_timeout, fallback_to_approx=False)
    records = []
    n_size_skip = n_timeout = 0

    for i, G in enumerate(raw_graphs):
        n = G.number_of_nodes()
        if n < min_nodes or n > max_nodes:
            n_size_skip += 1
            continue

        t0 = time.perf_counter()
        try:
            result = solver.solve(G)
        except Exception:
            n_timeout += 1
            continue

        mvc_set = result.cover
        for v in G.nodes():
            G.nodes[v]["label"] = 1 if v in mvc_set else 0

        records.append({
            "n_nodes": n,
            "edges":   [[u, v] for u, v in G.edges()],
            "mvc":     sorted(mvc_set),
            "source":  f"tudataset:{name}",
        })

        if (i + 1) % 50 == 0:
            print(f"  [{name}] {len(records)} solved / {i + 1} processed "
                  f"(timeout={n_timeout})")

    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[tudataset] Cached {len(records)} records → {cache_path}  "
          f"(size_skip={n_size_skip}, timeout={n_timeout})")

    # Reload through the cache path to apply size/count filters uniformly
    return load_tudataset(name, data_dir, max_graphs, min_nodes, max_nodes,
                          solver_timeout, recompute=False)

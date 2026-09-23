"""
download_realworld.py
---------------------
Generate real-world and structured graphs with ILP-computed MVC labels.
Saves results to data/realworld/{name}.json.

All graphs are built from NetworkX built-in generators — no external downloads
required.

Usage:
    python download_realworld.py              # generate all built-in datasets
    python download_realworld.py --dataset karate
    python download_realworld.py --force       # recompute even if exists
    python download_realworld.py --timeout 60  # per-graph ILP timeout (seconds)
    python download_realworld.py --approx      # use 2-approx instead of ILP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

_DATA_DIR = Path(__file__).parent / "data" / "realworld"

# ---------------------------------------------------------------------------
# Graph catalogue
# Each entry: (name, factory_fn)
# factory_fn() -> nx.Graph
# ---------------------------------------------------------------------------

def _make_karate() -> nx.Graph:
    return nx.karate_club_graph()


def _make_lesmis() -> nx.Graph:
    return nx.les_miserables_graph()


def _make_florentine() -> nx.Graph:
    return nx.florentine_families_graph()


def _make_davis() -> nx.Graph:
    # Davis southern women is a bipartite graph — use it as-is (proper Graph)
    return nx.davis_southern_women_graph()


def _make_petersen() -> nx.Graph:
    return nx.petersen_graph()


def _make_dodecahedron() -> nx.Graph:
    return nx.dodecahedral_graph()


def _make_icosahedron() -> nx.Graph:
    return nx.icosahedral_graph()


def _make_complete_bipartite_5_5() -> nx.Graph:
    return nx.complete_bipartite_graph(5, 5)


def _make_grid_5x5() -> nx.Graph:
    G = nx.grid_2d_graph(5, 5)
    return nx.convert_node_labels_to_integers(G)


def _make_grid_7x7() -> nx.Graph:
    G = nx.grid_2d_graph(7, 7)
    return nx.convert_node_labels_to_integers(G)


def _make_grid_10x10() -> nx.Graph:
    G = nx.grid_2d_graph(10, 10)
    return nx.convert_node_labels_to_integers(G)


def _make_cycle_20() -> nx.Graph:
    return nx.cycle_graph(20)


def _make_wheel_30() -> nx.Graph:
    return nx.wheel_graph(30)


def _make_barbell_15() -> nx.Graph:
    return nx.barbell_graph(15, 5)


CATALOGUE: List[Tuple[str, callable]] = [
    ("karate",                _make_karate),
    ("lesmis",                _make_lesmis),
    ("florentine",            _make_florentine),
    ("davis",                 _make_davis),
    ("petersen",              _make_petersen),
    ("dodecahedron",          _make_dodecahedron),
    ("icosahedron",           _make_icosahedron),
    ("complete_bipartite_5_5", _make_complete_bipartite_5_5),
    ("grid_5x5",              _make_grid_5x5),
    ("grid_7x7",              _make_grid_7x7),
    ("grid_10x10",            _make_grid_10x10),
    ("cycle_20",              _make_cycle_20),
    ("wheel_30",              _make_wheel_30),
    ("barbell_15",            _make_barbell_15),
]

CATALOGUE_DICT: Dict[str, callable] = {name: fn for name, fn in CATALOGUE}


# ---------------------------------------------------------------------------
# Graph pre-processing
# ---------------------------------------------------------------------------

def preprocess(G: nx.Graph, name: str) -> Optional[nx.Graph]:
    """
    Normalise a raw NetworkX graph:
      1. Convert to undirected simple Graph
      2. Remove self-loops
      3. Relabel nodes to integers 0..n-1
      4. Extract largest connected component
      5. Return None if fewer than 5 nodes remain
    """
    # Ensure undirected
    G = nx.Graph(G)
    # Remove self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Relabel to integers (handles tuple node labels from grid graphs)
    G = nx.convert_node_labels_to_integers(G)
    # Largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
    if G.number_of_nodes() < 5:
        print(f"  [skip] '{name}' has only {G.number_of_nodes()} nodes after processing")
        return None
    return G


# ---------------------------------------------------------------------------
# MVC solving
# ---------------------------------------------------------------------------

def solve_mvc(G: nx.Graph, backend: str, timeout: int):
    """Return a SolveResult for graph G."""
    from solver.mvc_solver import MVCSolver
    solver = MVCSolver(
        backend=backend,
        timeout=timeout,
        fallback_to_approx=(backend == "ilp"),
    )
    return solver.solve(G)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def graph_to_record(G: nx.Graph, name: str, cover) -> dict:
    """Serialise a graph + MVC cover to the standard JSON record format."""
    return {
        "n_nodes": G.number_of_nodes(),
        "edges":   [[int(u), int(v)] for u, v in G.edges()],
        "mvc":     sorted(int(v) for v in cover),
        "source":  f"realworld:{name}",
    }


# ---------------------------------------------------------------------------
# Per-dataset save (individual .json files)
# ---------------------------------------------------------------------------

def process_single(
    name: str,
    factory_fn: callable,
    backend: str,
    timeout: int,
    force: bool,
    data_dir: Path,
) -> Optional[dict]:
    """
    Generate, solve, and save one dataset to data_dir/{name}.json.

    Returns the record dict on success, None on skip/failure.
    """
    out_path = data_dir / f"{name}.json"
    if out_path.exists() and not force:
        print(f"  [skip] '{name}' already exists at {out_path}  (use --force to recompute)")
        # Load and return first record for summary
        with open(out_path) as f:
            records = json.load(f)
        return records[0] if records else None

    print(f"\n  Generating '{name}' ...")
    try:
        G_raw = factory_fn()
    except Exception as e:
        print(f"  [error] Could not build '{name}': {e}")
        return None

    G = preprocess(G_raw, name)
    if G is None:
        return None

    print(f"  Solving MVC for '{name}'  (n={G.number_of_nodes()}, "
          f"m={G.number_of_edges()}, backend={backend}, timeout={timeout}s) ...")
    try:
        result = solve_mvc(G, backend, timeout)
    except Exception as e:
        print(f"  [error] MVC solve failed for '{name}': {e}")
        return None

    record = graph_to_record(G, name, result.cover)
    with open(out_path, "w") as f:
        json.dump([record], f, indent=2)
    print(f"  Saved → {out_path}  "
          f"(n={record['n_nodes']}, m={len(record['edges'])}, "
          f"mvc_size={len(record['mvc'])}, backend={result.backend_used})")
    return record


# ---------------------------------------------------------------------------
# Consolidated builtin.json
# ---------------------------------------------------------------------------

def save_builtin(records: List[dict], data_dir: Path) -> None:
    """Save all records to data_dir/builtin.json."""
    out_path = data_dir / "builtin.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n  All records saved → {out_path}  ({len(records)} graphs)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(records: List[dict]) -> None:
    if not records:
        print("\nNo records to summarise.")
        return

    print("\n" + "=" * 62)
    print(f"  {'Name':<28} {'N':>5} {'M':>6} {'MVC':>5}")
    print("=" * 62)
    for rec in records:
        src = rec.get("source", "?").replace("realworld:", "")
        n = rec["n_nodes"]
        m = len(rec["edges"])
        mvc = len(rec["mvc"])
        print(f"  {src:<28} {n:>5} {m:>6} {mvc:>5}")
    print("=" * 62)
    print(f"  {'TOTAL':<28} {'':>5} {'':>6} {sum(len(r['mvc']) for r in records):>5}")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate real-world graphs with ILP MVC labels."
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=f"Only generate this dataset. Known: {', '.join(CATALOGUE_DICT)}",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite even if output file already exists.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-graph ILP timeout in seconds (default: 120).",
    )
    p.add_argument(
        "--approx",
        action="store_true",
        help="Use 2-approximation instead of ILP (faster but not optimal).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DATA_DIR,
        help=f"Output directory (default: {_DATA_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backend = "approx" if args.approx else "ilp"
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset is not None:
        if args.dataset not in CATALOGUE_DICT:
            print(f"[download_realworld] Unknown dataset '{args.dataset}'.")
            print(f"  Known datasets: {', '.join(CATALOGUE_DICT)}")
            sys.exit(1)
        entries = [(args.dataset, CATALOGUE_DICT[args.dataset])]
    else:
        entries = CATALOGUE

    print(f"[download_realworld] Generating {len(entries)} dataset(s)  "
          f"(backend={backend}, timeout={args.timeout}s) ...")

    all_records: List[dict] = []
    for name, factory_fn in entries:
        rec = process_single(name, factory_fn, backend, args.timeout, args.force, data_dir)
        if rec is not None:
            all_records.append(rec)

    # Always rebuild builtin.json when generating all datasets
    if args.dataset is None and all_records:
        # Collect all individual records (each file may have one record)
        builtin_records: List[dict] = []
        for name, _ in CATALOGUE:
            p = data_dir / f"{name}.json"
            if p.exists():
                with open(p) as f:
                    builtin_records.extend(json.load(f))
        if builtin_records:
            save_builtin(builtin_records, data_dir)

    print_summary(all_records)
    print(f"[download_realworld] Done.  "
          f"Files saved to: {data_dir.resolve()}")


if __name__ == "__main__":
    main()

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


def _make_grid_8x8() -> nx.Graph:
    G = nx.grid_2d_graph(8, 8)
    return nx.convert_node_labels_to_integers(G)


def _make_triangular_lattice() -> nx.Graph:
    G = nx.triangular_lattice_graph(6, 8)
    return nx.convert_node_labels_to_integers(G)


def _make_hexagonal_lattice() -> nx.Graph:
    G = nx.hexagonal_lattice_graph(5, 6)
    return nx.convert_node_labels_to_integers(G)


def _make_turan_graph() -> nx.Graph:
    return nx.turan_graph(50, 5)


def _make_random_regular_50_3() -> nx.Graph:
    return nx.random_regular_graph(3, 50, seed=42)


def _make_random_regular_75_4() -> nx.Graph:
    return nx.random_regular_graph(4, 76, seed=42)


def _make_random_regular_100_3() -> nx.Graph:
    return nx.random_regular_graph(3, 100, seed=42)


# ── Watts-Strogatz small-world graphs ────────────────────────────────────────
def _ws(n: int, k: int, p: float, seed: int) -> callable:
    def _make() -> nx.Graph:
        return nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _make


# ── Barabási-Albert scale-free graphs ────────────────────────────────────────
def _ba(n: int, m: int, seed: int) -> callable:
    def _make() -> nx.Graph:
        return nx.barabasi_albert_graph(n, m, seed=seed)
    return _make


# ── LFR benchmark graphs (community structure) ───────────────────────────────
def _lfr(n: int, tau1: float, tau2: float, mu: float,
         avg_degree: int, seed: int) -> callable:
    def _make() -> nx.Graph:
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n, tau1, tau2, mu,
                average_degree=avg_degree,
                min_community=max(10, n // 8),
                seed=seed,
            )
            return nx.Graph(G)
        except Exception as exc:
            raise RuntimeError(f"LFR failed (n={n}, mu={mu}): {exc}") from exc
    return _make


# ---------------------------------------------------------------------------
# Full catalogue — 14 named/structured + 9 extra structured + 10 WS + 10 BA +
# 15 LFR = 58 graphs (some LFR may fail and be skipped).
# ---------------------------------------------------------------------------
CATALOGUE: List[Tuple[str, callable]] = [
    # ── Named / classical structured graphs ──────────────────────────────────
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
    ("grid_8x8",              _make_grid_8x8),
    ("grid_10x10",            _make_grid_10x10),
    ("cycle_20",              _make_cycle_20),
    ("wheel_30",              _make_wheel_30),
    ("barbell_15",            _make_barbell_15),
    ("triangular_lattice",    _make_triangular_lattice),
    ("hexagonal_lattice",     _make_hexagonal_lattice),
    ("turan_50_5",            _make_turan_graph),
    ("rreg_50_3",             _make_random_regular_50_3),
    ("rreg_76_4",             _make_random_regular_75_4),
    ("rreg_100_3",            _make_random_regular_100_3),
    # ── Watts-Strogatz small-world graphs (10 instances) ─────────────────────
    ("ws_50_4_01",            _ws(50,  4, 0.10, seed=1)),
    ("ws_50_6_02",            _ws(50,  6, 0.20, seed=2)),
    ("ws_75_4_01",            _ws(75,  4, 0.10, seed=3)),
    ("ws_75_6_02",            _ws(75,  6, 0.20, seed=4)),
    ("ws_100_4_01",           _ws(100, 4, 0.10, seed=5)),
    ("ws_100_6_02",           _ws(100, 6, 0.20, seed=6)),
    ("ws_100_8_03",           _ws(100, 8, 0.30, seed=7)),
    ("ws_150_6_02",           _ws(150, 6, 0.20, seed=8)),
    ("ws_200_6_01",           _ws(200, 6, 0.10, seed=9)),
    ("ws_200_8_02",           _ws(200, 8, 0.20, seed=10)),
    # ── Barabási-Albert scale-free graphs (10 instances) ─────────────────────
    ("ba_50_2",               _ba(50,  2, seed=11)),
    ("ba_50_3",               _ba(50,  3, seed=12)),
    ("ba_75_2",               _ba(75,  2, seed=13)),
    ("ba_75_3",               _ba(75,  3, seed=14)),
    ("ba_100_2",              _ba(100, 2, seed=15)),
    ("ba_100_3",              _ba(100, 3, seed=16)),
    ("ba_150_2",              _ba(150, 2, seed=17)),
    ("ba_150_3",              _ba(150, 3, seed=18)),
    ("ba_200_2",              _ba(200, 2, seed=19)),
    ("ba_200_3",              _ba(200, 3, seed=20)),
    # ── LFR benchmark graphs — community structure (15 instances) ────────────
    # Parameters: (n, tau1, tau2, mu, avg_degree, seed)
    # tau1≈3 (power-law degree), tau2≈1.5 (power-law communities), mu=mixing
    ("lfr_50_mu01",           _lfr( 50, 3.0, 1.5, 0.10, 5, seed=21)),
    ("lfr_50_mu02",           _lfr( 50, 3.0, 1.5, 0.20, 5, seed=22)),
    ("lfr_50_mu03",           _lfr( 50, 3.0, 1.5, 0.30, 5, seed=23)),
    ("lfr_75_mu01",           _lfr( 75, 3.0, 1.5, 0.10, 5, seed=24)),
    ("lfr_75_mu02",           _lfr( 75, 3.0, 1.5, 0.20, 5, seed=25)),
    ("lfr_75_mu03",           _lfr( 75, 3.0, 1.5, 0.30, 5, seed=26)),
    ("lfr_100_mu01",          _lfr(100, 3.0, 1.5, 0.10, 6, seed=27)),
    ("lfr_100_mu02",          _lfr(100, 3.0, 1.5, 0.20, 6, seed=28)),
    ("lfr_100_mu03",          _lfr(100, 3.0, 1.5, 0.30, 6, seed=29)),
    ("lfr_150_mu01",          _lfr(150, 3.0, 1.5, 0.10, 6, seed=30)),
    ("lfr_150_mu02",          _lfr(150, 3.0, 1.5, 0.20, 6, seed=31)),
    ("lfr_150_mu03",          _lfr(150, 3.0, 1.5, 0.30, 6, seed=32)),
    ("lfr_200_mu01",          _lfr(200, 3.0, 1.5, 0.10, 6, seed=33)),
    ("lfr_200_mu02",          _lfr(200, 3.0, 1.5, 0.20, 6, seed=34)),
    ("lfr_200_mu03",          _lfr(200, 3.0, 1.5, 0.30, 6, seed=35)),
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

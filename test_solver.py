"""
test_solver.py
--------------
Quick smoke test for the ILP solver on graphs with known optimal MVC sizes.

    G1: path P4  (4 nodes, 3 edges)   → optimal MVC = 2 nodes
    G2: triangle K3 (3 nodes, 3 edges) → optimal MVC = 2 nodes
    G3: complete K5 (5 nodes, 10 edges) → optimal MVC = 4 nodes

Run:
    python test_solver.py
"""

from __future__ import annotations

import sys
import networkx as nx

from pruning.heuristic import verify_cover
from solver.mvc_solver import MVCSolver


def run_test(name: str, G: nx.Graph, expected_size: int, backend: str) -> bool:
    solver = MVCSolver(backend=backend, timeout=30, fallback_to_approx=False)
    result = solver.solve(G)
    valid = verify_cover(G, result.cover)
    ok = valid and result.size == expected_size
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name}  "
        f"cover_size={result.size} (expected={expected_size})  "
        f"valid={valid}  backend={result.backend_used}  "
        f"time={result.runtime_seconds*1000:.1f}ms"
    )
    if not ok:
        print(f"         cover={sorted(result.cover)}")
        print(f"         edges={list(G.edges())}")
    return ok


def main() -> None:
    tests = [
        ("P4 (path 4)",   nx.path_graph(4),     2),
        ("K3 (triangle)", nx.complete_graph(3),  2),
        ("K5 (complete)", nx.complete_graph(5),  4),
    ]

    all_passed = True
    for backend in ("ilp", "approx"):
        print(f"\n--- backend: {backend} ---")
        for name, G, expected in tests:
            ok = run_test(name, G, expected, backend)
            if backend == "ilp":  # only ILP must be exact
                all_passed = all_passed and ok

    print()
    if all_passed:
        print("All ILP tests PASSED.")
        sys.exit(0)
    else:
        print("Some ILP tests FAILED — check solver output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

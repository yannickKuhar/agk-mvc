"""
compute_tudataset_mvc.py
------------------------
Download a TUDataset and compute optimal MVC labels using our ILP solver.
Labels are cached to data/tudatasets/{NAME}_mvc.json.

Usage:
    python compute_tudataset_mvc.py --dataset MUTAG
    python compute_tudataset_mvc.py --dataset NCI1 --max-graphs 500
    python compute_tudataset_mvc.py --dataset MUTAG --recompute
"""

from __future__ import annotations

import argparse

from data.tudataset import load_tudataset, KNOWN_DATASETS
from data.loader import graph_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MVC labels for a TUDataset")
    p.add_argument("--dataset", required=True,
                   help=f"Dataset name. Known: {', '.join(KNOWN_DATASETS)}")
    p.add_argument("--max-graphs", type=int, default=None)
    p.add_argument("--min-nodes", type=int, default=5)
    p.add_argument("--max-nodes", type=int, default=500)
    p.add_argument("--solver-timeout", type=int, default=60,
                   help="ILP timeout per graph in seconds (default: 60)")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute labels even if cache exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graphs = load_tudataset(
        name=args.dataset,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        solver_timeout=args.solver_timeout,
        recompute=args.recompute,
    )
    print(f"\nResult: {len(graphs)} graphs from {args.dataset}")
    if graphs:
        summary = graph_summary(graphs)
        print(f"  Nodes: min={summary['min_nodes']}, max={summary['max_nodes']}, "
              f"mean={summary['avg_nodes']:.1f}")
        print(f"  Edges: mean={summary['avg_edges']:.1f}")
        print(f"  MVC positive fraction: {summary['avg_pos_fraction']:.3f}")


if __name__ == "__main__":
    main()

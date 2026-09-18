"""
generate_synthetic.py
---------------------
Generate synthetic graph datasets with ILP-computed MVC labels.

Splits:
  small   n=[20,50]   200 graphs/family  → data/synthetic/small.json
  medium  n=[50,150]  100 graphs/family  → data/synthetic/medium.json
  large   n=[150,500]  50 graphs/family  → data/synthetic/large.json

Usage:
    python generate_synthetic.py --split small
    python generate_synthetic.py --split all
    python generate_synthetic.py --split small --families erdos_renyi,barabasi_albert
    python generate_synthetic.py --split medium --n-per-family 50 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.synthetic import generate_dataset, _FAMILIES

SPLIT_CONFIGS = {
    "small": {
        "n_range": (20, 50),
        "n_per_family": 200,
        "solver_timeout": 60,
        "density_range": (0.1, 0.4),
    },
    "medium": {
        "n_range": (50, 150),
        "n_per_family": 100,
        "solver_timeout": 120,
        "density_range": (0.1, 0.4),
    },
    "large": {
        "n_range": (150, 500),
        "n_per_family": 50,
        "solver_timeout": 300,
        "density_range": (0.1, 0.4),
    },
    # Graphs where ILP takes 1–15 s — the regime where pruning gives real speedup.
    # n=50–80, higher density (p=0.3–0.5) confirmed to produce 1–15 s ILP times.
    "hard": {
        "n_range": (50, 80),
        "n_per_family": 75,
        "solver_timeout": 300,
        "density_range": (0.3, 0.5),
    },
}

_OUT_DIR = Path("data/synthetic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic MVC datasets")
    p.add_argument("--split", choices=["small", "medium", "large", "hard", "all"],
                   default="small", help="Which split to generate (default: small)")
    p.add_argument("--families", type=str, default=None,
                   help="Comma-separated families (default: all four)")
    p.add_argument("--n-per-family", type=int, default=None,
                   help="Override n_per_family for the chosen split")
    p.add_argument("--solver-timeout", type=int, default=None,
                   help="Override ILP timeout per graph (seconds)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_split(split_name: str, args: argparse.Namespace) -> None:
    cfg = SPLIT_CONFIGS[split_name]
    families     = args.families.split(",") if args.families else _FAMILIES
    n_per_family = args.n_per_family or cfg["n_per_family"]
    timeout      = args.solver_timeout or cfg["solver_timeout"]

    density_range = cfg.get("density_range", (0.1, 0.4))

    print(f"\n=== Generating split '{split_name}' ===")
    print(f"  families={families}, n_range={cfg['n_range']}, "
          f"n_per_family={n_per_family}, timeout={timeout}s, "
          f"density_range={density_range}")

    output_path = _OUT_DIR / f"{split_name}.json"
    graphs = generate_dataset(
        families=families,
        n_graphs=n_per_family,
        n_range=cfg["n_range"],
        solver_timeout=timeout,
        density_range=density_range,
        output_path=output_path,
        seed=args.seed,
    )
    if not graphs:
        print(f"  WARNING: 0 graphs saved for split '{split_name}'. "
              f"All ILP solves timed out — try increasing --solver-timeout.")
        return
    sizes = [G.number_of_nodes() for G in graphs]
    print(f"  Done: {len(graphs)} graphs, "
          f"n in [{min(sizes)}, {max(sizes)}], mean={sum(sizes)/len(sizes):.1f}")


def main() -> None:
    args = parse_args()
    splits = list(SPLIT_CONFIGS) if args.split == "all" else [args.split]
    for split in splits:
        run_split(split, args)


if __name__ == "__main__":
    main()

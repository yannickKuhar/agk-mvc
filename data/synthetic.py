"""
data/synthetic.py
-----------------
Generate synthetic random graphs with ILP-computed optimal MVC labels.

Families:
  erdos_renyi      G(n, p)      random with edge probability p ∈ [0.1, 0.4]
  barabasi_albert  G(n, m)      scale-free via preferential attachment
  watts_strogatz   G(n, k, p)   small-world
  random_regular   G(n, d)      d-regular

Each generated graph:
  - Has been solved exactly by our ILP (solver/mvc_solver.py)
  - Has node attribute 'label' = 1 if in MVC, 0 otherwise
  - Has graph attribute 'dataset' = 'synthetic:{family}'

Saved as data/synthetic/{split}.json in the same format as data/erdos/train.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx

_FAMILIES = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "random_regular"]

_DATA_DIR = Path(__file__).parent / "synthetic"


def _generate_one(
    family: str,
    n: int,
    rng: random.Random,
    density_range: Tuple[float, float] = (0.1, 0.4),
) -> Optional[nx.Graph]:
    """Generate one random graph of the given family. Returns None on failure."""
    seed = rng.randint(0, 2**31 - 1)
    try:
        if family == "erdos_renyi":
            p = rng.uniform(*density_range)
            G = nx.erdos_renyi_graph(n, p, seed=seed)

        elif family == "barabasi_albert":
            m = max(1, min(n // 4, rng.randint(1, 5)))
            G = nx.barabasi_albert_graph(n, m, seed=seed)

        elif family == "watts_strogatz":
            k = rng.randint(2, min(n - 1, 8))
            k = k if k % 2 == 0 else k + 1  # must be even
            k = max(2, min(k, n - 1))
            p = rng.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n, k, p, seed=seed)

        elif family == "random_regular":
            d = rng.randint(2, min(n - 1, 6))
            if (n * d) % 2 != 0:
                d = max(2, d - 1)
            G = nx.random_regular_graph(d, n, seed=seed)

        else:
            raise ValueError(f"Unknown family: {family}")

        G = nx.convert_node_labels_to_integers(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G if G.number_of_edges() > 0 else None

    except Exception:
        return None


def generate_dataset(
    families: List[str],
    n_graphs: int,
    n_range: Tuple[int, int],
    solver_timeout: int = 120,
    density_range: Tuple[float, float] = (0.1, 0.4),
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> List[nx.Graph]:
    """
    Generate random graphs with ILP-computed optimal MVC labels.

    Parameters
    ----------
    families       : graph families to sample from
    n_graphs       : number of graphs per family
    n_range        : (min_nodes, max_nodes) inclusive
    solver_timeout : ILP timeout per graph in seconds; graph is skipped on timeout
    output_path    : save records to this JSON file if given
    seed           : random seed

    Returns
    -------
    List of nx.Graph with node attribute 'label' ∈ {0, 1}
    and graph attribute 'dataset' = 'synthetic:{family}'.
    """
    from solver.mvc_solver import MVCSolver

    rng = random.Random(seed)
    solver = MVCSolver(backend="ilp", timeout=solver_timeout, fallback_to_approx=False)
    n_min, n_max = n_range

    all_graphs: List[nx.Graph] = []
    records: List[dict] = []

    for family in families:
        print(f"[synthetic] {family}: generating {n_graphs} graphs, "
              f"n=[{n_min},{n_max}], timeout={solver_timeout}s")
        generated = 0
        skipped_gen = 0
        skipped_timeout = 0
        max_attempts = n_graphs * 20

        for attempt in range(max_attempts):
            if generated >= n_graphs:
                break

            n = rng.randint(n_min, n_max)
            G = _generate_one(family, n, rng, density_range=density_range)
            if G is None:
                skipped_gen += 1
                continue

            t0 = time.perf_counter()
            try:
                result = solver.solve(G)
            except Exception:
                skipped_timeout += 1
                continue
            solve_time = time.perf_counter() - t0

            mvc_set = result.cover
            for v in G.nodes():
                G.nodes[v]["label"] = 1 if v in mvc_set else 0
            G.graph["dataset"] = f"synthetic:{family}"

            all_graphs.append(G)
            records.append({
                "n_nodes": G.number_of_nodes(),
                "edges":   [[u, v] for u, v in G.edges()],
                "mvc":     sorted(mvc_set),
                "source":  f"synthetic:{family}:n{n}:t{solve_time:.3f}s",
            })
            generated += 1

            if generated % 50 == 0 or generated == n_graphs:
                print(f"  {generated}/{n_graphs}  "
                      f"(skip_gen={skipped_gen}, skip_timeout={skipped_timeout})")

        if generated < n_graphs:
            print(f"  [WARNING] Only {generated}/{n_graphs} graphs produced "
                  f"(timeout on {skipped_timeout}).")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[synthetic] Saved {len(records)} records → {output_path}")

    return all_graphs

"""
features/ajwani.py
------------------
13 node features from O'Connor, Coleman, Strash, Ray & Ajwani (CPAIOR 2026).
"A Scalable Learning Approach for Efficient Computation of Independent Set
and Cover Variants."

Feature set (Table 1):
  Structural (9):
    0  lcc              – local clustering coefficient
    1  degree_centrality
    2  core_number      – k-shell (normalised by max core number)
    3  degree_rank      – normalised rank of degree in [0, 1]
    4  min_nbr_deg_rank – min degree rank over neighbours (0 for isolated)
    5  max_nbr_deg_rank – max degree rank over neighbours
    6  avg_nbr_deg_rank – mean degree rank over neighbours
    7  pagerank
    8  mis_freq         – maximal IS membership frequency via Luby's algorithm

  MWUA LPR approximation (4):
    9   mwua_avg_x      – average fractional MIS value across MWUA iterations
    10  mwua_avg_w      – mean  incident edge weight (final MWUA weights)
    11  mwua_min_w      – min   incident edge weight
    12  mwua_max_w      – max   incident edge weight

Implementation note:
  The MWUA (Algorithm 1 in the paper) approximates the LP relaxation for the
  Maximum Independent Set problem.  For MVC, the complement 1 - mwua_avg_x
  approximates the MVC LP variable.  We expose the raw MIS-LP features (as
  the paper does) and let the classifier learn the mapping.
"""
from __future__ import annotations

import time
from typing import List

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feature_names() -> List[str]:
    return [
        "lcc", "degree_centrality", "core_number",
        "degree_rank", "min_nbr_deg_rank", "max_nbr_deg_rank", "avg_nbr_deg_rank",
        "pagerank",
        "mis_freq",
        "mwua_avg_x", "mwua_avg_w", "mwua_min_w", "mwua_max_w",
    ]


def compute_ajwani_features(
    G: nx.Graph,
    mis_freq_runs: int = 20,
    mis_freq_seed: int = 0,
    mwua_t_max: float = 2.0,
) -> np.ndarray:
    """
    Compute the 13-dim Ajwani et al. feature vector for every node in G.

    Parameters
    ----------
    G               : undirected simple graph (nodes labelled arbitrarily)
    mis_freq_runs   : number of Luby MIS runs for the frequency estimate
    mis_freq_seed   : random seed for Luby runs (reproducibility)
    mwua_t_max      : wall-clock seconds budget for the MWUA loop per graph

    Returns
    -------
    np.ndarray of shape (n_nodes, 13), float64, nodes in list(G.nodes()) order
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return np.empty((0, 13), dtype=np.float64)

    node_idx = {v: i for i, v in enumerate(nodes)}

    # ── Structural features ──────────────────────────────────────────────────

    # 0: LCC
    lcc_dict = nx.clustering(G)
    lcc = np.array([lcc_dict[v] for v in nodes], dtype=np.float64)

    # 1: Degree centrality (= degree / (n-1))
    dc_dict = nx.degree_centrality(G)
    deg_cent = np.array([dc_dict[v] for v in nodes], dtype=np.float64)

    # 2: Core number (k-shell), normalised to [0, 1]
    cn_dict = nx.core_number(G)
    cn = np.array([cn_dict[v] for v in nodes], dtype=np.float64)
    max_cn = cn.max() if cn.max() > 0 else 1.0
    cn_norm = cn / max_cn

    # 3–6: Degree rank and neighbour degree ranks
    raw_deg = np.array([G.degree(v) for v in nodes], dtype=np.float64)
    deg_rank = _normalised_rank(raw_deg)

    nbr_min = np.zeros(n, dtype=np.float64)
    nbr_max = np.zeros(n, dtype=np.float64)
    nbr_avg = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(nodes):
        nbrs = list(G.neighbors(v))
        if nbrs:
            ranks = deg_rank[[node_idx[u] for u in nbrs]]
            nbr_min[i] = ranks.min()
            nbr_max[i] = ranks.max()
            nbr_avg[i] = ranks.mean()

    # 7: PageRank
    try:
        pr_dict = nx.pagerank(G, max_iter=200)
        pagerank = np.array([pr_dict[v] for v in nodes], dtype=np.float64)
    except Exception:
        # Fallback: uniform
        pagerank = np.full(n, 1.0 / n, dtype=np.float64)

    # 8: MIS membership frequency (Luby's algorithm)
    mis_freq = _mis_frequency(G, nodes, node_idx,
                               n_runs=mis_freq_runs, seed=mis_freq_seed)

    # ── MWUA LPR features ────────────────────────────────────────────────────

    mwua = _mwua_features(G, nodes, node_idx, t_max=mwua_t_max)

    # ── Assemble ─────────────────────────────────────────────────────────────
    X = np.column_stack([
        lcc,        # 0
        deg_cent,   # 1
        cn_norm,    # 2
        deg_rank,   # 3
        nbr_min,    # 4
        nbr_max,    # 5
        nbr_avg,    # 6
        pagerank,   # 7
        mis_freq,   # 8
        mwua,       # 9-12
    ])
    return X.astype(np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalised_rank(values: np.ndarray) -> np.ndarray:
    """
    Sort the unique values, assign each element its rank, normalise to [0, 1].
    Ties receive the same rank.  A single unique value maps to 0.0.
    """
    unique = np.unique(values)
    if len(unique) == 1:
        return np.zeros(len(values), dtype=np.float64)
    rank_map = {v: i / (len(unique) - 1) for i, v in enumerate(unique)}
    return np.array([rank_map[v] for v in values], dtype=np.float64)


def _mis_frequency(
    G: nx.Graph,
    nodes: list,
    node_idx: dict,
    n_runs: int,
    seed: int,
) -> np.ndarray:
    """
    Estimate MIS membership frequency by running Luby's randomised maximal IS
    algorithm n_runs times.

    Each run: assign uniform random priority r(v); add v to MIS if r(v) is
    strictly higher than all neighbours still in the graph; remove v and its
    neighbours; repeat until the graph is empty.
    """
    n = len(nodes)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Pre-build adjacency as sets of integer indices for speed
    adj_idx: list[set] = [set() for _ in range(n)]
    for u, v in G.edges():
        ui, vi = node_idx[u], node_idx[v]
        adj_idx[ui].add(vi)
        adj_idx[vi].add(ui)

    count = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for _ in range(n_runs):
        remaining = set(range(n))
        r = rng.random(n)

        while remaining:
            # Find winners: highest priority among self + remaining neighbours
            winners = {
                i for i in remaining
                if all(r[i] > r[j]
                       for j in adj_idx[i] if j in remaining)
            }

            count[list(winners)] += 1

            # Remove winners and their remaining neighbours
            to_remove: set[int] = set()
            for i in winners:
                to_remove.add(i)
                to_remove.update(adj_idx[i] & remaining)
            remaining -= to_remove

            # Refresh random values for next round (Luby's requirement)
            r = rng.random(n)

    return count / n_runs


def _mwua_features(
    G: nx.Graph,
    nodes: list,
    node_idx: dict,
    t_max: float = 2.0,
    convergence_tol: float = 1e-5,
) -> np.ndarray:
    """
    Approximate the MIS LP relaxation via the Multiplicative Weights Update
    algorithm (Algorithm 1 in the paper).

    For each edge (u, v) the constraint is x_u + x_v ≤ 1.
    Edge weights track constraint pressure; vertices with low incident weight
    sum are less constrained and tend to receive high x_v (likely in IS).

    Returns (n, 4) array: [avg_x, avg_w, min_w, max_w] per vertex.
    """
    n = len(nodes)
    edges = list(G.edges())
    m = len(edges)

    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    if m == 0:
        # No edges: every vertex is trivially in the MIS with x_v = 1
        return np.column_stack([
            np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n),
        ]).astype(np.float64)

    # Incident edge lists per vertex
    incident: list[list[int]] = [[] for _ in range(n)]
    edge_ui = np.empty(m, dtype=np.intp)
    edge_vi = np.empty(m, dtype=np.intp)
    for ei, (u, v) in enumerate(edges):
        ui, vi = node_idx[u], node_idx[v]
        edge_ui[ei] = ui
        edge_vi[ei] = vi
        incident[ui].append(ei)
        incident[vi].append(ei)

    incident_arr = [np.array(inc, dtype=np.intp) for inc in incident]

    w = np.ones(m, dtype=np.float64)
    solutions: list[np.ndarray] = []
    prev_x: np.ndarray | None = None
    t0 = time.time()

    while (time.time() - t0) < t_max:
        # s_v = sum of incident edge weights (per Algorithm 1, line 5)
        s = np.array(
            [w[incident_arr[i]].sum() if len(incident_arr[i]) else 0.0
             for i in range(n)],
            dtype=np.float64,
        )

        w_total = w.sum()
        s_norm = s / w_total if w_total > 1e-12 else np.zeros(n)

        # x_v for MIS: vertices with low s_norm are less constrained → higher x
        x = np.maximum(0.0, 1.0 - s_norm)
        solutions.append(x)

        # Convergence check
        if prev_x is not None and np.abs(x - prev_x).max() < convergence_tol:
            break
        prev_x = x

        # Update: increase w_e when constraint x_u + x_v ≤ 1 is violated
        violation = np.maximum(0.0, x[edge_ui] + x[edge_vi] - 1.0)
        w *= (1.0 + violation)

        # Normalise to prevent overflow
        w_max = w.max()
        if w_max > 1e8:
            w /= w_max

    avg_x = np.mean(solutions, axis=0) if solutions else np.zeros(n)

    # Per-vertex statistics of final edge weights
    avg_w = np.array(
        [w[incident_arr[i]].mean() if len(incident_arr[i]) else 0.0
         for i in range(n)],
        dtype=np.float64,
    )
    min_w = np.array(
        [w[incident_arr[i]].min() if len(incident_arr[i]) else 0.0
         for i in range(n)],
        dtype=np.float64,
    )
    max_w = np.array(
        [w[incident_arr[i]].max() if len(incident_arr[i]) else 0.0
         for i in range(n)],
        dtype=np.float64,
    )

    return np.column_stack([avg_x, avg_w, min_w, max_w]).astype(np.float64)

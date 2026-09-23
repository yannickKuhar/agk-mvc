"""
features/lauri.py
-----------------
9 handcrafted node features from Lauri et al. (2023).

Reference:
  Lauri, J., Dutta, S., Dobbe, R., & Gadde, C. (2023).
  "Learning fine-grained search space pruning and heuristics for
  combinatorial optimization." Journal of Heuristics, 29, 313–347.

Graph-theoretic: n_nodes, n_edges, degree, lcc, eigenvec_centrality
Statistical:     chi2_degree, chi2_degree_neighbors,
                 chi2_lcc, chi2_lcc_neighbors
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List


def feature_names() -> List[str]:
    return [
        "n_nodes", "n_edges", "degree", "lcc", "eigenvec_centrality",
        "chi2_degree", "chi2_degree_nbr", "chi2_lcc", "chi2_lcc_nbr",
    ]


def compute_lauri_features(G: nx.Graph) -> np.ndarray:
    """
    Returns (n_nodes, 9) float64 array, nodes in list(G.nodes()) order.

    Features (columns):
      0  n_nodes              – number of vertices (global, same for all nodes)
      1  n_edges              – number of edges   (global, same for all nodes)
      2  degree(v)
      3  lcc(v)               – local clustering coefficient
      4  eigenvec_centrality  – falls back to normalised degree centrality on
                                failure (disconnected graph, no edges, etc.)
      5  chi2_degree          – (degree(v) - mean_degree)² / max(mean_degree, 1e-9)
      6  chi2_degree_nbr      – mean of chi2_degree(u) over neighbours of v;
                                0.0 for isolated nodes
      7  chi2_lcc             – (lcc(v) - mean_lcc)²   / max(mean_lcc, 1e-9)
      8  chi2_lcc_nbr         – mean of chi2_lcc(u) over neighbours of v;
                                0.0 for isolated nodes
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return np.empty((0, 9), dtype=np.float64)

    node_idx = {v: i for i, v in enumerate(nodes)}

    # ── Global features ──────────────────────────────────────────────────────
    n_nodes = float(n)
    n_edges = float(G.number_of_edges())

    # ── Degree ───────────────────────────────────────────────────────────────
    deg = np.array([G.degree(v) for v in nodes], dtype=np.float64)

    # ── Local clustering coefficient ─────────────────────────────────────────
    lcc_dict = nx.clustering(G)
    lcc = np.array([lcc_dict[v] for v in nodes], dtype=np.float64)

    # ── Eigenvector centrality ────────────────────────────────────────────────
    try:
        ec_dict = nx.eigenvector_centrality(G, max_iter=1000)
        ec = np.array([ec_dict[v] for v in nodes], dtype=np.float64)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException, Exception):
        # Fall back to degree centrality (normalised by n-1)
        denom = max(n - 1, 1)
        ec = deg / denom

    # ── Chi-squared deviations ────────────────────────────────────────────────
    mean_deg = deg.mean()
    mean_lcc = lcc.mean()

    chi2_deg = (deg - mean_deg) ** 2 / max(mean_deg, 1e-9)
    chi2_lcc = (lcc - mean_lcc) ** 2 / max(mean_lcc, 1e-9)

    # ── Neighbour-averaged chi2 ───────────────────────────────────────────────
    chi2_deg_nbr = np.zeros(n, dtype=np.float64)
    chi2_lcc_nbr = np.zeros(n, dtype=np.float64)

    for i, v in enumerate(nodes):
        nbrs = list(G.neighbors(v))
        if nbrs:
            nbr_ids = [node_idx[u] for u in nbrs]
            chi2_deg_nbr[i] = chi2_deg[nbr_ids].mean()
            chi2_lcc_nbr[i] = chi2_lcc[nbr_ids].mean()
        # else stays 0.0 (isolated node)

    # ── Assemble matrix ───────────────────────────────────────────────────────
    X = np.column_stack([
        np.full(n, n_nodes),   # F1
        np.full(n, n_edges),   # F2
        deg,                   # F3
        lcc,                   # F4
        ec,                    # F5
        chi2_deg,              # F6
        chi2_deg_nbr,          # F7
        chi2_lcc,              # F8
        chi2_lcc_nbr,          # F9
    ])

    return X.astype(np.float64)

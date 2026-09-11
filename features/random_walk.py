"""
features/random_walk.py
-----------------------
Per-node random walk statistics for use as node features.

Features computed per node v:
  1.  degree                      — deg(v)
  2.  degree_normalized           — deg(v) / (n-1)
  3.  return_probability_t2       — P(walk returns to v in 2 steps)
  4.  return_probability_t4       — P(walk returns to v in 4 steps)
  5.  stationary_probability      — pi(v) = deg(v) / (2|E|)
  6.  local_clustering            — C(v)
  7.  avg_neighbor_degree         — mean degree of neighbors
  8.  max_neighbor_degree         — max degree among neighbors
  9.  min_neighbor_degree         — min degree among neighbors
  10. degree_centrality           — NetworkX degree centrality
  11. closeness_centrality        — (approximate, uses BFS distances)
  12. betweenness_centrality      — (sampled for large graphs)
  13. eigenvector_centrality      — (power iteration, may not converge → fallback 0)
  14. pagerank                    — PageRank score
  15. core_number                 — k-core decomposition
  16. eccentricity                — max BFS distance (only for connected G; else -1)
  17. avg_shortest_path_to_nbrs   — mean distance to 2-hop neighbors
  18. hitting_time_proxy          — 1 / stationary_probability (∝ avg hitting time)
  19. commute_time_proxy          — 2|E| / (deg(v) * avg_neighbor_degree)
  20. triangle_count              — number of triangles containing v
"""

from __future__ import annotations

import warnings
from typing import List

import networkx as nx
import numpy as np

RW_DIM = 20


class RandomWalkNodeFeatures:
    """
    Compute per-node random walk and structural features.

    Parameters
    ----------
    betweenness_k : int or None
        Number of pivot nodes to sample for betweenness estimation.
        None = exact (slow for large graphs). Default 50.
    eigenvector_max_iter : int
        Max iterations for eigenvector centrality. Default 100.
    closeness_use_distance : bool
        If True, compute closeness from BFS. If False, set to 0 for speed.
    """

    def __init__(
        self,
        betweenness_k: int = 50,
        eigenvector_max_iter: int = 100,
        closeness_use_distance: bool = True,
    ):
        self.betweenness_k = betweenness_k
        self.eigenvector_max_iter = eigenvector_max_iter
        self.closeness_use_distance = closeness_use_distance

    def fit_transform(self, G: nx.Graph) -> np.ndarray:
        """
        Compute RW feature matrix for all nodes in G.

        Returns
        -------
        X : np.ndarray of shape (n_nodes, 20)
        """
        nodes = list(G.nodes())
        n = len(nodes)
        node_to_idx = {v: i for i, v in enumerate(nodes)}
        X = np.zeros((n, RW_DIM), dtype=np.float64)

        if n == 0:
            return X

        m = G.number_of_edges()

        # --- Degree features ---
        degrees = dict(G.degree())
        deg_arr = np.array([degrees[v] for v in nodes], dtype=np.float64)
        X[:, 0] = deg_arr                                        # raw degree
        X[:, 1] = deg_arr / max(n - 1, 1)                       # normalized

        # --- Stationary probability ---
        denom = 2.0 * m if m > 0 else 1.0
        stat_prob = deg_arr / denom
        X[:, 4] = stat_prob                                      # pi(v)

        # --- Return probabilities (diagonal of P^t) ---
        # P^2[v,v] = sum_{u~v} 1/(deg(v)*deg(u)) = 1/deg(v) * sum_{u~v} 1/deg(u)
        for i, v in enumerate(nodes):
            dv = degrees[v]
            if dv == 0:
                continue
            nbr_inv_sum = sum(1.0 / max(degrees[u], 1) for u in G.neighbors(v))
            X[i, 2] = nbr_inv_sum / dv                          # P^2[v,v]
            # P^4[v,v] ≈ (P^2[v,v])^2  (crude approximation)
            X[i, 3] = X[i, 2] ** 2                              # P^4[v,v]

        # --- Local clustering coefficient ---
        clustering = nx.clustering(G)
        X[:, 5] = np.array([clustering[v] for v in nodes])

        # --- Neighbor degree statistics ---
        for i, v in enumerate(nodes):
            nbrs = list(G.neighbors(v))
            if nbrs:
                nd = [degrees[u] for u in nbrs]
                X[i, 6] = np.mean(nd)
                X[i, 7] = np.max(nd)
                X[i, 8] = np.min(nd)
            else:
                X[i, 6] = 0.0
                X[i, 7] = 0.0
                X[i, 8] = 0.0

        # --- Centrality measures ---
        # Degree centrality
        deg_cent = nx.degree_centrality(G)
        X[:, 9] = np.array([deg_cent[v] for v in nodes])

        # Closeness centrality
        if self.closeness_use_distance:
            close_cent = nx.closeness_centrality(G)
            X[:, 10] = np.array([close_cent[v] for v in nodes])

        # Betweenness centrality (sampled)
        try:
            if self.betweenness_k is not None and n > self.betweenness_k:
                between_cent = nx.betweenness_centrality(G, k=self.betweenness_k)
            else:
                between_cent = nx.betweenness_centrality(G)
            X[:, 11] = np.array([between_cent[v] for v in nodes])
        except Exception:
            pass

        # Eigenvector centrality
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eig_cent = nx.eigenvector_centrality(
                    G, max_iter=self.eigenvector_max_iter, tol=1e-4
                )
            X[:, 12] = np.array([eig_cent[v] for v in nodes])
        except (nx.PowerIterationFailedConvergence, Exception):
            pass  # leave as 0

        # PageRank
        try:
            pr = nx.pagerank(G, max_iter=100)
            X[:, 13] = np.array([pr[v] for v in nodes])
        except Exception:
            pass

        # --- k-core number ---
        try:
            core = nx.core_number(G)
            X[:, 14] = np.array([core[v] for v in nodes], dtype=np.float64)
        except Exception:
            pass

        # --- Eccentricity (only for connected graphs) ---
        if nx.is_connected(G):
            try:
                ecc = nx.eccentricity(G)
                X[:, 15] = np.array([ecc[v] for v in nodes], dtype=np.float64)
            except Exception:
                X[:, 15] = -1.0
        else:
            X[:, 15] = -1.0

        # --- Avg shortest path to 2-hop neighbors ---
        for i, v in enumerate(nodes):
            two_hop = set()
            for u in G.neighbors(v):
                two_hop |= set(G.neighbors(u))
            two_hop.discard(v)
            if two_hop:
                dists = []
                for w in two_hop:
                    try:
                        dists.append(nx.shortest_path_length(G, v, w))
                    except nx.NetworkXNoPath:
                        pass
                X[i, 16] = np.mean(dists) if dists else 0.0

        # --- Hitting time proxy: 1 / pi(v) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            hit_proxy = np.where(stat_prob > 0, 1.0 / stat_prob, 0.0)
        # Normalize by max to keep scale reasonable
        max_hit = hit_proxy.max()
        X[:, 17] = hit_proxy / max(max_hit, 1.0)

        # --- Commute time proxy: 2|E| / (deg(v) * avg_neighbor_degree) ---
        for i, v in enumerate(nodes):
            dv = degrees[v]
            avg_nd = X[i, 6]  # already computed
            if dv > 0 and avg_nd > 0:
                X[i, 18] = (2.0 * m) / (dv * avg_nd)

        # --- Triangle count ---
        triangles = nx.triangles(G)
        X[:, 19] = np.array([triangles[v] for v in nodes], dtype=np.float64)

        return X


def feature_names() -> List[str]:
    return [
        "rw_degree",
        "rw_degree_normalized",
        "rw_return_prob_t2",
        "rw_return_prob_t4",
        "rw_stationary_prob",
        "rw_local_clustering",
        "rw_avg_neighbor_degree",
        "rw_max_neighbor_degree",
        "rw_min_neighbor_degree",
        "rw_degree_centrality",
        "rw_closeness_centrality",
        "rw_betweenness_centrality",
        "rw_eigenvector_centrality",
        "rw_pagerank",
        "rw_core_number",
        "rw_eccentricity",
        "rw_avg_dist_2hop_nbrs",
        "rw_hitting_time_proxy",
        "rw_commute_time_proxy",
        "rw_triangle_count",
    ]

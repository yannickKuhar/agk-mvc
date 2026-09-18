"""
pruning/structural.py
---------------------
Fast structural pruner for MVC pre-processing.

Uses only local graph features (no ML model, no feature extraction pipeline)
to estimate P(node ∈ optimal MVC) and prune unlikely nodes.

Feature weights are taken from mean XGBoost gain importance observed across
10 training runs on Erdős/synthetic/TUDataset benchmarks.  Duplicate features
that measure the same quantity are merged:

  rw_degree + gdv_orbit_0               73.0  (node degree)
  gdv_orbit_1 + rw_triangle_count       41.0  (triangles per node)
  rw_return_prob_t2 + rw_return_prob_t4 24.6  (RW return probability)
  rw_min_neighbor_degree                20.5
  gdv_orbit_3                           11.4  (2-paths, v as endpoint)
  rw_core_number                         9.9
  rw_avg_neighbor_degree                 7.7

All features are min-max normalised to [0, 1] within the graph before
combining, so the threshold has the same semantics as ConfidencePruner.

Time complexity: O(m·sqrt(m)) dominated by triangle enumeration.
"""
from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

from pruning.heuristic import ConfidencePruner, PruningResult


_WEIGHTS: Dict[str, float] = {
    "degree":         73.0,
    "triangles":      41.0,
    "return_prob_t2": 24.6,
    "min_nbr_degree": 20.5,
    "orbit_3":        11.4,
    "core_number":     9.9,
    "avg_nbr_degree":  7.7,
}
_TOTAL_WEIGHT = sum(_WEIGHTS.values())


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def structural_scores(G: nx.Graph) -> Dict:
    """
    Compute a structural MVC-membership score for every node.

    Returns {node: score} where score ∈ [0, 1].  Higher means more likely
    to be in the optimal MVC.  No model or feature pipeline required.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # ---- degree -------------------------------------------------------
    deg = np.array([G.degree(v) for v in nodes], dtype=float)

    # ---- triangles ----------------------------------------------------
    tri_dict = nx.triangles(G)
    tri = np.array([tri_dict[v] for v in nodes], dtype=float)

    # ---- neighbour-degree features ------------------------------------
    min_nbr = np.zeros(n)
    avg_nbr = np.zeros(n)
    ret_p2  = np.zeros(n)   # P(return at t=2)
    orbit3  = np.zeros(n)   # sum_{u in N(v)} (deg(u) - 1)

    for i, v in enumerate(nodes):
        nbrs = list(G.neighbors(v))
        if not nbrs:
            continue
        nd = [G.degree(u) for u in nbrs]
        min_nbr[i] = min(nd)
        avg_nbr[i] = sum(nd) / len(nd)
        # P(X_2 = v | X_0 = v) = (1/deg(v)) * sum_{u} 1/deg(u)
        ret_p2[i]  = sum(1.0 / max(d, 1) for d in nd) / max(deg[i], 1)
        orbit3[i]  = sum(max(d - 1, 0) for d in nd)

    # ---- k-core -------------------------------------------------------
    core_dict = nx.core_number(G)
    core = np.array([core_dict[v] for v in nodes], dtype=float)

    # ---- weighted combination -----------------------------------------
    score = (
        _WEIGHTS["degree"]         * _minmax(deg)     +
        _WEIGHTS["triangles"]      * _minmax(tri)     +
        _WEIGHTS["return_prob_t2"] * _minmax(ret_p2)  +
        _WEIGHTS["min_nbr_degree"] * _minmax(min_nbr) +
        _WEIGHTS["orbit_3"]        * _minmax(orbit3)  +
        _WEIGHTS["core_number"]    * _minmax(core)    +
        _WEIGHTS["avg_nbr_degree"] * _minmax(avg_nbr)
    ) / _TOTAL_WEIGHT

    return {v: float(score[i]) for i, v in enumerate(nodes)}


class StructuralPruner:
    """
    Fast MVC pruner using local structural graph features only.

    No training, no feature extraction pipeline.  Computes structural
    scores in O(m·sqrt(m)) and delegates pruning logic to ConfidencePruner.

    Parameters
    ----------
    threshold : float
        Nodes with score < threshold are pruning candidates (default 0.10).
        Same semantics as ConfidencePruner.threshold.
    """

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        self._pruner = ConfidencePruner(threshold=threshold)

    def prune(self, G: nx.Graph) -> PruningResult:
        """Score every node structurally, then prune."""
        node_probs = structural_scores(G)
        return self._pruner.prune(G, node_probs)

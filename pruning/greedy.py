"""
pruning/greedy.py
-----------------
Degree-based greedy MVC heuristic pruner.

Algorithm
---------
Repeat until all edges are covered:
  1. Select the vertex v with the highest number of *currently uncovered* edges.
  2. Add v to the cover.
  3. Mark all edges incident to v as covered.

Scoring
-------
Being selected in round r out of R total rounds yields a confidence score:

    score(v) = (R - r + 1) / R      ∈ (0, 1]

so the first vertex chosen (highest degree, most certain to be in any cover)
gets score 1.0, and the last gets 1/R.  Vertices never selected get score 0.0
(greedy says they are not needed in the cover).

This maps directly to ConfidencePruner semantics:
  score > fix_threshold  →  lock into cover (high confidence IN)
  score < prune_threshold →  remove from ILP (high confidence OUT)

Time complexity: O((n + m) log n) with a lazy max-heap.
No model, no feature extraction pipeline required.
"""
from __future__ import annotations

import heapq
from typing import Dict

import networkx as nx

from pruning.heuristic import ConfidencePruner, PruningResult


def greedy_degree_scores(G: nx.Graph) -> Dict:
    """
    Run the degree-greedy MVC heuristic and return a confidence score per node.

    Returns
    -------
    dict  {node: float in [0, 1]}
        Nodes selected early (high degree) receive scores close to 1.
        Nodes not selected receive score 0.0.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Active adjacency sets — shrink as edges become covered
    active_adj: Dict = {v: set(G.neighbors(v)) for v in nodes}

    # Lazy max-heap: entries are (-effective_degree, vertex)
    heap = [(-len(active_adj[v]), v) for v in nodes]
    heapq.heapify(heap)

    in_cover: set = set()
    round_selected: Dict = {}
    round_num = 0

    # Count uncovered edges (each undirected edge counted once)
    uncovered_count = G.number_of_edges()

    while uncovered_count > 0:
        # Pop until we find an active vertex (lazy deletion)
        while heap:
            neg_deg, v = heapq.heappop(heap)
            if v not in in_cover:
                # Verify degree is still current; re-push if stale
                actual_deg = len(active_adj[v])
                if actual_deg != -neg_deg:
                    heapq.heappush(heap, (-actual_deg, v))
                    continue
                break
        else:
            break  # No more candidates (shouldn't happen if uncovered_count > 0)

        if len(active_adj[v]) == 0:
            # Remaining vertices are isolated — no more edges to cover
            break

        round_num += 1
        in_cover.add(v)
        round_selected[v] = round_num

        # Cover all edges incident to v; update neighbours
        for u in list(active_adj[v]):
            active_adj[u].discard(v)
            uncovered_count -= 1
            # Push updated degree for u onto heap (lazy)
            heapq.heappush(heap, (-len(active_adj[u]), u))
        active_adj[v] = set()

    # Assign confidence scores
    n_rounds = round_num if round_num > 0 else 1
    scores: Dict = {}
    for v in nodes:
        if v in round_selected:
            r = round_selected[v]
            scores[v] = (n_rounds - r + 1) / n_rounds
        else:
            scores[v] = 0.0

    return scores


class GreedyDegreePruner:
    """
    Fast MVC pruner using the degree-greedy cover heuristic.

    No training, no feature pipeline.  Runs in O((n + m) log n).

    Parameters
    ----------
    threshold : float
        Nodes with score < threshold are pruning candidates (default 0.10).
        Nodes not selected by greedy get score 0.0, so they are always
        candidates when threshold > 0.
    fix_threshold : float
        Nodes with score >= fix_threshold are locked into the cover and
        excluded from the ILP (default 1.1 = disabled).  Setting this to
        e.g. 0.85 locks vertices selected in the first ~15% of greedy rounds.
    """

    def __init__(self, threshold: float = 0.10, fix_threshold: float = 1.1):
        self.threshold = threshold
        self.fix_threshold = fix_threshold
        self._pruner = ConfidencePruner(
            threshold=threshold,
            fix_threshold=fix_threshold,
        )

    def prune(self, G: nx.Graph) -> PruningResult:
        """Score every node via greedy degree heuristic, then prune."""
        node_scores = greedy_degree_scores(G)
        return self._pruner.prune(G, node_scores)

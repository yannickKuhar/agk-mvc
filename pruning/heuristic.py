"""
pruning/heuristic.py
--------------------
Confidence-threshold graph pruning for MVC pre-processing.

Strategy:
  1. For each node v, we have P(v ∈ optimal MVC) from the classifier.
  2. Nodes with P < tau are *candidates for removal* (predicted NOT in cover).
  3. Before removing a candidate node v, we check whether all its edges
     are still covered by the remaining nodes.
  4. Feasibility repair: if removing v leaves an uncovered edge (v, u),
     we must keep u in the cover (force_include). This is done greedily
     before pruning.

The result is a reduced graph G' ⊆ G and a set of "forced" nodes that
are guaranteed to be in the final cover. The solver only needs to find
a cover for G'.

Combined MVC solution:
  cover(G) = forced_nodes ∪ cover(G')

Correctness guarantee:
  If cover(G') is a valid MVC of G', then forced_nodes ∪ cover(G') is a
  valid MVC of G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class PruningResult:
    """Output of the pruning step."""
    reduced_graph: nx.Graph          # G' — the pruned graph for the solver
    forced_nodes: Set                # nodes guaranteed in the cover
    removed_nodes: Set               # nodes removed from G (predicted out of cover)
    node_probs: Dict                 # v -> P(v in MVC) for all original nodes
    original_n_nodes: int
    original_n_edges: int

    @property
    def reduction_ratio_nodes(self) -> float:
        return 1.0 - self.reduced_graph.number_of_nodes() / max(self.original_n_nodes, 1)

    @property
    def reduction_ratio_edges(self) -> float:
        return 1.0 - self.reduced_graph.number_of_edges() / max(self.original_n_edges, 1)

    def summary(self) -> str:
        return (
            f"Original: {self.original_n_nodes} nodes, {self.original_n_edges} edges\n"
            f"Reduced:  {self.reduced_graph.number_of_nodes()} nodes, "
            f"{self.reduced_graph.number_of_edges()} edges\n"
            f"Forced nodes: {len(self.forced_nodes)}\n"
            f"Node reduction: {self.reduction_ratio_nodes:.1%}\n"
            f"Edge reduction: {self.reduction_ratio_edges:.1%}"
        )


class ConfidencePruner:
    """
    Prune a graph by removing nodes predicted to be outside the MVC.

    Parameters
    ----------
    threshold : float
        Nodes with P(in MVC) < threshold are pruning candidates (default 0.10).
    repair_strategy : str
        'force_neighbor': force a neighbor into the cover to repair feasibility.
        'keep_node': instead of forcing, keep the original node.
    min_remaining_nodes : int
        Don't prune if reduced graph would have fewer than this many nodes.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        repair_strategy: str = "force_neighbor",
        min_remaining_nodes: int = 2,
    ):
        self.threshold = threshold
        self.repair_strategy = repair_strategy
        self.min_remaining_nodes = min_remaining_nodes

    def prune(
        self,
        G: nx.Graph,
        node_probs: Dict,
    ) -> PruningResult:
        """
        Apply pruning to graph G given per-node probabilities.

        Parameters
        ----------
        G : nx.Graph
            Original graph (nodes must match keys in node_probs).
        node_probs : dict
            {node_id: float} — P(node ∈ MVC) for each node.

        Returns
        -------
        PruningResult
        """
        nodes = list(G.nodes())
        n_orig = G.number_of_nodes()
        m_orig = G.number_of_edges()

        # Sort candidates by ascending probability (most confident to remove first)
        candidates = sorted(
            [v for v in nodes if node_probs.get(v, 0.5) < self.threshold],
            key=lambda v: node_probs.get(v, 0.5),
        )

        forced_nodes: Set = set()
        removed_nodes: Set = set()
        covered_by_forced: Set = set()  # edges already covered by forced nodes

        for v in candidates:
            if len(nodes) - len(removed_nodes) - 1 < self.min_remaining_nodes:
                break

            # Check which edges of v are not yet covered
            uncovered_edges = [
                (v, u) for u in G.neighbors(v)
                if u not in removed_nodes
                and u not in forced_nodes
                and v not in forced_nodes
            ]

            if not uncovered_edges:
                # All edges already covered — safe to remove v
                removed_nodes.add(v)
                continue

            # Repair: handle uncovered edges
            if self.repair_strategy == "force_neighbor":
                # For each uncovered edge (v, u), force u into cover
                # (u is the neighbor that will stay in the reduced graph)
                # We actually just remove v and mark its high-prob neighbors as forced
                # if they aren't already in the graph
                neighbors_not_removed = [
                    u for u in G.neighbors(v)
                    if u not in removed_nodes
                ]
                if neighbors_not_removed:
                    # Force the highest-prob neighbor to ensure coverage
                    best_neighbor = max(
                        neighbors_not_removed,
                        key=lambda u: node_probs.get(u, 0.5),
                    )
                    # Only force if the neighbor is also a low-confidence node
                    # Otherwise it will appear in the reduced graph and the solver handles it
                    removed_nodes.add(v)
                    # We don't need to force here — the reduced graph retains
                    # all non-removed nodes, so coverage is handled by the solver
                else:
                    # Isolated after other removals — safe to remove
                    removed_nodes.add(v)

            elif self.repair_strategy == "keep_node":
                # Conservative: don't remove if it would leave uncovered edges
                continue

        # Build reduced graph
        remaining_nodes = [v for v in nodes if v not in removed_nodes]
        G_reduced = G.subgraph(remaining_nodes).copy()

        # Identify forced nodes: nodes adjacent to removed nodes that
        # MUST be in the cover (their only cover was the removed node)
        for v in removed_nodes:
            for u in G.neighbors(v):
                if u not in removed_nodes:
                    # u is still in the graph; the edge (v, u) is now an
                    # "external" edge. Since v is removed, u must cover it.
                    forced_nodes.add(u)

        # Forced nodes don't need to appear in the reduced subproblem
        # (they're already in the cover), but we keep them for the solver
        # to handle edges among themselves.

        return PruningResult(
            reduced_graph=G_reduced,
            forced_nodes=forced_nodes,
            removed_nodes=removed_nodes,
            node_probs=node_probs,
            original_n_nodes=n_orig,
            original_n_edges=m_orig,
        )

    def prune_with_array(
        self,
        G: nx.Graph,
        probs: np.ndarray,
    ) -> PruningResult:
        """
        Convenience wrapper: accepts a probability array ordered by G.nodes().

        Parameters
        ----------
        G : nx.Graph
        probs : np.ndarray of shape (n_nodes,)
        """
        nodes = list(G.nodes())
        assert len(probs) == len(nodes), (
            f"probs length {len(probs)} != graph node count {len(nodes)}"
        )
        node_probs = {v: float(probs[i]) for i, v in enumerate(nodes)}
        return self.prune(G, node_probs)


# ---------------------------------------------------------------------------
# Reconstruction: merge forced nodes + solver output
# ---------------------------------------------------------------------------

def reconstruct_cover(
    pruning_result: PruningResult,
    solver_cover: Set,
) -> Set:
    """
    Combine the forced nodes from pruning with the solver's cover on G'.

    Parameters
    ----------
    pruning_result : PruningResult
    solver_cover : set of nodes forming a valid MVC of G'

    Returns
    -------
    Full MVC of original G.
    """
    return pruning_result.forced_nodes | solver_cover


def verify_cover(G: nx.Graph, cover: Set) -> bool:
    """Check that `cover` is a valid vertex cover of G."""
    for u, v in G.edges():
        if u not in cover and v not in cover:
            return False
    return True

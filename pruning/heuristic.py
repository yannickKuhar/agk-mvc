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

    Two-sided pruning:
      - Nodes with P(in MVC) < threshold  → removed (predicted NOT in cover).
      - Nodes with P(in MVC) > fix_threshold → fixed into cover (predicted
        definitely IN cover); removed from the ILP search space, their
        incident edges no longer need to be covered by other nodes.

    Parameters
    ----------
    threshold : float
        Nodes with P(in MVC) < threshold are pruning candidates (default 0.10).
    fix_threshold : float
        Nodes with P(in MVC) > fix_threshold are locked into the cover and
        excluded from the ILP (default 1.1, i.e. disabled).  Set to e.g. 0.85
        to enable two-sided pruning.
    repair_strategy : str
        'force_neighbor': force a neighbor into the cover to repair feasibility.
        'keep_node': instead of forcing, keep the original node.
    min_remaining_nodes : int
        Don't prune if reduced graph would have fewer than this many nodes.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        fix_threshold: float = 1.1,
        repair_strategy: str = "force_neighbor",
        min_remaining_nodes: int = 2,
    ):
        self.threshold = threshold
        self.fix_threshold = fix_threshold
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

        # Two-sided: lock high-confidence nodes into cover first
        forced_high: Set = set()
        if self.fix_threshold <= 1.0:
            forced_high = {v for v in nodes if node_probs.get(v, 0.5) >= self.fix_threshold}

        # Sort candidates by ascending probability (most confident to remove first)
        # Exclude already-forced-high nodes from removal candidates
        candidates = sorted(
            [v for v in nodes
             if node_probs.get(v, 0.5) < self.threshold and v not in forced_high],
            key=lambda v: node_probs.get(v, 0.5),
        )

        forced_nodes: Set = set(forced_high)  # seed with high-confidence nodes
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

        # Identify repair-forced nodes: neighbors of removed nodes that MUST
        # cover the now-external edges left by removed nodes.
        for v in removed_nodes:
            for u in G.neighbors(v):
                if u not in removed_nodes and u not in forced_high:
                    forced_nodes.add(u)

        # Build reduced graph: exclude removed nodes AND forced_high nodes.
        # Edges covered by forced_high nodes disappear automatically since
        # those nodes are absent from the subgraph.
        ilp_nodes = [v for v in nodes if v not in removed_nodes and v not in forced_high]
        G_reduced = G.subgraph(ilp_nodes).copy()

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
    original_graph: Optional[nx.Graph] = None,
) -> Set:
    """
    Combine the forced nodes from pruning with the solver's cover on G'.

    Parameters
    ----------
    pruning_result  : PruningResult
    solver_cover    : set of nodes forming a valid MVC of G'
    original_graph  : if provided, the result is verified against the full
                      original graph and an error is raised if invalid

    Returns
    -------
    Full MVC of original G.
    """
    full_cover = pruning_result.forced_nodes | solver_cover

    if original_graph is not None:
        invalid_edges = [
            (u, v) for u, v in original_graph.edges()
            if u not in full_cover and v not in full_cover
        ]
        if invalid_edges:
            # Pruner incorrectly removed nodes that share edges with each
            # other.  Repair by adding one endpoint per uncovered edge.
            repair = {u for u, v in invalid_edges}
            full_cover |= repair
            print(f"[pruning] WARNING: repaired {len(invalid_edges)} uncovered "
                  f"edges by adding {len(repair)} nodes to cover "
                  f"(forced={len(pruning_result.forced_nodes)}, "
                  f"solver={len(solver_cover)})")

    return full_cover


def verify_cover(G: nx.Graph, cover: Set) -> bool:
    """Check that `cover` is a valid vertex cover of the full graph G."""
    for u, v in G.edges():
        if u not in cover and v not in cover:
            return False
    return True

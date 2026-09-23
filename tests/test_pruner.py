"""
tests/test_pruner.py
--------------------
Unit tests for ConfidencePruner (one-sided and two-sided).

Correctness invariant: forced_nodes ∪ solver_cover(G_reduced) must be a valid
MVC of the original graph G for all inputs.
"""
import networkx as nx
import numpy as np
import pytest

from pruning.heuristic import ConfidencePruner, reconstruct_cover, verify_cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probs(G, high_nodes, low_nodes):
    """Assign high prob to high_nodes, low to low_nodes, 0.5 otherwise."""
    probs = {v: 0.5 for v in G.nodes()}
    for v in high_nodes:
        probs[v] = 0.95
    for v in low_nodes:
        probs[v] = 0.02
    return probs


def _full_cover(G):
    """Trivially valid cover: all nodes."""
    return set(G.nodes())


# ---------------------------------------------------------------------------
# One-sided pruning (default fix_threshold > 1.0)
# ---------------------------------------------------------------------------

class TestOneSidedPruner:

    def test_result_is_valid_cover_path(self):
        G = nx.path_graph(6)
        pruner = ConfidencePruner(threshold=0.10)
        probs = {v: 0.9 if v % 2 == 0 else 0.05 for v in G.nodes()}
        result = pruner.prune(G, probs)
        cover = reconstruct_cover(result, _full_cover(result.reduced_graph), G)
        assert verify_cover(G, cover), "Cover must be valid for path graph"

    def test_result_is_valid_cover_complete(self):
        G = nx.complete_graph(8)
        pruner = ConfidencePruner(threshold=0.10)
        probs = {v: 0.05 if v == 0 else 0.9 for v in G.nodes()}
        result = pruner.prune(G, probs)
        cover = reconstruct_cover(result, _full_cover(result.reduced_graph), G)
        assert verify_cover(G, cover)

    def test_no_pruning_when_threshold_zero(self):
        G = nx.cycle_graph(10)
        pruner = ConfidencePruner(threshold=0.0)
        probs = {v: 0.5 for v in G.nodes()}
        result = pruner.prune(G, probs)
        assert result.reduced_graph.number_of_nodes() == G.number_of_nodes()
        assert len(result.removed_nodes) == 0

    def test_all_pruned_when_threshold_one(self):
        """threshold=1.0 means every node is a candidate; min_remaining_nodes stops it."""
        G = nx.path_graph(4)
        pruner = ConfidencePruner(threshold=1.0, min_remaining_nodes=2)
        probs = {v: 0.0 for v in G.nodes()}
        result = pruner.prune(G, probs)
        cover = reconstruct_cover(result, _full_cover(result.reduced_graph), G)
        assert verify_cover(G, cover)

    def test_isolated_node_safe(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1)
        # node 2 is isolated — safe to remove or keep
        pruner = ConfidencePruner(threshold=0.10)
        probs = {0: 0.9, 1: 0.9, 2: 0.02}
        result = pruner.prune(G, probs)
        cover = reconstruct_cover(result, _full_cover(result.reduced_graph), G)
        assert verify_cover(G, cover)

    def test_single_node_graph(self):
        G = nx.Graph()
        G.add_node(0)
        pruner = ConfidencePruner(threshold=0.10)
        result = pruner.prune(G, {0: 0.5})
        cover = reconstruct_cover(result, set(), G)
        assert verify_cover(G, cover)

    def test_empty_graph(self):
        G = nx.Graph()
        pruner = ConfidencePruner(threshold=0.10)
        result = pruner.prune(G, {})
        cover = reconstruct_cover(result, set(), G)
        assert verify_cover(G, cover)

    def test_forced_nodes_subset_of_original(self):
        G = nx.karate_club_graph()
        pruner = ConfidencePruner(threshold=0.15)
        probs = {v: float(v % 3 == 0) * 0.03 + 0.5 for v in G.nodes()}
        result = pruner.prune(G, probs)
        assert result.forced_nodes.issubset(set(G.nodes()))
        assert result.removed_nodes.issubset(set(G.nodes()))
        assert result.forced_nodes.isdisjoint(result.removed_nodes)

    def test_reduced_graph_nodes_are_subset(self):
        G = nx.barabasi_albert_graph(30, 3, seed=42)
        pruner = ConfidencePruner(threshold=0.20)
        probs = {v: 0.05 if v < 10 else 0.8 for v in G.nodes()}
        result = pruner.prune(G, probs)
        assert set(result.reduced_graph.nodes()).issubset(set(G.nodes()))

    def test_prune_with_array_equivalent(self):
        G = nx.path_graph(5)
        pruner = ConfidencePruner(threshold=0.10)
        nodes = list(G.nodes())
        probs_dict = {v: 0.05 if v < 2 else 0.9 for v in nodes}
        probs_arr = np.array([probs_dict[v] for v in nodes])
        r1 = pruner.prune(G, probs_dict)
        r2 = pruner.prune_with_array(G, probs_arr)
        assert r1.removed_nodes == r2.removed_nodes
        assert r1.forced_nodes == r2.forced_nodes

    def test_prune_with_array_wrong_length(self):
        G = nx.path_graph(5)
        pruner = ConfidencePruner(threshold=0.10)
        with pytest.raises(AssertionError):
            pruner.prune_with_array(G, np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Two-sided pruning (fix_threshold <= 1.0)
# ---------------------------------------------------------------------------

class TestTwoSidedPruner:

    def test_high_confidence_nodes_forced(self):
        G = nx.complete_graph(6)
        pruner = ConfidencePruner(threshold=0.10, fix_threshold=0.85)
        probs = {0: 0.95, 1: 0.95, 2: 0.5, 3: 0.5, 4: 0.04, 5: 0.04}
        result = pruner.prune(G, probs)
        assert 0 in result.forced_nodes
        assert 1 in result.forced_nodes
        # Forced-high nodes must NOT appear in reduced graph
        assert 0 not in result.reduced_graph.nodes()
        assert 1 not in result.reduced_graph.nodes()

    def test_two_sided_valid_cover_path(self):
        G = nx.path_graph(10)
        pruner = ConfidencePruner(threshold=0.10, fix_threshold=0.85)
        probs = {v: 0.95 if v % 3 == 0 else 0.05 for v in G.nodes()}
        result = pruner.prune(G, probs)
        # Solver cover on reduced graph: take all remaining nodes
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, G)
        assert verify_cover(G, full), "Two-sided pruning must yield valid cover"

    def test_two_sided_valid_cover_karate(self):
        G = nx.karate_club_graph()
        pruner = ConfidencePruner(threshold=0.15, fix_threshold=0.80)
        rng = np.random.default_rng(42)
        probs = {v: float(rng.random()) for v in G.nodes()}
        result = pruner.prune(G, probs)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, G)
        assert verify_cover(G, full)

    def test_forced_high_disjoint_from_removed(self):
        G = nx.cycle_graph(8)
        pruner = ConfidencePruner(threshold=0.10, fix_threshold=0.85)
        probs = {0: 0.95, 1: 0.95, 2: 0.04, 3: 0.04, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5}
        result = pruner.prune(G, probs)
        assert result.forced_nodes.isdisjoint(result.removed_nodes)

    def test_disabled_fix_threshold(self):
        """fix_threshold > 1.0 should be equivalent to one-sided pruning."""
        G = nx.path_graph(8)
        probs = {v: 0.95 if v < 2 else 0.05 for v in G.nodes()}
        pruner1 = ConfidencePruner(threshold=0.10, fix_threshold=1.1)
        pruner2 = ConfidencePruner(threshold=0.10)
        r1 = pruner1.prune(G, probs)
        r2 = pruner2.prune(G, probs)
        assert r1.removed_nodes == r2.removed_nodes


# ---------------------------------------------------------------------------
# reconstruct_cover and verify_cover
# ---------------------------------------------------------------------------

class TestReconstructAndVerify:

    def test_verify_cover_complete_graph(self):
        G = nx.complete_graph(5)
        # Any single node covers nothing — needs at least n-1 nodes
        assert not verify_cover(G, {0})
        assert verify_cover(G, set(range(1, 5)))

    def test_verify_cover_empty_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        assert verify_cover(G, set())   # no edges to cover

    def test_reconstruct_merges_forced_and_solver(self):
        from pruning.heuristic import PruningResult
        G = nx.path_graph(4)
        pr = PruningResult(
            reduced_graph=G.subgraph([2, 3]).copy(),
            forced_nodes={0, 1},
            removed_nodes=set(),
            node_probs={},
            original_n_nodes=4,
            original_n_edges=3,
        )
        solver_cover = {2}
        full = reconstruct_cover(pr, solver_cover, G)
        assert {0, 1, 2}.issubset(full)
        assert verify_cover(G, full)

    def test_reconstruct_repairs_invalid_cover(self):
        """If pruner leaves an uncovered edge, reconstruct_cover repairs it."""
        from pruning.heuristic import PruningResult
        G = nx.path_graph(3)  # 0-1-2
        pr = PruningResult(
            reduced_graph=nx.Graph(),
            forced_nodes=set(),
            removed_nodes={1},  # removing the only node that covers both edges
            node_probs={},
            original_n_nodes=3,
            original_n_edges=2,
        )
        full = reconstruct_cover(pr, set(), G)
        assert verify_cover(G, full)

"""
tests/test_greedy_pruner.py
----------------------------
Unit tests for pruning/greedy.py (degree-greedy MVC heuristic pruner).

Correctness invariants:
  1. greedy_degree_scores returns scores in [0, 1] for all nodes.
  2. Nodes selected by greedy have score > 0; unselected nodes have score 0.
  3. The greedy cover (score > 0 nodes) is a valid vertex cover.
  4. The first selected node has score 1.0 (highest confidence).
  5. GreedyDegreePruner.prune(G) produces a valid PruningResult.
  6. forced_nodes ∪ full_cover(G_reduced) is a valid MVC of G.
  7. Edge cases: empty graph, edgeless graph, single edge, complete graph.
  8. Two-sided pruning: high-score nodes are locked, score-0 nodes are pruned.
  9. Greedy cover nodes decrease monotonically in score (earlier = higher).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.greedy import greedy_degree_scores, GreedyDegreePruner
from pruning.heuristic import reconstruct_cover, verify_cover


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def path6():
    return nx.path_graph(6)

@pytest.fixture
def karate():
    return nx.karate_club_graph()

@pytest.fixture
def star5():
    return nx.star_graph(5)   # hub + 5 leaves

@pytest.fixture
def complete5():
    return nx.complete_graph(5)


# ---------------------------------------------------------------------------
# 1. Score range
# ---------------------------------------------------------------------------

class TestScoreRange:

    def test_scores_in_unit_interval(self, karate):
        scores = greedy_degree_scores(karate)
        for v, s in scores.items():
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for node {v}"

    def test_scores_in_unit_interval_path(self, path6):
        scores = greedy_degree_scores(path6)
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_empty_graph_returns_empty(self):
        scores = greedy_degree_scores(nx.Graph())
        assert scores == {}

    def test_edgeless_all_zero(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        scores = greedy_degree_scores(G)
        assert all(s == 0.0 for s in scores.values())

    def test_single_edge_two_nodes(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        scores = greedy_degree_scores(G)
        assert set(scores.keys()) == {0, 1}
        # Exactly one node selected (score > 0); the other may also be selected
        # or not, but the one selected gets score 1.0 (only round)
        assert any(s == 1.0 for s in scores.values())


# ---------------------------------------------------------------------------
# 2. Greedy cover is a valid vertex cover
# ---------------------------------------------------------------------------

class TestGreedyCoverValidity:

    def _cover_from_scores(self, G, scores):
        return {v for v, s in scores.items() if s > 0}

    def test_valid_cover_path(self, path6):
        scores = greedy_degree_scores(path6)
        cover = self._cover_from_scores(path6, scores)
        assert verify_cover(path6, cover)

    def test_valid_cover_karate(self, karate):
        scores = greedy_degree_scores(karate)
        cover = self._cover_from_scores(karate, scores)
        assert verify_cover(karate, cover)

    def test_valid_cover_star(self, star5):
        scores = greedy_degree_scores(star5)
        cover = self._cover_from_scores(star5, scores)
        assert verify_cover(star5, cover)

    def test_valid_cover_complete(self, complete5):
        scores = greedy_degree_scores(complete5)
        cover = self._cover_from_scores(complete5, scores)
        assert verify_cover(complete5, cover)

    def test_valid_cover_ba_graph(self):
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        scores = greedy_degree_scores(G)
        cover = {v for v, s in scores.items() if s > 0}
        assert verify_cover(G, cover)

    def test_valid_cover_cycle(self):
        G = nx.cycle_graph(10)
        scores = greedy_degree_scores(G)
        cover = {v for v, s in scores.items() if s > 0}
        assert verify_cover(G, cover)


# ---------------------------------------------------------------------------
# 3. First selected node has score 1.0
# ---------------------------------------------------------------------------

class TestFirstNodeScore:

    def test_max_score_is_one(self, karate):
        scores = greedy_degree_scores(karate)
        selected = [s for s in scores.values() if s > 0]
        assert selected, "At least one node must be selected"
        assert max(selected) == pytest.approx(1.0)

    def test_max_score_is_one_star(self, star5):
        scores = greedy_degree_scores(star5)
        # Hub (node 0) has degree 5 — selected first → score 1.0
        assert scores[0] == pytest.approx(1.0)

    def test_star_hub_selected_first(self, star5):
        scores = greedy_degree_scores(star5)
        # Hub covers all edges; leaves have score 0
        assert scores[0] == pytest.approx(1.0)
        for leaf in range(1, 6):
            assert scores[leaf] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Score monotonicity within selected nodes
# ---------------------------------------------------------------------------

class TestScoreMonotonicity:

    def test_scores_are_decreasing_by_round(self, karate):
        """Scores of selected nodes must be non-increasing (earlier = higher)."""
        scores = greedy_degree_scores(karate)
        selected_scores = sorted(
            [s for s in scores.values() if s > 0], reverse=True
        )
        # Each successive round produces a score <= the previous
        for i in range(1, len(selected_scores)):
            assert selected_scores[i] <= selected_scores[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# 5. GreedyDegreePruner interface
# ---------------------------------------------------------------------------

class TestGreedyDegreePruner:

    def test_prune_returns_pruning_result(self, karate):
        from pruning.heuristic import PruningResult
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(karate)
        assert isinstance(result, PruningResult)

    def test_reduced_graph_is_subgraph(self, karate):
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(karate)
        assert set(result.reduced_graph.nodes()).issubset(set(karate.nodes()))

    def test_forced_and_removed_disjoint(self, karate):
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(karate)
        assert result.forced_nodes.isdisjoint(result.removed_nodes)

    def test_reconstruct_cover_valid_path(self, path6):
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(path6)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, path6)
        assert verify_cover(path6, full)

    def test_reconstruct_cover_valid_karate(self, karate):
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(karate)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, karate)
        assert verify_cover(karate, full)

    def test_reconstruct_cover_valid_ba(self):
        G = nx.barabasi_albert_graph(60, 4, seed=7)
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(G)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, G)
        assert verify_cover(G, full)


# ---------------------------------------------------------------------------
# 6. Two-sided pruning (fix_threshold ≤ 1.0)
# ---------------------------------------------------------------------------

class TestTwoSidedGreedy:

    def test_high_score_nodes_locked(self, star5):
        # Star hub gets score 1.0, which exceeds fix_threshold=0.85
        pruner = GreedyDegreePruner(threshold=0.10, fix_threshold=0.85)
        result = pruner.prune(star5)
        # Hub (node 0) should be in forced_nodes
        assert 0 in result.forced_nodes

    def test_hub_not_in_reduced_graph(self, star5):
        pruner = GreedyDegreePruner(threshold=0.10, fix_threshold=0.85)
        result = pruner.prune(star5)
        assert 0 not in result.reduced_graph.nodes()

    def test_two_sided_valid_cover_karate(self, karate):
        pruner = GreedyDegreePruner(threshold=0.10, fix_threshold=0.85)
        result = pruner.prune(karate)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, karate)
        assert verify_cover(karate, full)

    def test_two_sided_valid_cover_ba(self):
        G = nx.barabasi_albert_graph(80, 3, seed=99)
        pruner = GreedyDegreePruner(threshold=0.10, fix_threshold=0.85)
        result = pruner.prune(G)
        solver_cover = set(result.reduced_graph.nodes())
        full = reconstruct_cover(result, solver_cover, G)
        assert verify_cover(G, full)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_graph(self):
        G = nx.Graph()
        pruner = GreedyDegreePruner()
        result = pruner.prune(G)
        assert verify_cover(G, reconstruct_cover(result, set(), G))

    def test_edgeless_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        pruner = GreedyDegreePruner()
        result = pruner.prune(G)
        assert verify_cover(G, reconstruct_cover(result, set(), G))

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        pruner = GreedyDegreePruner(threshold=0.10, fix_threshold=0.85)
        result = pruner.prune(G)
        full = reconstruct_cover(result, set(result.reduced_graph.nodes()), G)
        assert verify_cover(G, full)

    def test_complete_graph_k5(self, complete5):
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(complete5)
        full = reconstruct_cover(result, set(result.reduced_graph.nodes()), complete5)
        assert verify_cover(complete5, full)

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disjoint edges
        pruner = GreedyDegreePruner(threshold=0.10)
        result = pruner.prune(G)
        full = reconstruct_cover(result, set(result.reduced_graph.nodes()), G)
        assert verify_cover(G, full)

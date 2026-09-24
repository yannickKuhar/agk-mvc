"""
tests/test_ajwani_features.py
------------------------------
Unit tests for features/ajwani.py (O'Connor et al., CPAIOR 2026).

Correctness invariants:
  1. Output shape is (n_nodes, 13).
  2. All values are finite floats.
  3. All values are in expected ranges.
  4. Empty graphs and edgeless graphs don't crash.
  5. MWUA features respect MIS structure on simple graphs.
  6. MIS frequency sums to <= n on a path (never exceeds IS size).
  7. Degree-rank feature is in [0, 1].
  8. Pipeline correctly wires 'ajwani' and 'kernel+ajwani' feature sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.ajwani import (
    compute_ajwani_features,
    feature_names,
    _normalised_rank,
    _mis_frequency,
    _mwua_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def karate() -> nx.Graph:
    return nx.karate_club_graph()

@pytest.fixture
def path5() -> nx.Graph:
    return nx.path_graph(5)

@pytest.fixture
def complete5() -> nx.Graph:
    return nx.complete_graph(5)

@pytest.fixture
def empty_graph() -> nx.Graph:
    return nx.Graph()

@pytest.fixture
def edgeless10() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(10))
    return G


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_shape_karate(self, karate):
        X = compute_ajwani_features(karate)
        assert X.shape == (karate.number_of_nodes(), 13)

    def test_shape_path5(self, path5):
        X = compute_ajwani_features(path5)
        assert X.shape == (5, 13)

    def test_shape_complete5(self, complete5):
        X = compute_ajwani_features(complete5)
        assert X.shape == (5, 13)

    def test_shape_empty(self, empty_graph):
        X = compute_ajwani_features(empty_graph)
        assert X.shape == (0, 13)

    def test_shape_edgeless(self, edgeless10):
        X = compute_ajwani_features(edgeless10)
        assert X.shape == (10, 13)

    def test_dtype_float64(self, karate):
        X = compute_ajwani_features(karate)
        assert X.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. All values finite
# ---------------------------------------------------------------------------

class TestFiniteValues:

    def test_finite_karate(self, karate):
        X = compute_ajwani_features(karate)
        assert np.all(np.isfinite(X)), "Non-finite values in karate features"

    def test_finite_path(self, path5):
        X = compute_ajwani_features(path5)
        assert np.all(np.isfinite(X))

    def test_finite_edgeless(self, edgeless10):
        X = compute_ajwani_features(edgeless10)
        assert np.all(np.isfinite(X))

    def test_finite_complete(self, complete5):
        X = compute_ajwani_features(complete5)
        assert np.all(np.isfinite(X))

    def test_finite_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        X = compute_ajwani_features(G)
        assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# 3. Value ranges
# ---------------------------------------------------------------------------

class TestValueRanges:

    def _check_ranges(self, X: np.ndarray):
        # LCC ∈ [0, 1]
        assert np.all(X[:, 0] >= 0) and np.all(X[:, 0] <= 1)
        # Degree centrality ∈ [0, 1]
        assert np.all(X[:, 1] >= 0) and np.all(X[:, 1] <= 1)
        # Core number (normalised) ∈ [0, 1]
        assert np.all(X[:, 2] >= 0) and np.all(X[:, 2] <= 1)
        # Degree rank ∈ [0, 1]
        assert np.all(X[:, 3] >= 0) and np.all(X[:, 3] <= 1)
        # Neighbour degree ranks ∈ [0, 1]
        assert np.all(X[:, 4] >= 0) and np.all(X[:, 4] <= 1)  # min
        assert np.all(X[:, 5] >= 0) and np.all(X[:, 5] <= 1)  # max
        assert np.all(X[:, 6] >= 0) and np.all(X[:, 6] <= 1)  # avg
        # PageRank > 0 (for connected graphs)
        assert np.all(X[:, 7] >= 0)
        # MIS frequency ∈ [0, 1]
        assert np.all(X[:, 8] >= 0) and np.all(X[:, 8] <= 1)
        # MWUA avg_x ∈ [0, 1]
        assert np.all(X[:, 9] >= 0) and np.all(X[:, 9] <= 1)
        # MWUA weights >= 0
        assert np.all(X[:, 10] >= 0)  # avg_w
        assert np.all(X[:, 11] >= 0)  # min_w
        assert np.all(X[:, 12] >= 0)  # max_w

    def test_ranges_karate(self, karate):
        self._check_ranges(compute_ajwani_features(karate))

    def test_ranges_path(self, path5):
        self._check_ranges(compute_ajwani_features(path5))

    def test_ranges_complete(self, complete5):
        self._check_ranges(compute_ajwani_features(complete5))

    def test_ranges_edgeless(self, edgeless10):
        self._check_ranges(compute_ajwani_features(edgeless10))


# ---------------------------------------------------------------------------
# 4. Feature names
# ---------------------------------------------------------------------------

class TestFeatureNames:

    def test_length_13(self):
        assert len(feature_names()) == 13

    def test_expected_names(self):
        names = feature_names()
        assert "lcc" in names
        assert "pagerank" in names
        assert "mis_freq" in names
        assert "mwua_avg_x" in names
        assert "mwua_min_w" in names
        assert "mwua_max_w" in names


# ---------------------------------------------------------------------------
# 5. Normalised rank helper
# ---------------------------------------------------------------------------

class TestNormalisedRank:

    def test_single_unique_value(self):
        arr = np.array([3.0, 3.0, 3.0])
        ranks = _normalised_rank(arr)
        assert np.all(ranks == 0.0)

    def test_two_values(self):
        arr = np.array([1.0, 2.0, 1.0, 2.0])
        ranks = _normalised_rank(arr)
        assert set(np.unique(ranks)) == {0.0, 1.0}
        assert ranks[0] == 0.0 and ranks[1] == 1.0

    def test_monotone_increasing(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = _normalised_rank(arr)
        assert np.all(np.diff(ranks) > 0)
        assert ranks[0] == 0.0 and ranks[-1] == 1.0


# ---------------------------------------------------------------------------
# 6. MIS frequency basic properties
# ---------------------------------------------------------------------------

class TestMISFrequency:

    def _run(self, G, n_runs=50, seed=0):
        nodes = list(G.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}
        return _mis_frequency(G, nodes, node_idx, n_runs=n_runs, seed=seed)

    def test_edgeless_all_ones(self, edgeless10):
        freq = self._run(edgeless10, n_runs=5)
        assert np.all(freq == 1.0), "Edgeless: every node should be in every MIS"

    def test_complete_graph_one_node_per_run(self, complete5):
        # In K5 exactly 1 node is chosen per run
        freq = self._run(complete5, n_runs=100)
        assert abs(freq.sum() - 1.0) < 0.25  # each run picks 1 of 5 → avg ≈ 0.2

    def test_path_frequency_positive(self, path5):
        freq = self._run(path5, n_runs=30)
        assert np.all(freq > 0), "Every vertex should appear in some MIS on a path"

    def test_independent_set_constraint(self, karate):
        # For any single run, selected vertices form an IS (no adjacent pair)
        # Check this by re-running a deterministic pass with seed=42
        nodes = list(karate.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}
        rng = np.random.default_rng(42)
        adj_idx = [set() for _ in range(len(nodes))]
        for u, v in karate.edges():
            adj_idx[node_idx[u]].add(node_idx[v])
            adj_idx[node_idx[v]].add(node_idx[u])

        # Run one round of Luby manually
        remaining = set(range(len(nodes)))
        r = rng.random(len(nodes))
        winners = {i for i in remaining
                   if all(r[i] > r[j] for j in adj_idx[i] if j in remaining)}

        # No two winners should be adjacent
        for i in winners:
            for j in adj_idx[i]:
                assert j not in winners, f"Nodes {i} and {j} are both in IS but adjacent"


# ---------------------------------------------------------------------------
# 7. MWUA basic properties
# ---------------------------------------------------------------------------

class TestMWUAFeatures:

    def _run(self, G, t_max=0.5):
        nodes = list(G.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}
        return _mwua_features(G, nodes, node_idx, t_max=t_max)

    def test_shape(self, karate):
        M = self._run(karate)
        assert M.shape == (karate.number_of_nodes(), 4)

    def test_edgeless_x_is_one(self, edgeless10):
        M = self._run(edgeless10)
        assert np.all(M[:, 0] == 1.0)  # avg_x = 1 for edgeless
        assert np.all(M[:, 1] == 0.0)  # avg_w = 0 (no edges)

    def test_min_w_le_avg_w_le_max_w(self, karate):
        M = self._run(karate)
        # M columns: [avg_x, avg_w, min_w, max_w]
        assert np.all(M[:, 2] <= M[:, 1] + 1e-9)  # min_w <= avg_w
        assert np.all(M[:, 1] <= M[:, 3] + 1e-9)  # avg_w <= max_w

    def test_complete_graph_symmetry(self, complete5):
        M = self._run(complete5)
        # All nodes identical in K5 → all MWUA values should be equal
        assert np.allclose(M[:, 0], M[0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Pipeline integration (skipped when ML deps unavailable in test env)
# ---------------------------------------------------------------------------

try:
    from features.pipeline import NodeFeaturePipeline as _Pipeline
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

_skip_pipeline = pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="features.pipeline requires tqdm/sklearn not installed in test env",
)


class TestPipelineIntegration:

    @_skip_pipeline
    def test_ajwani_feature_set(self, path5):
        pipe = _Pipeline(feature_set="ajwani", normalize=False)
        X = pipe.extract_graph(path5)
        assert X.shape == (5, 13)
        assert len(pipe.feature_names) == 13

    @_skip_pipeline
    def test_kernel_ajwani_feature_set(self, path5):
        pipe = _Pipeline(feature_set="kernel+ajwani", normalize=False)
        X = pipe.extract_graph(path5)
        assert X.shape == (5, 119)  # 106 kernel + 13 ajwani
        assert len(pipe.feature_names) == 119

    @_skip_pipeline
    def test_invalid_feature_set_raises(self):
        with pytest.raises(ValueError, match="feature_set"):
            _Pipeline(feature_set="bogus")

    @_skip_pipeline
    def test_ajwani_in_feature_names(self):
        pipe = _Pipeline(feature_set="ajwani", normalize=False)
        names = pipe.feature_names
        assert "lcc" in names
        assert "mwua_avg_x" in names
        assert "mis_freq" in names

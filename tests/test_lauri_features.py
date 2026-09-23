"""
tests/test_lauri_features.py
----------------------------
Unit tests for features/lauri.py (Lauri et al. 2023, 9 handcrafted features).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from features.lauri import compute_lauri_features, feature_names

# ── helpers ──────────────────────────────────────────────────────────────────

N_FEATURES = 9
COL = {name: i for i, name in enumerate(feature_names())}


def _col(X: np.ndarray, name: str) -> np.ndarray:
    """Return a named column from the feature matrix."""
    return X[:, COL[name]]


# ── 1. feature_names() ───────────────────────────────────────────────────────

def test_feature_names_count():
    names = feature_names()
    assert len(names) == N_FEATURES


def test_feature_names_are_strings():
    for name in feature_names():
        assert isinstance(name, str) and name


# ── 2. Output shape ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,m", [
    (5, 0),   # empty (no edges)
    (5, 4),   # path
    (10, 15), # arbitrary
])
def test_output_shape_parametric(n, m):
    rng = np.random.default_rng(42)
    G = nx.gnm_random_graph(n, m, seed=42)
    X = compute_lauri_features(G)
    assert X.shape == (n, N_FEATURES)


def test_output_shape_karate():
    G = nx.karate_club_graph()
    X = compute_lauri_features(G)
    assert X.shape == (G.number_of_nodes(), N_FEATURES)


# ── 3. Karate club: all finite ────────────────────────────────────────────────

def test_karate_all_finite():
    G = nx.karate_club_graph()
    X = compute_lauri_features(G)
    assert np.all(np.isfinite(X)), "Karate club features contain NaN or inf"


# ── 4. Complete graph K5 ─────────────────────────────────────────────────────

def test_k5_degree():
    G = nx.complete_graph(5)
    X = compute_lauri_features(G)
    deg_col = _col(X, "degree")
    np.testing.assert_array_equal(deg_col, 4.0)


def test_k5_lcc():
    G = nx.complete_graph(5)
    X = compute_lauri_features(G)
    lcc_col = _col(X, "lcc")
    np.testing.assert_allclose(lcc_col, 1.0, atol=1e-9)


def test_k5_chi2_degree_zero():
    # All degrees equal → chi2_degree should be 0 for all nodes
    G = nx.complete_graph(5)
    X = compute_lauri_features(G)
    np.testing.assert_allclose(_col(X, "chi2_degree"), 0.0, atol=1e-9)


def test_k5_global_features():
    G = nx.complete_graph(5)
    X = compute_lauri_features(G)
    np.testing.assert_array_equal(_col(X, "n_nodes"), 5.0)
    np.testing.assert_array_equal(_col(X, "n_edges"), float(G.number_of_edges()))


# ── 5. Empty graph (no edges) ─────────────────────────────────────────────────

def test_empty_graph_degree_zero():
    G = nx.empty_graph(6)
    X = compute_lauri_features(G)
    np.testing.assert_array_equal(_col(X, "degree"), 0.0)


def test_empty_graph_lcc_zero():
    G = nx.empty_graph(6)
    X = compute_lauri_features(G)
    np.testing.assert_array_equal(_col(X, "lcc"), 0.0)


def test_empty_graph_eigenvec_fallback_finite():
    # eigenvector_centrality must fall back gracefully
    G = nx.empty_graph(6)
    X = compute_lauri_features(G)
    assert np.all(np.isfinite(X)), "Empty graph features contain NaN or inf"


def test_empty_graph_chi2_nbr_zero():
    # Isolated nodes have no neighbours → neighbour-averaged chi2 must be 0
    G = nx.empty_graph(6)
    X = compute_lauri_features(G)
    np.testing.assert_array_equal(_col(X, "chi2_degree_nbr"), 0.0)
    np.testing.assert_array_equal(_col(X, "chi2_lcc_nbr"), 0.0)


# ── 6. Single-node graph ──────────────────────────────────────────────────────

def test_single_node_no_crash():
    G = nx.Graph()
    G.add_node(0)
    X = compute_lauri_features(G)
    assert X.shape == (1, N_FEATURES)
    assert np.all(np.isfinite(X))


# ── 7. Path graph P4 ──────────────────────────────────────────────────────────

def test_path_p4_degrees():
    # P4: nodes 0-1-2-3; degrees are 1,2,2,1
    G = nx.path_graph(4)
    nodes = list(G.nodes())
    X = compute_lauri_features(G)
    deg_col = _col(X, "degree")
    expected = np.array([float(G.degree(v)) for v in nodes])
    np.testing.assert_array_equal(deg_col, expected)


def test_path_p4_shape():
    G = nx.path_graph(4)
    X = compute_lauri_features(G)
    assert X.shape == (4, N_FEATURES)


def test_path_p4_all_finite():
    G = nx.path_graph(4)
    X = compute_lauri_features(G)
    assert np.all(np.isfinite(X))


# ── 8. Star graph S5 (hub + 4 leaves) ────────────────────────────────────────

def _star_hub_leaf_indices():
    """Return (hub_idx, leaf_indices) for star_graph(4) in list(G.nodes()) order."""
    G = nx.star_graph(4)
    nodes = list(G.nodes())
    hub = 0  # star_graph centres on node 0
    hub_idx = nodes.index(hub)
    leaf_indices = [i for i, v in enumerate(nodes) if v != hub]
    return G, hub_idx, leaf_indices


def test_star_hub_degree():
    G, hub_idx, _ = _star_hub_leaf_indices()
    X = compute_lauri_features(G)
    assert _col(X, "degree")[hub_idx] == 4.0


def test_star_leaf_degree():
    G, _, leaf_indices = _star_hub_leaf_indices()
    X = compute_lauri_features(G)
    for li in leaf_indices:
        assert _col(X, "degree")[li] == 1.0


def test_star_chi2_degree_hub_vs_leaf():
    # Hub and leaves have different degrees → their chi2_degree must differ
    G, hub_idx, leaf_indices = _star_hub_leaf_indices()
    X = compute_lauri_features(G)
    chi2 = _col(X, "chi2_degree")
    hub_chi2 = chi2[hub_idx]
    leaf_chi2 = chi2[leaf_indices[0]]
    assert hub_chi2 != leaf_chi2


def test_star_all_finite():
    G = nx.star_graph(4)
    X = compute_lauri_features(G)
    assert np.all(np.isfinite(X))


# ── 9. Disconnected graph ─────────────────────────────────────────────────────

def test_disconnected_no_crash():
    # Two disjoint K3 components
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    X = compute_lauri_features(G)
    assert X.shape == (6, N_FEATURES)


def test_disconnected_all_finite():
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    X = compute_lauri_features(G)
    assert np.all(np.isfinite(X))


def test_disconnected_global_features():
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    X = compute_lauri_features(G)
    np.testing.assert_array_equal(_col(X, "n_nodes"), 6.0)
    np.testing.assert_array_equal(_col(X, "n_edges"), float(G.number_of_edges()))


# ── 10. Isolated node in an otherwise connected graph ────────────────────────

def test_isolated_node_in_connected_graph():
    # K4 plus one isolated node
    G = nx.complete_graph(4)
    G.add_node(99)
    X = compute_lauri_features(G)
    assert X.shape == (5, N_FEATURES)
    assert np.all(np.isfinite(X))


def test_isolated_node_features():
    G = nx.complete_graph(4)
    G.add_node(99)
    nodes = list(G.nodes())
    iso_idx = nodes.index(99)
    X = compute_lauri_features(G)
    assert _col(X, "degree")[iso_idx] == 0.0
    assert _col(X, "lcc")[iso_idx] == 0.0
    assert _col(X, "chi2_degree_nbr")[iso_idx] == 0.0
    assert _col(X, "chi2_lcc_nbr")[iso_idx] == 0.0


# ── 11. Dtype check ───────────────────────────────────────────────────────────

def test_output_dtype():
    G = nx.karate_club_graph()
    X = compute_lauri_features(G)
    assert X.dtype == np.float64


# ── 12. Global features are constant across all nodes ────────────────────────

def test_global_features_constant():
    G = nx.petersen_graph()
    X = compute_lauri_features(G)
    assert np.all(_col(X, "n_nodes") == _col(X, "n_nodes")[0])
    assert np.all(_col(X, "n_edges") == _col(X, "n_edges")[0])

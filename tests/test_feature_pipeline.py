"""
tests/test_feature_pipeline.py
-------------------------------
Unit tests for NodeFeaturePipeline — feature_set parameter and Lauri integration.
"""
import networkx as nx
import numpy as np
import pytest

from features.pipeline import NodeFeaturePipeline


SMALL_GRAPH = nx.karate_club_graph()


class TestFeatureSetParameter:

    def test_invalid_feature_set_raises(self):
        with pytest.raises(ValueError, match="feature_set"):
            NodeFeaturePipeline(feature_set="unknown")

    def test_kernel_shape(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape == (SMALL_GRAPH.number_of_nodes(), pipeline.n_features)
        assert pipeline.n_features == 106  # VSKO(13) + GDV(73) + RW(20)

    def test_lauri_shape(self):
        pipeline = NodeFeaturePipeline(feature_set="lauri")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape == (SMALL_GRAPH.number_of_nodes(), 9)
        assert pipeline.n_features == 9

    def test_both_shape(self):
        pipeline = NodeFeaturePipeline(feature_set="both")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape == (SMALL_GRAPH.number_of_nodes(), 115)  # 106 + 9
        assert pipeline.n_features == 115

    def test_kernel_feature_names_length(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel")
        assert len(pipeline.feature_names) == 106

    def test_lauri_feature_names_length(self):
        pipeline = NodeFeaturePipeline(feature_set="lauri")
        assert len(pipeline.feature_names) == 9

    def test_both_feature_names_length(self):
        pipeline = NodeFeaturePipeline(feature_set="both")
        assert len(pipeline.feature_names) == 115

    def test_both_feature_names_unique(self):
        pipeline = NodeFeaturePipeline(feature_set="both")
        names = pipeline.feature_names
        assert len(names) == len(set(names)), "Feature names must be unique"

    def test_no_nan_kernel(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert np.isfinite(X).all(), "Kernel features must be finite"

    def test_no_nan_lauri(self):
        pipeline = NodeFeaturePipeline(feature_set="lauri")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert np.isfinite(X).all(), "Lauri features must be finite"

    def test_no_nan_both(self):
        pipeline = NodeFeaturePipeline(feature_set="both")
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert np.isfinite(X).all(), "Combined features must be finite"

    def test_both_first_106_match_kernel(self):
        """First 106 columns of 'both' should equal 'kernel' output."""
        G = nx.path_graph(8)
        p_kernel = NodeFeaturePipeline(feature_set="kernel", normalize=False)
        p_both   = NodeFeaturePipeline(feature_set="both",   normalize=False)
        X_kernel = p_kernel.extract_graph(G)
        X_both   = p_both.extract_graph(G)
        np.testing.assert_array_almost_equal(X_both[:, :106], X_kernel)

    def test_both_last_9_match_lauri(self):
        """Last 9 columns of 'both' should equal 'lauri' output."""
        G = nx.path_graph(8)
        p_lauri = NodeFeaturePipeline(feature_set="lauri",  normalize=False)
        p_both  = NodeFeaturePipeline(feature_set="both",   normalize=False)
        X_lauri = p_lauri.extract_graph(G)
        X_both  = p_both.extract_graph(G)
        np.testing.assert_array_almost_equal(X_both[:, 106:], X_lauri)


class TestKernelSubsets:
    """Test disabling individual kernel families within feature_set='kernel'."""

    def test_no_vsko(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=False, use_gdv=True, use_rw=True)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 93  # 73 + 20

    def test_no_gdv(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=True, use_gdv=False, use_rw=True)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 33  # 13 + 20

    def test_no_rw(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=True, use_gdv=True, use_rw=False)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 86  # 13 + 73

    def test_vsko_only(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=True, use_gdv=False, use_rw=False)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 13

    def test_gdv_only(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=False, use_gdv=True, use_rw=False)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 73

    def test_rw_only(self):
        pipeline = NodeFeaturePipeline(feature_set="kernel",
                                       use_vsko=False, use_gdv=False, use_rw=True)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 20

    def test_kernel_subsets_ignored_for_lauri(self):
        """no_vsko/no_gdv/no_rw flags have no effect when feature_set='lauri'."""
        pipeline = NodeFeaturePipeline(feature_set="lauri",
                                       use_vsko=False, use_gdv=False, use_rw=False)
        X = pipeline.extract_graph(SMALL_GRAPH)
        assert X.shape[1] == 9


class TestNormalization:

    def test_fit_transform_scales(self):
        G = nx.karate_club_graph()
        pipeline = NodeFeaturePipeline(feature_set="lauri", normalize=True)
        X_raw = pipeline.extract_graph(G)
        X_scaled = pipeline.fit_transform_features(X_raw)
        # After StandardScaler: mean ~0, std ~1 per column (columns with std>0)
        std = X_scaled.std(axis=0)
        # Some columns may be constant (e.g. n_nodes) — exclude those
        non_const = std > 0
        if non_const.any():
            np.testing.assert_allclose(X_scaled[:, non_const].mean(axis=0), 0,
                                       atol=1e-10)

    def test_transform_uses_fitted_scaler(self):
        G = nx.karate_club_graph()
        pipeline = NodeFeaturePipeline(feature_set="lauri", normalize=True)
        X = pipeline.extract_graph(G)
        pipeline.fit_transform_features(X)
        X2 = pipeline.transform(X)
        assert X2.shape == X.shape

    def test_no_normalization(self):
        G = nx.path_graph(5)
        pipeline = NodeFeaturePipeline(feature_set="lauri", normalize=False)
        X_raw = pipeline.extract_graph(G)
        X_transform = pipeline.transform(X_raw)
        np.testing.assert_array_equal(X_raw, X_transform)


class TestExtractGraphs:

    def test_extract_graphs_batch(self):
        graphs = [nx.path_graph(5), nx.cycle_graph(6), nx.complete_graph(4)]
        for g in graphs:
            for v in g.nodes():
                g.nodes[v]["label"] = v % 2
        pipeline = NodeFeaturePipeline(feature_set="lauri", normalize=False)
        X, y, graph_ids = pipeline.extract_graphs(graphs, show_progress=False)
        total_nodes = sum(g.number_of_nodes() for g in graphs)
        assert X.shape == (total_nodes, 9)
        assert len(y) == total_nodes
        assert len(graph_ids) == total_nodes
        assert set(graph_ids) == {0, 1, 2}

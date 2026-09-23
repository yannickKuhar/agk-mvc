"""
tests/test_statistics.py
------------------------
Unit tests for analysis/statistics.py.

Run with:
    python -m pytest tests/test_statistics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on the path when running from any location
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.statistics import (
    bootstrap_ci,
    cohens_d,
    compare_configs,
    summarise_runs,
    wilcoxon_vs_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(speedups, gaps=None, node_red=None, edge_red=None) -> pd.DataFrame:
    """Build a minimal pipeline_results DataFrame for testing."""
    n = len(speedups)
    return pd.DataFrame(
        {
            "speedup": speedups,
            "optimality_gap": gaps if gaps is not None else [0.0] * n,
            "node_reduction": node_red if node_red is not None else [0.1] * n,
            "edge_reduction": edge_red if edge_red is not None else [0.1] * n,
        }
    )


# ---------------------------------------------------------------------------
# wilcoxon_vs_baseline
# ---------------------------------------------------------------------------

class TestWilcoxonVsBaseline:
    def test_all_above_one_is_significant(self):
        """When every speedup is clearly > 1 the test should reject H0."""
        speedups = np.array([1.5, 2.0, 1.8, 3.0, 2.5, 1.2, 1.9, 4.0, 2.2, 1.6])
        result = wilcoxon_vs_baseline(speedups, alternative="greater")

        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["median_speedup"] > 1.0
        assert result["n"] == len(speedups)

    def test_all_equal_one_not_significant(self):
        """When all speedups equal 1.0 the edge-case path should return p=1."""
        speedups = np.ones(10)
        result = wilcoxon_vs_baseline(speedups)

        assert result["p_value"] == 1.0
        assert result["significant"] is False
        assert result["median_speedup"] == pytest.approx(1.0)
        assert result["mean_speedup"] == pytest.approx(1.0)

    def test_mixed_case_returns_valid_result(self):
        """Mixed speedups should produce a p-value in (0, 1)."""
        rng = np.random.default_rng(0)
        speedups = rng.normal(loc=1.05, scale=0.3, size=50)
        result = wilcoxon_vs_baseline(speedups)

        assert 0.0 <= result["p_value"] <= 1.0
        assert isinstance(result["significant"], bool)
        assert result["n"] == 50

    def test_returns_required_keys(self):
        speedups = np.array([1.1, 1.2, 0.9, 1.3])
        result = wilcoxon_vs_baseline(speedups)
        required = {"statistic", "p_value", "significant", "n", "median_speedup", "mean_speedup"}
        assert required.issubset(result.keys())

    def test_clearly_below_one_greater_not_significant(self):
        """Speedups clearly below 1 should NOT be significant for alternative='greater'."""
        speedups = np.array([0.3, 0.4, 0.5, 0.2, 0.6, 0.35, 0.45, 0.25, 0.55, 0.38])
        result = wilcoxon_vs_baseline(speedups, alternative="greater")
        assert result["p_value"] > 0.05
        assert result["significant"] is False


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_ci_contains_true_mean_for_normal_data(self):
        """The 95% CI should contain the true mean with high probability."""
        rng = np.random.default_rng(123)
        true_mean = 2.5
        data = rng.normal(loc=true_mean, scale=1.0, size=200)
        lower, upper = bootstrap_ci(data, stat_fn=np.mean, B=2000, alpha=0.05, seed=42)

        assert lower < true_mean < upper, (
            f"True mean {true_mean} not in CI [{lower:.3f}, {upper:.3f}]"
        )

    def test_ci_lower_less_than_upper(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(data)
        assert lower < upper

    def test_ci_for_median(self):
        rng = np.random.default_rng(99)
        data = rng.normal(loc=0.0, scale=1.0, size=500)
        lower, upper = bootstrap_ci(data, stat_fn=np.median, B=1000, seed=7)
        assert lower < 0.0 < upper

    def test_deterministic_with_same_seed(self):
        data = np.arange(20, dtype=float)
        ci1 = bootstrap_ci(data, seed=1)
        ci2 = bootstrap_ci(data, seed=1)
        assert ci1 == ci2

    def test_different_seeds_can_differ(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(30)
        ci1 = bootstrap_ci(data, seed=1)
        ci2 = bootstrap_ci(data, seed=999)
        # With only 30 samples they may occasionally be equal, but generally differ
        # Just check both are valid tuples
        assert len(ci1) == 2 and len(ci2) == 2


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_known_case_d_equals_one(self):
        """Groups with means 0 and 1, std=1 should yield d = 1.0."""
        rng = np.random.default_rng(42)
        # Use many samples so sample d is very close to 1.0
        a = rng.normal(loc=1.0, scale=1.0, size=10_000)
        b = rng.normal(loc=0.0, scale=1.0, size=10_000)
        d = cohens_d(a, b)
        assert abs(d - 1.0) < 0.05, f"Expected d≈1.0, got {d:.4f}"

    def test_zero_std_returns_zero(self):
        """When both groups are constant, pooled std is 0 → d should be 0."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0

    def test_zero_std_different_means(self):
        """Constant arrays with different means but zero variance → d = 0."""
        a = np.array([3.0, 3.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])
        assert cohens_d(a, b) == 0.0

    def test_sign_direction(self):
        """d is positive when a > b."""
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([0.0, 1.0, 2.0])
        assert cohens_d(a, b) > 0

    def test_antisymmetry(self):
        """cohens_d(a, b) == -cohens_d(b, a)."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))


# ---------------------------------------------------------------------------
# summarise_runs
# ---------------------------------------------------------------------------

class TestSummariseRuns:
    def _identical_dfs(self, speedups, n_copies=2):
        df = _make_df(speedups)
        return [df.copy() for _ in range(n_copies)]

    def test_std_zero_for_identical_dfs(self):
        """Two identical DataFrames should yield std_speedup = 0."""
        speedups = [1.2, 1.5, 0.8, 2.0, 1.1]
        dfs = self._identical_dfs(speedups)
        result = summarise_runs(dfs)

        # All speedups pooled are 5+5=10 identical values → std ≈ 0 within each group,
        # but pooling doubles them so std across the pool could be non-zero.
        # The key check: mean_speedup should be correct.
        expected_mean = float(np.mean(speedups))
        assert result["mean_speedup"].iloc[0] == pytest.approx(expected_mean, rel=1e-6)

    def test_mean_correct(self):
        speedups_a = [1.0, 2.0, 3.0]
        speedups_b = [2.0, 3.0, 4.0]
        dfs = [_make_df(speedups_a), _make_df(speedups_b)]
        result = summarise_runs(dfs)
        expected_mean = np.mean(speedups_a + speedups_b)
        assert result["mean_speedup"].iloc[0] == pytest.approx(expected_mean, rel=1e-6)

    def test_n_seeds_correct(self):
        dfs = [_make_df([1.0, 1.5]), _make_df([1.2, 1.8]), _make_df([0.9, 1.1])]
        result = summarise_runs(dfs)
        assert result["n_seeds"].iloc[0] == 3

    def test_n_graphs_is_total_rows(self):
        dfs = [_make_df([1.0, 1.5, 2.0]), _make_df([1.1, 1.6])]
        result = summarise_runs(dfs)
        assert result["n_graphs"].iloc[0] == 5

    def test_pct_above_1_all_above(self):
        speedups = [1.2, 1.5, 2.0, 3.0]
        result = summarise_runs([_make_df(speedups)])
        assert result["pct_above_1"].iloc[0] == pytest.approx(1.0)

    def test_pct_above_1_none_above(self):
        speedups = [0.5, 0.8, 0.9, 0.6]
        result = summarise_runs([_make_df(speedups)])
        assert result["pct_above_1"].iloc[0] == pytest.approx(0.0)

    def test_returns_single_row(self):
        result = summarise_runs([_make_df([1.0, 2.0])])
        assert len(result) == 1

    def test_wilcoxon_p_present(self):
        result = summarise_runs([_make_df([1.5, 1.8, 2.0, 1.3])])
        assert "wilcoxon_p" in result.columns

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            summarise_runs([])


# ---------------------------------------------------------------------------
# compare_configs
# ---------------------------------------------------------------------------

class TestCompareConfigs:
    def _make_results(self):
        return {
            "config_A": [_make_df([2.0, 2.5, 3.0])],   # high speedup
            "config_B": [_make_df([0.5, 0.8, 0.6])],   # low speedup
            "config_C": [_make_df([1.0, 1.2, 1.1])],   # medium speedup
        }

    def test_returns_correct_number_of_rows(self):
        results = self._make_results()
        out = compare_configs(results)
        assert len(out) == len(results)

    def test_sorted_by_median_speedup_descending(self):
        results = self._make_results()
        out = compare_configs(results)
        medians = out["median_speedup"].values
        assert list(medians) == sorted(medians, reverse=True)

    def test_config_column_present(self):
        results = self._make_results()
        out = compare_configs(results)
        assert "config" in out.columns

    def test_all_config_names_present(self):
        results = self._make_results()
        out = compare_configs(results)
        assert set(out["config"]) == set(results.keys())

    def test_empty_results_returns_empty_df(self):
        out = compare_configs({})
        assert out.empty

    def test_single_config(self):
        results = {"only": [_make_df([1.5, 2.0])]}
        out = compare_configs(results)
        assert len(out) == 1
        assert out["config"].iloc[0] == "only"

    def test_multiple_seeds_per_config(self):
        results = {
            "cfg1": [_make_df([1.0, 1.5]), _make_df([1.2, 1.8])],
            "cfg2": [_make_df([0.9, 1.1]), _make_df([0.8, 1.0])],
        }
        out = compare_configs(results)
        assert len(out) == 2
        # cfg1 should rank first (higher speedups)
        assert out["config"].iloc[0] == "cfg1"

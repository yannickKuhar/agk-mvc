"""
analysis/statistics.py
----------------------
Statistical tests and confidence intervals for MVC pipeline experiments.

Functions:
  wilcoxon_vs_baseline(speedups)  -> test if median speedup > 1.0
  bootstrap_ci(values, stat, B, alpha) -> (lower, upper)
  cohens_d(a, b) -> effect size
  summarise_runs(dfs) -> aggregate DataFrame across seeds
  sensitivity_table(results_dict) -> DataFrame for threshold sweep
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test against the null hypothesis that median speedup = 1.0
# ---------------------------------------------------------------------------

def wilcoxon_vs_baseline(
    speedups: np.ndarray,
    alternative: str = "greater",
) -> dict:
    """
    Test H0: median(speedups) = 1.0 using the Wilcoxon signed-rank test.

    The test is performed on the differences ``speedups - 1.0``, so the null
    becomes H0: median difference = 0.

    Parameters
    ----------
    speedups : array-like of float
        Per-graph speedup values (baseline_runtime / pipeline_runtime).
    alternative : {'greater', 'less', 'two-sided'}
        Direction of the alternative hypothesis.

    Returns
    -------
    dict with keys
        statistic      : float  – Wilcoxon W statistic
        p_value        : float
        significant    : bool   – p_value < 0.05
        n              : int    – number of observations
        median_speedup : float
        mean_speedup   : float
    """
    speedups = np.asarray(speedups, dtype=float)
    n = len(speedups)
    median_speedup = float(np.median(speedups))
    mean_speedup = float(np.mean(speedups))

    differences = speedups - 1.0

    # Edge case: all values identical (no variation to rank)
    if np.all(differences == differences[0]):
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "n": n,
            "median_speedup": median_speedup,
            "mean_speedup": mean_speedup,
        }

    try:
        result = stats.wilcoxon(differences, alternative=alternative, zero_method="wilcox")
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except ValueError:
        # scipy raises ValueError when all differences are zero
        p_value = 1.0
        statistic = float("nan")

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n": n,
        "median_speedup": median_speedup,
        "mean_speedup": mean_speedup,
    }


# ---------------------------------------------------------------------------
# Percentile bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval.

    Parameters
    ----------
    values  : array-like of float
    stat_fn : callable applied to each bootstrap sample (default: np.mean)
    B       : number of bootstrap resamples
    alpha   : significance level; returns (alpha/2, 1-alpha/2) percentiles
    seed    : random seed for reproducibility

    Returns
    -------
    (lower, upper) : tuple of float
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(B, len(values)), replace=True)
    boot_stats = np.apply_along_axis(stat_fn, axis=1, arr=samples)
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d using pooled standard deviation.

    Parameters
    ----------
    a, b : array-like of float
        Two groups to compare.

    Returns
    -------
    float : Cohen's d (positive means a > b). Returns 0.0 when pooled std is 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    var_a = np.var(a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0.0
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    if pooled_std == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Aggregate multiple seed DataFrames into a single-row summary
# ---------------------------------------------------------------------------

_KEY_METRICS = [
    "speedup",
    "optimality_gap",
    "node_reduction",
    "edge_reduction",
]


def summarise_runs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate a list of per-seed DataFrames (one row per graph each) into a
    single-row summary DataFrame.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each element is the ``pipeline_results.csv`` for one seed run.

    Returns
    -------
    pd.DataFrame with one row containing:
        mean_speedup, std_speedup, median_speedup, pct_above_1,
        mean_gap, std_gap,
        mean_node_reduction, mean_edge_reduction,
        n_graphs, n_seeds,
        wilcoxon_p
    """
    if not dfs:
        raise ValueError("dfs must be a non-empty list of DataFrames")

    # Pool all speedups across seeds for the Wilcoxon test
    all_speedups = np.concatenate([df["speedup"].dropna().values for df in dfs])
    all_gaps = np.concatenate([df["optimality_gap"].dropna().values for df in dfs])
    all_node_red = np.concatenate([df["node_reduction"].dropna().values for df in dfs])
    all_edge_red = np.concatenate([df["edge_reduction"].dropna().values for df in dfs])

    wilcoxon_result = wilcoxon_vs_baseline(all_speedups)

    summary = {
        "mean_speedup": float(np.mean(all_speedups)),
        "std_speedup": float(np.std(all_speedups, ddof=1)) if len(all_speedups) > 1 else 0.0,
        "median_speedup": float(np.median(all_speedups)),
        "pct_above_1": float(np.mean(all_speedups > 1.0)),
        "mean_gap": float(np.mean(all_gaps)),
        "std_gap": float(np.std(all_gaps, ddof=1)) if len(all_gaps) > 1 else 0.0,
        "mean_node_reduction": float(np.mean(all_node_red)),
        "mean_edge_reduction": float(np.mean(all_edge_red)),
        "n_graphs": int(sum(len(df) for df in dfs)),
        "n_seeds": int(len(dfs)),
        "wilcoxon_p": wilcoxon_result["p_value"],
    }
    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Compare multiple configurations
# ---------------------------------------------------------------------------

def compare_configs(results: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """
    Compare multiple pipeline configurations across their seed runs.

    Parameters
    ----------
    results : dict mapping config_name -> list of per-seed DataFrames

    Returns
    -------
    pd.DataFrame with one row per config, sorted by median_speedup descending.
    Columns: config + all columns from summarise_runs.
    """
    rows = []
    for config_name, dfs in results.items():
        summary = summarise_runs(dfs)
        summary.insert(0, "config", config_name)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values("median_speedup", ascending=False).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Sensitivity table (threshold sweep)
# ---------------------------------------------------------------------------

def sensitivity_table(results_dict: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """
    Build a sensitivity / ablation table from a dict keyed by a sweep parameter
    (e.g. threshold value) mapping to seed DataFrames.

    Parameters
    ----------
    results_dict : dict mapping parameter_label -> list of DataFrames

    Returns
    -------
    pd.DataFrame sorted by parameter label, with columns from summarise_runs.
    """
    return compare_configs(results_dict).rename(columns={"config": "parameter"}).sort_values(
        "parameter"
    ).reset_index(drop=True)

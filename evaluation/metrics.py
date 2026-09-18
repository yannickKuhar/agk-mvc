"""
evaluation/metrics.py
---------------------
Evaluation metrics for the MVC pipeline:

  1. Node classification metrics  (F1, AUC, precision, recall)
  2. MVC solution quality         (optimality gap, cover size ratio)
  3. Solver speedup               (runtime reduction after pruning)
  4. Pruning statistics           (node/edge reduction ratios)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    f1: float
    auc: float
    precision: float
    recall: float
    threshold: float

    def __str__(self):
        return (f"F1={self.f1:.4f}  AUC={self.auc:.4f}  "
                f"P={self.precision:.4f}  R={self.recall:.4f}  "
                f"τ={self.threshold:.3f}")


@dataclass
class MVCQualityMetrics:
    """
    Compares the pipeline's MVC solution to a reference (ground truth or
    baseline solver on full graph).
    """
    pipeline_cover_size: int
    reference_cover_size: int           # optimal or approximation on full G
    optimality_gap: float               # (pipeline - reference) / reference
    is_valid: bool                      # does the cover actually cover all edges?
    pipeline_runtime: float             # seconds for pruning + solving G'
    baseline_runtime: float             # seconds for solving full G
    speedup: float                      # baseline_runtime / pipeline_runtime

    def __str__(self):
        valid_str = "✓" if self.is_valid else "✗ INVALID"
        return (
            f"Cover size: {self.pipeline_cover_size} (pipeline) vs "
            f"{self.reference_cover_size} (reference)  "
            f"Gap={self.optimality_gap:+.2%}  "
            f"Valid={valid_str}  "
            f"Speedup={self.speedup:.2f}x"
        )


@dataclass
class PruningMetrics:
    node_reduction: float   # fraction of nodes removed
    edge_reduction: float   # fraction of edges removed
    forced_node_frac: float # fraction of nodes force-included
    n_removed: int
    n_forced: int
    n_remaining: int

    def __str__(self):
        return (
            f"Node reduction={self.node_reduction:.1%}  "
            f"Edge reduction={self.edge_reduction:.1%}  "
            f"Forced={self.n_forced}  Remaining={self.n_remaining}"
        )


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> ClassificationMetrics:
    """Compute node classification metrics."""
    mask = y_true >= 0
    y_true, probs = y_true[mask], probs[mask]
    preds = (probs >= threshold).astype(int)

    return ClassificationMetrics(
        f1=float(f1_score(y_true, preds, zero_division=0)),
        auc=float(roc_auc_score(y_true, probs)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        threshold=threshold,
    )


def mvc_quality_metrics(
    G_original: nx.Graph,
    pipeline_cover: Set,
    pipeline_runtime: float,
    reference_cover_size: int,
    baseline_runtime: float,
) -> MVCQualityMetrics:
    """Compute MVC quality and speedup metrics."""
    from pruning.heuristic import verify_cover

    is_valid = verify_cover(G_original, pipeline_cover)
    p_size = len(pipeline_cover)
    gap = (p_size - reference_cover_size) / max(reference_cover_size, 1)

    return MVCQualityMetrics(
        pipeline_cover_size=p_size,
        reference_cover_size=reference_cover_size,
        optimality_gap=gap,
        is_valid=is_valid,
        pipeline_runtime=pipeline_runtime,
        baseline_runtime=baseline_runtime,
        speedup=baseline_runtime / max(pipeline_runtime, 1e-9),
    )


def pruning_metrics(pruning_result) -> PruningMetrics:
    """Compute pruning statistics from a PruningResult."""
    n_orig = pruning_result.original_n_nodes
    n_removed = len(pruning_result.removed_nodes)
    n_forced = len(pruning_result.forced_nodes)
    n_remaining = pruning_result.reduced_graph.number_of_nodes()

    return PruningMetrics(
        node_reduction=n_removed / max(n_orig, 1),
        edge_reduction=pruning_result.reduction_ratio_edges,
        forced_node_frac=n_forced / max(n_orig, 1),
        n_removed=n_removed,
        n_forced=n_forced,
        n_remaining=n_remaining,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    graphs: List[nx.Graph],
    pipeline,                   # NodeFeaturePipeline
    classifier,                 # MVCNodeClassifier
    pruner,                     # ConfidencePruner
    solver,                     # MVCSolver
    baseline_solver,            # MVCSolver (solves full graph)
    show_per_graph: bool = False,
    min_nodes: int = 0,         # skip graphs smaller than this
) -> pd.DataFrame:
    """
    Full end-to-end evaluation on a list of test graphs.

    Speedup definition: baseline_solver_time / (pruning_time + reduced_solver_time).
    Feature extraction is excluded from both sides — it is a one-time cost shared
    across all downstream uses and is reported separately.

    Returns a DataFrame with one row per graph.
    """
    import time
    from pruning.heuristic import reconstruct_cover, verify_cover

    rows = []
    skipped = 0
    for g_idx, G in enumerate(graphs):
        nodes = list(G.nodes())
        n = len(nodes)

        if n < min_nodes:
            skipped += 1
            continue

        y_true = np.array([G.nodes[v].get("label", -1) for v in nodes])

        # --- Feature extraction + classification (excluded from speedup) ---
        feature_extraction_time = 0.0
        clf_metrics_obj = None
        probs = None
        if pipeline is not None and classifier is not None:
            t_feat_start = time.perf_counter()
            X = pipeline.extract_graph(G)
            X_scaled = pipeline.transform(X)
            feature_extraction_time = time.perf_counter() - t_feat_start
            probs = classifier.predict_proba(X_scaled)
            clf_metrics_obj = classification_metrics(y_true, probs, classifier.best_threshold_)

        # --- Pruning + reduced solve (this IS the pipeline runtime) ---
        t_prune_start = time.perf_counter()
        if pruner is None:
            # Baseline only — no pruning
            from pruning.heuristic import PruningResult
            import networkx as nx_inner
            pruning_result = PruningResult(
                reduced_graph=G.copy(),
                forced_nodes=set(),
                removed_nodes=set(),
                node_probs={v: 0.5 for v in G.nodes()},
                original_n_nodes=n,
                original_n_edges=G.number_of_edges(),
            )
        elif probs is not None:
            pruning_result = pruner.prune_with_array(G, probs)
        else:
            # Structural pruner: computes its own scores from graph structure
            pruning_result = pruner.prune(G)
        G_reduced = pruning_result.reduced_graph
        solve_result = solver.solve(G_reduced)
        pipeline_runtime = time.perf_counter() - t_prune_start   # prune + solve only

        pipeline_cover = reconstruct_cover(pruning_result, solve_result.cover,
                                           original_graph=G)

        # --- Baseline: solve full graph without any pruning ---
        t_baseline_start = time.perf_counter()
        baseline_result = baseline_solver.solve(G)
        baseline_runtime = time.perf_counter() - t_baseline_start

        # --- Quality ---
        is_valid = verify_cover(G, pipeline_cover)
        p_size = len(pipeline_cover)
        b_size = baseline_result.size
        gap = (p_size - b_size) / max(b_size, 1)
        speedup = baseline_runtime / max(pipeline_runtime, 1e-9)

        pm = pruning_metrics(pruning_result)

        row = {
            "graph_id": g_idx,
            "source": G.graph.get("dataset", "unknown"),
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "n_nodes_reduced": G_reduced.number_of_nodes(),
            "n_edges_reduced": G_reduced.number_of_edges(),
            # Classification (None when using structural/no pruner)
            "clf_f1":        clf_metrics_obj.f1        if clf_metrics_obj else float("nan"),
            "clf_auc":       clf_metrics_obj.auc       if clf_metrics_obj else float("nan"),
            "clf_precision": clf_metrics_obj.precision if clf_metrics_obj else float("nan"),
            "clf_recall":    clf_metrics_obj.recall    if clf_metrics_obj else float("nan"),
            # MVC quality
            "pipeline_cover_size": p_size,
            "baseline_cover_size": b_size,
            "optimality_gap": gap,
            "cover_valid": is_valid,
            # Runtime breakdown
            "feature_extraction_time": feature_extraction_time,
            "pipeline_runtime": pipeline_runtime,
            "baseline_runtime": baseline_runtime,
            "speedup": speedup,
            # Pruning
            "node_reduction": pm.node_reduction,
            "edge_reduction": pm.edge_reduction,
            "n_forced": pm.n_forced,
        }
        rows.append(row)

        if show_per_graph:
            valid_sym = "✓" if is_valid else "✗"
            clf_str = f"F1={clf_metrics_obj.f1:.3f}  " if clf_metrics_obj else ""
            print(f"  G{g_idx:03d} n={n:3d}  {clf_str}"
                  f"Gap={gap:+.1%}  Speedup={speedup:.2f}x  {valid_sym}")

    if skipped:
        print(f"[eval] Skipped {skipped} graphs with fewer than {min_nodes} nodes.")

    df = pd.DataFrame(rows)
    return df


def print_summary_by_source(df: pd.DataFrame) -> None:
    """Print per-source breakdown of evaluation metrics."""
    if df.empty or "source" not in df.columns:
        return
    print("\n" + "=" * 70)
    print("PER-SOURCE BREAKDOWN")
    print("=" * 70)
    hdr = (f"{'Source':<22} | {'N':>4} | {'F1':>6} | {'AUC':>6} | "
           f"{'Valid':>6} | {'Gap':>7} | {'Speedup':>7}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for src, grp in df.groupby("source"):
        valid_pct = grp["cover_valid"].mean()
        print(f"  {src:<20} | {len(grp):>4} | "
              f"{grp['clf_f1'].mean():>6.3f} | "
              f"{grp['clf_auc'].mean():>6.3f} | "
              f"{valid_pct:>5.0%} | "
              f"{grp['optimality_gap'].mean():>+6.1%} | "
              f"{grp['speedup'].mean():>6.2f}x")
    print(sep)
    print(f"  {'ALL':<20} | {len(df):>4} | "
          f"{df['clf_f1'].mean():>6.3f} | "
          f"{df['clf_auc'].mean():>6.3f} | "
          f"{df['cover_valid'].mean():>5.0%} | "
          f"{df['optimality_gap'].mean():>+6.1%} | "
          f"{df['speedup'].mean():>6.2f}x")
    print("=" * 70)


def print_summary(df: pd.DataFrame) -> None:
    """Print aggregate evaluation summary."""
    print("\n" + "=" * 60)
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Graphs evaluated:     {len(df)}")
    if df.empty:
        print("\n  No graphs were evaluated — check --eval-min-nodes vs dataset size.")
        print("  The Erdos dataset contains graphs with 4–34 nodes.")
        print("=" * 60)
        return
    print(f"\n--- Node Classification ---")
    print(f"  Mean F1:            {df['clf_f1'].mean():.4f} ± {df['clf_f1'].std():.4f}")
    print(f"  Mean AUC:           {df['clf_auc'].mean():.4f} ± {df['clf_auc'].std():.4f}")
    print(f"\n--- MVC Quality ---")
    print(f"  Valid covers:       {df['cover_valid'].mean():.1%}")
    print(f"  Mean opt. gap:      {df['optimality_gap'].mean():+.2%}")
    print(f"  Median opt. gap:    {df['optimality_gap'].median():+.2%}")
    print(f"\n--- Pruning ---")
    print(f"  Mean node reduction:{df['node_reduction'].mean():.1%}")
    print(f"  Mean edge reduction:{df['edge_reduction'].mean():.1%}")
    print(f"\n--- Solver Speedup ---")
    print(f"  Mean speedup:       {df['speedup'].mean():.2f}x")
    print(f"  Median speedup:     {df['speedup'].median():.2f}x")
    print("=" * 60)

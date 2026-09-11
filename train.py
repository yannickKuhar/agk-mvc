"""
train.py
--------
End-to-end training script.

Usage:
    python train.py [options]

Options:
    --max-graphs INT        Max graphs to load (default: 500)
    --ego-hops INT          Ego graph radius for VSKO (default: 2)
    --prune-threshold FLOAT Pruning confidence threshold (default: 0.10)
    --output-dir STR        Directory for saved model and results (default: results/)
    --no-vsko               Disable VSKO features
    --no-gdv                Disable GDV features
    --no-rw                 Disable RW features
    --orca-path STR         Path to ORCA binary (default: orca)
    --use-orca              Enable ORCA binary (default: pure Python fallback)
    --cv-only               Only run cross-validation, don't save a final model
    --skip-cv               Skip cross-validation (faster)
    --solver-backend STR    'ilp' or 'approx' (default: ilp)
    --solver-timeout INT    ILP solver timeout in seconds (default: 60)
    --seed INT              Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.loader import load_erdos, graph_summary, get_node_labels
from features.pipeline import NodeFeaturePipeline
from models.classifier import MVCNodeClassifier
from pruning.heuristic import ConfidencePruner
from solver.mvc_solver import MVCSolver
from evaluation.metrics import evaluate_pipeline, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Train MVC Graph Kernel Pipeline")
    p.add_argument("--max-graphs", type=int, default=500)
    p.add_argument("--ego-hops", type=int, default=2)
    p.add_argument("--prune-threshold", type=float, default=0.10)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--no-vsko", action="store_true")
    p.add_argument("--no-gdv", action="store_true")
    p.add_argument("--no-rw", action="store_true")
    p.add_argument("--orca-path", type=str, default="orca")
    p.add_argument("--use-orca", action="store_true")
    p.add_argument("--cv-only", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--solver-backend", type=str, default="ilp", choices=["ilp", "approx"])
    p.add_argument("--solver-timeout", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Step 1: Loading data ===")
    graphs = load_erdos(split="train", max_graphs=args.max_graphs)
    summary = graph_summary(graphs)
    print(json.dumps(summary, indent=2))

    # Train/test split at graph level
    n = len(graphs)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed
    )
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    print("\n=== Step 2: Extracting node features ===")
    pipeline = NodeFeaturePipeline(
        ego_hops=args.ego_hops,
        use_orca_binary=args.use_orca,
        orca_path=args.orca_path,
        use_vsko=not args.no_vsko,
        use_gdv=not args.no_gdv,
        use_rw=not args.no_rw,
        normalize=True,
    )
    print(f"Feature dimensions: {pipeline.n_features}")
    print(f"Feature names: {pipeline.feature_names[:5]} ... [{pipeline.n_features} total]")

    t0 = time.perf_counter()
    X_train, y_train, graph_ids_train = pipeline.extract_graphs(train_graphs)
    print(f"Train feature extraction: {time.perf_counter() - t0:.1f}s")
    print(f"Train nodes: {len(X_train)} | Positive rate: {y_train[y_train>=0].mean():.3f}")

    X_train_scaled = pipeline.fit_transform_features(X_train)

    t0 = time.perf_counter()
    X_test, y_test, graph_ids_test = pipeline.extract_graphs(test_graphs)
    print(f"Test feature extraction: {time.perf_counter() - t0:.1f}s")
    X_test_scaled = pipeline.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Cross-validation
    # ------------------------------------------------------------------
    if not args.skip_cv:
        print("\n=== Step 3: Cross-validation ===")
        clf_cv = MVCNodeClassifier(random_state=args.seed)
        cv_results = clf_cv.cross_validate(
            X_train_scaled, y_train, graph_ids_train,
            feature_names=pipeline.feature_names,
        )
        (out_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2))
        print(f"CV results saved to {out_dir}/cv_results.json")

    if args.cv_only:
        print("--cv-only flag set. Exiting after cross-validation.")
        return

    # ------------------------------------------------------------------
    # 4. Train final model
    # ------------------------------------------------------------------
    print("\n=== Step 4: Training final model ===")
    classifier = MVCNodeClassifier(random_state=args.seed)
    classifier.fit(
        X_train_scaled,
        y_train,
        graph_ids_train,
        feature_names=pipeline.feature_names,
    )
    classifier.save(str(out_dir / "xgb_model.json"))

    # Test set classification metrics
    test_metrics = classifier.evaluate(X_test_scaled, y_test)
    print(f"\nTest set metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1:  {test_metrics['f1']:.4f}")
    print(f"  Threshold: {test_metrics['threshold']:.3f}")
    print(test_metrics["report"])

    (out_dir / "test_classification_metrics.json").write_text(
        json.dumps({k: v for k, v in test_metrics.items() if k != "report"}, indent=2)
    )

    # SHAP feature importance
    print("\n=== Step 4b: SHAP feature importance ===")
    sample_idx = np.random.choice(len(X_test_scaled), min(200, len(X_test_scaled)), replace=False)
    classifier.explain(X_test_scaled[sample_idx], plot=True)

    fi_df = classifier.feature_importance_df()
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"Top 10 features:")
    print(fi_df.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Evaluate full pipeline (pruning + solving)
    # ------------------------------------------------------------------
    print("\n=== Step 5: End-to-end pipeline evaluation ===")
    pruner = ConfidencePruner(threshold=args.prune_threshold)
    solver = MVCSolver(
        backend=args.solver_backend,
        timeout=args.solver_timeout,
        fallback_to_approx=True,
    )
    baseline_solver = MVCSolver(
        backend=args.solver_backend,
        timeout=args.solver_timeout,
        fallback_to_approx=True,
    )

    # Evaluate on a subset of test graphs (ILP can be slow)
    eval_graphs = test_graphs[:min(50, len(test_graphs))]
    print(f"Evaluating on {len(eval_graphs)} test graphs ...")

    results_df = evaluate_pipeline(
        eval_graphs,
        pipeline=pipeline,
        classifier=classifier,
        pruner=pruner,
        solver=solver,
        baseline_solver=baseline_solver,
        show_per_graph=True,
    )

    results_df.to_csv(out_dir / "pipeline_results.csv", index=False)
    print_summary(results_df)
    print(f"\nDetailed results saved to {out_dir}/pipeline_results.csv")

    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

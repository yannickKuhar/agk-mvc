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
    --output-dir STR        Directory for saved model and results (default: results/<datasets>_<timestamp>)
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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import graph_summary, get_node_labels
from data.unified import DatasetLoader
from features.pipeline import NodeFeaturePipeline
from models.classifier import MVCNodeClassifier
from pruning.heuristic import ConfidencePruner
from pruning.structural import StructuralPruner
from solver.mvc_solver import MVCSolver
from evaluation.metrics import evaluate_pipeline, print_summary, print_summary_by_source


def parse_args():
    p = argparse.ArgumentParser(description="Train MVC Graph Kernel Pipeline")
    p.add_argument("--datasets", type=str, default="erdos",
                   help="Comma-separated dataset sources (default: erdos). "
                        "Examples: erdos,synthetic:small  |  erdos,tudataset:MUTAG  |  "
                        "erdos,synthetic:small,synthetic:medium,pace")
    p.add_argument("--max-graphs", type=int, default=500)
    p.add_argument("--min-nodes", type=int, default=5,
                   help="Skip graphs with fewer nodes than this (default: 5)")
    p.add_argument("--max-nodes", type=int, default=500,
                   help="Skip graphs larger than this (default: 500)")
    p.add_argument("--ego-hops", type=int, default=2)
    p.add_argument("--prune-threshold", type=float, default=0.10)
    p.add_argument("--fix-threshold", type=float, default=1.1,
                   help="Nodes with P(in MVC) > this are locked into cover and excluded "
                        "from the ILP (two-sided pruning). Default 1.1 = disabled. "
                        "Try 0.85 for two-sided pruning.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: results/<datasets>_<timestamp>)")
    p.add_argument("--feature-set", type=str, default="kernel",
                   choices=["kernel", "lauri", "both"],
                   help="Feature family: 'kernel' (VSKO+GDV+RW, default), "
                        "'lauri' (9 handcrafted from Lauri et al. 2023), "
                        "or 'both' (115 dims total)")
    p.add_argument("--no-vsko", action="store_true")
    p.add_argument("--no-gdv", action="store_true")
    p.add_argument("--no-rw", action="store_true")
    p.add_argument("--orca-path", type=str, default="orca")
    p.add_argument("--use-orca", action="store_true")
    p.add_argument("--cv-only", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--pruner", type=str, default="ml", choices=["ml", "structural", "none"],
                   help="Pruner to use during pipeline evaluation: "
                        "'ml' (default, XGBoost-based), "
                        "'structural' (fast heuristic, no model needed), "
                        "'none' (baseline ILP only)")
    p.add_argument("--solver-backend", type=str, default="ilp", choices=["ilp", "approx"])
    p.add_argument("--solver-timeout", type=int, default=60)
    p.add_argument("--ilp-solver", type=str, default="auto",
                   choices=["auto", "cbc", "glpk", "highs", "gurobi", "scip", "xpress"],
                   help="ILP solver: auto (default — fastest available), cbc, glpk, "
                        "highs, gurobi, scip, xpress")
    p.add_argument("--eval-min-nodes", type=int, default=0,
                   help="Only evaluate graphs with at least this many nodes (default: 0 = all)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _make_run_dir(datasets_str: str, pruner: str = "ml",
                  ilp_solver: str = "auto", prune_threshold: float = 0.10,
                  fix_threshold: float = 1.1, feature_set: str = "kernel",
                  no_vsko: bool = False, no_gdv: bool = False, no_rw: bool = False,
                  seed: int = 42, base: str = "results") -> Path:
    tag = datasets_str.replace(",", "_").replace(":", "-").replace(" ", "")
    if pruner != "ml":
        tag += f"_{pruner}-pruner"
    if ilp_solver != "auto":
        tag += f"_{ilp_solver}-solver"
    if feature_set != "kernel":
        tag += f"_{feature_set}-feat"
    # Encode which kernel sub-families are active (A2/A3/A4 disambiguation)
    if feature_set in ("kernel", "both") and (no_vsko or no_gdv or no_rw):
        active = []
        if not no_vsko:
            active.append("vsko")
        if not no_gdv:
            active.append("gdv")
        if not no_rw:
            active.append("rw")
        tag += "_" + "+".join(active) + "-only" if active else "_no-kernel"
    if prune_threshold != 0.10:
        tag += f"_pt{prune_threshold:.2f}"
    if fix_threshold <= 1.0:
        tag += f"_ft{fix_threshold:.2f}"
    if seed != 42:
        tag += f"_s{seed}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{tag}_{ts}"


def main():
    args = parse_args()
    if args.output_dir is None:
        out_dir = _make_run_dir(args.datasets, pruner=args.pruner,
                                ilp_solver=args.ilp_solver,
                                prune_threshold=args.prune_threshold,
                                fix_threshold=args.fix_threshold,
                                feature_set=args.feature_set,
                                no_vsko=args.no_vsko,
                                no_gdv=args.no_gdv,
                                no_rw=args.no_rw,
                                seed=args.seed)
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Output dir: {out_dir}")
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Step 1: Loading data ===")
    sources = [s.strip() for s in args.datasets.split(",") if s.strip()]
    loader = DatasetLoader()
    train_graphs, test_graphs = loader.load(
        sources=sources,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        train_ratio=0.8,
        seed=args.seed,
    )
    if not train_graphs:
        print("\n[train] No training graphs loaded — cannot continue.")
        return
    summary = graph_summary(train_graphs + test_graphs)
    print(json.dumps(summary, indent=2))
    if summary["avg_nodes"] < 30:
        print(f"[train] NOTE: mean graph size is {summary['avg_nodes']:.1f} nodes. "
              "ILP solves in <50ms at this scale; speedup reflects pruning overhead.")
    print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")

    # ------------------------------------------------------------------
    # 2-4. Feature extraction + model training (skipped for structural pruner)
    # ------------------------------------------------------------------
    pipeline = None
    classifier = None

    if args.pruner in ("ml", None):
        print("\n=== Step 2: Extracting node features ===")
        print(f"[train] Feature set: {args.feature_set}")
        pipeline = NodeFeaturePipeline(
            feature_set=args.feature_set,
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

        print("\n=== Step 4: Training final model ===")
        classifier = MVCNodeClassifier(random_state=args.seed)
        classifier.fit(
            X_train_scaled, y_train, graph_ids_train,
            feature_names=pipeline.feature_names,
        )
        classifier.save(str(out_dir / "xgb_model.json"))

        test_metrics = classifier.evaluate(X_test_scaled, y_test)
        print(f"\nTest set metrics:")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  F1:  {test_metrics['f1']:.4f}")
        print(f"  Threshold: {test_metrics['threshold']:.3f}")
        print(test_metrics["report"])
        (out_dir / "test_classification_metrics.json").write_text(
            json.dumps({k: v for k, v in test_metrics.items() if k != "report"}, indent=2)
        )

        print("\n=== Step 4b: SHAP feature importance ===")
        sample_idx = np.random.choice(len(X_test_scaled), min(200, len(X_test_scaled)), replace=False)
        classifier.explain(X_test_scaled[sample_idx], plot=True,
                           plot_path=str(out_dir / "shap_summary.png"))
        fi_df = classifier.feature_importance_df()
        fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
        print(f"Top 10 features:")
        print(fi_df.head(10).to_string(index=False))

    else:
        print(f"\n[train] --pruner {args.pruner}: skipping feature extraction and model training.")

    # ------------------------------------------------------------------
    # 5. Evaluate full pipeline (pruning + solving)
    # ------------------------------------------------------------------
    print("\n=== Step 5: End-to-end pipeline evaluation ===")

    if args.pruner == "ml":
        pruner = ConfidencePruner(threshold=args.prune_threshold,
                                  fix_threshold=args.fix_threshold)
    elif args.pruner == "structural":
        pruner = StructuralPruner(threshold=args.prune_threshold)
        print(f"[train] Using StructuralPruner (threshold={args.prune_threshold})")
    else:
        pruner = None
        print("[train] No pruner — baseline ILP only.")

    solver = MVCSolver(
        backend=args.solver_backend,
        timeout=args.solver_timeout,
        fallback_to_approx=True,
        ilp_solver=args.ilp_solver,
    )
    baseline_solver = MVCSolver(
        backend=args.solver_backend,
        timeout=args.solver_timeout,
        fallback_to_approx=True,
        ilp_solver=args.ilp_solver,
    )

    eval_graphs = test_graphs[:min(50, len(test_graphs))]
    if args.eval_min_nodes > 0:
        eval_graphs_filtered = [G for G in eval_graphs
                                if G.number_of_nodes() >= args.eval_min_nodes]
        if not eval_graphs_filtered:
            max_n = max(G.number_of_nodes() for G in eval_graphs) if eval_graphs else 0
            print(f"\n[train] --eval-min-nodes {args.eval_min_nodes} skips all "
                  f"{len(eval_graphs)} test graphs (largest has {max_n} nodes). "
                  f"The Erdos dataset tops out at 34 nodes. "
                  f"Try --eval-min-nodes 0 to evaluate all graphs.")
            return
    print(f"Evaluating on {len(eval_graphs)} test graphs "
          f"(min_nodes={args.eval_min_nodes}) ...")

    results_df = evaluate_pipeline(
        eval_graphs,
        pipeline=pipeline,
        classifier=classifier,
        pruner=pruner,
        solver=solver,
        baseline_solver=baseline_solver,
        show_per_graph=True,
        min_nodes=args.eval_min_nodes,
    )

    results_df.to_csv(out_dir / "pipeline_results.csv", index=False)
    print_summary(results_df)
    print_summary_by_source(results_df)
    print(f"\nDetailed results saved to {out_dir}/pipeline_results.csv")

    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
infer.py
--------
Run the trained pipeline on a new graph.

Usage:
    python infer.py --graph-file my_graph.edgelist --model results/xgb_model.json
    python infer.py --demo  # run on a synthetic random graph

Input formats supported:
    .edgelist   — each line: "u v"
    .graphml    — NetworkX GraphML format
    .gml        — NetworkX GML format
    .adjlist    — adjacency list

Output:
    - Node probabilities (P in MVC)
    - Pruning summary
    - MVC solution (cover set)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from features.pipeline import NodeFeaturePipeline
from models.classifier import MVCNodeClassifier
from pruning.heuristic import ConfidencePruner, reconstruct_cover, verify_cover
from solver.mvc_solver import MVCSolver


def parse_args():
    p = argparse.ArgumentParser(description="MVC Pipeline Inference")
    p.add_argument("--graph-file", type=str, default=None,
                   help="Path to graph file")
    p.add_argument("--model", type=str, default="results/xgb_model.json",
                   help="Path to trained XGBoost model")
    p.add_argument("--prune-threshold", type=float, default=0.10)
    p.add_argument("--solver-backend", type=str, default="ilp", choices=["ilp", "approx"])
    p.add_argument("--solver-timeout", type=int, default=60)
    p.add_argument("--ego-hops", type=int, default=2)
    p.add_argument("--demo", action="store_true",
                   help="Run on a synthetic Erdos-Renyi graph for demonstration")
    p.add_argument("--output-json", type=str, default=None,
                   help="Save results to JSON file")
    return p.parse_args()


def load_graph(path: str) -> nx.Graph:
    """Load a graph from file based on extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".edgelist":
        return nx.read_edgelist(path, nodetype=int)
    elif ext == ".graphml":
        return nx.read_graphml(path)
    elif ext == ".gml":
        return nx.read_gml(path)
    elif ext == ".adjlist":
        return nx.read_adjlist(path, nodetype=int)
    else:
        raise ValueError(f"Unsupported graph format: {ext}. "
                         f"Supported: .edgelist, .graphml, .gml, .adjlist")


def make_demo_graph(n: int = 30, p: float = 0.15, seed: int = 42) -> nx.Graph:
    """Create a random Erdos-Renyi graph for demonstration."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Relabel to ensure contiguous 0-indexed nodes
    G = nx.convert_node_labels_to_integers(G)
    print(f"[demo] Generated Erdos-Renyi graph: n={G.number_of_nodes()}, "
          f"m={G.number_of_edges()}")
    return G


def run_inference(
    G: nx.Graph,
    classifier: MVCNodeClassifier,
    pipeline: NodeFeaturePipeline,
    pruner: ConfidencePruner,
    solver: MVCSolver,
    baseline_solver: MVCSolver,
    verbose: bool = True,
) -> dict:
    """
    Run the full pipeline on a single graph.
    Returns a dict with all results.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    t_total_start = time.perf_counter()

    # Feature extraction
    X = pipeline.extract_graph(G)
    X_scaled = pipeline.transform(X)

    # Classification
    probs = classifier.predict_proba(X_scaled)
    preds = (probs >= classifier.best_threshold_).astype(int)

    # Pruning
    t_prune = time.perf_counter()
    pruning_result = pruner.prune_with_array(G, probs)
    G_reduced = pruning_result.reduced_graph

    # Solve reduced graph
    solve_result = solver.solve(G_reduced)
    pipeline_cover = reconstruct_cover(pruning_result, solve_result.cover)
    t_pipeline = time.perf_counter() - t_prune

    # Baseline (solve full graph)
    t_baseline = time.perf_counter()
    baseline_result = baseline_solver.solve(G)
    t_baseline = time.perf_counter() - t_baseline

    # Verify
    is_valid = verify_cover(G, pipeline_cover)

    results = {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "node_probabilities": {int(v): float(probs[i]) for i, v in enumerate(nodes)},
        "predicted_in_cover": [int(v) for v, p in zip(nodes, preds) if p == 1],
        "pruning": {
            "removed_nodes": list(pruning_result.removed_nodes),
            "forced_nodes": list(pruning_result.forced_nodes),
            "reduced_n_nodes": G_reduced.number_of_nodes(),
            "reduced_n_edges": G_reduced.number_of_edges(),
            "node_reduction_pct": pruning_result.reduction_ratio_nodes * 100,
            "edge_reduction_pct": pruning_result.reduction_ratio_edges * 100,
        },
        "mvc_solution": {
            "pipeline_cover": sorted(pipeline_cover),
            "pipeline_cover_size": len(pipeline_cover),
            "baseline_cover_size": baseline_result.size,
            "optimality_gap_pct": (
                (len(pipeline_cover) - baseline_result.size)
                / max(baseline_result.size, 1) * 100
            ),
            "cover_valid": is_valid,
            "pipeline_runtime_s": t_pipeline,
            "baseline_runtime_s": t_baseline,
            "speedup": t_baseline / max(t_pipeline, 1e-9),
        },
    }

    if verbose:
        _print_results(results)

    return results


def _print_results(results: dict) -> None:
    print("\n" + "=" * 55)
    print("PIPELINE INFERENCE RESULTS")
    print("=" * 55)
    print(f"Graph: {results['n_nodes']} nodes, {results['n_edges']} edges")

    pr = results["pruning"]
    print(f"\n--- Pruning ---")
    print(f"  Nodes removed:   {len(pr['removed_nodes'])} "
          f"({pr['node_reduction_pct']:.1f}%)")
    print(f"  Nodes forced in: {len(pr['forced_nodes'])}")
    print(f"  Reduced graph:   {pr['reduced_n_nodes']} nodes, "
          f"{pr['reduced_n_edges']} edges")

    mv = results["mvc_solution"]
    valid_str = "✓ VALID" if mv["cover_valid"] else "✗ INVALID"
    print(f"\n--- MVC Solution ---")
    print(f"  Pipeline cover size:  {mv['pipeline_cover_size']}")
    print(f"  Baseline cover size:  {mv['baseline_cover_size']}")
    print(f"  Optimality gap:       {mv['optimality_gap_pct']:+.1f}%")
    print(f"  Cover validity:       {valid_str}")
    print(f"  Pipeline runtime:     {mv['pipeline_runtime_s']:.4f}s")
    print(f"  Baseline runtime:     {mv['baseline_runtime_s']:.4f}s")
    print(f"  Speedup:              {mv['speedup']:.2f}x")

    print(f"\nCover nodes: {mv['pipeline_cover']}")
    print("=" * 55)


def main():
    args = parse_args()

    # Load graph
    if args.demo:
        G = make_demo_graph()
    elif args.graph_file:
        print(f"Loading graph from {args.graph_file} ...")
        G = load_graph(args.graph_file)
    else:
        print("No graph specified. Use --graph-file or --demo.")
        return

    # Load trained model
    classifier = MVCNodeClassifier()
    classifier.load(args.model)

    # Reconstruct pipeline (normalization must match training)
    pipeline = NodeFeaturePipeline(
        ego_hops=args.ego_hops,
        normalize=True,
    )
    # Note: for real use, save/load the scaler alongside the model.
    # Here we fit on the single graph (less ideal but functional for demo).
    X = pipeline.extract_graph(G)
    _ = pipeline.fit_transform_features(X)

    pruner = ConfidencePruner(threshold=args.prune_threshold)
    solver = MVCSolver(backend=args.solver_backend, timeout=args.solver_timeout)
    baseline_solver = MVCSolver(backend=args.solver_backend, timeout=args.solver_timeout)

    results = run_inference(G, classifier, pipeline, pruner, solver, baseline_solver)

    if args.output_json:
        with open(args.output_json, "w") as f:
            # Convert sets to lists for JSON serialization
            json.dump(results, f, indent=2, default=list)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

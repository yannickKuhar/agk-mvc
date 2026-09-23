#!/bin/bash
#SBATCH --partition=amd
#SBATCH --job-name=mvc-journal
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# ---------------------------------------------------------------------------
# Journal-paper experiment suite
#
# Phase 1 — data preparation (~10 min if cached, ~44 h from scratch):
#   synthetic splits skip if .json exists; TUDatasets and real-world skip by
#   default (pass --recompute / --force to regenerate).
#
# Phase 2 — experiments (~120 jobs total across 8 parallel workers):
#   Groups 1-5 : baselines, pruner comparison, threshold sweep, solver
#                comparison, cross-domain
#   Group A    : ablation (Lauri features vs. kernel vs. both)
#   Group S    : sensitivity sweep (fix_threshold × prune_threshold curves)
#   Group R    : real-world graphs
#   Multi-seed : 5 seeds × key configs for statistical significance
#
# Time estimate (all data cached):
#   ~120 experiments / 8 workers × ~8 min avg = ~2 h
#   Multi-seed adds ~25 more jobs        = ~0.5 h
#   Total  ≈ 2.5–3 h  →  48 h limit is comfortable
#
# Workers: 8 (not 16) — each job loads full datasets + XGBoost model.
#   16 simultaneous jobs risk OOM on 48 GB.
# ---------------------------------------------------------------------------

echo "=== MVC journal experiment batch job ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "Working dir: $(pwd)"

mkdir -p results data/realworld

# ===========================================================================
# PHASE 1: Data preparation
# ===========================================================================
echo ""
echo "=== Phase 1: Preparing data ==="

# Erdos dataset
echo "--- Erdos ---"
python download_dataset.py --all-splits

# Synthetic datasets — skip if .json already exists (add --force to regenerate)
echo "--- Synthetic splits ---"
python generate_synthetic.py --split small  --solver-timeout 120
python generate_synthetic.py --split medium --solver-timeout 300
python generate_synthetic.py --split hard   --solver-timeout 300
python generate_synthetic.py --split large  --solver-timeout 600

# TUDataset MVC labels — skipped automatically if cache exists
echo "--- TUDataset MVC labels ---"
python compute_tudataset_mvc.py --dataset MUTAG
python compute_tudataset_mvc.py --dataset NCI1     --max-graphs 500
python compute_tudataset_mvc.py --dataset NCI109   --max-graphs 500
python compute_tudataset_mvc.py --dataset PROTEINS
python compute_tudataset_mvc.py --dataset COLLAB   --max-graphs 200 --solver-timeout 120

# Real-world graphs — skip if builtin.json exists
echo "--- Real-world graphs ---"
if [ -f data/realworld/builtin.json ]; then
    echo "[realworld] data/realworld/builtin.json exists — skipping generation."
else
    python download_realworld.py
fi

# ===========================================================================
# PHASE 2: Run all experiments
# ===========================================================================
echo ""
echo "=== Phase 2: Running experiments ==="

# Run all groups (skip real-world if generation failed, skip-large for speed)
python run_all.py --workers 8

echo ""
echo "=== All done: $(date) ==="

# ===========================================================================
# PHASE 3: Aggregate results across seeds
# ===========================================================================
echo ""
echo "=== Phase 3: Aggregating results ==="
python analysis/aggregate.py --results-dir results/ --out results/aggregate_summary.csv
echo "Summary saved to results/aggregate_summary.csv"

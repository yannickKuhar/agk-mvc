#!/bin/bash
#SBATCH --partition=amd
#SBATCH --job-name=mvc-experiments
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G

# ---------------------------------------------------------------------------
# Adjust the lines above for your cluster:
#   --time        48 h covers data generation + all experiments
#   --cpus-per-task  cores available to Python multiprocessing
#   --mem         48 G; COLLAB labelling + parallel ILP jobs need headroom
#   Add --partition, --account, --gres=gpu:... as required by your cluster.
# ---------------------------------------------------------------------------

echo "=== MVC experiment batch job ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "Working dir: $(pwd)"

mkdir -p results

# ===========================================================================
# PHASE 1: Data preparation
# ===========================================================================
echo ""
echo "=== Phase 1: Preparing data ==="

# Erdos dataset (HuggingFace)
echo "--- Erdos ---"
python download_dataset.py --all-splits

# Synthetic datasets
# small : n=20-50,  p=0.1-0.4,  200 graphs/family,  ~easy for ILP
# medium: n=50-150, p=0.1-0.4,  100 graphs/family,  ~borderline
# hard  : n=50-80,  p=0.3-0.5,   75 graphs/family,  1-15 s ILP time  ← key dataset
# large : n=150-500,p=0.1-0.4,   50 graphs/family,  ~always timeout
echo "--- Synthetic splits ---"
python generate_synthetic.py --split small  --solver-timeout 120
python generate_synthetic.py --split medium --solver-timeout 300
python generate_synthetic.py --split hard   --solver-timeout 300
python generate_synthetic.py --split large  --solver-timeout 600

# TUDatasets — compute optimal MVC labels with ILP
# MUTAG:    188 graphs, avg 18 nodes  — fast (~seconds total)
# NCI1:     4110 graphs, avg 30 nodes — use max-graphs 500 for speed
# NCI109:   4127 graphs, avg 30 nodes — similar to NCI1
# PROTEINS: 1113 graphs, avg 39 nodes — moderate
# COLLAB:   5000 graphs, avg 74 nodes, dense — SLOW (~20s/graph); 200 graphs ≈ 1 h
echo "--- TUDataset MVC labels ---"
python compute_tudataset_mvc.py --dataset MUTAG
python compute_tudataset_mvc.py --dataset NCI1     --max-graphs 500
python compute_tudataset_mvc.py --dataset NCI109   --max-graphs 500
python compute_tudataset_mvc.py --dataset PROTEINS
python compute_tudataset_mvc.py --dataset COLLAB   --max-graphs 200 --solver-timeout 120

# ===========================================================================
# PHASE 2: Run all experiments
# ===========================================================================
# --workers 4: run 4 train.py processes in parallel.
# Keep workers <= cpus-per-task / 2 so ILP threads don't fight each other.
echo ""
echo "=== Phase 2: Running experiments ==="
python run_all.py --workers 4

echo ""
echo "=== All done: $(date) ==="

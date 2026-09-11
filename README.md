# Graph Kernel MVC: Symmetry-Guided Vertex Cover Pruning

A research pipeline integrating graph kernel features (VSKO, GDV orbits, random walk statistics)
with XGBoost node classification to guide Minimum Vertex Cover (MVC) solver pruning.

## Pipeline Overview

```
Erdos Dataset (HuggingFace)
        │
        ▼
Node Feature Extraction
  ├── VSKO (13-dim)         — ego-graph symmetry type distribution (Kuhar & Cibej, 2026)
  ├── GDV Orbits (73-dim)   — graphlet degree vectors via ORCA
  └── Random Walk (varies)  — per-node RW statistics (return prob, hitting time, etc.)
        │
        ▼
XGBoost Binary Classifier
  └── P(node ∈ optimal vertex cover)
        │
        ▼
Pruning Heuristic
  └── Drop nodes with P < τ (default 0.10), repair feasibility
        │
        ▼
Reduced Graph → ILP/Exact MVC Solver
        │
        ▼
Evaluation
  ├── Node classification F1
  ├── MVC solution optimality gap
  └── Solver runtime reduction
```

## Project Structure

```
mvc_kernel_project/
├── data/
│   └── loader.py           # HuggingFace Erdos dataset loading & graph construction
├── features/
│   ├── vsko.py             # VSKO ego-graph symmetry kernel (node-level)
│   ├── gdv.py              # Graphlet Degree Vectors via ORCA (node-level)
│   ├── random_walk.py      # Per-node random walk statistics
│   └── pipeline.py         # Combines all features into one node feature matrix
├── models/
│   └── classifier.py       # XGBoost node classifier with CV + SHAP explanation
├── pruning/
│   └── heuristic.py        # Confidence-threshold pruning + feasibility repair
├── solver/
│   └── mvc_solver.py       # Exact MVC via ILP (PuLP); NetworkX approx fallback
├── evaluation/
│   └── metrics.py          # F1, optimality gap, runtime reduction reporting
├── train.py                # End-to-end training script
├── infer.py                # Run pipeline on a new graph
├── requirements.txt
└── README.md
```

## Symmetry Kernel (VSKO) — Node-Level Adaptation

The original VSKO (Kuhar & Cibej, 2026) operates at graph level. We lift it to node level:
- For each node v, extract the **2-hop ego graph** G_v
- Compute VSKO on G_v → 13-dim symmetry type distribution
- This captures the local symmetry structure around each node

VSKO symmetry types (index I, with 1-cycles included):
`(2,1), (3), (2,1,1), (2,2), (3,1), (4), (2,1,1,1), (2,2,1), (2,3), (3,1,1), (3,2), (4,1), (5)`

## Dataset

**Erdos** (PKU-ML/Erdos on HuggingFace): graphs with binary node labels indicating
membership in an optimal Minimum Vertex Cover solution.

## Installation

```bash
pip install -r requirements.txt
```

Note: ORCA must be compiled separately (see `features/gdv.py` for instructions).
A pure-Python fallback for GDV is included for environments without ORCA.

## Usage

### Train
```bash
python train.py --ego-hops 2 --prune-threshold 0.10 --output-dir results/
```

### Infer on a new graph
```bash
python infer.py --graph-file my_graph.edgelist --model results/xgb_model.json
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Ego-graph radius | 2-hop | Balances locality vs. feature richness |
| Pruning threshold τ | 0.10 | Conservative; nodes with P<10% very unlikely in cover |
| Feasibility repair | Force-include nodes covering dropped-node edges | Guarantees valid cover |
| Solver backend | PuLP ILP (exact) | Swappable; NetworkX approx for large graphs |
| Class imbalance | XGBoost `scale_pos_weight` | MVC labels often imbalanced |

## References

- Kuhar, Y. & Cibej, U. (2026). *Symmetry-based kernels for graph classification.* NPL2026.
- Hocevar, T. & Demsar, J. (2014). *A combinatorial approach to graphlet counting.* Bioinformatics.
- PKU-ML/Erdos dataset: https://huggingface.co/datasets/PKU-ML/Erdos

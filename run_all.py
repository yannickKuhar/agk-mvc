"""
run_all.py
----------
Run all training experiments in parallel.

Each job is a --datasets string passed to train.py.  Results land in
automatically named subdirectories under results/.

Usage:
    python run_all.py                     # sequential
    python run_all.py --workers 4         # 4 parallel jobs
    python run_all.py --skip-large        # omit large/slow combos
    python run_all.py --dry-run           # print commands without running
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Each entry is a dict of train.py flags.  'datasets' is required; the rest
# are optional overrides.  Jobs are run in the order listed.
EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # 1. Single-source baselines
    #    Establish per-domain performance before mixing.
    # -----------------------------------------------------------------------
    {"datasets": "erdos"},
    {"datasets": "synthetic:small"},
    {"datasets": "synthetic:medium"},
    {"datasets": "synthetic:hard"},           # graphs where ILP takes 1–15 s
    {"datasets": "tudataset:MUTAG"},
    {"datasets": "tudataset:NCI1"},
    {"datasets": "tudataset:COLLAB"},         # dense social graphs, avg 74 nodes

    # -----------------------------------------------------------------------
    # 2. Pruner comparison on hard graphs  (KEY EXPERIMENT)
    #    Same dataset, three pruning strategies — directly comparable speedup.
    # -----------------------------------------------------------------------
    {"datasets": "synthetic:hard", "pruner": "none"},       # true ILP baseline
    {"datasets": "synthetic:hard", "pruner": "ml"},         # learned pruner
    {"datasets": "synthetic:hard", "pruner": "structural"}, # fast heuristic

    # -----------------------------------------------------------------------
    # 2b. Solver comparison on hard graphs
    #     Same dataset + pruner, different ILP backends — reveals solver effect.
    # -----------------------------------------------------------------------
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "cbc"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "highs"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "gurobi"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "scip"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "xpress"},

    # -----------------------------------------------------------------------
    # 3. Cross-domain training — synthetic
    #    Train on progressively more data; test set always includes hard graphs.
    # -----------------------------------------------------------------------
    {"datasets": "erdos,synthetic:small"},
    {"datasets": "erdos,synthetic:small,synthetic:medium"},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard"},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:large"},          # slow
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,synthetic:large"},  # slow

    # -----------------------------------------------------------------------
    # 4. Cross-domain training — TUDatasets
    # -----------------------------------------------------------------------
    {"datasets": "erdos,tudataset:MUTAG"},
    {"datasets": "erdos,tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS"},
    {"datasets": "erdos,tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS,tudataset:COLLAB"},

    # -----------------------------------------------------------------------
    # 5. Full combinations  (max_graphs cap keeps runtime tractable)
    # -----------------------------------------------------------------------
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,"
                 "tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS",
     "max_graphs": 1000},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,"
                 "tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS,tudataset:COLLAB",
     "max_graphs": 1000},
]

# Datasets that make experiments slow — filtered out with --skip-large
LARGE_DATASETS = {"synthetic:large"}


def _build_cmd(exp: dict, python: str) -> list[str]:
    cmd = [python, "train.py", "--skip-cv",
           "--datasets", exp["datasets"]]
    if "max_graphs" in exp:
        cmd += ["--max-graphs", str(exp["max_graphs"])]
    if "pruner" in exp:
        cmd += ["--pruner", exp["pruner"]]
    if "ilp_solver" in exp:
        cmd += ["--ilp-solver", exp["ilp_solver"]]
    return cmd


def run_job(args: tuple) -> int:
    exp, python, dry_run = args
    cmd = _build_cmd(exp, python)
    label = exp["datasets"]
    if "pruner" in exp:
        label += f"  [pruner={exp['pruner']}]"
    if "ilp_solver" in exp:
        label += f"  [solver={exp['ilp_solver']}]"

    print(f"[run_all] START  {label}", flush=True)
    if dry_run:
        print(f"  CMD: {' '.join(cmd)}", flush=True)
        return 0

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    code = result.returncode
    status = "OK" if code == 0 else f"FAILED (exit {code})"
    print(f"[run_all] DONE   {label} → {status}", flush=True)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MVC training experiments")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent jobs (default: 1). Each job loads a full "
                             "dataset and model — raise only if you have enough RAM.")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip experiments that include large/slow datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    python = sys.executable  # same interpreter that launched this script

    experiments = EXPERIMENTS
    if args.skip_large:
        experiments = [
            e for e in experiments
            if not any(ld in e["datasets"] for ld in LARGE_DATASETS)
        ]

    jobs = [(exp, python, args.dry_run) for exp in experiments]
    print(f"[run_all] Total experiments: {len(jobs)}")
    if args.dry_run:
        print("[run_all] DRY RUN — commands only\n")

    workers = min(args.workers, len(jobs))
    if workers <= 1:
        results = [run_job(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(run_job, jobs)

    failed = [(experiments[i]["datasets"], c)
              for i, c in enumerate(results) if c != 0]
    print(f"\n[run_all] All done. "
          f"{len(results) - len(failed)}/{len(results)} succeeded.")
    if failed:
        print("[run_all] Failed:")
        for ds, code in failed:
            print(f"  {ds}  (exit {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()

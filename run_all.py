"""
run_all.py
----------
Run all training experiments in parallel.

Each job is a dict of train.py flags. 'datasets' is required; the rest are
optional overrides. Multi-seed experiments repeat key configs across seeds.

Usage:
    python run_all.py                     # sequential
    python run_all.py --workers 8         # 8 parallel jobs
    python run_all.py --group 2b          # only threshold sweep
    python run_all.py --skip-large        # omit large/slow combos
    python run_all.py --dry-run           # print commands without running
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Seeds for multi-seed experiments
# ---------------------------------------------------------------------------
MULTI_SEEDS = [42, 123, 456, 789, 1337]

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry is a dict of train.py flags. 'datasets' required.
# Optional: pruner, ilp_solver, feature_set, prune_threshold, fix_threshold,
#           eval_min_nodes, max_graphs
# Jobs run in order listed; multi-seed groups are expanded at runtime.

# ── Group 1: single-source baselines ────────────────────────────────────────
GROUP1 = [
    {"datasets": "erdos"},
    {"datasets": "synthetic:small"},
    {"datasets": "synthetic:medium"},
    {"datasets": "synthetic:hard"},
    {"datasets": "tudataset:MUTAG"},
    {"datasets": "tudataset:NCI1"},
    {"datasets": "tudataset:COLLAB"},
]

# ── Group 2: pruner comparison on hard graphs ────────────────────────────────
GROUP2 = [
    {"datasets": "synthetic:hard", "pruner": "none"},
    {"datasets": "synthetic:hard", "pruner": "ml"},
    {"datasets": "synthetic:hard", "pruner": "structural"},
]

# ── Group 2b: threshold sweep ────────────────────────────────────────────────
GROUP2B = [
    {"datasets": "synthetic:hard", "pruner": "ml", "prune_threshold": 0.20},
    {"datasets": "synthetic:hard", "pruner": "ml", "prune_threshold": 0.30},
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85},
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.20, "fix_threshold": 0.85},
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.30, "fix_threshold": 0.85},
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.20, "fix_threshold": 0.85, "eval_min_nodes": 30},
]

# ── Group 2c: solver comparison ──────────────────────────────────────────────
GROUP2C = [
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "cbc"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "highs"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "gurobi"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "scip"},
    {"datasets": "synthetic:hard", "pruner": "none", "ilp_solver": "xpress"},
]

# ── Group 3: cross-domain synthetic ─────────────────────────────────────────
GROUP3 = [
    {"datasets": "erdos,synthetic:small"},
    {"datasets": "erdos,synthetic:small,synthetic:medium"},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard"},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:large"},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,synthetic:large"},
]

# ── Group 4: cross-domain TUDatasets ────────────────────────────────────────
GROUP4 = [
    {"datasets": "erdos,tudataset:MUTAG"},
    {"datasets": "erdos,tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS"},
    {"datasets": "erdos,tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS,tudataset:COLLAB"},
]

# ── Group 5: full combinations ───────────────────────────────────────────────
GROUP5 = [
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,"
                 "tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS",
     "max_graphs": 1000},
    {"datasets": "erdos,synthetic:small,synthetic:medium,synthetic:hard,"
                 "tudataset:MUTAG,tudataset:NCI1,tudataset:PROTEINS,tudataset:COLLAB",
     "max_graphs": 1000},
]

# ── Group A: ablation — feature set comparison (KEY for journal paper) ───────
# All use synthetic:hard, ml pruner, best threshold config (pt=0.10, ft=0.85).
# A1-A7 are run with ALL 5 seeds (via MULTI_SEED_CONFIGS below).
# Do NOT run this list with a single seed — always expand via _expand_multi_seed.
GROUP_ABLATION = [
    # A1: no pruner — true ILP baseline (no features needed)
    {"datasets": "synthetic:hard", "pruner": "none"},
    # A2: VSKO only (no GDV, no RW)
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "kernel", "no_gdv": True, "no_rw": True},
    # A3: GDV only (no VSKO, no RW)
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "kernel", "no_vsko": True, "no_rw": True},
    # A4: RW only (no VSKO, no GDV)
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "kernel", "no_vsko": True, "no_gdv": True},
    # A5: full kernel features (VSKO+GDV+RW) — current best
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "kernel"},
    # A6: Lauri et al. (2023) handcrafted features only
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "lauri"},
    # A7: kernel + Lauri combined (full 115 dims)
    {"datasets": "synthetic:hard", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85,
     "feature_set": "both"},
]

# ── Group S: sensitivity sweep — fix_threshold and prune_threshold ───────────
# Finer-grained sweep for sensitivity curves in the paper
GROUP_SENSITIVITY = (
    # fix_threshold sweep (prune_threshold fixed at default 0.10)
    [{"datasets": "synthetic:hard", "pruner": "ml",
      "prune_threshold": 0.10, "fix_threshold": ft}
     for ft in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    +
    # prune_threshold sweep (no fix_threshold)
    [{"datasets": "synthetic:hard", "pruner": "ml",
      "prune_threshold": pt}
     for pt in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]]
)

# ── Group R: real-world graphs ───────────────────────────────────────────────
GROUP_REALWORLD = [
    {"datasets": "realworld", "pruner": "none"},
    {"datasets": "realworld", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85},
    {"datasets": "realworld", "pruner": "structural"},
    # Train on hard synthetic, evaluate on real-world (cross-domain)
    {"datasets": "synthetic:hard,realworld", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85},
]

# ── Multi-seed: ALL ablation conditions A1-A7 × MULTI_SEEDS ─────────────────
# Exactly 5 seeds per condition → 35 ablation jobs + 5 real-world jobs = 40 total.
# This is the authoritative job list for statistical analysis.
# Seeds: [42, 123, 456, 789, 1337] — each config runs once per seed.
MULTI_SEED_CONFIGS = GROUP_ABLATION + [
    # Real-world best config (cross-domain generalisation check)
    {"datasets": "realworld", "pruner": "ml",
     "prune_threshold": 0.10, "fix_threshold": 0.85},
]

# Datasets that make experiments slow — filtered with --skip-large
LARGE_DATASETS = {"synthetic:large"}

ALL_GROUPS = {
    "1":           GROUP1,
    "2":           GROUP2,
    "2b":          GROUP2B,
    "2c":          GROUP2C,
    "3":           GROUP3,
    "4":           GROUP4,
    "5":           GROUP5,
    "sensitivity": GROUP_SENSITIVITY,
    "realworld":   GROUP_REALWORLD,
    # "ablation" is handled specially in main() — always expanded to multi-seed
}


def _build_cmd(exp: dict, python: str, seed: int = 42) -> list[str]:
    cmd = [python, "train.py", "--skip-cv",
           "--datasets", exp["datasets"],
           "--seed", str(seed)]
    if "max_graphs" in exp:
        cmd += ["--max-graphs", str(exp["max_graphs"])]
    if "pruner" in exp:
        cmd += ["--pruner", exp["pruner"]]
    if "ilp_solver" in exp:
        cmd += ["--ilp-solver", exp["ilp_solver"]]
    if "feature_set" in exp:
        cmd += ["--feature-set", exp["feature_set"]]
    if "prune_threshold" in exp:
        cmd += ["--prune-threshold", str(exp["prune_threshold"])]
    if "fix_threshold" in exp:
        cmd += ["--fix-threshold", str(exp["fix_threshold"])]
    if "eval_min_nodes" in exp:
        cmd += ["--eval-min-nodes", str(exp["eval_min_nodes"])]
    return cmd


def _exp_label(exp: dict, seed: int = 42) -> str:
    label = exp["datasets"]
    if "pruner" in exp:
        label += f"  [pruner={exp['pruner']}]"
    if "ilp_solver" in exp:
        label += f"  [solver={exp['ilp_solver']}]"
    if "feature_set" in exp:
        label += f"  [feat={exp['feature_set']}]"
    if exp.get("no_vsko") or exp.get("no_gdv") or exp.get("no_rw"):
        active = []
        if not exp.get("no_vsko"): active.append("vsko")
        if not exp.get("no_gdv"):  active.append("gdv")
        if not exp.get("no_rw"):   active.append("rw")
        label += "  [kernel=" + "+".join(active) + "-only]"
    if "prune_threshold" in exp:
        label += f"  [pt={exp['prune_threshold']}]"
    if "fix_threshold" in exp:
        label += f"  [ft={exp['fix_threshold']}]"
    if seed != 42:
        label += f"  [seed={seed}]"
    return label


def run_job(args: tuple) -> int:
    exp, python, dry_run, seed = args
    cmd = _build_cmd(exp, python, seed=seed)
    label = _exp_label(exp, seed)

    # Extra flags that affect the command but not the label
    if exp.get("no_vsko"):
        cmd += ["--no-vsko"]
    if exp.get("no_gdv"):
        cmd += ["--no-gdv"]
    if exp.get("no_rw"):
        cmd += ["--no-rw"]

    print(f"[run_all] START  {label}", flush=True)
    if dry_run:
        print(f"  CMD: {' '.join(cmd)}", flush=True)
        return 0

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    code = result.returncode
    status = "OK" if code == 0 else f"FAILED (exit {code})"
    print(f"[run_all] DONE   {label} → {status}", flush=True)
    return code


def _expand_multi_seed(configs: list, seeds: list) -> list:
    """Expand each config into one job per seed."""
    return [(exp, seed) for exp in configs for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MVC training experiments")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent jobs (default: 1).")
    parser.add_argument("--group", type=str, default=None,
                        help="Run only this group: 1, 2, 2b, 2c, 3, 4, 5, "
                             "ablation, sensitivity, realworld, multiseed. "
                             "Default: all groups.")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip experiments that include synthetic:large")
    parser.add_argument("--skip-realworld", action="store_true",
                        help="Skip real-world experiments (if data not available)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed group "
                             "(default: 42,123,456,789,1337)")
    args = parser.parse_args()

    python = sys.executable
    seeds = (
        [int(s) for s in args.seeds.split(",")]
        if args.seeds
        else MULTI_SEEDS
    )

    # Build job list: (exp_dict, seed)
    job_pairs: list[tuple[dict, int]] = []

    if args.group in ("ablation", "multiseed"):
        # Ablation always runs multi-seed (A1-A7 × 5 seeds = 35 jobs)
        job_pairs = _expand_multi_seed(MULTI_SEED_CONFIGS, seeds)
    elif args.group is not None:
        if args.group not in ALL_GROUPS:
            print(f"[run_all] Unknown group '{args.group}'. "
                  f"Valid: {', '.join(sorted(ALL_GROUPS))}, ablation, multiseed")
            sys.exit(1)
        job_pairs = [(exp, 42) for exp in ALL_GROUPS[args.group]]
    else:
        # All groups in order; ablation handled below as multi-seed
        for grp in ["1", "2", "2b", "2c", "sensitivity", "realworld", "3", "4", "5"]:
            if grp == "realworld" and args.skip_realworld:
                continue
            job_pairs += [(exp, 42) for exp in ALL_GROUPS[grp]]
        # Multi-seed ablation: A1-A7 × MULTI_SEEDS (includes seed=42)
        job_pairs += _expand_multi_seed(MULTI_SEED_CONFIGS, seeds)

    if args.skip_large:
        job_pairs = [
            (exp, s) for exp, s in job_pairs
            if not any(ld in exp["datasets"] for ld in LARGE_DATASETS)
        ]

    if args.skip_realworld:
        job_pairs = [
            (exp, s) for exp, s in job_pairs
            if "realworld" not in exp["datasets"]
        ]

    jobs = [(exp, python, args.dry_run, seed) for exp, seed in job_pairs]
    print(f"[run_all] Total experiments: {len(jobs)}")
    if args.dry_run:
        print("[run_all] DRY RUN — commands only\n")

    workers = min(args.workers, len(jobs))
    if workers <= 1:
        results = [run_job(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(run_job, jobs)

    failed = [(_exp_label(job_pairs[i][0], job_pairs[i][1]), c)
              for i, c in enumerate(results) if c != 0]
    print(f"\n[run_all] All done. "
          f"{len(results) - len(failed)}/{len(results)} succeeded.")
    if failed:
        print("[run_all] Failed:")
        for label, code in failed:
            print(f"  {label}  (exit {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()

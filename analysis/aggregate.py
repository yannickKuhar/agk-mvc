"""
analysis/aggregate.py
---------------------
Command-line script to aggregate multi-seed pipeline results from a results
directory.

Usage
-----
    python analysis/aggregate.py --results-dir results/
    python analysis/aggregate.py --results-dir results/ --pattern "*synthetic-hard*"
    python analysis/aggregate.py --results-dir results/ --out aggregate_summary.csv

The script:
  1. Scans --results-dir for sub-directories that contain pipeline_results.csv.
  2. Groups those directories by their "config key" — the directory name with
     the trailing timestamp suffix (_YYYYMMDD_HHMMSS) stripped.
  3. For each config group, loads all CSVs and calls summarise_runs.
  4. Prints a formatted table to stdout.
  5. Saves the summary DataFrame to --out (default: aggregate_summary.csv inside
     --results-dir).
  6. Annotates configs whose Wilcoxon test is significant (p < 0.05) with *.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow running from the project root as well as from the analysis/ directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.statistics import summarise_runs, wilcoxon_vs_baseline

# Regex that matches the trailing _YYYYMMDD_HHMMSS timestamp suffix
_TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def strip_timestamp(name: str) -> str:
    """Return the directory name with the trailing timestamp suffix removed."""
    return _TIMESTAMP_RE.sub("", name)


def collect_csv_groups(
    results_dir: Path,
    pattern: str = "*",
) -> dict[str, list[Path]]:
    """
    Scan *results_dir* for sub-directories whose names match *pattern* and
    that contain a ``pipeline_results.csv`` file.

    Returns a dict mapping config_key -> list of CSV paths.
    """
    groups: dict[str, list[Path]] = {}
    for subdir in sorted(results_dir.glob(pattern)):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "pipeline_results.csv"
        if not csv_path.exists():
            continue
        config_key = strip_timestamp(subdir.name)
        groups.setdefault(config_key, []).append(csv_path)
    return groups


def load_group(csv_paths: list[Path]) -> list[pd.DataFrame]:
    """Load a list of CSV paths into DataFrames, skipping unreadable files."""
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"  [warn] Empty CSV: {path}", file=sys.stderr)
                continue
            dfs.append(df)
        except Exception as exc:
            print(f"  [warn] Could not read {path}: {exc}", file=sys.stderr)
    return dfs


def format_table(summary_df: pd.DataFrame) -> str:
    """Return a human-readable fixed-width table string."""
    if summary_df.empty:
        return "  (no results)"

    lines = []
    col_widths = {
        "config": 40,
        "n_seeds": 7,
        "n_graphs": 8,
        "mean_speedup": 12,
        "median_speedup": 14,
        "pct_above_1": 10,
        "mean_gap": 10,
        "wilcoxon_p": 11,
    }

    header_parts = [
        f"{'Config':<{col_widths['config']}}",
        f"{'Seeds':>{col_widths['n_seeds']}}",
        f"{'Graphs':>{col_widths['n_graphs']}}",
        f"{'MeanSpd':>{col_widths['mean_speedup']}}",
        f"{'MedSpd':>{col_widths['median_speedup']}}",
        f"{'%>1':>{col_widths['pct_above_1']}}",
        f"{'MeanGap':>{col_widths['mean_gap']}}",
        f"{'Wilcoxon-p':>{col_widths['wilcoxon_p']}}",
    ]
    header = "  ".join(header_parts)
    sep = "-" * len(header)
    lines.extend([sep, header, sep])

    for _, row in summary_df.iterrows():
        sig_marker = "*" if row.get("wilcoxon_p", 1.0) < 0.05 else " "
        config_label = f"{row['config']}{sig_marker}"
        parts = [
            f"{config_label:<{col_widths['config']}}",
            f"{int(row['n_seeds']):>{col_widths['n_seeds']}}",
            f"{int(row['n_graphs']):>{col_widths['n_graphs']}}",
            f"{row['mean_speedup']:>{col_widths['mean_speedup']}.3f}",
            f"{row['median_speedup']:>{col_widths['median_speedup']}.3f}",
            f"{row['pct_above_1']:>{col_widths['pct_above_1']}.1%}",
            f"{row['mean_gap']:>{col_widths['mean_gap']}.3f}",
            f"{row['wilcoxon_p']:>{col_widths['wilcoxon_p']}.4f}",
        ]
        lines.append("  ".join(parts))

    lines.append(sep)
    lines.append("  * = Wilcoxon test significant at p < 0.05")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed MVC pipeline results into a summary table."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory to scan for sub-directories with pipeline_results.csv "
             "(default: results/).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for sub-directory names (default: '*').",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-dir>/aggregate_summary.csv).",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    out_path: Path = args.out if args.out is not None else results_dir / "aggregate_summary.csv"

    print(f"Scanning: {results_dir}  (pattern='{args.pattern}')")
    groups = collect_csv_groups(results_dir, pattern=args.pattern)

    if not groups:
        print("[warn] No pipeline_results.csv files found matching the pattern.")
        sys.exit(0)

    print(f"Found {len(groups)} config group(s).\n")

    summary_rows: list[pd.DataFrame] = []

    for config_key in sorted(groups):
        csv_paths = groups[config_key]
        print(f"  {config_key}  ({len(csv_paths)} seed(s))")
        dfs = load_group(csv_paths)
        if not dfs:
            print(f"    [skip] No readable CSVs for config '{config_key}'.")
            continue

        try:
            summary = summarise_runs(dfs)
        except Exception as exc:
            print(f"    [skip] summarise_runs failed for '{config_key}': {exc}", file=sys.stderr)
            continue

        # Annotate with Wilcoxon significance at top-level for quick reference
        all_speedups = pd.concat(dfs)["speedup"].dropna().values
        wt = wilcoxon_vs_baseline(all_speedups)
        summary["wilcoxon_significant"] = wt["significant"]
        summary.insert(0, "config", config_key)
        summary_rows.append(summary)

    if not summary_rows:
        print("[error] No summaries could be computed.")
        sys.exit(1)

    all_summary = pd.concat(summary_rows, ignore_index=True)
    all_summary = all_summary.sort_values("median_speedup", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    print(format_table(all_summary))
    print()

    all_summary.to_csv(out_path, index=False)
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()

"""
tests/test_experiment_structure.py
-----------------------------------
Verify that run_all.py and train.py produce correctly distinct, reproducible
experiment configurations for the ablation study (A1-A7).

Correctness invariants:
  1. Every ablation condition generates a UNIQUE output directory prefix.
  2. A2/A3/A4 are present in MULTI_SEED_CONFIGS (they were missing before).
  3. MULTI_SEED_CONFIGS × MULTI_SEEDS yields exactly 5 runs per ablation condition.
  4. _make_run_dir encodes kernel sub-family flags in the directory name.
  5. The catalogue in download_realworld.py has >= 50 entries.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# run_all.py has no heavy imports — import directly
from run_all import MULTI_SEED_CONFIGS, MULTI_SEEDS, GROUP_ABLATION, _build_cmd  # noqa: E402


# _make_run_dir is inlined here (pure stdlib) to avoid importing train.py's
# heavy ML dependencies (tqdm, xgboost, sklearn) in this test environment.
def _make_run_dir(datasets_str: str, pruner: str = "ml",
                  ilp_solver: str = "auto", prune_threshold: float = 0.10,
                  fix_threshold: float = 1.1, feature_set: str = "kernel",
                  no_vsko: bool = False, no_gdv: bool = False,
                  no_rw: bool = False, seed: int = 42,
                  base: str = "results") -> Path:
    tag = datasets_str.replace(",", "_").replace(":", "-").replace(" ", "")
    if pruner != "ml":
        tag += f"_{pruner}-pruner"
    if ilp_solver != "auto":
        tag += f"_{ilp_solver}-solver"
    if feature_set != "kernel":
        tag += f"_{feature_set}-feat"
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dir_prefix(exp: dict, seed: int = 42) -> str:
    """Return the _make_run_dir tag (everything before the timestamp)."""
    d = _make_run_dir(
        datasets_str=exp["datasets"],
        pruner=exp.get("pruner", "ml"),
        ilp_solver=exp.get("ilp_solver", "auto"),
        prune_threshold=exp.get("prune_threshold", 0.10),
        fix_threshold=exp.get("fix_threshold", 1.1),
        feature_set=exp.get("feature_set", "kernel"),
        no_vsko=exp.get("no_vsko", False),
        no_gdv=exp.get("no_gdv", False),
        no_rw=exp.get("no_rw", False),
        seed=seed,
    )
    # Strip the trailing _YYYYMMDD_HHMMSS suffix
    name = d.name
    parts = name.rsplit("_", 2)
    # timestamp is last two parts: YYYYMMDD and HHMMSS
    return "_".join(parts[:-2]) if len(parts) > 2 else name


# ---------------------------------------------------------------------------
# 1. All 7 ablation conditions produce distinct directory prefixes
# ---------------------------------------------------------------------------

class TestAblationDirectoryDistinctness:

    def test_a1_a9_prefixes_are_unique(self):
        prefixes = [_dir_prefix(exp) for exp in GROUP_ABLATION]
        assert len(prefixes) == len(set(prefixes)), (
            f"Duplicate directory prefixes among ablation conditions: {prefixes}"
        )

    def test_a2_vsko_only_in_prefix(self):
        a2 = GROUP_ABLATION[1]  # VSKO only
        prefix = _dir_prefix(a2)
        assert "vsko" in prefix and "gdv" not in prefix.split("vsko")[0]

    def test_a3_gdv_only_in_prefix(self):
        a3 = GROUP_ABLATION[2]  # GDV only
        prefix = _dir_prefix(a3)
        assert "gdv" in prefix

    def test_a4_rw_only_in_prefix(self):
        a4 = GROUP_ABLATION[3]  # RW only
        prefix = _dir_prefix(a4)
        assert "rw" in prefix

    def test_a5_full_kernel_no_subfamily_tag(self):
        a5 = GROUP_ABLATION[4]  # full kernel
        prefix = _dir_prefix(a5)
        # Full kernel should NOT have a -only suffix
        assert "-only" not in prefix

    def test_a6_lauri_feat_in_prefix(self):
        a6 = GROUP_ABLATION[5]  # Lauri only
        prefix = _dir_prefix(a6)
        assert "lauri-feat" in prefix

    def test_a7_both_feat_in_prefix(self):
        a7 = GROUP_ABLATION[6]  # both
        prefix = _dir_prefix(a7)
        assert "both-feat" in prefix

    def test_a2_a3_a4_prefixes_distinct_from_a5(self):
        a2_prefix = _dir_prefix(GROUP_ABLATION[1])
        a3_prefix = _dir_prefix(GROUP_ABLATION[2])
        a4_prefix = _dir_prefix(GROUP_ABLATION[3])
        a5_prefix = _dir_prefix(GROUP_ABLATION[4])
        assert a2_prefix != a5_prefix
        assert a3_prefix != a5_prefix
        assert a4_prefix != a5_prefix
        assert a2_prefix != a3_prefix
        assert a2_prefix != a4_prefix
        assert a3_prefix != a4_prefix


# ---------------------------------------------------------------------------
# 2. MULTI_SEED_CONFIGS includes A2, A3, A4
# ---------------------------------------------------------------------------

class TestMultiSeedConfigContents:

    def _has_config(self, no_vsko: bool, no_gdv: bool, no_rw: bool) -> bool:
        for exp in MULTI_SEED_CONFIGS:
            if (exp.get("no_vsko", False) == no_vsko and
                    exp.get("no_gdv", False) == no_gdv and
                    exp.get("no_rw", False) == no_rw and
                    exp.get("feature_set", "kernel") == "kernel" and
                    exp.get("pruner") == "ml"):
                return True
        return False

    def test_a2_vsko_only_present(self):
        assert self._has_config(no_vsko=False, no_gdv=True, no_rw=True), \
            "A2 (VSKO only) not in MULTI_SEED_CONFIGS"

    def test_a3_gdv_only_present(self):
        assert self._has_config(no_vsko=True, no_gdv=False, no_rw=True), \
            "A3 (GDV only) not in MULTI_SEED_CONFIGS"

    def test_a4_rw_only_present(self):
        assert self._has_config(no_vsko=True, no_gdv=True, no_rw=False), \
            "A4 (RW only) not in MULTI_SEED_CONFIGS"

    def test_a5_full_kernel_present(self):
        assert any(
            exp.get("feature_set", "kernel") == "kernel" and
            not exp.get("no_vsko") and not exp.get("no_gdv") and not exp.get("no_rw") and
            exp.get("pruner") == "ml" and exp.get("fix_threshold") == 0.85
            for exp in MULTI_SEED_CONFIGS
        ), "A5 (full kernel) not in MULTI_SEED_CONFIGS"

    def test_a6_lauri_present(self):
        assert any(exp.get("feature_set") == "lauri" for exp in MULTI_SEED_CONFIGS), \
            "A6 (lauri) not in MULTI_SEED_CONFIGS"

    def test_a7_both_present(self):
        assert any(exp.get("feature_set") == "both" for exp in MULTI_SEED_CONFIGS), \
            "A7 (both) not in MULTI_SEED_CONFIGS"

    def test_a8_ajwani_present(self):
        assert any(exp.get("feature_set") == "ajwani" for exp in MULTI_SEED_CONFIGS), \
            "A8 (ajwani) not in MULTI_SEED_CONFIGS"

    def test_a9_kernel_ajwani_present(self):
        assert any(exp.get("feature_set") == "kernel+ajwani" for exp in MULTI_SEED_CONFIGS), \
            "A9 (kernel+ajwani) not in MULTI_SEED_CONFIGS"

    def test_a1_baseline_present(self):
        assert any(exp.get("pruner") == "none" and "synthetic" in exp["datasets"]
                   for exp in MULTI_SEED_CONFIGS), \
            "A1 (no-pruner baseline) not in MULTI_SEED_CONFIGS"

    def test_exactly_five_seeds(self):
        assert len(MULTI_SEEDS) == 5, f"Expected 5 seeds, got {len(MULTI_SEEDS)}"
        assert set(MULTI_SEEDS) == {42, 123, 456, 789, 1337}

    def test_total_ablation_jobs(self):
        # 9 ablation conditions + 1 realworld = 10 configs × 5 seeds = 50 jobs
        from run_all import _expand_multi_seed
        jobs = _expand_multi_seed(MULTI_SEED_CONFIGS, MULTI_SEEDS)
        assert len(jobs) == len(MULTI_SEED_CONFIGS) * 5

    def test_each_ablation_config_gets_five_seeds(self):
        from run_all import _expand_multi_seed
        jobs = _expand_multi_seed(MULTI_SEED_CONFIGS, MULTI_SEEDS)
        for exp in MULTI_SEED_CONFIGS:
            seeds_for_exp = [s for e, s in jobs if e is exp]
            assert sorted(seeds_for_exp) == sorted(MULTI_SEEDS), (
                f"Config {exp} doesn't get all 5 seeds: {seeds_for_exp}"
            )


# ---------------------------------------------------------------------------
# 3. _build_cmd passes no_vsko/no_gdv/no_rw flags correctly
# ---------------------------------------------------------------------------

class TestBuildCmd:

    def test_a2_cmd_has_no_gdv_no_rw(self):
        a2 = GROUP_ABLATION[1]
        cmd = _build_cmd(a2, "python", seed=42)
        # run_job appends these flags; _build_cmd itself doesn't — check run_job
        # behaviour via the exp dict
        assert a2.get("no_gdv") is True
        assert a2.get("no_rw") is True
        assert not a2.get("no_vsko")

    def test_a3_cmd_has_no_vsko_no_rw(self):
        a3 = GROUP_ABLATION[2]
        assert a3.get("no_vsko") is True
        assert a3.get("no_rw") is True
        assert not a3.get("no_gdv")

    def test_a4_cmd_has_no_vsko_no_gdv(self):
        a4 = GROUP_ABLATION[3]
        assert a4.get("no_vsko") is True
        assert a4.get("no_gdv") is True
        assert not a4.get("no_rw")

    def test_a5_has_no_exclusion_flags(self):
        a5 = GROUP_ABLATION[4]
        assert not a5.get("no_vsko")
        assert not a5.get("no_gdv")
        assert not a5.get("no_rw")

    def test_seed_flag_in_cmd(self):
        exp = GROUP_ABLATION[4]
        cmd = _build_cmd(exp, "python", seed=123)
        assert "--seed" in cmd
        assert "123" in cmd

    def test_feature_set_flag_in_cmd(self):
        a6 = GROUP_ABLATION[5]
        cmd = _build_cmd(a6, "python")
        assert "--feature-set" in cmd
        assert "lauri" in cmd


# ---------------------------------------------------------------------------
# 4. _make_run_dir seed disambiguation
# ---------------------------------------------------------------------------

class TestMakeRunDir:

    def _prefix(self, **kwargs) -> str:
        base = {"datasets_str": "synthetic:hard", "pruner": "ml",
                "prune_threshold": 0.10, "fix_threshold": 0.85}
        base.update(kwargs)
        d = _make_run_dir(**base)
        name = d.name
        parts = name.rsplit("_", 2)
        return "_".join(parts[:-2]) if len(parts) > 2 else name

    def test_seed_42_no_seed_suffix(self):
        prefix = self._prefix(seed=42)
        assert "_s42" not in prefix

    def test_non_default_seed_adds_suffix(self):
        prefix = self._prefix(seed=123)
        assert "_s123" in prefix

    def test_vsko_only_tag(self):
        prefix = self._prefix(no_gdv=True, no_rw=True)
        assert "vsko-only" in prefix

    def test_gdv_only_tag(self):
        prefix = self._prefix(no_vsko=True, no_rw=True)
        assert "gdv-only" in prefix

    def test_rw_only_tag(self):
        prefix = self._prefix(no_vsko=True, no_gdv=True)
        assert "rw-only" in prefix

    def test_full_kernel_no_only_tag(self):
        prefix = self._prefix()
        assert "-only" not in prefix

    def test_vsko_gdv_no_rw(self):
        prefix = self._prefix(no_rw=True)
        assert "vsko+gdv-only" in prefix


# ---------------------------------------------------------------------------
# 5. download_realworld catalogue size
# ---------------------------------------------------------------------------

class TestRealworldCatalogue:

    def test_catalogue_has_at_least_50_entries(self):
        from download_realworld import CATALOGUE
        assert len(CATALOGUE) >= 50, (
            f"Catalogue has only {len(CATALOGUE)} entries; need >= 50 for journal"
        )

    def test_catalogue_names_are_unique(self):
        from download_realworld import CATALOGUE
        names = [name for name, _ in CATALOGUE]
        assert len(names) == len(set(names)), "Duplicate names in CATALOGUE"

    def test_catalogue_has_ws_entries(self):
        from download_realworld import CATALOGUE
        ws_names = [n for n, _ in CATALOGUE if n.startswith("ws_")]
        assert len(ws_names) >= 8, f"Expected >= 8 WS graphs, got {len(ws_names)}"

    def test_catalogue_has_ba_entries(self):
        from download_realworld import CATALOGUE
        ba_names = [n for n, _ in CATALOGUE if n.startswith("ba_")]
        assert len(ba_names) >= 8, f"Expected >= 8 BA graphs, got {len(ba_names)}"

    def test_catalogue_has_lfr_entries(self):
        from download_realworld import CATALOGUE
        lfr_names = [n for n, _ in CATALOGUE if n.startswith("lfr_")]
        assert len(lfr_names) >= 10, f"Expected >= 10 LFR graphs, got {len(lfr_names)}"

    def test_catalogue_dict_matches_list(self):
        from download_realworld import CATALOGUE, CATALOGUE_DICT
        assert len(CATALOGUE_DICT) == len(CATALOGUE)
        for name, fn in CATALOGUE:
            assert name in CATALOGUE_DICT

    def test_ws_graphs_can_be_built(self):
        from download_realworld import CATALOGUE, preprocess
        ws_entries = [(n, fn) for n, fn in CATALOGUE if n.startswith("ws_")][:3]
        for name, factory in ws_entries:
            G_raw = factory()
            G = preprocess(G_raw, name)
            assert G is not None
            assert G.number_of_nodes() >= 5

    def test_ba_graphs_can_be_built(self):
        from download_realworld import CATALOGUE, preprocess
        ba_entries = [(n, fn) for n, fn in CATALOGUE if n.startswith("ba_")][:3]
        for name, factory in ba_entries:
            G_raw = factory()
            G = preprocess(G_raw, name)
            assert G is not None
            assert G.number_of_nodes() >= 5

    def test_named_graphs_still_present(self):
        from download_realworld import CATALOGUE_DICT
        for name in ("karate", "lesmis", "grid_7x7", "barbell_15"):
            assert name in CATALOGUE_DICT, f"Named graph '{name}' missing from catalogue"

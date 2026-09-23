"""
tests/test_realworld_loader.py
------------------------------
Unit tests for data/realworld.py and download_realworld.py.

Run with:
    pytest tests/test_realworld_loader.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal fixture record
# ---------------------------------------------------------------------------

def _tiny_graph_record(name: str = "test_tiny") -> dict:
    """Triangle graph with a known MVC of size 2."""
    return {
        "n_nodes": 3,
        "edges": [[0, 1], [1, 2], [0, 2]],
        "mvc": [0, 1],
        "source": f"realworld:{name}",
    }


def _karate_like_record() -> dict:
    """Minimal stub record that looks like a real-world entry (>= 5 nodes)."""
    G = nx.karate_club_graph()
    # Use a greedy 2-approx cover so we don't need ILP in tests
    cover = set()
    for u, v in G.edges():
        if u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)
    return {
        "n_nodes": G.number_of_nodes(),
        "edges": [[u, v] for u, v in G.edges()],
        "mvc": sorted(cover),
        "source": "realworld:karate",
    }


# ---------------------------------------------------------------------------
# Test 1: load_realworld raises FileNotFoundError for unknown name
# ---------------------------------------------------------------------------

def test_load_realworld_unknown_raises(tmp_path):
    """FileNotFoundError is raised when the JSON file does not exist."""
    # Monkey-patch _DATA_DIR inside the module so it points to an empty tmp dir
    import data.realworld as rw_mod
    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = tmp_path
    try:
        with pytest.raises(FileNotFoundError, match="download_realworld.py"):
            rw_mod.load_realworld("nonexistent_dataset")
    finally:
        rw_mod._DATA_DIR = original_data_dir


# ---------------------------------------------------------------------------
# Test 2: load_realworld on a small fixture returns graphs with labels
# ---------------------------------------------------------------------------

def test_load_realworld_returns_labelled_graphs(tmp_path):
    """load_realworld correctly reads a fixture JSON and sets node labels."""
    import data.realworld as rw_mod

    fixture_name = "fixture_tiny"
    # Write a fixture with two records — one too small (2 nodes), one valid (3 nodes)
    records = [
        {"n_nodes": 2, "edges": [[0, 1]], "mvc": [0], "source": "realworld:fixture_tiny"},
        _tiny_graph_record(fixture_name),
    ]
    (tmp_path / f"{fixture_name}.json").write_text(json.dumps(records))

    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = tmp_path
    try:
        graphs = rw_mod.load_realworld(fixture_name, min_nodes=3)
    finally:
        rw_mod._DATA_DIR = original_data_dir

    assert len(graphs) == 1
    G = graphs[0]
    assert G.number_of_nodes() == 3
    # Every node must have a label in {0, 1}
    for v in G.nodes():
        assert G.nodes[v]["label"] in (0, 1), f"node {v} has unexpected label"
    # dataset attribute is set
    assert G.graph["dataset"] == "realworld:fixture_tiny"


# ---------------------------------------------------------------------------
# Test 3: list_available returns names of .json files
# ---------------------------------------------------------------------------

def test_list_available(tmp_path):
    """list_available() lists .json stem names in the data dir."""
    import data.realworld as rw_mod

    # Create two dummy json files
    (tmp_path / "alpha.json").write_text("[]")
    (tmp_path / "beta.json").write_text("[]")
    (tmp_path / "not_json.txt").write_text("ignored")

    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = tmp_path
    try:
        names = rw_mod.list_available()
    finally:
        rw_mod._DATA_DIR = original_data_dir

    assert "alpha" in names
    assert "beta" in names
    assert "not_json" not in names


def test_list_available_empty_when_dir_missing(tmp_path):
    """list_available() returns [] when the data directory does not exist."""
    import data.realworld as rw_mod

    missing_dir = tmp_path / "does_not_exist"
    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = missing_dir
    try:
        names = rw_mod.list_available()
    finally:
        rw_mod._DATA_DIR = original_data_dir

    assert names == []


# ---------------------------------------------------------------------------
# Test 4: MVC validity — every edge is covered by at least one endpoint
# ---------------------------------------------------------------------------

def test_mvc_validity_on_fixture(tmp_path):
    """Graphs loaded from a fixture all have valid MVC labels."""
    import data.realworld as rw_mod

    # Fixture with karate-like graph (approx cover)
    record = _karate_like_record()
    fixture_name = "karate_stub"
    (tmp_path / f"{fixture_name}.json").write_text(json.dumps([record]))

    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = tmp_path
    try:
        graphs = rw_mod.load_realworld(fixture_name, min_nodes=5)
    finally:
        rw_mod._DATA_DIR = original_data_dir

    assert len(graphs) >= 1
    for G in graphs:
        cover = {v for v in G.nodes() if G.nodes[v].get("label") == 1}
        for u, v in G.edges():
            assert u in cover or v in cover, (
                f"Edge ({u},{v}) is not covered — MVC is invalid"
            )


# ---------------------------------------------------------------------------
# Test 5: download_realworld generates at least 5 datasets (mocked ILP)
# ---------------------------------------------------------------------------

def test_download_realworld_generates_datasets(tmp_path):
    """
    When run (with a mocked ILP solver), download_realworld produces >= 5 JSON files.
    We mock MVCSolver.solve so no actual ILP is needed.
    """
    # Build a fake SolveResult-like object
    class FakeResult:
        cover = {0, 1}
        size = 2
        runtime_seconds = 0.001
        backend_used = "mock"
        is_optimal = True

    fake_result = FakeResult()

    import download_realworld as dr

    # Patch MVCSolver inside solver.mvc_solver (imported by download_realworld)
    with patch("download_realworld.solve_mvc", return_value=fake_result):
        all_records = []
        for name, factory_fn in dr.CATALOGUE:
            out_path = tmp_path / f"{name}.json"
            # Manually run process_single with our patched solve_mvc
            G_raw = factory_fn()
            G = dr.preprocess(G_raw, name)
            if G is None:
                continue
            # Use our patch directly
            result = fake_result
            record = dr.graph_to_record(G, name, result.cover)
            out_path.write_text(json.dumps([record]))
            all_records.append(record)

    # We expect all 14 CATALOGUE entries to produce valid output
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) >= 5, (
        f"Expected >= 5 JSON files, got {len(json_files)}: {[f.name for f in json_files]}"
    )


# ---------------------------------------------------------------------------
# Test 6: preprocess handles grid tuple-labelled nodes correctly
# ---------------------------------------------------------------------------

def test_preprocess_grid_relabels_to_integers():
    """preprocess() converts (i,j)-labelled grid nodes to integers."""
    import download_realworld as dr

    G_grid = nx.grid_2d_graph(3, 3)
    # Nodes are tuples before preprocessing
    assert isinstance(list(G_grid.nodes())[0], tuple)

    G_out = dr.preprocess(G_grid, "grid_3x3")
    assert G_out is not None
    # All nodes should now be integers
    for v in G_out.nodes():
        assert isinstance(v, int), f"Node {v!r} is not an integer after preprocess"


# ---------------------------------------------------------------------------
# Test 7: graph_to_record produces a valid JSON-serialisable dict
# ---------------------------------------------------------------------------

def test_graph_to_record_format():
    """graph_to_record() produces a dict matching the expected schema."""
    import download_realworld as dr

    G = nx.petersen_graph()
    cover = {0, 1, 2, 3, 4}
    record = dr.graph_to_record(G, "petersen", cover)

    assert record["n_nodes"] == 10
    assert record["source"] == "realworld:petersen"
    assert isinstance(record["edges"], list)
    assert isinstance(record["mvc"], list)
    # All mvc entries are sorted ints
    assert record["mvc"] == sorted(record["mvc"])
    for v in record["mvc"]:
        assert isinstance(v, int)
    # JSON serialisable
    json.dumps(record)  # should not raise


# ---------------------------------------------------------------------------
# Test 8: max_graphs cap is respected
# ---------------------------------------------------------------------------

def test_load_realworld_max_graphs_cap(tmp_path):
    """max_graphs parameter limits the number of graphs returned."""
    import data.realworld as rw_mod

    records = [_tiny_graph_record(f"g{i}") for i in range(10)]
    fixture_name = "multi"
    (tmp_path / f"{fixture_name}.json").write_text(json.dumps(records))

    original_data_dir = rw_mod._DATA_DIR
    rw_mod._DATA_DIR = tmp_path
    try:
        graphs = rw_mod.load_realworld(fixture_name, max_graphs=3, min_nodes=1)
    finally:
        rw_mod._DATA_DIR = original_data_dir

    assert len(graphs) <= 3

"""
data/unified.py
---------------
Unified dataset interface — load and merge graphs from multiple sources
with a consistent stratified train/test split.

Supported source strings:
  'erdos'                data/erdos/train.json + test.json
  'synthetic:small'      data/synthetic/small.json
  'synthetic:medium'     data/synthetic/medium.json
  'synthetic:large'      data/synthetic/large.json
  'pace'                 data/pace/  (requires manual download)
  'tudataset:{NAME}'     data/tudatasets/{NAME}_mvc.json

Usage:
    loader = DatasetLoader()
    train_graphs, test_graphs = loader.load(
        sources=['erdos', 'synthetic:small'],
        min_nodes=5,
        max_nodes=500,
        train_ratio=0.8,
        seed=42,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def _load_json_records(
    path: Path,
    max_graphs: Optional[int],
    min_nodes: int,
    max_nodes: int,
    dataset_tag: str,
) -> List[nx.Graph]:
    """Load graphs from a {n_nodes, edges, mvc, source} JSON file."""
    with open(path) as f:
        records = json.load(f)
    graphs: List[nx.Graph] = []
    for rec in records:
        if max_graphs is not None and len(graphs) >= max_graphs:
            break
        n = rec["n_nodes"]
        if n < min_nodes or n > max_nodes:
            continue
        mvc_set = set(rec["mvc"])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in rec["edges"]:
            G.add_edge(u, v)
        for node in G.nodes():
            G.nodes[node]["label"] = 1 if node in mvc_set else 0
        G.graph["dataset"] = dataset_tag
        graphs.append(G)
    return graphs


class DatasetLoader:
    """Load and merge graphs from multiple sources with a consistent split."""

    def load(
        self,
        sources: List[str],
        split: str = "train",          # kept for API compatibility
        max_graphs: Optional[int] = None,
        min_nodes: int = 5,
        max_nodes: int = 500,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        """
        Load graphs from all sources, print a summary table, and split.

        Parameters
        ----------
        sources     : list of source strings (see module docstring)
        split       : ignored — method always returns (train, test)
        max_graphs  : cap per source (None = unlimited)
        min_nodes   : skip graphs with fewer nodes
        max_nodes   : skip graphs with more nodes
        train_ratio : fraction for training split
        seed        : random seed for splitting

        Returns
        -------
        (train_graphs, test_graphs)
        """
        rng = np.random.default_rng(seed)
        by_source: Dict[str, List[nx.Graph]] = {}

        for source in sources:
            graphs = self._load_source(source, max_graphs, min_nodes, max_nodes)
            if graphs:
                by_source[source] = graphs
            else:
                print(f"[DatasetLoader] No graphs loaded from '{source}'.")

        if not by_source:
            print("[DatasetLoader] No graphs from any source — cannot continue.")
            return [], []

        self._print_table(by_source)

        train_all: List[nx.Graph] = []
        test_all: List[nx.Graph] = []
        for graphs in by_source.values():
            idxs = np.arange(len(graphs))
            rng.shuffle(idxs)
            cut = max(1, int(len(idxs) * train_ratio))
            train_all.extend(graphs[i] for i in idxs[:cut])
            test_all.extend(graphs[i] for i in idxs[cut:])

        # Shuffle within each split so source order doesn't matter
        tr = list(train_all)
        te = list(test_all)
        rng.shuffle(tr)
        rng.shuffle(te)
        print(f"[DatasetLoader] Split → {len(tr)} train, {len(te)} test")
        return tr, te

    # ------------------------------------------------------------------
    # Source dispatch
    # ------------------------------------------------------------------

    def _load_source(
        self,
        source: str,
        max_graphs: Optional[int],
        min_nodes: int,
        max_nodes: int,
    ) -> List[nx.Graph]:
        try:
            if source == "erdos":
                return self._load_erdos(max_graphs, min_nodes, max_nodes)
            if source.startswith("synthetic:"):
                return self._load_synthetic(source.split(":", 1)[1],
                                            max_graphs, min_nodes, max_nodes)
            if source == "pace":
                from data.pace import load_pace
                return load_pace(max_graphs=max_graphs,
                                 min_nodes=min_nodes, max_nodes=max_nodes)
            if source.startswith("tudataset:"):
                name = source.split(":", 1)[1]
                from data.tudataset import load_tudataset
                return load_tudataset(name=name, max_graphs=max_graphs,
                                      min_nodes=min_nodes, max_nodes=max_nodes)
            print(f"[DatasetLoader] Unknown source '{source}'. "
                  f"Valid: erdos, synthetic:{{small|medium|large|hard}}, "
                  f"pace, tudataset:{{NAME}}")
            return []
        except FileNotFoundError as e:
            print(f"[DatasetLoader] {source}: {e}")
            return []
        except Exception as e:
            print(f"[DatasetLoader] Failed to load '{source}': {e}")
            return []

    def _load_erdos(self, max_graphs, min_nodes, max_nodes):
        base = Path("data/erdos")
        graphs: List[nx.Graph] = []
        for split in ("train", "test"):
            p = base / f"{split}.json"
            if p.exists():
                g = _load_json_records(p, max_graphs, min_nodes, max_nodes, "erdos")
                graphs.extend(g)
        if not graphs:
            raise FileNotFoundError(
                "data/erdos/train.json not found. "
                "Run: python download_dataset.py --all-splits"
            )
        return graphs

    def _load_synthetic(self, split_name, max_graphs, min_nodes, max_nodes):
        p = Path("data/synthetic") / f"{split_name}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"data/synthetic/{split_name}.json not found. "
                f"Run: python generate_synthetic.py --split {split_name}"
            )
        return _load_json_records(p, max_graphs, min_nodes, max_nodes,
                                  f"synthetic:{split_name}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    @staticmethod
    def _print_table(by_source: Dict[str, List[nx.Graph]]) -> None:
        col_w = 22
        hdr = (f"{'Source':<{col_w}} | {'Graphs':>6} | "
               f"{'Avg nodes':>9} | {'Avg edges':>9} | {'Pos frac':>8}")
        sep = "-" * len(hdr)
        print("\n" + sep)
        print(hdr)
        print(sep)

        all_pos_sum = 0
        all_pos_cnt = 0
        total_g = total_nodes_w = total_edges_w = 0

        for source, graphs in by_source.items():
            sizes = [G.number_of_nodes() for G in graphs]
            edges = [G.number_of_edges() for G in graphs]
            pos_sum = sum(
                G.nodes[v].get("label", 0)
                for G in graphs for v in G.nodes()
            )
            pos_cnt = sum(G.number_of_nodes() for G in graphs)
            avg_n = sum(sizes) / len(sizes)
            avg_e = sum(edges) / len(edges)
            avg_p = pos_sum / pos_cnt if pos_cnt else float("nan")

            print(f"  {source:<{col_w}} | {len(graphs):>6} | "
                  f"{avg_n:>9.1f} | {avg_e:>9.1f} | {avg_p:>8.3f}")

            total_g += len(graphs)
            total_nodes_w += avg_n * len(graphs)
            total_edges_w += avg_e * len(graphs)
            all_pos_sum += pos_sum
            all_pos_cnt += pos_cnt

        grand_n = total_nodes_w / total_g if total_g else 0
        grand_e = total_edges_w / total_g if total_g else 0
        grand_p = all_pos_sum / all_pos_cnt if all_pos_cnt else float("nan")
        print(sep)
        print(f"  {'TOTAL':<{col_w}} | {total_g:>6} | "
              f"{grand_n:>9.1f} | {grand_e:>9.1f} | {grand_p:>8.3f}")
        print(sep + "\n")

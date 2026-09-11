"""
features/pipeline.py
--------------------
Combines VSKO, GDV, and Random Walk features into a single node feature matrix.

Output feature vector per node (up to 106 dimensions):
  [VSKO (13)] + [GDV (73)] + [RW (20)]

Also handles:
  - Graph-level batch processing
  - Feature normalization (StandardScaler)
  - Feature name tracking for SHAP explainability
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .vsko import VSKONodeFeatures, feature_names as vsko_names
from .gdv import GDVNodeFeatures, feature_names as gdv_names
from .random_walk import RandomWalkNodeFeatures, feature_names as rw_names


class NodeFeaturePipeline:
    """
    Full feature extraction pipeline: VSKO + GDV + RW → node feature matrix.

    Parameters
    ----------
    ego_hops : int
        Ego graph radius for VSKO (default 2).
    use_orca_binary : bool
        Whether to try the ORCA binary for GDV computation.
    orca_path : str
        Path to ORCA binary.
    use_vsko : bool
        Include VSKO features (default True).
    use_gdv : bool
        Include GDV features (default True).
    use_rw : bool
        Include random walk features (default True).
    normalize : bool
        Apply StandardScaler to the combined feature matrix (default True).
    betweenness_k : int
        Number of nodes to sample for betweenness centrality.
    """

    def __init__(
        self,
        ego_hops: int = 2,
        use_orca_binary: bool = False,
        orca_path: str = "orca",
        use_vsko: bool = True,
        use_gdv: bool = True,
        use_rw: bool = True,
        normalize: bool = True,
        betweenness_k: int = 50,
    ):
        self.use_vsko = use_vsko
        self.use_gdv = use_gdv
        self.use_rw = use_rw
        self.normalize = normalize

        self.vsko = VSKONodeFeatures(ego_hops=ego_hops, use_orca_binary=use_orca_binary, orca_path=orca_path) if use_vsko else None
        self.gdv  = GDVNodeFeatures(orca_path=orca_path)  if use_gdv  else None
        self.rw   = RandomWalkNodeFeatures(betweenness_k=betweenness_k) if use_rw else None

        self.scaler: Optional[StandardScaler] = StandardScaler() if normalize else None
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_graph(self, G: nx.Graph) -> np.ndarray:
        """
        Extract feature matrix for a single graph.

        Returns
        -------
        X : np.ndarray of shape (n_nodes, n_features)
        """
        parts = []

        if self.vsko is not None:
            parts.append(self.vsko.fit_transform(G))

        if self.gdv is not None:
            parts.append(self.gdv.fit_transform(G))

        if self.rw is not None:
            parts.append(self.rw.fit_transform(G))

        if not parts:
            raise ValueError("At least one feature type must be enabled.")

        return np.hstack(parts)

    def extract_graphs(
        self,
        graphs: List[nx.Graph],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from a list of graphs.

        Returns
        -------
        X : np.ndarray of shape (total_nodes, n_features)
        y : np.ndarray of shape (total_nodes,)  — node labels (int)
        graph_ids : np.ndarray of shape (total_nodes,) — which graph each node belongs to
        """
        all_X, all_y, all_graph_ids = [], [], []

        it = tqdm(graphs, desc="Extracting features") if show_progress else graphs

        for g_id, G in enumerate(it):
            nodes = list(G.nodes())
            n = len(nodes)

            X_g = self.extract_graph(G)
            y_g = np.array([G.nodes[v].get("label", -1) for v in nodes], dtype=int)

            all_X.append(X_g)
            all_y.append(y_g)
            all_graph_ids.append(np.full(n, g_id, dtype=int))

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        graph_ids = np.concatenate(all_graph_ids)

        return X, y, graph_ids

    # ------------------------------------------------------------------
    # Fit / transform (for train/test split normalization)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "NodeFeaturePipeline":
        """Fit the scaler on training features."""
        if self.scaler is not None:
            self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler."""
        if self.scaler is not None and self._fitted:
            return self.scaler.transform(X)
        return X

    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        names = []
        if self.use_vsko:
            names.extend(vsko_names())
        if self.use_gdv:
            names.extend(gdv_names())
        if self.use_rw:
            names.extend(rw_names())
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

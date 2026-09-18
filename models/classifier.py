"""
models/classifier.py
--------------------
XGBoost binary node classifier.

Predicts P(node ∈ optimal MVC) for each node given its feature vector.

Features:
  - Handles class imbalance via scale_pos_weight
  - Cross-validation with early stopping on AUC
  - SHAP-based feature importance for explainability
  - Threshold tuning to maximize F1 on validation set
  - Graph-aware splitting (keeps nodes from same graph in same fold)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MVCNodeClassifier:
    """
    XGBoost classifier for predicting MVC node membership.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default 500; early stopping may reduce this).
    max_depth : int
        Tree depth (default 6).
    learning_rate : float
        XGBoost eta (default 0.05).
    subsample : float
        Row subsampling ratio (default 0.8).
    colsample_bytree : float
        Feature subsampling ratio per tree (default 0.8).
    early_stopping_rounds : int
        Stop if no improvement in validation AUC for this many rounds.
    n_cv_folds : int
        Number of cross-validation folds for evaluation (default 5).
    threshold : float or None
        Decision threshold for binary prediction. None = auto-tune on val set.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 30,
        n_cv_folds: int = 5,
        threshold: Optional[float] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.n_cv_folds = n_cv_folds
        self.threshold = threshold
        self.random_state = random_state

        self.model_: Optional[xgb.XGBClassifier] = None
        self.best_threshold_: float = threshold if threshold is not None else 0.5
        self.feature_names_: Optional[List[str]] = None
        self.cv_results_: List[Dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        graph_ids_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MVCNodeClassifier":
        """
        Train XGBoost classifier.

        Parameters
        ----------
        X_train : (n_train, n_features)
        y_train : (n_train,)  binary labels
        graph_ids_train : (n_train,) — graph ID for each node (for grouped CV)
        feature_names : optional list of feature name strings
        X_val, y_val : optional explicit validation set
        """
        self.feature_names_ = feature_names

        # Filter out unlabeled nodes
        mask = y_train >= 0
        X_train = X_train[mask]
        y_train = y_train[mask]
        graph_ids_train = graph_ids_train[mask]

        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"[classifier] Class balance — neg: {n_neg}, pos: {n_pos}, "
              f"scale_pos_weight: {scale_pos_weight:.3f}")

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
        )

        if X_val is not None and y_val is not None:
            mask_val = y_val >= 0
            eval_set = [(X_val[mask_val], y_val[mask_val])]
        else:
            # Use a graph-stratified split for early stopping validation
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            train_idx, val_idx = next(splitter.split(X_train, y_train, groups=graph_ids_train))
            X_tr, X_v = X_train[train_idx], X_train[val_idx]
            y_tr, y_v = y_train[train_idx], y_train[val_idx]
            eval_set = [(X_v, y_v)]
            # Retrain on full data after finding best n_estimators
            # For simplicity, we train on split and keep the model
            X_train, y_train = X_tr, y_tr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )

        print(f"[classifier] Trained with {self.model_.best_iteration} trees "
              f"(best val AUC).")

        # Auto-tune decision threshold if not specified
        if self.threshold is None:
            val_X, val_y = eval_set[0]
            probs = self.model_.predict_proba(val_X)[:, 1]
            self.best_threshold_ = self._tune_threshold(probs, val_y)
            print(f"[classifier] Auto-tuned threshold: {self.best_threshold_:.3f}")

        return self

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        graph_ids: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Graph-stratified k-fold cross-validation.

        Returns aggregated metrics across folds.
        """
        mask = y >= 0
        X, y, graph_ids = X[mask], y[mask], graph_ids[mask]

        splitter = StratifiedGroupKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups=graph_ids)
        ):
            X_tr, X_v = X[train_idx], X[val_idx]
            y_tr, y_v = y[train_idx], y[val_idx]

            clf = MVCNodeClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                early_stopping_rounds=self.early_stopping_rounds,
                threshold=None,
                random_state=self.random_state,
            )
            clf.fit(X_tr, y_tr, graph_ids[train_idx], feature_names, X_v, y_v)

            probs = clf.predict_proba(X_v)
            preds = (probs >= clf.best_threshold_).astype(int)

            metrics = {
                "fold": fold,
                "auc": roc_auc_score(y_v, probs),
                "f1": f1_score(y_v, preds, zero_division=0),
                "threshold": clf.best_threshold_,
            }
            fold_metrics.append(metrics)
            print(f"  Fold {fold+1}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

        self.cv_results_ = fold_metrics
        agg = {
            "mean_auc": float(np.mean([m["auc"] for m in fold_metrics])),
            "std_auc": float(np.std([m["auc"] for m in fold_metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
            "std_f1": float(np.std([m["f1"] for m in fold_metrics])),
        }
        print(f"[CV] AUC: {agg['mean_auc']:.4f} ± {agg['std_auc']:.4f}  |  "
              f"F1: {agg['mean_f1']:.4f} ± {agg['std_f1']:.4f}")
        return agg

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of belonging to optimal MVC."""
        assert self.model_ is not None, "Model not trained yet."
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary predictions using threshold."""
        probs = self.predict_proba(X)
        t = threshold if threshold is not None else self.best_threshold_
        return (probs >= t).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compute classification metrics on a labelled set."""
        mask = y >= 0
        X, y = X[mask], y[mask]
        probs = self.predict_proba(X)
        preds = self.predict(X)
        return {
            "auc": float(roc_auc_score(y, probs)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "report": classification_report(y, preds, zero_division=0),
            "threshold": self.best_threshold_,
        }

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        max_display: int = 20,
        plot: bool = True,
        plot_path: str = "shap_summary.png",
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values and plot feature importance.

        Returns SHAP value matrix of shape (n_samples, n_features).
        """
        if not SHAP_AVAILABLE:
            print("[classifier] Install shap (`pip install shap`) for explainability.")
            return None

        assert self.model_ is not None, "Model not trained yet."
        try:
            explainer = shap.TreeExplainer(self.model_)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"[classifier] SHAP failed (XGBoost/SHAP version mismatch?): {e}")
            print("[classifier] Skipping SHAP explanation.")
            return None

        if plot:
            import matplotlib
            matplotlib.use("Agg")  # headless — no display required
            import matplotlib.pyplot as plt
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names_,
                max_display=max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"[classifier] SHAP summary plot saved to {plot_path}")
            plt.close()

        return shap_values

    def feature_importance_df(self):
        """Return a DataFrame of feature importances (gain)."""
        import pandas as pd
        assert self.model_ is not None
        scores = self.model_.get_booster().get_score(importance_type="gain")
        names = self.feature_names_ or [f"f{i}" for i in range(len(scores))]
        df = pd.DataFrame(
            {"feature": list(scores.keys()), "importance": list(scores.values())}
        ).sort_values("importance", ascending=False)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(p))
        meta = {
            "threshold": self.best_threshold_,
            "feature_names": self.feature_names_,
            "cv_results": self.cv_results_,
        }
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[classifier] Model saved to {p}")

    def load(self, path: str) -> "MVCNodeClassifier":
        """Load model from disk."""
        p = Path(path)
        self.model_ = xgb.XGBClassifier()
        self.model_.load_model(str(p))
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.best_threshold_ = meta.get("threshold", 0.5)
            self.feature_names_ = meta.get("feature_names")
            self.cv_results_ = meta.get("cv_results", [])
        print(f"[classifier] Model loaded from {p}")
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Find threshold maximizing F1 on validation data."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
        best_idx = np.argmax(f1s)
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return 0.5

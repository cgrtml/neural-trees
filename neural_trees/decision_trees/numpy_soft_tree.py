"""
A fitted soft decision tree as plain numpy: predict, serialise, ship.

`to_hard_tree()` gives readable rules at the price of being a different model.
This is the other export: the *same* model, the mixture over leaves included,
with no PyTorch in the prediction path and a JSON form that round-trips. Use
it to deploy a fitted tree where torch is not installed, or to freeze one for
audit. Predictions agree with the torch model to float32 precision.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from neural_trees._missing import augment


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    # stable log(sigmoid(z)) = -softplus(-z)
    return -np.logaddexp(0.0, -z)


def _log_softmax(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis=1, keepdims=True))



class NumpySoftTree:
    """
    Torch-free copy of a fitted :class:`~neural_trees.SoftDecisionTree` or
    :class:`~neural_trees.SoftDecisionTreeRegressor` (``kind``).

    Parameters are the tree's own: gate weights ``(n_internal, n_features)``,
    gate biases, per-gate log temperatures, node logits ``(n_nodes, n_classes)``
    and the ``is_split`` flags that say which internal nodes act as leaves.
    """

    def __init__(self, depth: int, weights, biases, log_beta, node_logits, is_split,
                 classes: Sequence[Any], feature_names: Optional[Sequence[str]] = None,
                 kind: str = "classifier", y_mean=None, y_scale=None, single_output: bool = True,
                 missing_columns: Optional[Sequence[int]] = None):
        if kind not in ("classifier", "regressor"):
            raise ValueError(f"kind must be 'classifier' or 'regressor', got {kind!r}")
        self.kind = kind
        # For a regressor `node_logits` holds standardised leaf values and the
        # two arrays below map them back to target units.
        self.y_mean_ = None if y_mean is None else np.asarray(y_mean, dtype=np.float64)
        self.y_scale_ = None if y_scale is None else np.asarray(y_scale, dtype=np.float64)
        self.single_output_ = bool(single_output)
        self.depth = int(depth)
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.biases_ = np.asarray(biases, dtype=np.float64)
        self.log_beta_ = np.asarray(log_beta, dtype=np.float64)
        self.node_logits_ = np.asarray(node_logits, dtype=np.float64)
        self.is_split_ = np.asarray(is_split, dtype=bool)
        self.classes_ = np.asarray(list(classes))
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self.n_internal_ = 2 ** self.depth - 1
        self.n_leaves_ = 2 ** self.depth
        # features with missing values in training: their indicators are the
        # last len(missing_columns) gate inputs (see neural_trees._missing)
        self.missing_columns_ = np.asarray(missing_columns if missing_columns is not None else [],
                                           dtype=np.int64)
        self.n_features_in_ = self.weights_.shape[1] - len(self.missing_columns_)
        if self.weights_.shape[0] != self.n_internal_:
            raise ValueError(f"expected {self.n_internal_} gates for depth {self.depth}, got {self.weights_.shape[0]}")

    # ── prediction · only the gates that split, in log space ──
    def _compiled(self):
        """
        Gates that split, and for every acting leaf the signs of its path.

        A depth-8 tree has 255 gates, but after growth or pruning usually far
        fewer of them split. Computing only those, and summing each leaf's
        path as one matrix product, gives the same numbers as the full walk
        at a fraction of the cost.
        """
        cached = getattr(self, "_paths_cache", None)
        if cached is not None:
            return cached
        from neural_trees.decision_trees.onnx_export import _paths

        leaves = _paths(self.depth, self.is_split_)
        gates = np.array(sorted({g for _, path in leaves for g, _ in path}), dtype=np.int64)
        col = {int(g): i for i, g in enumerate(gates)}
        # path matrix over [log sigmoid(z) for each gate, log sigmoid(-z) for each gate]
        P = np.zeros((2 * len(gates), len(leaves)))
        for t, (_, path) in enumerate(leaves):
            for g, right in path:
                P[col[g] + (0 if right else len(gates)), t] = 1.0
        nodes = np.array([node for node, _ in leaves], dtype=np.int64)
        self._paths_cache = (gates, P, nodes)
        return self._paths_cache

    def _arrival(self, X):
        """(terms, log_mu) for every acting leaf: node index and log arrival mass."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        X = augment(X, self.missing_columns_)
        gates, P, nodes = self._compiled()
        if len(gates) == 0:
            return nodes, np.zeros((X.shape[0], 1))
        z = np.exp(self.log_beta_[gates]) * (X @ self.weights_[gates].T + self.biases_[gates])
        right = _log_sigmoid(z)
        return nodes, np.hstack([right, right - z]) @ P   # log sigmoid(-z) = log sigmoid(z) - z

    def predict_values(self, X) -> np.ndarray:
        """Regressor only: arrival-weighted leaf values, in target units."""
        if self.kind != "regressor":
            raise AttributeError("predict_values is for a regressor export; use predict_proba")
        nodes, log_mu = self._arrival(X)
        mu = np.exp(log_mu)                                    # (n, terms)
        out = mu @ self.node_logits_[nodes]                    # (n, n_outputs), standardised
        out = out * self.y_scale_ + self.y_mean_
        return out[:, 0] if self.single_output_ else out

    def predict_log_proba(self, X) -> np.ndarray:
        if self.kind != "classifier":
            raise AttributeError("predict_log_proba is for a classifier export; use predict_values")
        nodes, log_mu = self._arrival(X)                                # (n, terms)
        log_dist = _log_softmax(self.node_logits_)[nodes]                 # (terms, K)
        # log sum_t mu_t * p_t, both factors scaled by their maxima first
        m = log_mu.max(axis=1, keepdims=True)
        k = log_dist.max(axis=0, keepdims=True)
        with np.errstate(divide="ignore"):
            out = m + k + np.log(np.exp(log_mu - m) @ np.exp(log_dist - k))
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():                                                     # underflow: exact sum
            allp = log_mu[bad][:, :, None] + log_dist[None, :, :]
            mm = allp.max(axis=1, keepdims=True)
            out[bad] = (mm + np.log(np.exp(allp - mm).sum(axis=1, keepdims=True)))[:, 0, :]
        return out

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        if self.kind == "regressor":
            return self.predict_values(X)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # ── ONNX ──
    def to_onnx(self, opset: int = 17):
        """The same model as an ``onnx.ModelProto``; see :mod:`neural_trees.decision_trees.onnx_export`."""
        from neural_trees.decision_trees.onnx_export import to_onnx

        return to_onnx(self, opset=opset)

    def save_onnx(self, path: str, opset: int = 17) -> None:
        """Write :meth:`to_onnx` to a file."""
        from neural_trees.decision_trees.onnx_export import save_onnx

        save_onnx(self, path, opset=opset)

    # ── serialisation ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "neural-trees/soft-tree/1",
            "depth": self.depth,
            "weights": self.weights_.tolist(),
            "biases": self.biases_.tolist(),
            "log_beta": self.log_beta_.tolist(),
            "node_logits": self.node_logits_.tolist(),
            "is_split": self.is_split_.tolist(),
            "classes": self.classes_.tolist(),
            "feature_names": self.feature_names_,
            "kind": self.kind,
            "y_mean": None if self.y_mean_ is None else self.y_mean_.tolist(),
            "y_scale": None if self.y_scale_ is None else self.y_scale_.tolist(),
            "single_output": self.single_output_,
            "missing_columns": self.missing_columns_.tolist(),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict())
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NumpySoftTree:
        if d.get("format") != "neural-trees/soft-tree/1":
            raise ValueError(f"unknown format {d.get('format')!r}")
        return cls(d["depth"], d["weights"], d["biases"], d["log_beta"], d["node_logits"],
                   d["is_split"], d["classes"], d.get("feature_names"),
                   kind=d.get("kind", "classifier"), y_mean=d.get("y_mean"), y_scale=d.get("y_scale"),
                   single_output=d.get("single_output", True),
                   missing_columns=d.get("missing_columns"))

    @classmethod
    def from_json(cls, text_or_path: str) -> NumpySoftTree:
        s = text_or_path
        if not s.lstrip().startswith("{"):
            with open(s, encoding="utf-8") as f:
                s = f.read()
        return cls.from_dict(json.loads(s))

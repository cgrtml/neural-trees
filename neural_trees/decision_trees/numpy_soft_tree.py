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


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    # stable log(sigmoid(z)) = -softplus(-z)
    return -np.logaddexp(0.0, -z)


def _log_softmax(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis=1, keepdims=True))


class NumpySoftTree:
    """
    Torch-free copy of a fitted :class:`~neural_trees.SoftDecisionTree`.

    Parameters are the tree's own: gate weights ``(n_internal, n_features)``,
    gate biases, per-gate log temperatures, node logits ``(n_nodes, n_classes)``
    and the ``is_split`` flags that say which internal nodes act as leaves.
    """

    def __init__(self, depth: int, weights, biases, log_beta, node_logits, is_split,
                 classes: Sequence[Any], feature_names: Optional[Sequence[str]] = None):
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
        self.n_features_in_ = self.weights_.shape[1]
        if self.weights_.shape[0] != self.n_internal_:
            raise ValueError(f"expected {self.n_internal_} gates for depth {self.depth}, got {self.weights_.shape[0]}")

    # ── prediction · the same walk as the torch module, in log space ──
    def predict_log_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        logits = np.exp(self.log_beta_) * (X @ self.weights_.T + self.biases_)  # (n, n_internal)
        log_dist = _log_softmax(self.node_logits_)                               # (n_nodes, K)
        n = X.shape[0]
        log_mu = np.zeros((n, 1))
        parts: List[np.ndarray] = []
        start = 0
        for level in range(self.depth):
            width = 2 ** level
            lv = logits[:, start:start + width]
            splits = self.is_split_[start:start + width]
            if not splits.all():
                stopped = np.where(~splits)[0]
                idx = start + stopped
                parts.append(log_mu[:, stopped][:, :, None] + log_dist[idx][None, :, :])
            log_left, log_right = _log_sigmoid(-lv), _log_sigmoid(lv)
            children = np.stack([log_left, log_right], axis=2).reshape(n, 2 * width)
            log_mu = np.repeat(log_mu, 2, axis=1) + children
            if not splits.all():
                alive = np.repeat(splits, 2)
                log_mu = np.where(alive, log_mu, -np.inf)
            start += width
        bottom = np.arange(self.n_internal_, self.n_internal_ + self.n_leaves_)
        parts.append(log_mu[:, :, None] + log_dist[bottom][None, :, :])
        allp = np.concatenate(parts, axis=1)                                      # (n, n_terms, K)
        m = allp.max(axis=1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return (m + np.log(np.exp(allp - m).sum(axis=1, keepdims=True)))[:, 0, :]

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

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
                   d["is_split"], d["classes"], d.get("feature_names"))

    @classmethod
    def from_json(cls, text_or_path: str) -> NumpySoftTree:
        s = text_or_path
        if not s.lstrip().startswith("{"):
            with open(s, encoding="utf-8") as f:
                s = f.read()
        return cls.from_dict(json.loads(s))

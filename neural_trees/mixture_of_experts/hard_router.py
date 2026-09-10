"""
Hard export of a trained Hierarchical Mixture of Experts
========================================================

A trained mixture evaluates every expert for every sample and blends them by
the gating weights, so one prediction costs `branching_factor^depth` expert
forward passes. Once training is over, each gating node has a preferred child
for any given input, and taking that preference as a decision routes a sample
down a single path to a single expert.

That is a different model from the mixture, in the same way a hard tree is a
different model from a soft one: a blend of experts is not one expert. The
export reports how often the two agree rather than assuming they do.

The experts themselves stay small MLPs, so this is a routing tree over experts,
not a rule list. What it buys is evaluating one expert instead of all of them.
"""

import numpy as np
from sklearn.utils.validation import check_array


class HardRoutedExperts:
    """
    A trained mixture of experts with its gates read as hard routing decisions.

    Each gating node sends a sample to its highest-weighted child, and the
    expert reached at the leaf produces the prediction. Everything runs in
    numpy; no PyTorch is involved.

    Built by `HierarchicalMixtureOfExperts.to_hard_router()` rather than
    directly.

    Attributes
    ----------
    depth : int
    branching_factor : int
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    gate_weights_ : list of (W1, b1, W2, b2)
        One two-layer gating network per internal node, in breadth-first order.
    expert_weights_ : list of (W1, b1, W2, b2)
        One two-layer expert network per leaf.
    """

    def __init__(
        self, gate_weights, expert_weights, classes, n_features_in, depth, branching_factor
    ):
        self.gate_weights_ = gate_weights
        self.expert_weights_ = expert_weights
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        self.depth = int(depth)
        self.branching_factor = int(branching_factor)

    @staticmethod
    def _mlp(x: np.ndarray, params, activation) -> np.ndarray:
        weight_in, bias_in, weight_out, bias_out = params
        hidden = activation(x @ weight_in.T + bias_in)
        return hidden @ weight_out.T + bias_out

    def _route(self, X: np.ndarray) -> np.ndarray:
        """Index of the expert each sample is routed to, shape (n_samples,)."""
        b = self.branching_factor
        node = np.zeros(len(X), dtype=np.int64)  # index within the current level
        offset = 0  # first gate index of the current level

        for level in range(self.depth):
            choice = np.empty(len(X), dtype=np.int64)
            for position in np.unique(node):
                rows = node == position
                logits = self._mlp(X[rows], self.gate_weights_[offset + position], np.tanh)
                choice[rows] = logits.argmax(axis=1)
            node = node * b + choice
            offset += b ** level

        return node

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities from the single expert each sample reaches."""
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but HardRoutedExperts is expecting "
                f"{self.n_features_in_} features as input."
            )

        expert_index = self._route(X)
        proba = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        for expert in np.unique(expert_index):
            rows = expert_index == expert
            logits = self._mlp(X[rows], self.expert_weights_[expert], lambda h: np.maximum(h, 0.0))
            logits = logits - logits.max(axis=1, keepdims=True)
            exponentiated = np.exp(logits)
            proba[rows] = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X) -> np.ndarray:
        """Predicted class labels, shape (n_samples,)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def route_counts(self, X) -> np.ndarray:
        """
        How many samples reach each expert, shape (n_experts,).

        A mixture spreads every sample over all experts, so this is the first
        thing the export makes visible: whether the tree actually partitions
        the input or leans on one branch.
        """
        X = check_array(X)
        return np.bincount(self._route(X), minlength=len(self.expert_weights_))

    def export_text(self, feature_names=None, max_features=3, decimals=3) -> str:
        """
        Render the routing tree, showing which features drive each gate.

        A gating node is a two-layer network, not a single hyperplane, so there
        is no exact rule to print. What is shown is each gate's input-layer
        sensitivity per feature, summed over hidden units, which says what the
        gate is looking at.
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries, expected "
                f"{self.n_features_in_}"
            )

        b = self.branching_factor
        lines = []

        def describe_gate(gate_index: int) -> str:
            weight_in = self.gate_weights_[gate_index][0]
            sensitivity = np.abs(weight_in).sum(axis=0)
            order = np.argsort(-sensitivity)[:max_features]
            terms = ", ".join(
                f"{feature_names[i]} ({sensitivity[i]:.{decimals}f})" for i in order
            )
            return f"gate on {terms}"

        def walk(level: int, position: int, indent: str):
            if level == self.depth:
                lines.append(f"{indent}-> expert {position}")
                return
            offset = sum(b ** d for d in range(level))
            lines.append(f"{indent}{describe_gate(offset + position)}:")
            for child in range(b):
                lines.append(f"{indent}  child {child}:")
                walk(level + 1, position * b + child, indent + "    ")

        walk(0, 0, "")
        return "\n".join(lines)

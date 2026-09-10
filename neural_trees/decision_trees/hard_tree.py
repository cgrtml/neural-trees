"""
Hard export of a trained Soft Decision Tree
===========================================

A soft tree sends every sample to every leaf and mixes the results, which is
what makes it differentiable and what makes a prediction cost a PyTorch forward
pass over all 2^depth leaves. Once training is over, that machinery is no longer
doing any work: each internal node has settled on a hyperplane, and the sign of
that hyperplane is a decision.

`SoftDecisionTree.to_hard_tree()` takes those hyperplanes as they are and routes
each sample down a single path, in plain numpy. The result is not the same model
- a mixture is not a path, and the two disagree on samples that sat near a split
- so the export reports how often they agree rather than pretending otherwise.
"""

import numpy as np
from sklearn.utils.validation import check_array


class HardDecisionTree:
    """
    A trained Soft Decision Tree with its gates read as hard decisions.

    Each internal node routes right when `w . x + b > 0` and left otherwise, and
    each leaf carries the class distribution the soft tree learned for it.
    Prediction is a walk of `depth` steps in numpy, with no PyTorch involved.

    Built by `SoftDecisionTree.to_hard_tree()` rather than directly.

    Attributes
    ----------
    depth : int
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    weights_ : ndarray of shape (n_internal, n_features)
        One hyperplane per internal node, in breadth-first order.
    biases_ : ndarray of shape (n_internal,)
    node_distributions_ : ndarray of shape (n_nodes, n_classes)
        A distribution per node. `leaf_distributions_` selects the ones that
        act as leaves.
    is_split_ : ndarray of shape (n_internal,)
        Whether each internal node routes onward. All True for a complete tree.
    """

    def __init__(
        self, weights, biases, node_distributions, classes, n_features_in, is_split=None
    ):
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.biases_ = np.asarray(biases, dtype=np.float64)
        self.node_distributions_ = np.asarray(node_distributions, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        self.n_internal_ = len(self.weights_)
        self.depth = int(np.log2(self.n_internal_ + 1))
        self.is_split_ = (
            np.ones(self.n_internal_, dtype=bool)
            if is_split is None
            else np.asarray(is_split, dtype=bool)
        )

    @property
    def leaf_distributions_(self) -> np.ndarray:
        """Distributions of the nodes that actually behave as leaves."""
        return self.node_distributions_[self._acting_leaves()]

    def _acting_leaves(self) -> np.ndarray:
        leaves = []
        stack = [0]
        while stack:
            node = stack.pop()
            if node >= self.n_internal_ or not self.is_split_[node]:
                leaves.append(node)
                continue
            stack.extend([2 * node + 2, 2 * node + 1])
        return np.array(sorted(leaves))

    def _leaf_index(self, X: np.ndarray) -> np.ndarray:
        """
        Walk every sample to the node where its path stops.

        A node whose subtree was never grown keeps the sample instead of
        routing it on, so the walk is a fixpoint rather than a fixed number of
        levels.
        """
        node = np.zeros(len(X), dtype=np.int64)
        for _ in range(self.depth):
            moving = (node < self.n_internal_) & self.is_split_[np.minimum(node, self.n_internal_ - 1)]
            if not moving.any():
                break
            scores = (
                np.einsum("nf,nf->n", X[moving], self.weights_[node[moving]])
                + self.biases_[node[moving]]
            )
            node[moving] = 2 * node[moving] + 1 + (scores > 0).astype(np.int64)
        return node

    def predict_proba(self, X) -> np.ndarray:
        """
        Class probabilities of the reached leaf, shape (n_samples, n_classes).
        """
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but HardDecisionTree is expecting "
                f"{self.n_features_in_} features as input."
            )
        return self.node_distributions_[self._leaf_index(X)]

    def predict(self, X) -> np.ndarray:
        """Predicted class labels, shape (n_samples,)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def export_text(self, feature_names=None, max_features=3, decimals=3) -> str:
        """
        Render the tree as readable rules.

        Parameters
        ----------
        feature_names : list of str, optional
            Defaults to `x0`, `x1`, ...
        max_features : int, default=3
            Features shown per split, largest absolute weight first. A
            multivariate split uses every feature; printing all of them stops
            being readable, which is the thing this method is for.
        decimals : int, default=3

        Returns
        -------
        str
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries, expected "
                f"{self.n_features_in_}"
            )

        n_internal = self.n_internal_
        lines = []

        def describe_split(node: int) -> str:
            weights = self.weights_[node]
            order = np.argsort(-np.abs(weights))[:max_features]
            terms = [
                f"{weights[i]:+.{decimals}f}*{feature_names[i]}" for i in order
            ]
            omitted = ""
            if self.n_features_in_ > max_features:
                omitted = f" (+{self.n_features_in_ - max_features} more)"
            return f"{' '.join(terms)} {self.biases_[node]:+.{decimals}f}{omitted} > 0"

        def walk(node: int, indent: str, branch: str):
            if node >= n_internal or not self.is_split_[node]:
                distribution = self.node_distributions_[node]
                winner = self.classes_[int(np.argmax(distribution))]
                lines.append(
                    f"{indent}{branch}predict {winner!r} "
                    f"(p={distribution.max():.{decimals}f})"
                )
                return
            lines.append(f"{indent}{branch}if {describe_split(node)}:")
            walk(2 * node + 2, indent + "    ", "yes -> ")
            walk(2 * node + 1, indent + "    ", "no  -> ")

        walk(0, "", "")
        return "\n".join(lines)

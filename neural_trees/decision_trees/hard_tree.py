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
    leaf_distributions_ : ndarray of shape (n_leaves, n_classes)
    """

    def __init__(self, weights, biases, leaf_distributions, classes, n_features_in):
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.biases_ = np.asarray(biases, dtype=np.float64)
        self.leaf_distributions_ = np.asarray(leaf_distributions, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        self.depth = int(np.log2(len(self.leaf_distributions_)))

    def _leaf_index(self, X: np.ndarray) -> np.ndarray:
        """Walk every sample down to its leaf, one level at a time."""
        node = np.zeros(len(X), dtype=np.int64)
        for _ in range(self.depth):
            scores = np.einsum("nf,nf->n", X, self.weights_[node]) + self.biases_[node]
            node = 2 * node + 1 + (scores > 0).astype(np.int64)
        return node - (len(self.weights_))

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
        return self.leaf_distributions_[self._leaf_index(X)]

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

        n_internal = len(self.weights_)
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
            if node >= n_internal:
                distribution = self.leaf_distributions_[node - n_internal]
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

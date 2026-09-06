"""
Omnivariate Decision Trees
===========================
Implementation based on:
    Yıldız, O. T., & Alpaydın, E. (2001).
    Omnivariate Decision Trees.
    IEEE Transactions on Neural Networks, 12(6), 1539-1546.

Key idea:
    Standard decision trees are "univariate" (split on a single feature) or
    "multivariate" (split on a linear combination). Omnivariate trees adaptively
    choose the best split type (univariate, linear, or nonlinear MLP) at each
    node based on cross-validation, giving them maximum flexibility.

    Split types supported:
        - Univariate: split on a single feature threshold
        - Linear (LDA): split on a linear discriminant
        - Nonlinear (MLP): split on a 1-hidden-layer perceptron
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Dict, Any


class _OmnivariateNode:
    """A single node in an omnivariate decision tree."""

    def __init__(self, depth: int, max_depth: int, min_samples_split: int, cv_folds: int):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.cv_folds = cv_folds
        self.split_type: Optional[str] = None
        self.classifier = None
        self.is_leaf = False
        self.leaf_class = None
        self.left: Optional["_OmnivariateNode"] = None
        self.right: Optional["_OmnivariateNode"] = None

    def _two_group_labels(self, X: np.ndarray, y: np.ndarray):
        """
        Reduce the classes at this node to the two-group problem a binary
        split has to solve. Two classes map directly; more than two are
        grouped by clustering their centroids.
        """
        present = np.unique(y)
        if len(present) < 2:
            return None
        if len(present) == 2:
            return (y == present[1]).astype(int)

        centroids = np.vstack([X[y == c].mean(axis=0) for c in present])
        group_of_class = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(centroids)
        if len(np.unique(group_of_class)) < 2:
            return None
        mapping = {c: int(g) for c, g in zip(present, group_of_class)}
        return np.array([mapping[label] for label in y])

    def _select_best_splitter(self, X: np.ndarray, y_bin: np.ndarray):
        """Cross-validate the three split types on the two-group problem."""
        candidates = {
            "univariate": DecisionTreeClassifier(max_depth=1, random_state=42),
            "linear": LinearDiscriminantAnalysis(),
            "nonlinear": MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42),
        }
        # Folds are bounded by the rarest group, not by the number of groups,
        # otherwise StratifiedKFold raises on small or skewed nodes.
        min_group = int(np.bincount(y_bin).min())
        n_folds = min(self.cv_folds, min_group)
        if n_folds < 2:
            return "univariate", candidates["univariate"]

        best_type, best_score = "univariate", -np.inf
        for split_type, clf in candidates.items():
            try:
                score = cross_val_score(clf, X, y_bin, cv=n_folds, scoring="accuracy").mean()
            except Exception:
                continue
            if score > best_score:
                best_score, best_type = score, split_type

        return best_type, candidates[best_type]

    def fit(self, X: np.ndarray, y: np.ndarray):
        if (
            self.depth >= self.max_depth
            or len(X) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            self.is_leaf = True
            self.leaf_class = np.bincount(y).argmax()
            return self

        y_bin = self._two_group_labels(X, y)
        if y_bin is None:
            self.is_leaf = True
            self.leaf_class = np.bincount(y).argmax()
            return self

        self.split_type, self.classifier = self._select_best_splitter(X, y_bin)
        self.classifier.fit(X, y_bin)

        # The split is the node classifier's own decision: group 1 goes right,
        # group 0 goes left. Routing at predict time uses the same rule.
        mask_right = self.classifier.predict(X) == 1
        mask_left = ~mask_right

        if mask_right.sum() == 0 or mask_left.sum() == 0:
            self.is_leaf = True
            self.leaf_class = np.bincount(y).argmax()
            return self

        self.left = _OmnivariateNode(
            self.depth + 1, self.max_depth, self.min_samples_split, self.cv_folds
        ).fit(X[mask_left], y[mask_left])
        self.right = _OmnivariateNode(
            self.depth + 1, self.max_depth, self.min_samples_split, self.cv_folds
        ).fit(X[mask_right], y[mask_right])
        return self

    def predict_one(self, x: np.ndarray) -> int:
        node = self
        while not node.is_leaf:
            goes_right = node.classifier.predict(x.reshape(1, -1))[0] == 1
            node = node.right if goes_right else node.left
        return node.leaf_class


class OmnivariateDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Omnivariate Decision Tree Classifier (sklearn-compatible).

    At each node, automatically selects the best split type from:
    univariate (single feature), linear (LDA), or nonlinear (MLP) splits,
    chosen by cross-validation.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_split : int, default=10
        Minimum number of samples required to split a node.
    cv_folds : int, default=3
        Number of cross-validation folds used to select split type at each node.

    Examples
    --------
    >>> from neural_trees import OmnivariateDecisionTree
    >>> from sklearn.datasets import load_wine
    >>> X, y = load_wine(return_X_y=True)
    >>> odt = OmnivariateDecisionTree(max_depth=4)
    >>> odt.fit(X, y)
    >>> odt.score(X, y)

    References
    ----------
    Yıldız, O. T., & Alpaydın, E. (2001).
    Omnivariate Decision Trees.
    IEEE Transactions on Neural Networks, 12(6), 1539-1546.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        cv_folds: int = 3,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.cv_folds = cv_folds

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        self.root_ = _OmnivariateNode(
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            cv_folds=self.cv_folds,
        ).fit(X, y_enc)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        preds = np.array([self.root_.predict_one(x) for x in X])
        return self.le_.inverse_transform(preds)

    def get_split_type_distribution(self) -> Dict[str, int]:
        """Count how many nodes use each split type."""
        check_is_fitted(self)
        counts: Dict[str, int] = {"univariate": 0, "linear": 0, "nonlinear": 0}

        def traverse(node):
            if node is None or node.is_leaf:
                return
            if node.split_type:
                counts[node.split_type] = counts.get(node.split_type, 0) + 1
            traverse(node.left)
            traverse(node.right)

        traverse(self.root_)
        return counts

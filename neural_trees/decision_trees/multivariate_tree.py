"""
Multivariate Decision Trees
===========================
Implementation based on:
    Alpaydın, E., & Çetin, Ü. (1995).
    Multivariate Statistical Techniques for Constructive Induction.
    (see also Yıldız, O. T., & Alpaydın, E. (2001), Omnivariate Decision Trees,
    IEEE Transactions on Neural Networks, 12(6), 1539-1546.)

Key idea:
    A univariate tree (CART, C4.5) tests one feature at a time, so its
    decision boundary is a staircase of axis-aligned cuts. A multivariate tree
    tests a linear combination instead:

        go right if  w . x + b > 0

    The weight vector w at each node comes from a linear discriminant fitted
    on the samples that reach that node, so a single node can express an
    oblique boundary that a univariate tree needs many nodes to approximate.

    With more than two classes the discriminant needs a two-group problem, so
    the classes at a node are first partitioned into two superclasses by
    clustering their centroids. This is the standard reduction used when a
    binary tree is grown with discriminant splits.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import List, Optional, Tuple


def _gini(y: np.ndarray, n_classes: int) -> float:
    """Gini impurity of a label vector."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


class _MultivariateNode:
    """One node holding a linear discriminant split, or a class distribution."""

    def __init__(self, depth: int, params: dict):
        self.depth = depth
        self.params = params
        self.is_leaf = False
        self.distribution: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.left: Optional["_MultivariateNode"] = None
        self.right: Optional["_MultivariateNode"] = None

    def _make_leaf(self, y: np.ndarray) -> "_MultivariateNode":
        n_classes = self.params["n_classes"]
        counts = np.bincount(y, minlength=n_classes).astype(float)
        self.is_leaf = True
        self.distribution = counts / counts.sum() if counts.sum() else counts
        return self

    def _two_group_labels(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Reduce the classes present at this node to a two-group problem.

        Two classes map directly. More than two are split by clustering their
        centroids into two groups, which keeps similar classes on the same
        side of the discriminant.
        """
        present = np.unique(y)
        if len(present) < 2:
            return None
        if len(present) == 2:
            return (y == present[1]).astype(int)

        centroids = np.vstack([X[y == c].mean(axis=0) for c in present])
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.params["random_state"])
        group_of_class = kmeans.fit_predict(centroids)
        if len(np.unique(group_of_class)) < 2:
            return None

        mapping = {c: int(g) for c, g in zip(present, group_of_class)}
        return np.array([mapping[label] for label in y])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MultivariateNode":
        p = self.params
        if (
            self.depth >= p["max_depth"]
            or len(X) < p["min_samples_split"]
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        y_bin = self._two_group_labels(X, y)
        if y_bin is None:
            return self._make_leaf(y)

        try:
            lda = LinearDiscriminantAnalysis(solver="svd").fit(X, y_bin)
        except Exception:
            return self._make_leaf(y)

        # LinearDiscriminantAnalysis on a two-group problem exposes the split
        # directly as a single hyperplane.
        weights = np.asarray(lda.coef_).ravel()
        bias = float(np.asarray(lda.intercept_).ravel()[0])
        scores = X @ weights + bias
        mask_right = scores > 0

        n_right = int(mask_right.sum())
        n_left = len(X) - n_right
        if min(n_left, n_right) < p["min_samples_leaf"]:
            return self._make_leaf(y)

        # Only keep the split if it actually purifies the node.
        n_classes = p["n_classes"]
        parent_impurity = _gini(y, n_classes)
        child_impurity = (
            n_left / len(y) * _gini(y[~mask_right], n_classes)
            + n_right / len(y) * _gini(y[mask_right], n_classes)
        )
        if parent_impurity - child_impurity < p["min_impurity_decrease"]:
            return self._make_leaf(y)

        self.weights = weights
        self.bias = bias
        self.left = _MultivariateNode(self.depth + 1, p).fit(X[~mask_right], y[~mask_right])
        self.right = _MultivariateNode(self.depth + 1, p).fit(X[mask_right], y[mask_right])
        return self

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        node = self
        while not node.is_leaf:
            node = node.right if float(x @ node.weights + node.bias) > 0 else node.left
        return node.distribution


class MultivariateDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Multivariate Decision Tree Classifier (sklearn-compatible).

    Each internal node splits on a linear combination of all features,
    `w . x + b > 0`, where `w` is a linear discriminant fitted on the samples
    reaching that node. Boundaries are oblique rather than axis-aligned, so
    correlated features are handled in one node instead of a staircase of
    univariate cuts.

    Parameters
    ----------
    max_depth : int, default=4
        Maximum depth of the tree.
    min_samples_split : int, default=10
        Minimum samples required to attempt a split at a node.
    min_samples_leaf : int, default=3
        Minimum samples that must land on each side of a split.
    min_impurity_decrease : float, default=0.0
        Minimum weighted Gini decrease required to keep a split.
    random_state : int or None, default=None
        Seed for the centroid clustering used to build two-group problems
        when a node holds more than two classes.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    root_ : internal node object
    tree_depth_ : int
        Depth actually reached after fitting.
    n_nodes_ : int
        Number of internal (splitting) nodes.

    Examples
    --------
    >>> from neural_trees import MultivariateDecisionTree
    >>> from sklearn.datasets import load_wine
    >>> X, y = load_wine(return_X_y=True)
    >>> mdt = MultivariateDecisionTree(max_depth=3, random_state=0)
    >>> mdt.fit(X, y)
    >>> mdt.score(X, y)

    References
    ----------
    Alpaydın, E., & Çetin, Ü. (1995). Multivariate Statistical Techniques for
    Constructive Induction.
    Yıldız, O. T., & Alpaydın, E. (2001). Omnivariate Decision Trees. IEEE TNN.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 10,
        min_samples_leaf: int = 3,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the multivariate tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError(f"max_depth must be a positive integer, got {self.max_depth!r}")

        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "n_classes": len(self.classes_),
            "random_state": self.random_state,
        }
        self.root_ = _MultivariateNode(0, params).fit(X, y_enc)
        self.tree_depth_ = self._max_depth_reached(self.root_)
        self.n_nodes_ = len(self.get_split_weights())
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities from the reached leaf's class distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.vstack([self.root_.predict_proba_one(x) for x in X])

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def get_split_weights(self) -> List[Tuple[np.ndarray, float]]:
        """
        Return the hyperplane of every internal node as `(w, b)` pairs,
        in pre-order. Useful for reading off which features drive a split.
        """
        check_is_fitted(self)
        found: List[Tuple[np.ndarray, float]] = []

        def traverse(node):
            if node is None or node.is_leaf:
                return
            found.append((node.weights, node.bias))
            traverse(node.left)
            traverse(node.right)

        traverse(self.root_)
        return found

    @staticmethod
    def _max_depth_reached(node) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(
            MultivariateDecisionTree._max_depth_reached(node.left),
            MultivariateDecisionTree._max_depth_reached(node.right),
        )

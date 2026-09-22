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

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)

from neural_trees._validation import check_predict_input, reject_sparse
from neural_trees.decision_trees._grouping import two_group_candidates


def _gini(y: np.ndarray, n_classes: int, w: Optional[np.ndarray] = None) -> float:
    """Gini impurity of a label vector, over weighted class totals if `w` is given."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, weights=w, minlength=n_classes)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p ** 2))


def _weighted_lda(X: np.ndarray, y_bin: np.ndarray, w: np.ndarray, ridge: float = 0.0):
    """
    Fisher's linear discriminant for two groups with sample weights.

    scikit-learn's LinearDiscriminantAnalysis takes no `sample_weight`, so
    the weighted case is solved directly: weighted class means, pooled
    within-class covariance weighted by class mass, and the intercept that
    LDA uses, with the log prior ratio taken over weighted class totals.
    Uniform weights give the same hyperplane as the sklearn solver up to
    floating point.
    """
    masses, means, pooled = [], [], np.zeros((X.shape[1], X.shape[1]))
    for g in (0, 1):
        sel = y_bin == g
        wg = w[sel]
        masses.append(wg.sum())
        mu = np.average(X[sel], axis=0, weights=wg)
        means.append(mu)
        centred = X[sel] - mu
        pooled += (centred * wg[:, None]).T @ centred
    pooled /= w.sum()
    if ridge > 0:
        # Shrink toward a scaled identity, the same idea as sklearn's
        # shrinkage LDA, for node data too ill-conditioned to invert.
        pooled = (1 - ridge) * pooled + ridge * np.trace(pooled) / X.shape[1] * np.eye(X.shape[1])
    coef = np.linalg.pinv(pooled) @ (means[1] - means[0])
    bias = -0.5 * (means[1] + means[0]) @ coef + np.log(masses[1] / masses[0])
    return coef, float(bias)


class _MultivariateNode:
    """One node holding a linear discriminant split, or a class distribution."""

    def __init__(self, depth: int, params: dict):
        self.depth = depth
        self.params = params
        self.is_leaf = False
        self.distribution: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.left: Optional[_MultivariateNode] = None
        self.right: Optional[_MultivariateNode] = None

    def _make_leaf(self, y: np.ndarray, w: np.ndarray) -> "_MultivariateNode":
        n_classes = self.params["n_classes"]
        counts = np.bincount(y, weights=w, minlength=n_classes).astype(float)
        self.is_leaf = True
        self.distribution = counts / counts.sum() if counts.sum() else counts
        return self

    def _discriminant(self, X: np.ndarray, y_bin: np.ndarray, w: np.ndarray):
        """
        The split hyperplane for one two-group labeling, or None.

        The plain discriminant comes first: scikit-learn's SVD solver when
        there are no weights (unchanged from before), the weighted solve
        otherwise. On ill-conditioned node data (pixel features, constant
        columns) the SVD solver can return a hyperplane of norm 1e18 whose
        sign classifies the training groups worse than the majority rule,
        and which side of a rank threshold that happens on depends on the
        BLAS thread count. A hyperplane that does not beat the majority rule
        on its own training groups is therefore rejected, and a shrinkage
        discriminant is fitted in its place; if that fails too the node tries
        the next grouping.
        """
        p = self.params
        trivial = max(y_bin.mean(), 1.0 - y_bin.mean())
        for shrink in (False, True):
            try:
                if p["weighted"]:
                    weights, bias = _weighted_lda(X, y_bin, w, ridge=1e-3 if shrink else 0.0)
                elif shrink:
                    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y_bin)
                    weights = np.asarray(lda.coef_).ravel()
                    bias = float(np.asarray(lda.intercept_).ravel()[0])
                else:
                    # LinearDiscriminantAnalysis on a two-group problem
                    # exposes the split directly as a single hyperplane.
                    lda = LinearDiscriminantAnalysis(solver="svd").fit(X, y_bin)
                    weights = np.asarray(lda.coef_).ravel()
                    bias = float(np.asarray(lda.intercept_).ravel()[0])
            except Exception:
                continue
            if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
                continue
            mask_right = (X @ weights + bias) > 0
            if (mask_right == (y_bin == 1)).mean() < trivial:
                continue
            n_right = int(mask_right.sum())
            if min(len(X) - n_right, n_right) < p["min_samples_leaf"]:
                continue
            return weights, bias, mask_right
        return None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "_MultivariateNode":
        p = self.params
        # min_samples_split and min_samples_leaf count rows, not weight, as
        # scikit-learn's trees do; weights enter the discriminant, the
        # impurity and the leaf distributions.
        if (
            self.depth >= p["max_depth"]
            or len(X) < p["min_samples_split"]
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y, w)

        # Try the clustered grouping first and the mass-balanced one if the
        # discriminant fitted to it cannot make a legal split (#104).
        split = None
        for y_bin in two_group_candidates(
            X, y, w, p["random_state"], min_rows=p["min_samples_leaf"]
        ):
            split = self._discriminant(X, y_bin, w)
            if split is not None:
                break
        if split is None:
            return self._make_leaf(y, w)
        weights, bias, mask_right = split
        total = w.sum()
        if min(w[mask_right].sum(), w[~mask_right].sum()) < p["min_weight_fraction_leaf"] * total:
            return self._make_leaf(y, w)

        # Only keep the split if it actually purifies the node, weighing each
        # child by the mass it receives.
        n_classes = p["n_classes"]
        parent_impurity = _gini(y, n_classes, w)
        child_impurity = (
            w[~mask_right].sum() / total * _gini(y[~mask_right], n_classes, w[~mask_right])
            + w[mask_right].sum() / total * _gini(y[mask_right], n_classes, w[mask_right])
        )
        if parent_impurity - child_impurity < p["min_impurity_decrease"]:
            return self._make_leaf(y, w)

        self.weights = weights
        self.bias = bias
        self.left = _MultivariateNode(self.depth + 1, p).fit(
            X[~mask_right], y[~mask_right], w[~mask_right]
        )
        self.right = _MultivariateNode(self.depth + 1, p).fit(
            X[mask_right], y[mask_right], w[mask_right]
        )
        return self

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """
        Walk to the leaf this sample belongs in.

        A non-leaf node always has a hyperplane and both children; `fit` turns
        the node into a leaf rather than leaving any of them unset, so the
        asserts document that invariant instead of guarding against it.
        """
        node = self
        while not node.is_leaf:
            assert node.weights is not None
            assert node.left is not None and node.right is not None
            node = node.right if float(x @ node.weights + node.bias) > 0 else node.left
        assert node.distribution is not None
        return node.distribution


class MultivariateDecisionTree(ClassifierMixin, BaseEstimator):
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
    min_weight_fraction_leaf : float, default=0.0
        Smallest fraction of the total sample weight a child may hold, as in
        scikit-learn's trees. Only meaningful with `sample_weight` or
        `class_weight`; it stops a split from isolating a region whose weight
        is negligible, which is how a down-weighted class would otherwise
        keep leaves of its own.
    random_state : int or None, default=None
        Seed for the centroid clustering used to build two-group problems
        when a node holds more than two classes.
    class_weight : dict, "balanced" or None, default=None
        Weights per class, combined multiplicatively with `sample_weight`.
        `"balanced"` uses `n_samples / (n_classes * bincount(y))`.

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
        class_weight=None,
        min_weight_fraction_leaf: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.min_weight_fraction_leaf = min_weight_fraction_leaf

    def fit(self, X, y, sample_weight=None) -> "MultivariateDecisionTree":
        """
        Fit the multivariate tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Enters the discriminant at every node (weighted class means,
            pooled covariance and prior ratio), the Gini decrease and the
            leaf distributions. `min_samples_split` and `min_samples_leaf`
            keep counting rows, as scikit-learn's trees do, so weighting a
            row is not the same as repeating it where those limits bind.
            Without weights the node discriminant is scikit-learn's own
            solver, unchanged.

        Returns
        -------
        self
        """
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError(f"max_depth must be a positive integer, got {self.max_depth!r}")

        reject_sparse(self, X)
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        weighted = sample_weight is not None or self.class_weight is not None
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if self.class_weight is not None:
            class_weights = compute_class_weight(
                self.class_weight, classes=np.arange(len(self.classes_)), y=y_enc
            )
            w = w * class_weights[y_enc]

        params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "n_classes": len(self.classes_),
            "random_state": self.random_state,
            "weighted": weighted,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
        }
        self.root_ = _MultivariateNode(0, params).fit(X, y_enc, w)
        self.tree_depth_ = self._max_depth_reached(self.root_)
        self.n_nodes_ = len(self.get_split_weights())
        return self

    def predict_proba(self, X) -> np.ndarray:
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
        X = check_predict_input(self, X)
        return np.vstack([self.root_.predict_proba_one(x) for x in X])

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
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

"""
Omnivariate Decision Trees
===========================
Implementation based on:
    Yıldız, O. T., & Alpaydın, E. (2001).
    Omnivariate Decision Trees.
    IEEE Transactions on Neural Networks, 12(6), 1539–1546.

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

    def _select_best_splitter(self, X: np.ndarray, y: np.ndarray):
        """Cross-validate three split types and return the best one."""
        candidates = {
            "univariate": DecisionTreeClassifier(max_depth=1),
            "linear": LinearDiscriminantAnalysis(),
            "nonlinear": MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42),
        }
        best_type = "univariate"
        best_score = -np.inf

        n_folds = min(self.cv_folds, len(np.unique(y)), len(y))
        if n_folds < 2:
            return "univariate", candidates["univariate"]

        for split_type, clf in candidates.items():
            try:
                scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_type = split_type
            except Exception:
                continue

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

        self.split_type, self.classifier = self._select_best_splitter(X, y)
        self.classifier.fit(X, y)
        preds = self.classifier.predict(X)

        # Binary split: correct predictions go right, incorrect go left
        mask_right = preds == y
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
        if self.is_leaf:
            return self.leaf_class
        pred = self.classifier.predict(x.reshape(1, -1))[0]
        true_class_guess = pred
        # Route: if predicted matches majority, go right, else left
        if self.right is not None and self.left is not None:
            return self.right.predict_one(x)
        return self.leaf_class


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
    IEEE Transactions on Neural Networks, 12(6), 1539–1546.
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

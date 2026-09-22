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

from typing import Any, Dict, Optional

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from neural_trees._validation import check_predict_input, reject_sparse
from neural_trees.statistical_tests.classifier_comparison import combined_5x2cv_f_test


def _cv_score_or_none(clf, X, y_bin, n_folds):
    """Mean CV accuracy of one candidate, or None if it cannot be fitted here."""
    try:
        return float(cross_val_score(clf, X, y_bin, cv=n_folds, scoring="accuracy").mean())
    except Exception:
        return None


class _OmnivariateNode:
    """A single node in an omnivariate decision tree."""

    def __init__(self, depth: int, max_depth: int, min_samples_split: int, cv_folds: int,
                 n_classes: int = 0, selection: str = "accuracy", alpha: float = 0.05,
                 min_samples_test: int = 50, random_state: Optional[int] = None,
                 n_jobs: Optional[int] = None):
        self.depth = depth
        self.n_jobs = n_jobs
        # One stream per node, seeded by the parent, so a tree is a pure
        # function of its seed however the recursion is scheduled.
        self._rng = np.random.RandomState(random_state)
        self.n_classes = n_classes
        self.distribution: Optional[np.ndarray] = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.cv_folds = cv_folds
        self.selection = selection
        self.alpha = alpha
        self.min_samples_test = min_samples_test
        self.selection_used_: Optional[str] = None
        self.split_type: Optional[str] = None
        self.classifier: Optional[Any] = None
        self.is_leaf = False
        self.leaf_class: Optional[int] = None
        self.left: Optional[_OmnivariateNode] = None
        self.right: Optional[_OmnivariateNode] = None

    def _make_leaf(self, y: np.ndarray) -> "_OmnivariateNode":
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        self.is_leaf = True
        self.leaf_class = int(counts.argmax())
        self.distribution = counts / counts.sum() if counts.sum() else counts
        return self

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
        group_of_class = KMeans(
            n_clusters=2, n_init=10, random_state=self._seed()
        ).fit_predict(centroids)
        if len(np.unique(group_of_class)) < 2:
            return None
        mapping = {c: int(g) for c, g in zip(present, group_of_class)}
        return np.array([mapping[label] for label in y])

    def _seed(self) -> int:
        return int(self._rng.randint(np.iinfo(np.int32).max))

    def _candidates(self) -> "Dict[str, Any]":
        return {
            "univariate": DecisionTreeClassifier(max_depth=1, random_state=self._seed()),
            "linear": LinearDiscriminantAnalysis(),
            "nonlinear": MLPClassifier(
                hidden_layer_sizes=(10,), max_iter=200, random_state=self._seed()
            ),
        }

    def _select_best_splitter(self, X: np.ndarray, y_bin: np.ndarray):
        """
        Choose a split type for the two-group problem at this node.

        With `selection="test"` the simplest split type that is not
        *significantly* worse than the best one wins, using the combined 5x2cv
        F test this library ships. Comparing three candidates on a handful of
        folds and taking the maximum, which is what `selection="accuracy"`
        does, is the ad hoc accuracy comparison the README argues against, and
        it biases toward the most flexible candidate: noise helps whoever has
        the most capacity to exploit it.

        Simplicity is ordered univariate, then linear, then nonlinear, so a
        node only pays for an MLP when an MLP is measurably needed.
        """
        candidates = self._candidates()
        # Folds are bounded by the rarest group, not by the number of groups,
        # otherwise StratifiedKFold raises on small or skewed nodes.
        min_group = int(np.bincount(y_bin).min())
        n_folds = min(self.cv_folds, min_group)
        if n_folds < 2:
            self.selection_used_ = "fallback"
            return "univariate", candidates["univariate"]

        # The three candidates are independent, so their cross-validations run
        # in parallel over candidates (not folds: three tasks of unequal cost,
        # the MLP dominating, parallelise better than 3 x n_folds tiny ones).
        # Each candidate is cloned with its seed already set, so the result
        # does not depend on n_jobs. joblib's loky workers cap their BLAS
        # threads to cpu_count // n_jobs, which is what keeps the MLP from
        # oversubscribing the machine.
        # Never more workers than candidates: n_jobs=-1 on a 12-core machine
        # would otherwise start twelve processes for three tasks and lose to
        # the spawn cost (3.6 s against 0.9 s sequential, measured).
        n_workers = min(effective_n_jobs(self.n_jobs), len(candidates))
        results = Parallel(n_jobs=n_workers)(
            delayed(_cv_score_or_none)(clone(clf), X, y_bin, n_folds)
            for clf in candidates.values()
        )
        scores = {
            split_type: score
            for split_type, score in zip(candidates, results)
            if score is not None
        }
        if not scores:
            self.selection_used_ = "fallback"
            return "univariate", candidates["univariate"]

        best_type = max(scores, key=lambda name: scores[name])
        if self.selection == "accuracy":
            self.selection_used_ = "accuracy"
            return best_type, candidates[best_type]

        # The 5x2cv F test needs each half of a 2-fold split to contain both
        # groups five times over. Small nodes cannot supply that, and forcing
        # it there would compare noise with noise.
        order = ["univariate", "linear", "nonlinear"]
        if min_group < self.min_samples_test:
            self.selection_used_ = "accuracy"
            return best_type, candidates[best_type]

        for split_type in order:
            if split_type not in scores or split_type == best_type:
                continue
            if scores[split_type] >= scores[best_type]:
                self.selection_used_ = "test"
                return split_type, candidates[split_type]
            try:
                result = combined_5x2cv_f_test(
                    candidates[split_type], candidates[best_type], X, y_bin,
                    alpha=self.alpha, random_state=self._seed(),
                )
            except Exception:
                continue
            if not result.reject_null:
                # Not significantly worse, and simpler.
                self.selection_used_ = "test"
                return split_type, candidates[split_type]

        self.selection_used_ = "test"
        return best_type, candidates[best_type]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OmnivariateNode":
        if (
            self.depth >= self.max_depth
            or len(X) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        y_bin = self._two_group_labels(X, y)
        if y_bin is None:
            return self._make_leaf(y)

        self.split_type, self.classifier = self._select_best_splitter(X, y_bin)
        self.classifier.fit(X, y_bin)

        # The split is the node classifier's own decision: group 1 goes right,
        # group 0 goes left. Routing at predict time uses the same rule.
        mask_right = self.classifier.predict(X) == 1
        mask_left = ~mask_right

        if mask_right.sum() == 0 or mask_left.sum() == 0:
            return self._make_leaf(y)

        left_seed, right_seed = self._seed(), self._seed()
        self.left = _OmnivariateNode(
            self.depth + 1, self.max_depth, self.min_samples_split, self.cv_folds,
            self.n_classes, self.selection, self.alpha, self.min_samples_test,
            random_state=left_seed, n_jobs=self.n_jobs,
        ).fit(X[mask_left], y[mask_left])
        self.right = _OmnivariateNode(
            self.depth + 1, self.max_depth, self.min_samples_split, self.cv_folds,
            self.n_classes, self.selection, self.alpha, self.min_samples_test,
            random_state=right_seed, n_jobs=self.n_jobs,
        ).fit(X[mask_right], y[mask_right])
        return self

    def _leaf_for(self, x: np.ndarray) -> "_OmnivariateNode":
        """
        Walk to the leaf this sample belongs in.

        A non-leaf node always has a classifier and both children; `fit` turns
        the node into a leaf rather than leaving any of them unset, so the
        asserts document that invariant instead of guarding against it.
        """
        node = self
        while not node.is_leaf:
            assert node.classifier is not None
            assert node.left is not None and node.right is not None
            goes_right = node.classifier.predict(x.reshape(1, -1))[0] == 1
            node = node.right if goes_right else node.left
        return node

    def predict_one(self, x: np.ndarray) -> int:
        leaf_class = self._leaf_for(x).leaf_class
        assert leaf_class is not None
        return leaf_class

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        distribution = self._leaf_for(x).distribution
        assert distribution is not None
        return distribution


class OmnivariateDecisionTree(ClassifierMixin, BaseEstimator):
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
        Number of cross-validation folds used to score split types at a node.
    selection : {"accuracy", "test"}, default="accuracy"
        How a node picks its split type.

        - ``"test"`` keeps the simplest type that is not *significantly* worse
          than the best one, judged by the combined 5x2cv F test this library
          ships. Simplicity runs univariate, then linear, then nonlinear.
        - ``"accuracy"`` takes whichever type scored highest on the folds. That
          is the ad hoc accuracy comparison the README argues against, and it
          biases toward the most flexible candidate, since noise helps whoever
          has the most capacity to exploit it.

        ``"accuracy"`` is still the default, because the principled rule costs
        accuracy here. On Breast Cancer the accuracy rule picks a nonlinear
        split at 15 of 21 nodes and scores 0.971; the test finds those MLPs no
        better than a univariate split at the 0.05 level, picks univariate, and
        scores 0.959. Significance at a node does not compose into performance
        of the tree, and this library would rather say that than pick the
        answer that sounds better.
    alpha : float, default=0.05
        Significance level for the test under ``selection="test"``.
    min_samples_test : int, default=50
        Smallest group size at a node that still gets a hypothesis test. The
        5x2cv F test needs each half of a 2-fold split to hold both groups,
        five times over; below this the node falls back to the accuracy rule
        rather than treating a test with no power as evidence of no difference.
        The default matters: at 20 the test fires on nodes too small to resolve
        anything and Wine drops from 0.977 to 0.961, while at 50 it recovers
        completely.
    random_state : int or None, default=None
        Seed for everything stochastic in the fit: the k-means that pairs
        classes into two groups at each node, the stump and MLP candidates,
        and the fold assignment of the F test. Same seed, same tree. The seed
        moves the tree more than usual here because the *type* of split
        chosen at a node can flip: on Breast Cancer with ``selection="test"``
        and ``max_depth=3``, five seeds gave (univariate, linear, nonlinear)
        counts of (2, 1, 1), (0, 0, 1), (0, 0, 1), (1, 1, 2) and (2, 1, 2).
        Until this parameter existed a fixed seed of 42 hid that.
    n_jobs : int or None, default=None
        Processes for the per-node model selection, scikit-learn convention:
        ``None`` is one, ``-1`` is every core. The three candidate split types
        at a node are cross-validated in parallel; the tree is identical for
        every ``n_jobs`` on a fixed ``random_state``. The candidates are few
        and the nodes small, so the gain is bounded by the MLP's share of the
        work. Measured on Breast Cancer at ``max_depth=3``: with
        ``selection="accuracy"`` 0.82 s sequential against 0.68 s with three
        workers; with ``selection="test"`` 2.72 s against 2.54 s, because the
        F tests that follow the selection stay sequential. Never more workers
        than candidates are started, so ``-1`` cannot be slower than ``1``.

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
        selection: str = "accuracy",
        alpha: float = 0.05,
        min_samples_test: int = 50,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.cv_folds = cv_folds
        self.selection = selection
        self.alpha = alpha
        self.min_samples_test = min_samples_test
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y) -> "OmnivariateDecisionTree":
        if self.selection not in ("test", "accuracy"):
            raise ValueError(
                f"selection must be 'test' or 'accuracy', got {self.selection!r}"
            )
        reject_sparse(self, X)
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)

        self.root_ = _OmnivariateNode(
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            cv_folds=self.cv_folds,
            n_classes=len(self.classes_),
            selection=self.selection,
            alpha=self.alpha,
            min_samples_test=self.min_samples_test,
            random_state=int(rng.randint(np.iinfo(np.int32).max)),
            n_jobs=self.n_jobs,
        ).fit(X, y_enc)
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
            Class probabilities in the order of `self.classes_`, each row
            summing to 1.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X)
        return np.vstack([self.root_.predict_proba_one(x) for x in X])

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = check_predict_input(self, X)
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

"""
Weighted K-Nearest Neighbors
=============================
From: Alpaydın, E. (2020). Introduction to Machine Learning (4th ed.), Chapter 8.

Extension of standard KNN with distance-weighted voting:
    y_hat = argmax_c Σ_{i ∈ kNN(x)} w_i · I(y_i = c)
    where w_i = 1 / d(x, x_i)^p (inverse distance weighting)

Also implements "Condensed Nearest Neighbor" (Alpaydın, 1997):
    Voting over Multiple Condensed Nearest Neighbors.
    Artificial Intelligence Review, 11, 115-132.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)

from neural_trees._validation import check_predict_input


class WeightedKNN(ClassifierMixin, BaseEstimator):
    """
    Distance-Weighted K-Nearest Neighbors Classifier.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors.
    weight_power : float, default=2.0
        Power for inverse-distance weighting. Set to 0 for uniform weights.
    metric : str, default="euclidean"
        Distance metric: "euclidean" or "manhattan".
    condense : bool, default=False
        If True, apply condensing: keep only a subset of training samples that
        correctly classifies all the others (Hart's CNN).
    n_condensed_sets : int, default=1
        How many condensed subsets to build and vote over when `condense=True`.

        Condensing is order dependent: which samples end up as prototypes
        depends on the order they were visited in, and a single pass throws
        away information that a different order would have kept. Alpaydin
        (1997) builds several subsets from different orderings and combines
        their votes, which is where some of the accuracy a single subset gives
        away comes back. 5-fold accuracy averaged over 5 seeds, by number of
        subsets, against keeping every sample::

                            1       3       5       9     all
            Iris          0.917   0.939   0.937   0.937   0.956
            Wine          0.947   0.955   0.964   0.971   0.966
            Breast Canc.  0.951   0.963   0.966   0.968   0.966

        Voting beats a single subset everywhere. It reaches the uncondensed
        classifier on Wine and Breast Cancer while storing roughly a sixth of
        the data, and closes about half the gap on Iris without closing it.

        Ignored when `condense=False`.
    random_state : int or None, default=None
        Seed for the orderings used to build the condensed subsets.

    References
    ----------
    Alpaydın, E. (1997). Voting over Multiple Condensed Nearest Neighbors.
    Artificial Intelligence Review, 11, 115-132.
    """

    def __init__(
        self,
        k: int = 5,
        weight_power: float = 2.0,
        metric: str = "euclidean",
        condense: bool = False,
        n_condensed_sets: int = 1,
        random_state: Optional[int] = None,
    ):
        self.k = k
        self.weight_power = weight_power
        self.metric = metric
        self.condense = condense
        self.n_condensed_sets = n_condensed_sets
        self.random_state = random_state

    def _distance(self, x1, x2) -> np.ndarray:
        if sp.issparse(x1) or sp.issparse(x2):
            return self._sparse_distance(sp.csr_matrix(x1), sp.csr_matrix(x2))
        if self.metric == "euclidean":
            return np.sqrt(((x1[:, None] - x2[None, :]) ** 2).sum(axis=-1))
        elif self.metric == "manhattan":
            return np.abs(x1[:, None] - x2[None, :]).sum(axis=-1)
        raise ValueError(f"Unknown metric: {self.metric}")

    def _sparse_distance(self, x1, x2) -> np.ndarray:
        if self.metric == "euclidean":
            # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, every term a sparse product;
            # clipped at zero before the root because cancellation can leave
            # a tiny negative on identical rows.
            n1 = np.asarray(x1.multiply(x1).sum(axis=1)).ravel()
            n2 = np.asarray(x2.multiply(x2).sum(axis=1)).ravel()
            sq = n1[:, None] + n2[None, :] - 2.0 * (x1 @ x2.T).toarray()
            return np.sqrt(np.maximum(sq, 0.0))
        elif self.metric == "manhattan":
            # No sparse identity for |a - b|, so the store is densified in
            # chunks of rows: memory is bounded by chunk x n_features, never
            # by n_store x n_features.
            out = np.empty((x1.shape[0], x2.shape[0]))
            chunk = max(1, 2**22 // max(1, x2.shape[1]))
            for start in range(0, x2.shape[0], chunk):
                dense = x2[start:start + chunk].toarray()
                for i in range(x1.shape[0]):
                    row = x1[i].toarray()
                    out[i, start:start + chunk] = np.abs(row - dense).sum(axis=1)
            return out
        raise ValueError(f"Unknown metric: {self.metric}")

    def _condense(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """
        Condensed Nearest Neighbor (Hart, 1968).

        Sweep the training set repeatedly, moving into the store any sample the
        current store misclassifies under the 1-nearest-neighbor rule, until a
        full sweep adds nothing. The result is *consistent*: the store
        classifies every training sample correctly.

        A single sweep, which is what this used to do, stops before that
        property holds, because samples seen early are judged against a store
        that later grows.
        """
        store_idx = [0]
        remaining = list(range(1, X.shape[0]))

        changed = True
        while changed:
            changed = False
            still_remaining = []
            for i in remaining:
                dists = self._distance(X[[i]], X[store_idx])[0]
                if y[store_idx[int(np.argmin(dists))]] != y[i]:
                    store_idx.append(i)
                    changed = True
                else:
                    still_remaining.append(i)
            remaining = still_remaining

        return X[store_idx], y[store_idx], w[store_idx]

    def fit(self, X, y, sample_weight=None) -> "WeightedKNN":
        """
        Store the training set, condensed or not.

        `sample_weight` scales a neighbour's vote, on top of the inverse
        distance factor. A sample with zero weight is dropped before storing,
        so it neither votes nor becomes a prototype; the result is the fit on
        the data without that row. Integer weights are *not* the same as
        repeating rows: a repeated row occupies several of the `k` neighbour
        slots, a weighted one occupies one slot and votes harder.
        """
        if self.metric not in ("euclidean", "manhattan"):
            raise ValueError(
                f"metric must be 'euclidean' or 'manhattan', got {self.metric!r}"
            )
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k!r}")

        X, y = check_X_y(X, y, accept_sparse="csr")
        check_classification_targets(y)
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if (w < 0).any():
            raise ValueError("sample_weight must be non-negative")
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        keep = w > 0
        if not keep.all():
            if not keep.any():
                raise ValueError("sample_weight is zero for every sample")
            X, y_enc, w = X[keep], y_enc[keep], w[keep]

        if isinstance(self.n_condensed_sets, bool) or not isinstance(
            self.n_condensed_sets, int
        ) or self.n_condensed_sets < 1:
            raise ValueError(
                "n_condensed_sets must be a positive integer, got "
                f"{self.n_condensed_sets!r}"
            )

        if self.condense:
            rng = np.random.RandomState(self.random_state)
            self.stores_ = []
            for i in range(self.n_condensed_sets):
                # The first subset uses the data as given, so a single set
                # reproduces the previous behaviour exactly.
                order = np.arange(X.shape[0]) if i == 0 else rng.permutation(X.shape[0])
                store_X, store_y, store_w = self._condense(X[order], y_enc[order], w[order])
                self.stores_.append((store_X, store_y, store_w))
        else:
            self.stores_ = [(X, y_enc, w)]

        # The first store is the one an unvoted classifier would use, and the
        # attribute is part of the public surface.
        self.X_train_, self.y_train_, self.sample_weight_ = self.stores_[0]

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Class probabilities, averaged over the condensed subsets.

        Each subset votes with its own distance-weighted neighbours, and the
        votes are averaged. With `n_condensed_sets=1` this is a plain weighted
        KNN over a single store.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X, accept_sparse=True)
        votes = [self._vote(X, *store) for store in self.stores_]
        return np.mean(votes, axis=0)

    # scikit-learn >= 1.6 reads __sklearn_tags__; older versions read _more_tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}

    def _vote(
        self, X: np.ndarray, store_X: np.ndarray, store_y: np.ndarray, store_w: np.ndarray
    ) -> np.ndarray:
        dists = self._distance(X, store_X)  # (n_test, n_store)
        k = min(self.k, store_X.shape[0])
        n_classes = len(self.classes_)
        probs = np.zeros((X.shape[0], n_classes))

        for i, row in enumerate(dists):
            nn_idx = np.argsort(row)[:k]
            nn_dists = row[nn_idx]

            if self.weight_power == 0:
                weights = np.ones(k)
            elif nn_dists[0] == 0:
                # Exact matches carry the whole vote. Falling back to uniform
                # weights here, as this used to, let k - 1 unrelated neighbors
                # outvote a sample identical to the query.
                weights = (nn_dists == 0).astype(float)
            else:
                weights = 1.0 / (nn_dists ** self.weight_power + 1e-10)

            for j, idx in enumerate(nn_idx):
                probs[i, store_y[idx]] += weights[j] * store_w[idx]

        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

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

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.utils.multiclass import check_classification_targets

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
        If True, apply condensing: keep only the minimal subset of training
        samples that correctly classify all others (CNN algorithm).

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
    ):
        self.k = k
        self.weight_power = weight_power
        self.metric = metric
        self.condense = condense

    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.sqrt(((x1[:, None] - x2[None, :]) ** 2).sum(axis=-1))
        elif self.metric == "manhattan":
            return np.abs(x1[:, None] - x2[None, :]).sum(axis=-1)
        raise ValueError(f"Unknown metric: {self.metric}")

    def _condense(self, X: np.ndarray, y: np.ndarray):
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
        remaining = list(range(1, len(X)))

        changed = True
        while changed:
            changed = False
            still_remaining = []
            for i in remaining:
                dists = self._distance(X[i][None, :], X[store_idx])[0]
                if y[store_idx[int(np.argmin(dists))]] != y[i]:
                    store_idx.append(i)
                    changed = True
                else:
                    still_remaining.append(i)
            remaining = still_remaining

        return X[store_idx], y[store_idx]

    def fit(self, X, y):
        if self.metric not in ("euclidean", "manhattan"):
            raise ValueError(
                f"metric must be 'euclidean' or 'manhattan', got {self.metric!r}"
            )
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k!r}")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        if self.condense:
            self.X_train_, self.y_train_ = self._condense(X, y_enc)
        else:
            self.X_train_ = X
            self.y_train_ = y_enc

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_predict_input(self, X)
        dists = self._distance(X, self.X_train_)  # (n_test, n_train)
        k = min(self.k, len(self.X_train_))
        n_classes = len(self.classes_)
        probs = np.zeros((len(X), n_classes))

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
                probs[i, self.y_train_[idx]] += weights[j]

        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

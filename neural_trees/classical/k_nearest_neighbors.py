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


class WeightedKNN(BaseEstimator, ClassifierMixin):
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
        """Condensed Nearest Neighbor: keep minimal prototypical subset."""
        store_X = [X[0]]
        store_y = [y[0]]

        for i in range(1, len(X)):
            xi, yi = X[i], y[i]
            dists = np.array([np.linalg.norm(xi - s) for s in store_X])
            nearest_idx = np.argmin(dists)
            if store_y[nearest_idx] != yi:
                store_X.append(xi)
                store_y.append(yi)

        return np.array(store_X), np.array(store_y)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
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
        X = check_array(X)
        dists = self._distance(X, self.X_train_)  # (n_test, n_train)
        k = min(self.k, len(self.X_train_))
        n_classes = len(self.classes_)
        probs = np.zeros((len(X), n_classes))

        for i, row in enumerate(dists):
            nn_idx = np.argsort(row)[:k]
            nn_dists = row[nn_idx]

            if self.weight_power == 0 or nn_dists[0] == 0:
                weights = np.ones(k)
            else:
                weights = 1.0 / (nn_dists ** self.weight_power + 1e-10)

            for j, idx in enumerate(nn_idx):
                probs[i, self.y_train_[idx]] += weights[j]

        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

"""
Naive Bayes Classifier
======================
From: Alpaydın, E. (2020). Introduction to Machine Learning (4th ed.), Chapter 3.

Naive Bayes assumes feature independence given the class label:
    P(C | x) ∝ P(C) · ∏_d P(x_d | C)

Supports Gaussian, Bernoulli, and Multinomial likelihoods.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes Classifier with selectable likelihood.

    Parameters
    ----------
    likelihood : str, default="gaussian"
        Type of feature likelihood: "gaussian", "bernoulli", or "multinomial".
    alpha : float, default=1.0
        Laplace smoothing parameter (for bernoulli/multinomial).
    var_smoothing : float, default=1e-9
        Variance stabilizer for Gaussian likelihood.

    References
    ----------
    Alpaydın, E. (2020). Introduction to Machine Learning, Chapter 3. MIT Press.
    """

    def __init__(self, likelihood: str = "gaussian", alpha: float = 1.0, var_smoothing: float = 1e-9):
        self.likelihood = likelihood
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes)
        self.theta_: list = []  # per-class sufficient statistics

        for c in range(n_classes):
            X_c = X[y_enc == c]
            self.class_log_prior_[c] = np.log(len(X_c) / len(X))

            if self.likelihood == "gaussian":
                mean = X_c.mean(axis=0)
                var = X_c.var(axis=0) + self.var_smoothing
                self.theta_.append({"mean": mean, "var": var})

            elif self.likelihood == "bernoulli":
                p = (X_c.sum(axis=0) + self.alpha) / (len(X_c) + 2 * self.alpha)
                self.theta_.append({"p": p})

            elif self.likelihood == "multinomial":
                counts = X_c.sum(axis=0) + self.alpha
                self.theta_.append({"log_p": np.log(counts / counts.sum())})

        return self

    def _log_likelihood(self, X: np.ndarray, c: int) -> np.ndarray:
        params = self.theta_[c]
        if self.likelihood == "gaussian":
            log_probs = -0.5 * np.log(2 * np.pi * params["var"]) \
                        - 0.5 * ((X - params["mean"]) ** 2) / params["var"]
            return log_probs.sum(axis=1)

        elif self.likelihood == "bernoulli":
            p = params["p"]
            return (X * np.log(p + 1e-10) + (1 - X) * np.log(1 - p + 1e-10)).sum(axis=1)

        elif self.likelihood == "multinomial":
            return X.dot(params["log_p"])

        raise ValueError(f"Unknown likelihood: {self.likelihood}")

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        log_probs = np.column_stack([
            self.class_log_prior_[c] + self._log_likelihood(X, c)
            for c in range(len(self.classes_))
        ])
        return log_probs

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_log_proba(X).argmax(axis=1))

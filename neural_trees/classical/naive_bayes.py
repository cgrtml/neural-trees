"""
Naive Bayes Classifier
======================
From: Alpaydın, E. (2020). Introduction to Machine Learning (4th ed.), Chapter 3.

Naive Bayes assumes feature independence given the class label:
    P(C | x) ∝ P(C) · ∏_d P(x_d | C)

Supports Gaussian, Bernoulli, and Multinomial likelihoods.
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
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


def _weighted_sum(X, w):
    """w @ X for a dense or CSR X, as a 1-d array."""
    if sp.issparse(X):
        return np.asarray(X.T @ w).ravel()
    return w @ X


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
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

    def fit(self, X, y, sample_weight=None) -> "NaiveBayesClassifier":
        """
        Fit the class priors and per-class sufficient statistics.

        `sample_weight` enters exactly: priors are weighted class totals over
        the weighted total, and every per-class statistic is a weighted
        moment or count. Integer weights therefore reproduce the fit on the
        dataset with each row repeated that many times, up to the order of
        floating-point summation. A class whose total weight is zero keeps
        its place in `classes_` with a prior of zero.
        """
        if self.likelihood not in ("gaussian", "bernoulli", "multinomial"):
            raise ValueError(
                "likelihood must be 'gaussian', 'bernoulli' or 'multinomial', got "
                f"{self.likelihood!r}"
            )

        X, y = check_X_y(X, y, accept_sparse="csr")
        check_classification_targets(y)
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes)
        self.theta_: list = []  # per-class sufficient statistics

        for c in range(n_classes):
            X_c = X[y_enc == c]
            w_c = w[y_enc == c]
            total = w_c.sum()
            with np.errstate(divide="ignore"):
                self.class_log_prior_[c] = np.log(total / w.sum())

            if self.likelihood == "gaussian":
                if total > 0 and sp.issparse(X_c):
                    # Weighted moments without densifying: E[x^2] - E[x]^2.
                    mean = np.asarray(X_c.T @ w_c).ravel() / total
                    second = np.asarray(X_c.multiply(X_c).T @ w_c).ravel() / total
                    var = np.maximum(second - mean**2, 0.0)
                elif total > 0:
                    mean = np.average(X_c, axis=0, weights=w_c)
                    var = np.average((X_c - mean) ** 2, axis=0, weights=w_c)
                else:
                    mean, var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
                self.theta_.append({"mean": mean, "var": var + self.var_smoothing})

            elif self.likelihood == "bernoulli":
                p = (_weighted_sum(X_c, w_c) + self.alpha) / (total + 2 * self.alpha)
                self.theta_.append({"p": p})

            elif self.likelihood == "multinomial":
                counts = _weighted_sum(X_c, w_c) + self.alpha
                self.theta_.append({"log_p": np.log(counts / counts.sum())})

        return self

    def _log_likelihood(self, X, c: int) -> np.ndarray:
        params = self.theta_[c]
        if self.likelihood == "gaussian":
            mean, var = params["mean"], params["var"]
            if sp.issparse(X):
                # (x - m)^2 / v expanded so every term is a sparse product:
                # x^2 . (1/v)  -  2 x . (m/v)  +  sum(m^2 / v). Same quantity as
                # the dense branch, summed in a different order.
                quad = (
                    np.asarray(X.multiply(X) @ (1.0 / var)).ravel()
                    - 2.0 * np.asarray(X @ (mean / var)).ravel()
                    + float((mean**2 / var).sum())
                )
                return -0.5 * np.log(2 * np.pi * var).sum() - 0.5 * quad
            log_probs = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            return log_probs.sum(axis=1)

        elif self.likelihood == "bernoulli":
            p = params["p"]
            log_p, log_q = np.log(p + 1e-10), np.log(1 - p + 1e-10)
            if sp.issparse(X):
                # x log p + (1 - x) log q  =  x (log p - log q) + sum(log q)
                return np.asarray(X @ (log_p - log_q)).ravel() + float(log_q.sum())
            return (X * log_p + (1 - X) * log_q).sum(axis=1)

        elif self.likelihood == "multinomial":
            return np.asarray(X @ params["log_p"]).ravel()

        raise ValueError(f"Unknown likelihood: {self.likelihood}")

    # scikit-learn >= 1.6 reads __sklearn_tags__; older versions read _more_tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}

    def _joint_log_likelihood(self, X):
        """Unnormalized log P(y, x) per class, shape (n_samples, n_classes)."""
        return np.column_stack([
            self.class_log_prior_[c] + self._log_likelihood(X, c)
            for c in range(len(self.classes_))
        ])

    def predict_log_proba(self, X) -> np.ndarray:
        """
        Log of the posterior class probabilities, shape (n_samples, n_classes).

        Normalized, so `np.exp(predict_log_proba(X))` equals `predict_proba(X)`
        and each row of the exponential sums to 1. The unnormalized joint
        log-likelihood is available as `_joint_log_likelihood`.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X, accept_sparse=True)
        joint = self._joint_log_likelihood(X)
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities, shape (n_samples, n_classes)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_log_proba(X).argmax(axis=1))

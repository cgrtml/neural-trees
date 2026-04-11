"""
Statistical Tests for Comparing Classifiers
============================================
Implementations based on:

    Alpaydın, E. (1999).
    Combined 5x2cv F Test for Comparing Supervised Classification Learning Algorithms.
    Neural Computation, 11(8), 1885–1892.

    Dietterich, T. G. (1998).
    Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.
    Neural Computation, 10(7), 1895–1923.

These tests are the standard statistical tools for comparing two classifiers
on the same dataset. Alpaydın's combined 5×2cv F test is considered the gold
standard: more powerful than McNemar's test and more reliable than the paired t-test.

Usage
-----
>>> from neural_trees.statistical_tests import combined_5x2cv_f_test
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.svm import SVC
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> result = combined_5x2cv_f_test(DecisionTreeClassifier(), SVC(), X, y)
>>> print(result)
"""

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y
from dataclasses import dataclass
from typing import Any


@dataclass
class TestResult:
    """Result of a statistical comparison test."""

    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    test_name: str
    interpretation: str

    def __repr__(self):
        symbol = "✓ REJECT H0" if self.reject_null else "✗ FAIL TO REJECT H0"
        return (
            f"StatisticalTestResult(\n"
            f"  test       = {self.test_name}\n"
            f"  statistic  = {self.statistic:.4f}\n"
            f"  p-value    = {self.p_value:.4f}\n"
            f"  alpha      = {self.alpha}\n"
            f"  decision   = {symbol}\n"
            f"  note       = {self.interpretation}\n"
            f")"
        )


def combined_5x2cv_f_test(
    clf_A: Any,
    clf_B: Any,
    X,
    y,
    alpha: float = 0.05,
    random_state: int = 42,
) -> TestResult:
    """
    Alpaydın's Combined 5×2 Cross-Validation F Test.

    Compares two classifiers by repeating 2-fold CV 5 times (giving 10
    difference measurements) and computing an F statistic.

    H0: The two classifiers have equal expected error rates.
    H1: The classifiers differ.

    This test avoids the high variance of the standard paired t-test and
    the loss of power in McNemar's test. It is the recommended method
    for classifier comparison (Alpaydın, 1999).

    Parameters
    ----------
    clf_A : sklearn-compatible classifier
    clf_B : sklearn-compatible classifier
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    alpha : float, default=0.05
        Significance level.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    TestResult

    References
    ----------
    Alpaydın, E. (1999). Combined 5x2cv F Test for Comparing Supervised
    Classification Learning Algorithms. Neural Computation, 11(8), 1885–1892.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> result = combined_5x2cv_f_test(DecisionTreeClassifier(), KNeighborsClassifier(), X, y)
    >>> print(result)
    """
    X, y = check_X_y(X, y)
    rng = np.random.RandomState(random_state)

    diffs = []  # p^(i)_1 - p^(i)_2 for each fold in each repetition
    sq_diffs = []  # squared differences for variance estimation

    for rep in range(5):
        seed = rng.randint(0, 10000)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        rep_diffs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf_a = clone(clf_A)
            clf_b = clone(clf_B)
            clf_a.fit(X_train, y_train)
            clf_b.fit(X_train, y_train)

            err_a = 1.0 - (clf_a.predict(X_test) == y_test).mean()
            err_b = 1.0 - (clf_b.predict(X_test) == y_test).mean()
            rep_diffs.append(err_a - err_b)

        diffs.extend(rep_diffs)
        # Variance estimate for this repetition
        mean_rep = np.mean(rep_diffs)
        for d in rep_diffs:
            sq_diffs.append((d - mean_rep) ** 2)

    diffs = np.array(diffs)  # shape (10,)

    # F statistic = (sum p^2) / (2 * sum s^2)
    numerator = np.sum(diffs ** 2)
    denominator = 2.0 * sum(sq_diffs)

    if denominator < 1e-12:
        f_stat = 0.0
    else:
        f_stat = numerator / denominator

    # F distribution with 10 and 5 degrees of freedom
    p_value = 1 - stats.f.cdf(f_stat, dfn=10, dfd=5)
    reject = p_value < alpha

    return TestResult(
        statistic=f_stat,
        p_value=p_value,
        reject_null=reject,
        alpha=alpha,
        test_name="Alpaydın's Combined 5×2cv F Test",
        interpretation=(
            "Classifiers significantly differ" if reject
            else "No significant difference found"
        ),
    )


def mcnemar_test(
    y_true,
    y_pred_A,
    y_pred_B,
    alpha: float = 0.05,
) -> TestResult:
    """
    McNemar's Test for comparing two classifiers on the same test set.

    Examines the contingency table of disagreements between two classifiers.
    Less powerful than the 5×2cv F test but useful when you only have
    a fixed test set.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred_A : array-like of shape (n_samples,)
        Predictions from classifier A.
    y_pred_B : array-like of shape (n_samples,)
        Predictions from classifier B.
    alpha : float, default=0.05

    Returns
    -------
    TestResult
    """
    y_true = np.asarray(y_true)
    y_pred_A = np.asarray(y_pred_A)
    y_pred_B = np.asarray(y_pred_B)

    correct_A = y_pred_A == y_true
    correct_B = y_pred_B == y_true

    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = (~correct_A & correct_B).sum()
    n10 = (correct_A & ~correct_B).sum()

    if n01 + n10 == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # With continuity correction
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    reject = p_value < alpha

    return TestResult(
        statistic=chi2,
        p_value=p_value,
        reject_null=reject,
        alpha=alpha,
        test_name="McNemar's Test",
        interpretation=(
            "Classifiers significantly differ" if reject
            else "No significant difference found"
        ),
    )


def paired_t_test(
    clf_A: Any,
    clf_B: Any,
    X,
    y,
    k_folds: int = 10,
    alpha: float = 0.05,
    random_state: int = 42,
) -> TestResult:
    """
    Paired t-test for classifier comparison.

    Splits data into k folds, trains/tests both classifiers on each fold,
    and applies a paired t-test on the k accuracy differences.

    Note: This test has inflated Type I error due to the non-independence
    of overlapping training sets. Prefer the 5×2cv F test (Alpaydın, 1999).

    Parameters
    ----------
    clf_A, clf_B : sklearn-compatible classifiers
    X : array-like
    y : array-like
    k_folds : int, default=10
    alpha : float, default=0.05
    random_state : int, default=42

    Returns
    -------
    TestResult
    """
    X, y = check_X_y(X, y)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    diffs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        a = clone(clf_A)
        b = clone(clf_B)
        a.fit(X_train, y_train)
        b.fit(X_train, y_train)

        acc_a = (a.predict(X_test) == y_test).mean()
        acc_b = (b.predict(X_test) == y_test).mean()
        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)
    t_stat, p_value = stats.ttest_1samp(diffs, 0)
    reject = p_value < alpha

    return TestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        reject_null=reject,
        alpha=alpha,
        test_name=f"Paired t-test ({k_folds}-fold CV)",
        interpretation=(
            "Classifiers significantly differ" if reject
            else "No significant difference found"
        ),
    )

"""
Estimator contract checks that apply to every classifier in the library.

These mirror the parts of `sklearn.utils.estimator_checks.check_estimator`
that used to fail: predicting with the wrong number of features, fitting on a
continuous target, and predicting before fitting.
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from neural_trees import (
    GALNetwork,
    HierarchicalMixtureOfExperts,
    MultivariateDecisionTree,
    NaiveBayesClassifier,
    OmnivariateDecisionTree,
    SoftDecisionTree,
    WeightedKNN,
)

ESTIMATORS = [
    lambda: SoftDecisionTree(depth=2, max_epochs=3, random_state=0),
    lambda: MultivariateDecisionTree(max_depth=2, random_state=0),
    lambda: OmnivariateDecisionTree(max_depth=2),
    lambda: HierarchicalMixtureOfExperts(depth=1, max_epochs=3, random_state=0),
    lambda: GALNetwork(max_epochs=5, random_state=0),
    lambda: WeightedKNN(),
    lambda: NaiveBayesClassifier(),
]
IDS = [make().__class__.__name__ for make in ESTIMATORS]


@pytest.fixture(scope="module")
def iris():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_mixin_order_puts_classifier_first(make_estimator):
    """
    ClassifierMixin must precede BaseEstimator, otherwise the more general
    class wins the MRO and sklearn reports the estimator as misconfigured.
    """
    mro = type(make_estimator()).__mro__
    assert mro.index(ClassifierMixin) < mro.index(BaseEstimator)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_predict_rejects_wrong_number_of_features(make_estimator, iris):
    X, y = iris
    estimator = make_estimator().fit(X, y)

    with pytest.raises(ValueError, match="features, but"):
        estimator.predict(X[:, :2])


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_fit_rejects_a_continuous_target(make_estimator, iris):
    X, _ = iris
    continuous = np.linspace(0.0, 1.0, len(X))

    with pytest.raises(ValueError):
        make_estimator().fit(X, continuous)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_predict_before_fit_raises_not_fitted(make_estimator, iris):
    X, _ = iris

    with pytest.raises(NotFittedError):
        make_estimator().predict(X)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_predict_accepts_a_reversed_view(make_estimator, iris):
    """
    A reversed slice is a negatively strided array. torch refuses to build a
    tensor from one, so the torch-backed estimators used to raise ValueError on
    input as ordinary as `X[::-1]`.
    """
    X, y = iris
    estimator = make_estimator().fit(X, y)

    forward = estimator.predict(X[:40])
    reversed_back = estimator.predict(X[:40][::-1])[::-1]
    assert np.array_equal(forward, reversed_back)


def test_naive_bayes_log_proba_is_normalized(iris):
    """
    predict_log_proba must be the log of predict_proba, not the unnormalized
    joint log-likelihood: exponentiated rows used to sum to values above 3.
    """
    X, y = iris
    nb = NaiveBayesClassifier().fit(X, y)

    log_proba = nb.predict_log_proba(X)
    np.testing.assert_allclose(np.exp(log_proba).sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.exp(log_proba), nb.predict_proba(X), atol=1e-12)
    # The unnormalized quantity is still reachable for those who want it.
    assert nb._joint_log_likelihood(X).shape == log_proba.shape

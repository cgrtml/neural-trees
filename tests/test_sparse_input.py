"""scipy sparse input: accepted where cheap, refused clearly elsewhere (#100)."""
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import load_digits, load_iris
from sklearn.utils.estimator_checks import check_estimator

from neural_trees import (
    GALNetwork,
    HierarchicalMixtureOfExperts,
    MultivariateDecisionTree,
    NaiveBayesClassifier,
    OmnivariateDecisionTree,
    SoftDecisionTree,
    SoftDecisionTreeRegressor,
    WeightedKNN,
)


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return X[:600], y[:600]


@pytest.mark.parametrize("likelihood", ["gaussian", "bernoulli", "multinomial"])
def test_naive_bayes_sparse_equals_dense(digits, likelihood):
    X, y = digits
    if likelihood == "bernoulli":
        X = (X > 7).astype(float)
    Xs = sp.csr_matrix(X)
    dense = NaiveBayesClassifier(likelihood=likelihood).fit(X, y)
    sparse = NaiveBayesClassifier(likelihood=likelihood).fit(Xs, y)
    assert np.array_equal(dense.predict(X), sparse.predict(Xs))
    assert np.array_equal(dense.predict(X), sparse.predict(X))
    assert np.array_equal(dense.predict(Xs), sparse.predict(Xs))
    np.testing.assert_allclose(dense.predict_proba(X), sparse.predict_proba(Xs), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dense.class_log_prior_, sparse.class_log_prior_)
    for a, b in zip(dense.theta_, sparse.theta_):
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("condense", [False, True])
def test_knn_sparse_equals_dense(digits, metric, condense):
    X, y = digits
    Xs = sp.csr_matrix(X)
    dense = WeightedKNN(k=5, metric=metric, condense=condense, random_state=0).fit(X, y)
    sparse = WeightedKNN(k=5, metric=metric, condense=condense, random_state=0).fit(Xs, y)
    assert np.array_equal(dense.predict(X), sparse.predict(Xs))
    np.testing.assert_allclose(dense.predict_proba(X), sparse.predict_proba(Xs), rtol=1e-9, atol=1e-12)
    if condense:
        assert dense.X_train_.shape == sparse.X_train_.shape
        assert np.array_equal(dense.X_train_, sparse.X_train_.toarray())


def test_sparse_weighted_sum_with_zero_weights(digits):
    X, y = digits
    w = (np.arange(len(y)) % 3 != 0).astype(float)
    a = NaiveBayesClassifier().fit(X, y, sample_weight=w)
    b = NaiveBayesClassifier().fit(sp.csr_matrix(X), y, sample_weight=w)
    assert np.array_equal(a.predict(X), b.predict(X))
    c = WeightedKNN().fit(sp.csr_matrix(X), y, sample_weight=w)
    assert c.X_train_.shape[0] == int(w.sum())


@pytest.mark.parametrize(
    "make",
    [
        lambda: SoftDecisionTree(depth=2, max_epochs=2),
        lambda: SoftDecisionTreeRegressor(depth=2, max_epochs=2),
        lambda: HierarchicalMixtureOfExperts(depth=1, max_epochs=2),
        lambda: GALNetwork(max_epochs=2),
        lambda: MultivariateDecisionTree(max_depth=2),
        lambda: OmnivariateDecisionTree(max_depth=2),
    ],
    ids=["soft", "regressor", "hme", "gal", "multivariate", "omnivariate"],
)
def test_dense_only_models_name_the_fix(make):
    X, y = load_iris(return_X_y=True)
    Xs = sp.csr_matrix(X)
    with pytest.raises(TypeError, match="requires dense input.*toarray"):
        make().fit(Xs, y)
    est = make().fit(X, y)
    with pytest.raises(TypeError, match="requires dense input.*toarray"):
        est.predict(Xs)


def _failures(est):
    return [r["check_name"] for r in check_estimator(est, on_fail=None) if r["status"] == "failed"]


def test_sparse_tag_runs_the_sparse_checks():
    # With the tag declared, sklearn's sparse-data checks run and pass.
    assert _failures(NaiveBayesClassifier()) == []
    # The KNN's one design-time exception now appears twice, once per layout.
    assert _failures(WeightedKNN()) == [
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
    ]

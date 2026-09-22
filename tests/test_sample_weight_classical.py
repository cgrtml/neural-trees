"""sample_weight in the two classical models (#94)."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.utils.estimator_checks import check_estimator

from neural_trees import NaiveBayesClassifier, WeightedKNN


@pytest.fixture(scope="module")
def wine():
    X, y = load_wine(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


def _failures(est):
    return [r["check_name"] for r in check_estimator(est, on_fail=None) if r["status"] == "failed"]


# ── Naive Bayes: exact, so integer weights are repeated rows ──


@pytest.mark.parametrize("likelihood", ["gaussian", "bernoulli", "multinomial"])
def test_naive_bayes_integer_weights_equal_repeated_rows(likelihood):
    rng = np.random.RandomState(0)
    X, y = load_iris(return_X_y=True)
    if likelihood == "bernoulli":
        X = (X > X.mean(0)).astype(float)
    w = rng.randint(0, 4, size=len(y))
    a = NaiveBayesClassifier(likelihood=likelihood).fit(X, y, sample_weight=w)
    b = NaiveBayesClassifier(likelihood=likelihood).fit(np.repeat(X, w, axis=0), np.repeat(y, w))
    np.testing.assert_allclose(a.class_log_prior_, b.class_log_prior_, rtol=1e-12)
    for ta, tb in zip(a.theta_, b.theta_):
        for key in ta:
            np.testing.assert_allclose(ta[key], tb[key], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=1e-9)
    assert (a.predict(X) == b.predict(X)).all()


def test_naive_bayes_unit_weights_match_unweighted_exactly(wine):
    X, y = wine
    a = NaiveBayesClassifier().fit(X, y)
    b = NaiveBayesClassifier().fit(X, y, sample_weight=np.ones(len(y)))
    assert np.array_equal(a.class_log_prior_, b.class_log_prior_)
    for ta, tb in zip(a.theta_, b.theta_):
        for key in ta:
            np.testing.assert_allclose(ta[key], tb[key], rtol=1e-14)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=1e-12)


def test_naive_bayes_zero_weight_class_gets_zero_prior(wine):
    X, y = wine
    w = (y != 2).astype(float)
    m = NaiveBayesClassifier().fit(X, y, sample_weight=w)
    assert list(m.classes_) == [0, 1, 2]
    assert m.class_log_prior_[2] == -np.inf
    assert (m.predict_proba(X)[:, 2] == 0).all()
    assert 2 not in m.predict(X)


# ── KNN: weights scale the vote; zero weight removes the row ──


def test_knn_unit_weights_match_unweighted_exactly(wine):
    X, y = wine
    for condense in (False, True):
        a = WeightedKNN(k=5, condense=condense, random_state=0).fit(X, y)
        b = WeightedKNN(k=5, condense=condense, random_state=0).fit(X, y, sample_weight=np.ones(len(y)))
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
        assert np.array_equal(a.X_train_, b.X_train_)


def test_knn_zero_weight_equals_dropping_the_row(wine):
    X, y = wine
    rng = np.random.RandomState(1)
    w = (rng.rand(len(y)) > 0.3).astype(float)
    for condense in (False, True):
        a = WeightedKNN(k=5, condense=condense, random_state=0).fit(X, y, sample_weight=w)
        b = WeightedKNN(k=5, condense=condense, random_state=0).fit(X[w > 0], y[w > 0])
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
        assert np.array_equal(a.X_train_, b.X_train_)
        if condense:
            # No dropped row became a prototype.
            dropped = X[w == 0]
            assert not any((a.X_train_ == row).all(1).any() for row in dropped)


def test_knn_weight_scales_the_vote():
    # Query at the origin; one neighbour of class 0 at distance 1, two of
    # class 1 at distance 1. Unweighted the vote is 1:2; weight 3 on the
    # class-0 neighbour makes it 3:2.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    y = np.array([0, 1, 1])
    q = np.zeros((1, 2))
    plain = WeightedKNN(k=3, weight_power=0).fit(X, y).predict_proba(q)[0]
    np.testing.assert_allclose(plain, [1 / 3, 2 / 3])
    heavy = WeightedKNN(k=3, weight_power=0).fit(X, y, sample_weight=[3.0, 1.0, 1.0]).predict_proba(q)[0]
    np.testing.assert_allclose(heavy, [3 / 5, 2 / 5])
    assert WeightedKNN(k=3, weight_power=0).fit(X, y, sample_weight=[3.0, 1.0, 1.0]).predict(q)[0] == 0


def test_knn_rejects_negative_and_all_zero_weights(wine):
    X, y = wine
    with pytest.raises(ValueError, match="non-negative|[Nn]egative"):
        WeightedKNN().fit(X, y, sample_weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match="zero for every|non-zero"):
        WeightedKNN().fit(X, y, sample_weight=np.zeros(len(y)))


# ── scikit-learn's own checks, now including the sample_weight ones ──


def test_naive_bayes_passes_every_estimator_check():
    assert _failures(NaiveBayesClassifier()) == []


def test_knn_passes_every_estimator_check_but_row_repetition():
    """
    The one failure is by design: sklearn's check asserts that weighting a
    row is the same as repeating it. For a k-nearest-neighbour rule it is
    not, because a repeated row fills several of the k slots while a
    weighted one fills one and votes harder. The docstring says so.
    """
    assert _failures(WeightedKNN()) == [
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
    ]

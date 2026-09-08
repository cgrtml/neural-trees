"""Tests for distance-weighted KNN and its condensed-prototype mode."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import WeightedKNN


@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def test_fit_predict_and_proba(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    knn = WeightedKNN().fit(X_train, y_train)

    preds = knn.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset(set(np.unique(y_train)))
    assert knn.score(X_test, y_test) > 0.9

    proba = knn.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_both_metrics_work(iris_split, metric):
    X_train, X_test, y_train, y_test = iris_split
    knn = WeightedKNN(metric=metric).fit(X_train, y_train)
    assert knn.score(X_test, y_test) > 0.9


def test_manhattan_and_euclidean_differ_somewhere():
    """Otherwise the metric parameter would be decorative."""
    rng = np.random.RandomState(0)
    X = rng.randn(120, 8)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int)
    queries = rng.randn(40, 8)  # not in the training set, so no exact matches

    euclid = WeightedKNN(k=3, metric="euclidean").fit(X, y).predict_proba(queries)
    manhattan = WeightedKNN(k=3, metric="manhattan").fit(X, y).predict_proba(queries)
    assert not np.allclose(euclid, manhattan)


def test_uniform_weights_when_power_is_zero():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    query = np.array([[5.4]])

    weighted = WeightedKNN(k=4, weight_power=2.0).fit(X, y).predict_proba(query)
    uniform = WeightedKNN(k=4, weight_power=0.0).fit(X, y).predict_proba(query)

    np.testing.assert_allclose(uniform, [[0.5, 0.5]])
    assert not np.allclose(weighted, uniform)


def test_exact_match_carries_the_vote():
    """
    A query identical to a training sample must take that sample's label.
    Uniform fallback on a zero distance let k - 1 unrelated neighbors outvote it.
    """
    X = np.array([[0.0], [1.0], [1.1], [1.2], [1.3]])
    y = np.array([0, 1, 1, 1, 1])

    proba = WeightedKNN(k=5, weight_power=2.0).fit(X, y).predict_proba(np.array([[0.0]]))
    np.testing.assert_allclose(proba, [[1.0, 0.0]])


def test_condensed_store_is_consistent_and_smaller():
    """
    Hart's condensing keeps sweeping until the store classifies every training
    sample correctly. A single sweep, which is what this used to do, stops
    short of that.
    """
    X, y = load_iris(return_X_y=True)
    knn = WeightedKNN(condense=True).fit(X, y)

    assert len(knn.X_train_) < len(X)
    distances = knn._distance(X, knn.X_train_)
    nearest_label = knn.y_train_[distances.argmin(axis=1)]
    assert np.array_equal(nearest_label, knn.le_.transform(y))


def test_condensing_keeps_accuracy_close_to_the_full_set():
    X, y = load_iris(return_X_y=True)
    full = Pipeline([("s", StandardScaler()), ("k", WeightedKNN())])
    condensed = Pipeline([("s", StandardScaler()), ("k", WeightedKNN(condense=True))])

    full_score = cross_val_score(full, X, y, cv=5).mean()
    condensed_score = cross_val_score(condensed, X, y, cv=5).mean()
    assert condensed_score > full_score - 0.1


@pytest.mark.parametrize("bad", [{"metric": "cosine"}, {"k": 0}, {"k": -1}, {"k": 2.5}])
def test_invalid_parameters_are_rejected(bad):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError):
        WeightedKNN(**bad).fit(X, y)


def test_k_larger_than_training_set_is_clamped():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    proba = WeightedKNN(k=50).fit(X, y).predict_proba(np.array([[1.5]]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

"""Tests for the GAL (Grow and Learn) constructive network."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import GALNetwork


@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def test_fit_predict_shapes(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    gal = GALNetwork(max_epochs=60, random_state=0)
    gal.fit(X_train, y_train)

    preds = gal.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset(set(np.unique(y_train)))

    proba = gal.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_architecture_grows_and_is_recorded(iris_split):
    X_train, _, y_train, _ = iris_split
    gal = GALNetwork(initial_hidden=2, max_hidden=8, max_epochs=60, random_state=0)
    gal.fit(X_train, y_train)

    assert len(gal.architecture_history_) == 60
    assert all({"epoch", "n_hidden", "error"} <= set(h) for h in gal.architecture_history_)
    assert 1 <= gal.n_hidden_final_ <= 8
    # Growth is the point of GAL: the network should not stay at its seed size
    # while the error is above grow_threshold.
    assert gal.n_hidden_final_ > 2


def test_get_params_set_params_roundtrip():
    gal = GALNetwork(initial_hidden=3, learning_rate=0.05)
    params = gal.get_params()
    assert params["initial_hidden"] == 3

    gal.set_params(initial_hidden=5)
    assert gal.get_params()["initial_hidden"] == 5
    assert clone(gal).get_params()["initial_hidden"] == 5


def test_works_in_pipeline_and_grid_search(iris_split):
    X_train, _, y_train, _ = iris_split
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gal", GALNetwork(max_epochs=20, random_state=0)),
    ])
    search = GridSearchCV(pipe, {"gal__initial_hidden": [2, 4]}, cv=2)
    search.fit(X_train, y_train)

    assert search.best_params_["gal__initial_hidden"] in (2, 4)
    assert 0.0 <= search.best_score_ <= 1.0


def test_random_state_makes_training_reproducible(iris_split):
    X_train, X_test, y_train, _ = iris_split
    first = GALNetwork(max_epochs=30, random_state=7).fit(X_train, y_train)
    second = GALNetwork(max_epochs=30, random_state=7).fit(X_train, y_train)

    assert first.n_hidden_final_ == second.n_hidden_final_
    assert np.array_equal(first.predict(X_test), second.predict(X_test))

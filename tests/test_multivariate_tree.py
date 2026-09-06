"""Tests for the multivariate (linear discriminant split) decision tree."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from neural_trees import MultivariateDecisionTree


@pytest.fixture(scope="module")
def wine_split():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def test_fit_predict_multiclass(wine_split):
    X_train, X_test, y_train, y_test = wine_split
    mdt = MultivariateDecisionTree(max_depth=3, random_state=0).fit(X_train, y_train)

    preds = mdt.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset(set(np.unique(y_train)))
    assert mdt.score(X_test, y_test) > 0.8


def test_predict_proba_is_a_distribution(wine_split):
    X_train, X_test, y_train, _ = wine_split
    mdt = MultivariateDecisionTree(max_depth=3, random_state=0).fit(X_train, y_train)

    proba = mdt.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_splits_are_oblique_and_readable(wine_split):
    """Every internal node must expose a full-length weight vector and a bias."""
    X_train, _, y_train, _ = wine_split
    mdt = MultivariateDecisionTree(max_depth=3, random_state=0).fit(X_train, y_train)

    splits = mdt.get_split_weights()
    assert len(splits) == mdt.n_nodes_ >= 1
    for weights, bias in splits:
        assert weights.shape == (X_train.shape[1],)
        assert np.isfinite(weights).all() and np.isfinite(bias)
        # A univariate stump would put all its mass on one feature.
        assert np.count_nonzero(np.abs(weights) > 1e-8) > 1


def test_beats_cart_on_correlated_oblique_data():
    """
    The point of a multivariate split: one hyperplane where a univariate tree
    needs a staircase. The classes here are separated by a rotated boundary.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(400, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    mdt = cross_val_score(MultivariateDecisionTree(max_depth=2, random_state=0), X, y, cv=5)
    cart = cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=0), X, y, cv=5)

    assert mdt.mean() > cart.mean()
    assert mdt.mean() > 0.95


def test_respects_depth_and_leaf_constraints(wine_split):
    X_train, _, y_train, _ = wine_split
    shallow = MultivariateDecisionTree(max_depth=1, random_state=0).fit(X_train, y_train)
    assert shallow.tree_depth_ <= 1

    conservative = MultivariateDecisionTree(
        max_depth=4, min_impurity_decrease=1.0, random_state=0
    ).fit(X_train, y_train)
    assert conservative.n_nodes_ == 0  # no split can decrease Gini by 1.0
    assert conservative.tree_depth_ == 0


def test_single_class_and_tiny_input():
    X, _ = make_classification(n_samples=40, n_features=5, n_informative=3, random_state=0)
    y = np.zeros(len(X), dtype=int)

    mdt = MultivariateDecisionTree(random_state=0).fit(X, y)
    assert mdt.n_nodes_ == 0
    assert np.array_equal(mdt.predict(X), y)


@pytest.mark.parametrize("max_depth", [0, -1, 2.5, "3", None, True])
def test_max_depth_validation(max_depth):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="max_depth must be a positive integer"):
        MultivariateDecisionTree(max_depth=max_depth).fit(X, y)


def test_reproducible_and_pipeline_compatible():
    X, y = load_wine(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", MultivariateDecisionTree(max_depth=3, random_state=0)),
    ])
    search = GridSearchCV(pipe, {"tree__max_depth": [2, 3]}, cv=3)
    search.fit(X, y)
    assert search.best_params_["tree__max_depth"] in (2, 3)

    first = clone(pipe).fit(X, y).predict(X)
    second = clone(pipe).fit(X, y).predict(X)
    assert np.array_equal(first, second)

"""Tests for Soft Decision Tree."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from alpaydin_ml import SoftDecisionTree


def test_fit_predict_iris():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sdt = SoftDecisionTree(depth=3, max_epochs=20, verbose=False)
    sdt.fit(X_train, y_train)
    preds = sdt.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset(set(y_train))


def test_predict_proba_sums_to_one():
    X, y = load_iris(return_X_y=True)
    sdt = SoftDecisionTree(depth=3, max_epochs=10)
    sdt.fit(X, y)
    proba = sdt.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_accuracy_above_baseline():
    """SDT should beat random chance on iris (>33%)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sdt = SoftDecisionTree(depth=4, max_epochs=30)
    sdt.fit(X_train, y_train)
    acc = sdt.score(X_test, y_test)
    assert acc > 0.5, f"Expected >50% accuracy, got {acc:.2f}"


def test_training_history():
    X, y = load_iris(return_X_y=True)
    sdt = SoftDecisionTree(depth=3, max_epochs=5)
    sdt.fit(X, y)
    assert len(sdt.training_history_) == 5
    assert all("loss" in h and "accuracy" in h for h in sdt.training_history_)


def test_leaf_distributions_shape():
    X, y = load_iris(return_X_y=True)
    sdt = SoftDecisionTree(depth=3, max_epochs=5)
    sdt.fit(X, y)
    leaf_dists = sdt.get_leaf_distributions()
    n_leaves = 2 ** 3
    n_classes = 3
    assert leaf_dists.shape == (n_leaves, n_classes)
    np.testing.assert_allclose(leaf_dists.sum(axis=1), 1.0, atol=1e-5)


def test_split_weights_shape():
    X, y = load_iris(return_X_y=True)
    depth = 3
    sdt = SoftDecisionTree(depth=depth, max_epochs=5)
    sdt.fit(X, y)
    weights = sdt.get_split_weights()
    n_internal = 2 ** depth - 1
    assert len(weights) == n_internal
    assert all(w.shape == (X.shape[1],) for w in weights)


def test_different_depths():
    X, y = load_wine(return_X_y=True)
    for depth in [2, 3, 5]:
        sdt = SoftDecisionTree(depth=depth, max_epochs=10)
        sdt.fit(X, y)
        assert sdt.score(X, y) > 0.3

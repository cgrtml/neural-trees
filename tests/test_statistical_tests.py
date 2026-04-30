"""Tests for statistical classifier comparison tests."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from neural_trees.statistical_tests import (
    combined_5x2cv_f_test,
    mcnemar_test,
    paired_t_test,
)


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris
    return load_iris(return_X_y=True)


def test_5x2cv_returns_test_result(iris_data):
    X, y = iris_data
    result = combined_5x2cv_f_test(
        DecisionTreeClassifier(random_state=0),
        KNeighborsClassifier(),
        X, y,
    )
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert hasattr(result, "reject_null")
    assert 0 <= result.p_value <= 1
    assert result.statistic >= 0


def test_5x2cv_identical_classifiers_same_seed(iris_data):
    """Identical classifiers should not reject H0."""
    X, y = iris_data
    result = combined_5x2cv_f_test(
        DecisionTreeClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        X, y,
    )
    # Two identical classifiers have zero difference, should not reject
    assert not result.reject_null or result.p_value > 0.01


def test_mcnemar_test():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    y_pred_A = (y_true + rng.randint(0, 2, 200)) % 2  # ~50% accuracy
    y_pred_B = y_true.copy()  # perfect
    result = mcnemar_test(y_true, y_pred_A, y_pred_B)
    assert result.reject_null  # should detect difference
    assert result.p_value < 0.05


def test_mcnemar_identical_predictions():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    y_pred = rng.randint(0, 2, 200)
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert not result.reject_null  # identical → no difference


def test_paired_t_test(iris_data):
    X, y = iris_data
    result = paired_t_test(
        DecisionTreeClassifier(random_state=0),
        DummyClassifier(strategy="most_frequent"),
        X, y,
    )
    assert result.reject_null  # DT should clearly beat dummy
    assert result.p_value < 0.01


def test_repr_contains_key_info():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 100)
    result = mcnemar_test(y, y, y)
    r = repr(result)
    assert "p-value" in r
    assert "statistic" in r

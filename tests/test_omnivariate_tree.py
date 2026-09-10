"""Tests for the omnivariate decision tree."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import OmnivariateDecisionTree


@pytest.fixture(scope="module")
def wine_split():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def test_predictions_follow_the_learned_splits(wine_split):
    """
    Regression test for routing: predict_one used to always descend into the
    right child, so every sample landed in the same leaf regardless of its
    features. A working tree must produce more than one distinct prediction.
    """
    X_train, X_test, y_train, y_test = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)

    preds = odt.predict(X_test)
    assert preds.shape == y_test.shape
    assert len(np.unique(preds)) > 1
    assert odt.score(X_test, y_test) > 0.8


def test_beats_the_majority_class_baseline(wine_split):
    X_train, X_test, y_train, y_test = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)

    majority = np.bincount(y_test).max() / len(y_test)
    assert odt.score(X_test, y_test) > majority


def test_split_type_distribution_counts_internal_nodes(wine_split):
    X_train, _, y_train, _ = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)

    counts = odt.get_split_type_distribution()
    assert set(counts) == {"univariate", "linear", "nonlinear"}
    assert sum(counts.values()) >= 1


def test_pure_node_becomes_a_leaf():
    X, _ = load_iris(return_X_y=True)
    y = np.zeros(len(X), dtype=int)

    odt = OmnivariateDecisionTree(max_depth=3).fit(X, y)
    assert sum(odt.get_split_type_distribution().values()) == 0
    assert np.array_equal(odt.predict(X), y)


def test_works_in_a_pipeline():
    X, y = load_iris(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("tree", OmnivariateDecisionTree(max_depth=2)),
    ])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.7


def test_predict_proba_is_a_distribution(wine_split):
    """
    This was the one estimator in the library without predict_proba, so it
    could not be used for ROC AUC, calibration or soft voting.
    """
    X_train, X_test, y_train, _ = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)

    proba = odt.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_predict_proba_agrees_with_predict(wine_split):
    X_train, X_test, y_train, _ = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)

    from_proba = odt.classes_[odt.predict_proba(X_test).argmax(axis=1)]
    assert np.array_equal(from_proba, odt.predict(X_test))


def test_works_with_roc_auc(wine_split):
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test = wine_split
    odt = OmnivariateDecisionTree(max_depth=3).fit(X_train, y_train)
    auc = roc_auc_score(y_test, odt.predict_proba(X_test), multi_class="ovr")
    assert 0.0 <= auc <= 1.0


def test_test_based_selection_prefers_simpler_splits(wine_split):
    """
    The library ships a hypothesis test and its README argues against ad hoc
    accuracy comparisons; `selection="test"` makes the node selection follow
    that advice, keeping the simplest split type that is not significantly
    worse.
    """
    X_train, _, y_train, _ = wine_split

    by_accuracy = OmnivariateDecisionTree(max_depth=3, selection="accuracy").fit(
        X_train, y_train
    )
    by_test = OmnivariateDecisionTree(
        max_depth=3, selection="test", min_samples_test=2
    ).fit(X_train, y_train)

    simple_accuracy = by_accuracy.get_split_type_distribution()["univariate"]
    simple_test = by_test.get_split_type_distribution()["univariate"]
    assert simple_test >= simple_accuracy


def test_small_nodes_fall_back_rather_than_trusting_a_powerless_test(wine_split):
    """
    Below min_samples_test the 5x2cv test cannot resolve anything, and treating
    "failed to reject" as "no difference" there is how the tree ends up choosing
    the simplest split everywhere.
    """
    X_train, _, y_train, _ = wine_split

    huge_threshold = OmnivariateDecisionTree(
        max_depth=3, selection="test", min_samples_test=10**6
    ).fit(X_train, y_train)
    by_accuracy = OmnivariateDecisionTree(max_depth=3, selection="accuracy").fit(
        X_train, y_train
    )

    # With the test never firing, the two must agree exactly.
    assert (
        huge_threshold.get_split_type_distribution()
        == by_accuracy.get_split_type_distribution()
    )


def test_selection_is_validated():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="selection must be"):
        OmnivariateDecisionTree(selection="bogus").fit(X, y)


def test_both_selections_still_classify(wine_split):
    X_train, X_test, y_train, y_test = wine_split
    for selection in ("accuracy", "test"):
        tree = OmnivariateDecisionTree(max_depth=3, selection=selection).fit(
            X_train, y_train
        )
        assert tree.score(X_test, y_test) > 0.8
        np.testing.assert_allclose(tree.predict_proba(X_test).sum(axis=1), 1.0, atol=1e-9)

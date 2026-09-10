"""Tests for exporting a trained mixture of experts as hard routing."""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_trees import HardRoutedExperts, HierarchicalMixtureOfExperts


@pytest.fixture(scope="module")
def wine_fit():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    moe = HierarchicalMixtureOfExperts(
        depth=2, branching_factor=2, max_epochs=80, random_state=0
    ).fit(X_train, y_train)
    return moe, X_train, X_test, y_train, y_test


def test_export_has_the_shape_of_the_mixture(wine_fit):
    moe, _, _, _, _ = wine_fit
    router = moe.to_hard_router()

    assert isinstance(router, HardRoutedExperts)
    assert router.depth == moe.depth
    assert router.branching_factor == moe.branching_factor
    assert len(router.gate_weights_) == moe.model_.n_gates
    assert len(router.expert_weights_) == moe.model_.n_experts
    np.testing.assert_array_equal(router.classes_, moe.classes_)


def test_routing_follows_each_gate_argmax(wine_fit):
    """The exported path must be the one the trained gates prefer."""
    import torch

    moe, _, X_test, _, _ = wine_fit
    router = moe.to_hard_router()
    moe.model_.eval()

    with torch.no_grad():
        log_gates = moe.model_._log_gate_probabilities(torch.FloatTensor(X_test))
    gates = log_gates.numpy()

    b = router.branching_factor
    expected = np.zeros(len(X_test), dtype=np.int64)
    offset = 0
    for level in range(router.depth):
        choice = np.array(
            [gates[i, offset + expected[i], :].argmax() for i in range(len(X_test))]
        )
        expected = expected * b + choice
        offset += b ** level

    np.testing.assert_array_equal(router._route(X_test), expected)


def test_predictions_are_one_expert_not_a_blend(wine_fit):
    moe, _, X_test, _, _ = wine_fit
    router = moe.to_hard_router()

    proba = router.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(
        router.predict(X_test), router.classes_[proba.argmax(axis=1)]
    )


def test_agrees_with_the_mixture_on_most_samples(wine_fit):
    moe, _, X_test, _, y_test = wine_fit
    router = moe.to_hard_router()

    agreement = (router.predict(X_test) == moe.predict(X_test)).mean()
    assert agreement > 0.9
    assert router.score(X_test, y_test) > moe.score(X_test, y_test) - 0.1


def test_route_counts_show_how_the_tree_partitions(wine_fit):
    """
    A mixture spreads every sample over all experts, so nothing in it says
    whether the tree actually partitions the input. The export does.
    """
    moe, _, X_test, _, _ = wine_fit
    router = moe.to_hard_router()

    counts = router.route_counts(X_test)
    assert counts.shape == (moe.model_.n_experts,)
    assert counts.sum() == len(X_test)


def test_prediction_needs_no_torch(wine_fit):
    import sys

    moe, _, _, _, _ = wine_fit
    router = moe.to_hard_router()

    for params in router.gate_weights_ + router.expert_weights_:
        assert all(isinstance(array, np.ndarray) for array in params)
    assert sys.modules["neural_trees.mixture_of_experts.hard_router"].__dict__.get("torch") is None


def test_export_text_describes_every_gate_and_expert(wine_fit):
    moe, _, _, _, _ = wine_fit
    router = moe.to_hard_router()
    names = [f"feature_{i}" for i in range(router.n_features_in_)]

    text = router.export_text(feature_names=names, max_features=2)
    assert text.count("gate on") == moe.model_.n_gates
    assert text.count("-> expert") == moe.model_.n_experts
    assert "feature_" in text

    with pytest.raises(ValueError, match="feature_names has"):
        router.export_text(feature_names=["too", "few"])


def test_wrong_feature_count_is_rejected(wine_fit):
    moe, _, X_test, _, _ = wine_fit
    router = moe.to_hard_router()

    with pytest.raises(ValueError, match="features, but"):
        router.predict(X_test[:, :4])


@pytest.mark.parametrize("depth,branching_factor", [(1, 2), (2, 4), (3, 2)])
def test_export_works_for_other_tree_shapes(depth, branching_factor):
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    moe = HierarchicalMixtureOfExperts(
        depth=depth, branching_factor=branching_factor, max_epochs=20, random_state=0
    ).fit(X, y)

    router = moe.to_hard_router()
    assert router.route_counts(X).sum() == len(X)
    assert set(router.predict(X)).issubset(set(router.classes_))

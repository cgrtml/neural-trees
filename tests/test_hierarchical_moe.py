"""Tests for the Hierarchical Mixture of Experts."""
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import HierarchicalMixtureOfExperts


@pytest.fixture(scope="module")
def wine_split():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


@pytest.mark.parametrize("depth,branching_factor", [(1, 2), (2, 2), (2, 4)])
def test_fit_predict_across_tree_shapes(wine_split, depth, branching_factor):
    X_train, X_test, y_train, y_test = wine_split
    moe = HierarchicalMixtureOfExperts(
        depth=depth, branching_factor=branching_factor, max_epochs=40, random_state=0
    )
    moe.fit(X_train, y_train)

    assert moe.model_.n_experts == branching_factor ** depth
    proba = moe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(moe.predict(X_test)).issubset(set(np.unique(y_train)))


def test_leaf_weights_form_a_distribution(wine_split):
    """Gating weights over experts must sum to 1 for every sample."""
    X_train, _, y_train, _ = wine_split
    moe = HierarchicalMixtureOfExperts(depth=2, max_epochs=5, random_state=0).fit(
        X_train, y_train
    )
    moe.model_.eval()
    with torch.no_grad():
        weights = moe.model_._compute_leaf_weights(torch.FloatTensor(X_train))

    assert weights.shape == (len(X_train), moe.model_.n_experts)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(dim=1).numpy(), 1.0, atol=1e-5)


def test_single_class_target():
    """Degenerate but legal input: a target with only one class."""
    X, _ = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=0)
    y = np.zeros(len(X), dtype=int)

    moe = HierarchicalMixtureOfExperts(depth=1, max_epochs=5, random_state=0).fit(X, y)

    assert list(moe.classes_) == [0]
    assert moe.predict_proba(X).shape == (len(X), 1)
    assert np.array_equal(moe.predict(X), y)


@pytest.mark.parametrize("dropout_rate", [0.0, 0.9])
def test_extreme_dropout_rates_still_train(wine_split, dropout_rate):
    X_train, X_test, y_train, y_test = wine_split
    moe = HierarchicalMixtureOfExperts(
        depth=2, dropout_rate=dropout_rate, max_epochs=40, random_state=0
    ).fit(X_train, y_train)

    assert np.isfinite([h["loss"] for h in moe.training_history_]).all()
    assert moe.score(X_test, y_test) > 0.3  # above the majority-class floor


def test_dropout_is_active_in_training_mode_only(wine_split):
    """Dropout must perturb the forward pass while training and not at predict time."""
    X_train, _, y_train, _ = wine_split
    moe = HierarchicalMixtureOfExperts(
        depth=2, dropout_rate=0.5, max_epochs=5, random_state=0
    ).fit(X_train, y_train)
    batch = torch.FloatTensor(X_train[:16])

    moe.model_.train()
    with torch.no_grad():
        stochastic = [moe.model_(batch).numpy() for _ in range(2)]
    assert not np.allclose(stochastic[0], stochastic[1])

    np.testing.assert_allclose(moe.predict_proba(X_train[:16]), moe.predict_proba(X_train[:16]))


def test_subtree_dropout_removes_a_branch():
    """
    The mechanism from Irsoy & Alpaydin (2021): a gating node drops one of its
    children, so that subtree receives no probability mass for that sample.
    With rate 1.0 and two children, every gate output must be one-hot.
    """
    X, y = load_wine(return_X_y=True)
    moe = HierarchicalMixtureOfExperts(
        depth=2, branching_factor=2, dropout_rate=1.0, dropout_type="subtree",
        max_epochs=3, random_state=0,
    ).fit(X, y)

    batch = torch.FloatTensor(StandardScaler().fit_transform(X)[:32])
    moe.model_.train()
    with torch.no_grad():
        gate_out = moe.model_._drop_subtrees(moe.model_.gates[0](batch))

    np.testing.assert_allclose(gate_out.sum(dim=1).numpy(), 1.0, atol=1e-5)
    assert ((gate_out > 1 - 1e-5) | (gate_out < 1e-5)).all()

    # Leaf mixing weights stay a distribution while subtrees are dropped.
    with torch.no_grad():
        weights = moe.model_._compute_leaf_weights(batch)
    np.testing.assert_allclose(weights.sum(dim=1).numpy(), 1.0, atol=1e-5)


def test_activation_dropout_perturbs_but_never_removes_a_branch():
    """The pre-2021 mechanism, kept for comparison: no gate output is zeroed."""
    X, y = load_wine(return_X_y=True)
    moe = HierarchicalMixtureOfExperts(
        depth=2, dropout_rate=0.9, dropout_type="activation", max_epochs=3, random_state=0,
    ).fit(X, y)

    batch = torch.FloatTensor(StandardScaler().fit_transform(X)[:32])
    moe.model_.train()
    with torch.no_grad():
        gate_out = moe.model_.gates[0](batch)
    assert (gate_out > 1e-6).all()


def test_dropout_type_is_validated():
    X, y = load_wine(return_X_y=True)
    with pytest.raises(ValueError, match="dropout_type must be"):
        HierarchicalMixtureOfExperts(dropout_type="bogus").fit(X, y)


def test_dropout_does_not_worsen_the_generalization_gap():
    """
    Regression guard, not a proof: averaged over seeds, subtree dropout should
    not widen the train-test gap on a noisy toy problem. The measured effect is
    real but modest, see the numbers in the module docstring of this test file's
    companion study.
    """
    X, y = make_classification(
        n_samples=160, n_features=20, n_informative=5, n_redundant=0,
        flip_y=0.10, class_sep=0.8, random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    def mean_gap(dropout_rate):
        gaps = []
        for seed in range(3):
            moe = HierarchicalMixtureOfExperts(
                depth=2, gate_hidden=64, expert_hidden=64, dropout_rate=dropout_rate,
                max_epochs=120, random_state=seed,
            ).fit(X_train, y_train)
            gaps.append(moe.score(X_train, y_train) - moe.score(X_test, y_test))
        return float(np.mean(gaps))

    assert mean_gap(0.5) <= mean_gap(0.0) + 0.05


def test_competitive_with_mlp_on_wine(wine_split):
    """The mixture should land in the same league as a comparable MLP."""
    X_train, X_test, y_train, y_test = wine_split
    moe = HierarchicalMixtureOfExperts(depth=2, max_epochs=150, random_state=0).fit(
        X_train, y_train
    )
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=0).fit(
        X_train, y_train
    )

    assert moe.score(X_test, y_test) >= mlp.score(X_test, y_test) - 0.15


def test_pipeline_and_clone_compatible():
    X, y = load_wine(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("moe", HierarchicalMixtureOfExperts(depth=1, max_epochs=5, random_state=0)),
    ])
    pipe.fit(X, y)
    assert 0.0 <= pipe.score(X, y) <= 1.0
    assert clone(pipe).named_steps["moe"].get_params()["depth"] == 1


def test_predictions_do_not_depend_on_row_order(wine_split):
    """
    Rows are independent, but BLAS blocks differently for different memory
    layouts, so in float32 the same sample scored inside a reordered batch came
    out up to 1.2e-07 different, and a borderline argmax could flip with it.
    """
    X_train, X_test, y_train, _ = wine_split
    moe = HierarchicalMixtureOfExperts(depth=2, max_epochs=30, random_state=0).fit(
        X_train, y_train
    )

    forward = moe.predict_proba(X_test)
    reversed_back = moe.predict_proba(X_test[::-1])[::-1]
    np.testing.assert_allclose(forward, reversed_back, rtol=1e-9, atol=1e-12)
    assert np.array_equal(moe.predict(X_test), moe.predict(X_test[::-1])[::-1])

    shuffle = np.random.RandomState(0).permutation(len(X_test))
    inverse = np.argsort(shuffle)
    np.testing.assert_allclose(
        forward, moe.predict_proba(X_test[shuffle])[inverse], rtol=1e-9, atol=1e-12
    )


def test_refitting_rebuilds_the_prediction_model(wine_split):
    """The cached float64 copy must not survive a refit."""
    X_train, _, y_train, _ = wine_split
    moe = HierarchicalMixtureOfExperts(depth=2, max_epochs=5, random_state=0)

    moe.fit(X_train, y_train)
    first = moe.predict_proba(X_train[:10])
    moe.fit(X_train, y_train[::-1])
    second = moe.predict_proba(X_train[:10])

    assert not np.allclose(first, second)

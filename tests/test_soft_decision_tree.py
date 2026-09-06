"""Tests for Soft Decision Tree."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from neural_trees import SoftDecisionTree


@pytest.mark.parametrize("depth", [0, -1, 1.5, "3", None, True])
def test_depth_validation_rejects_invalid_values(depth):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        SoftDecisionTree(depth=depth).fit(X, y)


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


def test_training_reproducible_with_random_state():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    first = SoftDecisionTree(random_state=0, depth=3, max_epochs=10)
    second = SoftDecisionTree(random_state=0, depth=3, max_epochs=10)

    first.fit(X_train, y_train)
    second.fit(X_train, y_train)

    assert np.array_equal(first.predict(X_test), second.predict(X_test))


def test_depth_one_single_split():
    """A depth=1 tree is one split and two leaves, and must still train."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_classes=2, random_state=0
    )
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=0)

    sdt = SoftDecisionTree(depth=1, max_epochs=20, random_state=0)
    sdt.fit(X_train, y_train)
    preds = sdt.predict(X_test)

    assert preds.shape == (X_test.shape[0],)
    assert set(preds).issubset(set(np.unique(y_train)))
    assert sdt.get_leaf_distributions().shape == (2, 2)
    assert len(sdt.get_split_weights()) == 1


def test_pipeline_and_grid_search_compatible():
    """The estimator must survive clone(), Pipeline, and GridSearchCV."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SoftDecisionTree(depth=2, max_epochs=5, random_state=0)),
    ])
    search = GridSearchCV(pipe, {"model__depth": [2, 3]}, cv=2)
    search.fit(X, y)

    assert search.best_params_["model__depth"] in (2, 3)
    assert 0.0 <= search.best_score_ <= 1.0


def test_feature_importances_favor_informative_features():
    """Noise columns must not carry the weight that informative ones do."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=3, n_redundant=0,
        n_repeated=0, shuffle=False, random_state=0,
    )
    sdt = SoftDecisionTree(depth=3, max_epochs=60, random_state=0).fit(X, y)

    importances = sdt.feature_importances_
    assert importances.shape == (10,)
    assert (importances >= 0).all()
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)
    # make_classification with shuffle=False puts the informative columns first.
    assert importances[:3].sum() > importances[3:].sum()


def test_early_stopping_records_validation_and_can_stop_early():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=4, flip_y=0.15, random_state=0
    )
    sdt = SoftDecisionTree(
        depth=4, max_epochs=200, early_stopping=True, n_iter_no_change=5,
        validation_fraction=0.25, random_state=0,
    ).fit(X, y)

    assert sdt.n_iter_ < 200
    assert len(sdt.training_history_) == sdt.n_iter_
    assert all("val_loss" in h for h in sdt.training_history_)
    assert sdt.score(X, y) > 0.7
    # The restored parameters are the best epoch's, not the last one's.
    losses = [h["val_loss"] for h in sdt.training_history_]
    assert losses.index(min(losses)) < len(losses) - 1


def test_early_stopping_validation_fraction_is_validated():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="validation_fraction"):
        SoftDecisionTree(early_stopping=True, validation_fraction=1.5).fit(X, y)


def test_deep_tree_stays_numerically_healthy():
    """
    A depth-12 leaf is reached with probability on the order of 2^-12, and the
    product of gate probabilities down a saturated path underflows float32.
    Accumulating in log space keeps the mixture and its gradients usable.
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    sdt = SoftDecisionTree(depth=12, max_epochs=1, random_state=0).fit(X, y)

    X_t = torch.FloatTensor(X)
    log_probs = sdt.model_.log_forward(X_t)
    assert torch.isfinite(log_probs).all()

    loss = torch.nn.functional.nll_loss(log_probs, torch.LongTensor(y))
    loss.backward()
    gate_grads = sdt.model_.gates.weight.grad
    assert torch.isfinite(gate_grads).all()
    assert (gate_grads.abs().sum(dim=1) > 0).all()  # no node is cut off from the loss

    proba = sdt.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_learnable_temperature_is_a_trained_parameter():
    X, y = load_iris(return_X_y=True)

    warm = SoftDecisionTree(depth=3, max_epochs=30, learn_temperature=True, random_state=0)
    warm.fit(X, y)
    assert warm.model_.log_beta.requires_grad
    assert not np.allclose(warm.model_.log_beta.detach().numpy(), 0.0)

    plain = SoftDecisionTree(depth=3, max_epochs=30, learn_temperature=False, random_state=0)
    plain.fit(X, y)
    assert not plain.model_.log_beta.requires_grad
    np.testing.assert_allclose(plain.model_.log_beta.detach().numpy(), 0.0)

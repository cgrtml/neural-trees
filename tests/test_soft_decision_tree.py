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


def test_uniform_sample_weight_matches_the_unweighted_fit():
    X, y = load_iris(return_X_y=True)
    plain = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(X, y)
    weighted = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(
        X, y, sample_weight=np.full(len(X), 3.0)
    )

    np.testing.assert_allclose(
        plain.predict_proba(X), weighted.predict_proba(X), rtol=1e-5, atol=1e-6
    )


def test_weighting_a_row_equals_duplicating_it_in_the_loss():
    """
    The mathematical claim behind sample_weight, tested on the loss itself
    rather than on two training runs, which would differ by batch composition
    alone.
    """
    import torch
    import torch.nn.functional as F

    X, y = load_iris(return_X_y=True)
    model = SoftDecisionTree(depth=3, max_epochs=5, random_state=0).fit(X, y)

    weights = np.ones(len(X))
    weights[:20] = 2.0
    duplicated_X = np.vstack([X, X[:20]])
    duplicated_y = np.concatenate([y, y[:20]])

    with torch.no_grad():
        per_sample = F.nll_loss(
            model.model_.log_forward(torch.FloatTensor(X)),
            torch.LongTensor(y),
            reduction="none",
        )
        w = torch.FloatTensor(weights)
        weighted_loss = (per_sample * w).sum() / w.sum()

        duplicated_loss = F.nll_loss(
            model.model_.log_forward(torch.FloatTensor(duplicated_X)),
            torch.LongTensor(duplicated_y),
        )

    torch.testing.assert_close(weighted_loss, duplicated_loss, rtol=1e-5, atol=1e-6)


def test_zero_weight_removes_a_class():
    X, y = load_iris(return_X_y=True)
    weights = np.where(y == 2, 0.0, 1.0)

    sdt = SoftDecisionTree(depth=3, max_epochs=40, random_state=0).fit(
        X, y, sample_weight=weights
    )

    # Class 2 remains a known label, but nothing should ever be predicted into it.
    assert list(sdt.classes_) == [0, 1, 2]
    assert 2 not in set(sdt.predict(X))


def test_class_weight_balanced_lifts_minority_recall():
    """
    Without reweighting, a class holding 7% of the samples contributes too
    little loss to matter and the tree half ignores it, while overall accuracy
    stays high enough to hide that.
    """
    from sklearn.datasets import make_classification
    from sklearn.metrics import recall_score
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=4, n_redundant=0,
        weights=[0.94, 0.06], flip_y=0.02, class_sep=0.7, random_state=0,
    )
    X = StandardScaler().fit_transform(X)

    plain = SoftDecisionTree(depth=4, max_epochs=60, random_state=0).fit(X, y)
    balanced = SoftDecisionTree(
        depth=4, max_epochs=60, random_state=0, class_weight="balanced"
    ).fit(X, y)

    plain_recall = recall_score(y, plain.predict(X), pos_label=1)
    balanced_recall = recall_score(y, balanced.predict(X), pos_label=1)

    assert balanced_recall > plain_recall + 0.2
    # And it should not pay for that with a collapse in overall accuracy.
    assert balanced.score(X, y) > plain.score(X, y) - 0.05


def test_sample_weight_survives_early_stopping():
    X, y = load_iris(return_X_y=True)
    sdt = SoftDecisionTree(
        depth=3, max_epochs=40, early_stopping=True, validation_fraction=0.2,
        random_state=0,
    ).fit(X, y, sample_weight=np.linspace(0.5, 1.5, len(X)))

    assert sdt.n_iter_ >= 1
    assert sdt.score(X, y) > 0.7


def test_sample_weight_length_is_validated():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError):
        SoftDecisionTree(depth=2, max_epochs=2).fit(X, y, sample_weight=np.ones(5))

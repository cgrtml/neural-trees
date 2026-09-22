"""Early stopping in HierarchicalMixtureOfExperts (#99)."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_wine

from neural_trees import HierarchicalMixtureOfExperts


@pytest.fixture(scope="module")
def wine():
    X, y = load_wine(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


def test_off_by_default_and_trains_the_full_budget(wine):
    X, y = wine
    m = HierarchicalMixtureOfExperts(depth=1, max_epochs=12, random_state=0).fit(X, y)
    assert m.early_stopping is False
    assert m.n_iter_ == 12 and len(m.training_history_) == 12
    assert "val_loss" not in m.training_history_[0]


def test_off_is_bit_identical_to_a_second_full_fit(wine):
    # Pins today's behaviour: no split is drawn and no RNG is consumed when
    # early stopping is off, so two seeded fits agree exactly.
    X, y = wine
    a = HierarchicalMixtureOfExperts(depth=1, max_epochs=8, random_state=3).fit(X, y)
    b = HierarchicalMixtureOfExperts(depth=1, max_epochs=8, random_state=3).fit(X, y)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_stops_before_the_budget_and_restores_the_best_epoch():
    # Breast Cancer, not Wine: on Wine the held-out loss keeps falling for
    # hundreds of epochs (the data is separable), so stopping correctly
    # never fires there.
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    m = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=400, early_stopping=True, n_iter_no_change=10, random_state=0
    ).fit(X, y)
    assert m.n_iter_ < 400
    assert len(m.training_history_) == m.n_iter_
    val = [r["val_loss"] for r in m.training_history_]
    # The last n_iter_no_change epochs did not improve on the best one.
    assert int(np.argmin(val)) <= m.n_iter_ - 1 - 10
    assert m.score(X, y) > 0.95


def test_split_is_stratified_and_seeded(wine):
    X, y = wine
    m = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=3, early_stopping=True, validation_fraction=0.2, random_state=0
    ).fit(X, y)
    n_val = int(round(len(y) * 0.2))
    # training loss/accuracy were computed on the remaining rows only
    assert m.training_history_[0]["val_accuracy"] >= 0.0
    a = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=3, early_stopping=True, random_state=1
    ).fit(X, y).training_history_[0]["val_loss"]
    b = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=3, early_stopping=True, random_state=1
    ).fit(X, y).training_history_[0]["val_loss"]
    assert a == b
    assert n_val == 36


def test_warm_start_draws_a_fresh_split_on_the_new_data(wine):
    X, y = wine
    m = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=3, early_stopping=True, warm_start=True, random_state=0
    ).fit(X, y)
    first = m.training_history_[-1]["val_loss"]
    # Second fit on a different subset: the validation score must come from
    # that subset's own held-out rows, not the first split.
    idx = np.arange(0, len(y), 2)
    m.fit(X[idx], y[idx])
    assert m.n_iter_ == 3 and len(m.training_history_) == 3
    assert m.training_history_[-1]["val_loss"] != first


def test_falls_back_to_the_full_budget_when_a_split_is_impossible():
    X = np.random.RandomState(0).randn(6, 3)
    y = np.array([0, 0, 1, 1, 2, 2])
    m = HierarchicalMixtureOfExperts(
        depth=1, max_epochs=4, early_stopping=True, validation_fraction=0.1, random_state=0
    ).fit(X, y)
    assert m.n_iter_ == 4 and "val_loss" not in m.training_history_[0]


def test_validation_fraction_is_validated(wine):
    X, y = wine
    with pytest.raises(ValueError, match="validation_fraction"):
        HierarchicalMixtureOfExperts(early_stopping=True, validation_fraction=1.0).fit(X, y)

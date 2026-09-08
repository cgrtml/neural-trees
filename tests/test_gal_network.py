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


def test_learns_a_separable_problem(iris_split):
    """
    Regression test for the training loop. GAL used to take exactly one
    full-batch SGD step per epoch, so growth and pruning decisions were made on
    a network that had barely moved from its initialization. On three well
    separated blobs it scored 0.333, which is chance for three classes.
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.1, random_state=0)
    gal = GALNetwork(max_epochs=100, random_state=0).fit(X, y)

    assert gal.score(X, y) > 0.9


def test_beats_the_majority_baseline_on_iris(iris_split):
    X_train, X_test, y_train, y_test = iris_split
    gal = GALNetwork(max_epochs=100, random_state=0).fit(X_train, y_train)

    majority = np.bincount(y_test).max() / len(y_test)
    assert gal.score(X_test, y_test) > majority


def test_error_decreases_over_training(iris_split):
    X_train, _, y_train, _ = iris_split
    gal = GALNetwork(max_epochs=100, random_state=0).fit(X_train, y_train)

    history = [h["error"] for h in gal.architecture_history_]
    assert history[-1] < history[0]


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


def test_validation_policy_builds_a_smaller_network(iris_split):
    """
    The point of deciding on held-out evidence is that the network stops
    growing once extra units stop paying for themselves.
    """
    X_train, _, y_train, _ = iris_split

    threshold = GALNetwork(max_epochs=150, growth_policy="error_threshold", random_state=0)
    validated = GALNetwork(max_epochs=150, growth_policy="validation", random_state=0)
    threshold.fit(X_train, y_train)
    validated.fit(X_train, y_train)

    assert validated.n_hidden_final_ < threshold.n_hidden_final_
    assert validated.growth_policy_ == "validation"


def test_validation_policy_records_its_decisions():
    """
    Uses separable blobs rather than Iris: on Iris at these settings validation
    loss keeps improving by more than `tol` at every check, so the policy
    correctly never touches the architecture and there is no decision to record.
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.1, random_state=0)
    gal = GALNetwork(
        max_epochs=150, growth_policy="validation", check_interval=5, random_state=0
    ).fit(X, y)

    actions = {h["action"] for h in gal.architecture_history_}
    assert actions <= {"train", "keep", "grow", "prune", "capped"}
    assert {"grow", "prune"} & actions, "no architecture decision was ever taken"
    assert all("val_loss" in h for h in gal.architecture_history_)
    assert gal.best_val_loss_ == pytest.approx(
        min(h["val_loss"] for h in gal.architecture_history_), abs=1e-6
    )


def test_validation_policy_stops_when_changes_stop_paying(iris_split):
    X_train, _, y_train, _ = iris_split
    gal = GALNetwork(
        max_epochs=2000, growth_policy="validation", patience=2, check_interval=5,
        random_state=0,
    ).fit(X_train, y_train)

    assert gal.n_iter_ < 2000


def test_falls_back_when_the_data_cannot_support_a_validation_split():
    """A class with a single member cannot be stratified into a held-out split."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 0, 1])

    gal = GALNetwork(max_epochs=10, growth_policy="validation", random_state=0).fit(X, y)
    assert gal.growth_policy_ == "error_threshold"
    assert gal.predict(X).shape == y.shape


@pytest.mark.parametrize(
    "bad", [{"growth_policy": "bogus"}, {"validation_fraction": 0.0}, {"validation_fraction": 1.5}]
)
def test_invalid_growth_settings_are_rejected(bad, iris_split):
    X_train, _, y_train, _ = iris_split
    with pytest.raises(ValueError):
        GALNetwork(**bad).fit(X_train, y_train)


def test_pruning_uses_contribution_not_bare_variance():
    """
    A nearly constant unit with large outgoing weights still shifts every
    logit, so activation variance alone is the wrong thing to prune on.
    """
    X, y = load_iris(return_X_y=True)
    gal = GALNetwork(max_epochs=20, random_state=0).fit(X, y)

    import torch

    contributions = gal._contributions(gal.model_, torch.FloatTensor(X))
    activations = gal._hidden_activations(gal.model_, torch.FloatTensor(X))
    assert contributions.shape == (gal.n_hidden_final_,)
    assert (contributions >= 0).all()
    # Contribution is the spread scaled by outgoing weight, so it differs from
    # the bare spread whenever the outgoing weights are not all equal.
    assert not torch.allclose(contributions, activations.std(dim=0))

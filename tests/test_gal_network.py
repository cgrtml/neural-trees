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


def test_validation_policy_sizes_the_network_from_evidence(iris_split):
    """
    The point of deciding on held-out evidence is that the size comes from the
    data rather than from a fixed error threshold. Which of the two ends up
    larger depends on the problem, so what is asserted here is that the policy
    ran and that its size responds to the knob that controls it.
    """
    X_train, _, y_train, _ = iris_split

    eager = GALNetwork(
        max_epochs=150, growth_policy="validation", error_patience=2, random_state=0
    ).fit(X_train, y_train)
    patient = GALNetwork(
        max_epochs=150, growth_policy="validation", error_patience=8, random_state=0
    ).fit(X_train, y_train)

    assert eager.growth_policy_ == "validation"
    assert eager.n_hidden_final_ >= patient.n_hidden_final_


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


def test_residual_growth_does_not_disturb_the_network():
    """
    A residual-fitted unit enters with zero outgoing weights, so the network
    computes exactly the same function the moment it grows. A random unit
    perturbs every logit on arrival.
    """
    import torch

    X, y = load_iris(return_X_y=True)
    gal = GALNetwork(max_epochs=10, random_state=0).fit(X, y)
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(gal.le_.transform(y))

    before = gal.model_(X_t).detach()
    residual_grown = gal._grown(gal.model_, 3, torch.device("cpu"), X_t, y_t)
    random_gal = GALNetwork(growth_init="random", max_epochs=10, random_state=0).fit(X, y)
    random_grown = random_gal._grown(random_gal.model_, 3, torch.device("cpu"), X_t, y_t)

    assert residual_grown[0].weight.shape[0] == gal.model_[0].weight.shape[0] + 1
    torch.testing.assert_close(residual_grown(X_t).detach(), before)
    assert not torch.allclose(random_grown(X_t).detach(), random_gal.model_(X_t).detach())


def test_candidate_correlates_with_the_residual_better_than_a_random_unit():
    """The candidate is fitted to what the network still gets wrong, so it
    should explain more of the residual than an arbitrary unit does."""
    import torch

    X, y = load_iris(return_X_y=True)
    gal = GALNetwork(max_epochs=20, random_state=0).fit(X, y)
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(gal.le_.transform(y))
    residual = gal._residual(gal.model_, X_t, y_t, 3)

    _, _, fitted_score = gal._train_candidate(X_t, residual, torch.device("cpu"))

    torch.manual_seed(0)
    random_unit = torch.nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        activation = torch.sigmoid(random_unit(X_t)).squeeze(-1)
        centered = activation - activation.mean()
        covariance = (centered.unsqueeze(1) * residual).sum(dim=0).abs().sum()
        random_score = (covariance / centered.pow(2).sum().sqrt().clamp_min(1e-8)).item()

    assert np.isfinite(fitted_score)
    assert fitted_score > random_score


def test_residual_growth_reaches_a_smaller_network_on_iris():
    """
    The headline of fitting units to the residual: the same or better accuracy
    from far fewer units, because each one arrives already useful.
    """
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    residual = GALNetwork(max_epochs=150, growth_init="residual", random_state=0).fit(X, y)
    random_init = GALNetwork(max_epochs=150, growth_init="random", random_state=0).fit(X, y)

    assert residual.n_hidden_final_ < random_init.n_hidden_final_
    assert residual.score(X, y) >= random_init.score(X, y)
    assert np.isfinite(residual.last_candidate_score_)


def test_growth_init_is_validated():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="growth_init must be"):
        GALNetwork(growth_init="bogus").fit(X, y)


def test_saturated_candidate_does_not_poison_growth():
    """
    Regression test: the correlation objective divides by the norm of the
    centred activation. Clamping after the square root left sqrt(0) in the
    graph, whose gradient is infinite, so a candidate that saturated into a
    constant activation returned NaN and took every later candidate with it,
    leaving growth with no unit to install.
    """
    import torch

    gal = GALNetwork(random_state=0)
    gal.n_features_in_ = 2
    # Inputs far enough out that the sigmoid saturates to a constant.
    X_t = torch.full((32, 2), 1e6)
    residual = torch.randn(32, 3)

    weight, bias, score = gal._train_candidate(X_t, residual, torch.device("cpu"))

    assert weight is not None and bias is not None
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
    assert np.isfinite(score)


def test_optimizer_momentum_survives_an_architecture_change(iris_split):
    """
    Growth and pruning replace the module, and a fresh optimizer would start
    every surviving unit from a standstill. The buffers are reshaped the same
    way the weights are.
    """
    import torch

    X_train, _, y_train, _ = iris_split
    gal = GALNetwork(max_epochs=20, random_state=0).fit(X_train, y_train)

    model = gal.model_
    optimizer = gal._make_optimizer(model)
    # Give the optimizer some history to carry.
    for param in model.parameters():
        optimizer.state[param]["momentum_buffer"] = torch.full_like(param, 0.5)

    grown = gal._grown(model, 3, torch.device("cpu"))
    carried = gal._carry_optimizer(optimizer, model, grown)
    old_units = model[0].weight.shape[0]

    buffer = carried.state[list(grown.parameters())[0]]["momentum_buffer"]
    assert buffer.shape == grown[0].weight.shape
    assert torch.allclose(buffer[:old_units], torch.full_like(buffer[:old_units], 0.5))
    # The unit that has just appeared has no history to carry.
    assert torch.allclose(buffer[old_units:], torch.zeros_like(buffer[old_units:]))

    keep = [0]
    pruned = gal._rebuild_with_units(model, keep, 3, torch.device("cpu"))
    carried = gal._carry_optimizer(optimizer, model, pruned, keep)
    buffer = carried.state[list(pruned.parameters())[0]]["momentum_buffer"]
    assert buffer.shape == pruned[0].weight.shape
    assert torch.allclose(buffer, torch.full_like(buffer, 0.5))


def test_growth_triggers_when_the_error_plateaus_under_a_falling_loss():
    """
    A capacity-starved network keeps getting more confident about the same
    mistakes. On six separable blobs the validation loss fell from 1.81 to 1.17
    while the error sat at 0.46, and a loss-only rule never grew past three
    units.
    """
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_blobs(n_samples=400, centers=6, cluster_std=0.6, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    combined = GALNetwork(
        max_epochs=150, growth_policy="validation", random_state=0
    ).fit(X_train, y_train)
    loss_only = GALNetwork(
        max_epochs=150, growth_policy="validation", error_patience=10**6, random_state=0
    ).fit(X_train, y_train)

    assert combined.n_hidden_final_ > loss_only.n_hidden_final_
    assert combined.score(X_test, y_test) > loss_only.score(X_test, y_test)


def test_error_patience_trades_size_against_accuracy():
    """Raising it keeps the network smaller; the docstring says so, so test it."""
    X, y = load_iris(return_X_y=True)

    eager = GALNetwork(
        max_epochs=150, growth_policy="validation", error_patience=2, random_state=0
    ).fit(X, y)
    patient = GALNetwork(
        max_epochs=150, growth_policy="validation", error_patience=8, random_state=0
    ).fit(X, y)

    assert eager.n_hidden_final_ >= patient.n_hidden_final_

"""Growth and exports for the soft tree regressor (#103, second half)."""
import json

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from neural_trees import HardRegressionTree, NumpySoftTree, SoftDecisionTreeRegressor


def _step_function(n=600, seed=0):
    # Piecewise-constant target: a tree of depth 2 fits it, one leaf does not.
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    y = np.where(X[:, 0] > 0, 2.0, -2.0) + np.where(X[:, 1] > 0, 1.0, -1.0) + 0.1 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


def test_growth_validation_and_fallback(diabetes):
    X, y = diabetes
    for bad, kw in (("growth", {"growth": "level"}), ("growth_init", {"growth_init": "uniform"}),
                    ("growth_budget", {"growth_budget": "half"})):
        with pytest.raises(ValueError, match=bad):
            SoftDecisionTreeRegressor(max_epochs=1, **kw).fit(X, y)
    # too few rows for a validation split: falls back to the full depth
    m = SoftDecisionTreeRegressor(depth=2, max_epochs=3, growth="per_leaf").fit(X[:8], y[:8])
    assert m.growth_ == "none" and m.tree_depth_ == 2


@pytest.mark.parametrize("growth", ["incremental", "per_leaf"])
def test_growth_beats_a_single_leaf_and_records_its_shape(growth):
    X, y = _step_function()
    m = SoftDecisionTreeRegressor(
        depth=4, max_epochs=120, learning_rate=0.05, growth=growth, growth_budget="full",
        random_state=0,
    ).fit(X, y)
    assert m.growth_ == growth
    n_splits = int(m.model_.is_split.sum())
    assert 1 <= n_splits <= 15
    assert m.score(X, y) > 0.9
    assert m.training_history_[-1]["depth"] == m.tree_depth_


@pytest.mark.parametrize("growth_init", ["random", "residual", "residual_gate"])
def test_split_initialisation_is_a_perturbed_copy_of_the_parent(growth_init):
    X, y = _step_function()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=1, growth="per_leaf", growth_init=growth_init, random_state=0)
    m.fit(X, y)
    # Rebuild the first split by hand and check the children straddle the parent.
    import torch

    model = m._new_module(3, "cpu")
    with torch.no_grad():
        model.is_split.fill_(False)
        model.node_logits[0] = 0.7
    X_t, Y_t = torch.FloatTensor(X), torch.FloatTensor(((y - y.mean()) / y.std()).reshape(-1, 1))
    m._initialise_split(model, 0, X_t, Y_t, torch.ones(len(y)))
    left, right = model.node_logits[1].item(), model.node_logits[2].item()
    assert bool(model.is_split[0])
    assert abs((left + right) / 2 - 0.7) < 1e-5 and left != right
    if growth_init == "residual_gate":
        assert float(model.gates.weight[0].abs().sum()) > 0
    else:
        assert float(model.gates.weight[0].abs().sum()) == 0.0


def test_numpy_export_is_the_same_model_and_round_trips(diabetes, tmp_path):
    X, y = diabetes
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=30, random_state=0).fit(X, y)
    npt = m.to_numpy(feature_names=[f"f{i}" for i in range(X.shape[1])])
    assert npt.kind == "regressor"
    np.testing.assert_allclose(npt.predict(X), m.predict(X), rtol=1e-4, atol=1e-3)
    back = NumpySoftTree.from_json(npt.to_json())
    np.testing.assert_array_equal(back.predict(X), npt.predict(X))
    d = json.loads(npt.to_json())
    assert d["kind"] == "regressor" and d["y_mean"] is not None
    with pytest.raises(AttributeError):
        npt.predict_proba(X)


def test_numpy_export_multi_output():
    X, y = _step_function()
    Y = np.c_[y, -y]
    m = SoftDecisionTreeRegressor(depth=2, max_epochs=20, random_state=0).fit(X, Y)
    npt = m.to_numpy()
    assert npt.predict(X).shape == (len(X), 2)
    np.testing.assert_allclose(npt.predict(X), m.predict(X), rtol=1e-4, atol=1e-3)


def test_hard_tree_export_reports_agreement_and_prints_values():
    X, y = _step_function()
    m = SoftDecisionTreeRegressor(depth=2, max_epochs=120, learning_rate=0.05, random_state=0).fit(X, y)
    hard = m.to_hard_tree()
    assert isinstance(hard, HardRegressionTree)
    assert hard.predict(X).shape == (len(X),)
    # On a step function the hard reading of a trained soft tree is close but
    # not identical (measured 0.89 correlation at 80 epochs): the export is a
    # different model, which is the point of it reporting its agreement.
    assert np.corrcoef(hard.predict(X), m.predict(X))[0, 1] > 0.85
    # score is R^2 in target units; the value itself is low here (0.49
    # measured) because the soft leaves compensate for gates that share mass
    # and the hard walk lands on those attenuated values.
    pred = hard.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert abs(hard.score(X, y) - r2) < 1e-12
    text = hard.export_text(feature_names=["a", "b", "c"])
    assert "value = " in text and "if " in text
    with pytest.raises(AttributeError):
        hard.predict_proba(X)
    with pytest.raises(ValueError, match="features"):
        hard.predict(X[:, :2])


def test_grown_tree_exports_too():
    X, y = _step_function()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=60, learning_rate=0.05, growth="per_leaf", random_state=0).fit(X, y)
    np.testing.assert_allclose(m.to_numpy().predict(X), m.predict(X), rtol=1e-4, atol=1e-3)
    assert m.to_hard_tree().predict(X).shape == (len(X),)

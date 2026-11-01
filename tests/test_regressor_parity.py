"""SoftDecisionTreeRegressor: split_features, init='cart', init_prune and missing values."""
import numpy as np
import pytest

from neural_trees import NumpySoftTree, SoftDecisionTreeRegressor


def _data(seed=0, n=500, nan=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2.0 * (X[:, 0] > 0.3) + X[:, 1] + 0.1 * rng.normal(size=n)
    if nan:
        miss = rng.random(n) < 0.3
        y = y + 1.5 * miss                    # missingness of x3 carries signal
        X[miss, 3] = np.nan
    return X, y


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


@pytest.mark.parametrize("k", [1, 2])
def test_split_features(k):
    X, y = _data()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=40, split_features=k, random_state=0).fit(X, y)
    W = m.model_.gates.weight.detach().numpy()
    assert ((W != 0).sum(axis=1) <= k).all()
    assert np.allclose(m.to_numpy().predict(X), m.predict(X), atol=1e-4)


def test_init_cart_copies_a_sharp_threshold():
    X, y = _data()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=40, init="cart", random_state=0).fit(X, y)
    assert _r2(y, m.predict(X)) > 0.8
    assert hasattr(m, "init_ccp_alpha_")


def test_init_cart_with_one_feature_keeps_it():
    X, y = _data()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=10, init="cart", split_features=1,
                                  init_prune=False, random_state=0).fit(X, y)
    assert ((m.model_.gates.weight.detach().numpy() != 0).sum(axis=1) <= 1).all()


def test_init_cart_multi_output():
    X, y = _data()
    Y = np.column_stack([y, -y])
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=20, init="cart", random_state=0).fit(X, Y)
    assert m.predict(X).shape == (len(X), 2)


def test_missing_values_carry_signal_and_export():
    X, y = _data(nan=True)
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=60, random_state=0).fit(X, y)
    assert list(m.missing_columns_) == [3]
    assert _r2(y, m.predict(X)) > 0.75
    num = m.to_numpy()
    assert np.allclose(num.predict(X), m.predict(X), atol=1e-4)
    assert np.allclose(NumpySoftTree.from_json(num.to_json()).predict(X), m.predict(X), atol=1e-4)
    assert m.to_hard_tree().predict(X).shape == (len(X),)


def test_missing_values_onnx():
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    X, y = _data(nan=True)
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=10, random_state=0).fit(X, y)
    sess = ort.InferenceSession(m.to_onnx().SerializeToString())
    out = sess.run(["value"], {"X": X.astype(np.float32)})[0]
    assert np.allclose(out, m.predict(X), atol=1e-3)


@pytest.mark.parametrize("params,match", [
    ({"split_features": 0}, "split_features"),
    ({"init": "zeros"}, "init must be"),
    ({"init": "cart", "growth": "per_leaf"}, "growth='none'"),
    ({"init_sharpness": -1.0}, "init_sharpness"),
])
def test_parameter_validation(params, match):
    X, y = _data(n=60)
    with pytest.raises(ValueError, match=match):
        SoftDecisionTreeRegressor(depth=2, max_epochs=2, **params).fit(X, y)

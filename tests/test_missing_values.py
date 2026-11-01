"""Missing values in SoftDecisionTree: NaN -> 0 plus a learned per-gate indicator."""
import numpy as np
import pytest

from neural_trees import NumpySoftTree, SoftDecisionTree


def _data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    missing = rng.random(n) < 0.3
    # missingness carries signal: x2 is missing mostly for class 1
    X[:, 2] = np.where(missing & (y == 1), np.nan, X[:, 2])
    X[rng.random(n) < 0.1, 0] = np.nan
    return X, y


def test_fits_and_predicts_with_nan():
    X, y = _data()
    m = SoftDecisionTree(depth=3, max_epochs=30, random_state=0).fit(X, y)
    assert list(m.missing_columns_) == [0, 2]
    assert m.n_features_in_ == 4
    assert m.model_.gates.weight.shape[1] == 6
    assert m.feature_importances_.shape == (4,)
    assert (m.predict(X) == y).mean() > 0.85


def test_indicator_lets_missingness_carry_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = rng.integers(0, 2, 500)
    X[y == 1, 1] = np.nan                     # only the missingness is informative
    m = SoftDecisionTree(depth=2, max_epochs=40, random_state=0).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.95


def test_exports_agree_with_the_model_on_nan_rows():
    X, y = _data()
    m = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(X, y)
    num = m.to_numpy()
    assert np.allclose(num.predict_proba(X), m.predict_proba(X), atol=1e-5)
    back = NumpySoftTree.from_json(num.to_json())
    assert (back.predict(X) == m.predict(X)).all()
    hard = m.to_hard_tree()
    assert hard.predict(X).shape == (len(X),)
    assert "is missing" in hard.export_text(max_features=6)


def test_onnx_export_handles_nan():
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    X, y = _data()
    m = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(X, y)
    sess = ort.InferenceSession(m.to_onnx().SerializeToString())
    out = sess.run(["probabilities"], {"X": X.astype(np.float32)})[0]
    assert np.allclose(out, m.predict_proba(X), atol=1e-4)


def test_explain_names_the_missing_indicator_and_skips_it_in_counterfactuals():
    X, y = _data()
    m = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(X, y)
    row = X[np.isnan(X[:, 2])][:1]
    e = m.explain(row, feature_names=["a", "b", "c", "d"])
    assert "c is missing" in e[0].attributions
    if e[0].counterfactual is not None:
        assert e[0].counterfactual.feature in {"a", "b", "c", "d"}
        assert not np.isnan(e[0].counterfactual.from_value)


def test_no_nan_in_training_means_no_indicators_and_nan_later_is_zero_filled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    m = SoftDecisionTree(depth=2, max_epochs=10, random_state=0).fit(X, y)
    assert len(m.missing_columns_) == 0 and m.model_.gates.weight.shape[1] == 3
    Xn = X.copy()
    Xn[0, 1] = np.nan
    filled = Xn.copy()
    filled[0, 1] = 0.0
    assert np.allclose(m.predict_proba(Xn), m.predict_proba(filled))


def test_infinity_is_still_rejected():
    X, y = _data()
    X[0, 1] = np.inf
    with pytest.raises(ValueError):
        SoftDecisionTree(depth=2, max_epochs=2).fit(X, y)


def test_init_cart_and_split_features_work_with_nan():
    X, y = _data()
    m = SoftDecisionTree(depth=3, max_epochs=10, init="cart", split_features=2, random_state=0).fit(X, y)
    assert ((m.model_.gates.weight.detach().numpy() != 0).sum(axis=1) <= 2).all()
    assert (m.predict(X) == y).mean() > 0.8

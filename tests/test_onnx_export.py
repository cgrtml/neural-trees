"""The ONNX export is the same model, run by ONNX Runtime without torch or this library."""
import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris, load_wine

from neural_trees import NumpySoftTree, SoftDecisionTree, SoftDecisionTreeRegressor

# A plain importorskip is not enough: a broken protobuf makes `import onnx`
# raise TypeError rather than ImportError, and the whole session would fail
# at collection on a machine that never asked for ONNX.
try:
    import onnx
    import onnxruntime as ort
except Exception as exc:  # noqa: BLE001 - any failure means "not available here"
    pytest.skip(f"onnx and onnxruntime are not usable here: {exc}", allow_module_level=True)


def _run(model, X):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, {"X": np.asarray(X, dtype=np.float32)})))


def _scaled(loader):
    X, y = loader(return_X_y=True)
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd, y


@pytest.mark.parametrize("growth", ["none", "per_leaf", "incremental"])
def test_classifier_matches_torch_and_numpy(growth):
    X, y = _scaled(load_wine)
    m = SoftDecisionTree(depth=4, max_epochs=30, growth=growth, learn_temperature=True, random_state=0).fit(X, y)
    out = _run(m.to_onnx(), X)
    np.testing.assert_allclose(out["probabilities"], m.predict_proba(X), atol=2e-5)
    np.testing.assert_allclose(out["probabilities"], m.to_numpy().predict_proba(X), atol=2e-5)
    assert (out["label"] == m.predict(X)).all()
    assert out["label"].dtype.kind == "i"


def test_string_classes_come_back_as_strings():
    X, y = _scaled(load_iris)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    m = SoftDecisionTree(depth=2, max_epochs=10, random_state=0).fit(X, names)
    out = _run(m.to_onnx(), X)
    assert set(out["label"]) <= set(names)
    assert (out["label"] == m.predict(X)).mean() > 0.99


def test_a_tree_with_no_split_predicts_its_root():
    X, y = _scaled(load_iris)
    m = SoftDecisionTree(depth=3, max_epochs=1, growth="per_leaf", random_state=0).fit(X[:12], y[:12])
    if int(m.model_.is_split.sum()) == 0:
        out = _run(m.to_onnx(), X)
        np.testing.assert_allclose(out["probabilities"], m.predict_proba(X), atol=2e-5)
    else:
        pytest.skip("the growth found a split on twelve rows; the no-split branch is covered elsewhere")


def test_regressor_single_and_multi_output():
    X, y = _scaled(load_digits)
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=20, random_state=0).fit(X, X[:, 5])
    out = _run(m.to_onnx(), X)
    assert out["value"].shape == (len(X),)
    np.testing.assert_allclose(out["value"], m.predict(X), rtol=1e-4, atol=1e-3)
    m2 = SoftDecisionTreeRegressor(depth=2, max_epochs=10, random_state=0).fit(X, X[:, :2])
    out2 = _run(m2.to_onnx(), X)
    assert out2["value"].shape == (len(X), 2)
    np.testing.assert_allclose(out2["value"], m2.predict(X), rtol=1e-4, atol=1e-3)


def test_metadata_and_file_round_trip(tmp_path):
    X, y = _scaled(load_wine)
    m = SoftDecisionTree(depth=2, max_epochs=5, random_state=0).fit(X, y)
    path = tmp_path / "tree.onnx"
    m.save_onnx(str(path), feature_names=[f"f{i}" for i in range(13)])
    loaded = onnx.load(str(path))
    onnx.checker.check_model(loaded)
    meta = {e.key: e.value for e in loaded.metadata_props}
    assert meta["kind"] == "classifier" and meta["feature_names"].startswith("f0,f1") and meta["classes"] == "0,1,2"
    # the numpy tree exports the same graph without torch in the loop
    again = NumpySoftTree.from_json(m.to_numpy().to_json()).to_onnx()
    np.testing.assert_allclose(_run(again, X)["probabilities"], _run(loaded, X)["probabilities"], atol=1e-6)

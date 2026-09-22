"""The numpy export is the same model, and it round-trips through JSON."""
import json

import numpy as np
import pytest
from sklearn.datasets import load_digits, load_wine

from neural_trees import NumpySoftTree, SoftDecisionTree


def _scaled(loader):
    X, y = loader(return_X_y=True)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd, y


@pytest.mark.parametrize("growth", ["none", "per_leaf"])
def test_numpy_predictions_match_torch(growth):
    X, y = _scaled(load_wine)
    m = SoftDecisionTree(depth=4, max_epochs=40, growth=growth, learn_temperature=True,
                         random_state=0).fit(X, y)
    npt = m.to_numpy()
    assert np.allclose(npt.predict_proba(X), m.predict_proba(X), atol=1e-5)
    assert (npt.predict(X) == m.predict(X)).all()
    assert npt.predict_log_proba(X).shape == (len(y), 3)


def test_json_round_trip_is_exact(tmp_path):
    X, y = _scaled(load_digits)
    m = SoftDecisionTree(depth=3, max_epochs=20, random_state=0).fit(X, y)
    npt = m.to_numpy(feature_names=[f"px{i}" for i in range(64)])
    text = npt.to_json()
    back = NumpySoftTree.from_json(text)
    assert np.array_equal(back.predict_proba(X), npt.predict_proba(X))
    assert back.feature_names_ == npt.feature_names_ and back.classes_.tolist() == list(range(10))
    path = tmp_path / "tree.json"
    npt.to_json(str(path))
    assert np.array_equal(NumpySoftTree.from_json(str(path)).predict(X), npt.predict(X))
    d = json.loads(text)
    assert d["format"] == "neural-trees/soft-tree/1"


def test_rejects_wrong_shapes_and_formats():
    X, y = _scaled(load_wine)
    npt = SoftDecisionTree(depth=2, max_epochs=5, random_state=0).fit(X, y).to_numpy()
    with pytest.raises(ValueError, match="shape"):
        npt.predict(X[:, :5])
    with pytest.raises(ValueError, match="format"):
        NumpySoftTree.from_dict({"format": "something-else"})

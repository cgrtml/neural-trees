"""The explanation says only things that can be checked against the model."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine

from neural_trees import Explanation, SoftDecisionTree


def _scaled(loader):
    X, y = loader(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


@pytest.fixture(scope="module")
def iris_model():
    X, y = _scaled(load_iris)
    return SoftDecisionTree(depth=3, max_epochs=40, random_state=0).fit(X, y), X, y


def test_explanation_agrees_with_the_model(iris_model):
    model, X, y = iris_model
    names = ["sl", "sw", "pl", "pw"]
    exps = model.explain(X[:20], feature_names=names)
    assert len(exps) == 20 and all(isinstance(e, Explanation) for e in exps)
    preds = model.predict(X[:20])
    probas = model.predict_proba(X[:20])
    for e, p, pr in zip(exps, preds, probas):
        assert e.predicted_class == p
        assert abs(e.probabilities[p] - pr.max()) < 1e-6
        assert 0.0 < e.leaf_probability <= 1.0
        assert abs(sum(e.leaf_distribution.values()) - 1.0) < 1e-5
        assert 1 <= len(e.path) <= model.depth
        for s in e.path:
            assert 0.0 < s.probability < 1.0 and s.went in ("left", "right")
            assert len(s.terms) == 3 and all(t["feature"] in names for t in s.terms)
        assert set(e.attributions) == set(names)


def test_dominant_leaf_is_the_leaf_with_most_mass(iris_model):
    import torch
    model, X, _ = iris_model
    e = model.explain(X[7])
    m = model.model_
    with torch.no_grad():
        _, _, terminal = m._walk(torch.FloatTensor(X[7:8]))
    idx = np.concatenate([i.numpy() for _, i in terminal])
    mu = np.concatenate([lm.exp().numpy() for lm, _ in terminal], axis=1)[0]
    assert e.leaf == idx[np.argmax(mu)]
    assert abs(e.leaf_probability - mu.max()) < 1e-6


def test_counterfactual_really_flips_the_class(iris_model):
    model, X, _ = iris_model
    exps = model.explain(X)
    found = [e for e in exps if e.counterfactual is not None]
    assert found, "no counterfactual found on any Iris sample"
    for e, x in zip(exps, X):
        if e.counterfactual is None:
            continue
        c = e.counterfactual
        x_cf = x.copy()
        x_cf[c.index] = c.to_value
        assert model.predict(x_cf.reshape(1, -1))[0] == c.new_class
        assert c.new_class != e.predicted_class
        assert c.feature == f"x{c.index}"


def test_single_row_returns_one_explanation_and_text(iris_model):
    model, X, _ = iris_model
    e = model.explain(X[0], feature_names=["sl", "sw", "pl", "pw"])
    assert isinstance(e, Explanation)
    text = e.to_text()
    assert "predicted" in text and "gate" in text and "counterfactual" in text
    assert str(e) == text
    d = e.to_dict()
    assert d["predicted_class"] == e.predicted_class and "feature_names" not in d


def test_feature_names_are_validated(iris_model):
    model, X, _ = iris_model
    with pytest.raises(ValueError, match="feature_names"):
        model.explain(X[0], feature_names=["a", "b"])


def test_explain_works_on_an_unbalanced_per_leaf_tree():
    X, y = _scaled(load_wine)
    model = SoftDecisionTree(depth=4, max_epochs=60, growth="per_leaf", random_state=0).fit(X, y)
    exps = model.explain(X[:10], counterfactual=False)
    for e in exps:
        assert all(c is None for c in [e.counterfactual])
        assert 0 <= e.leaf < model.model_.n_internal + model.model_.n_leaves
        assert len(e.path) <= model.depth

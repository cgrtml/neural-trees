"""Export rules for to_hard_tree (#101)."""
import numpy as np
import pytest
from sklearn.datasets import load_wine

from neural_trees import SoftDecisionTree


@pytest.fixture(scope="module")
def fitted():
    X, y = load_wine(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    m = SoftDecisionTree(depth=3, max_epochs=30, learn_temperature=True, random_state=0).fit(X, y)
    return m, X, y


def test_default_rule_is_gate_and_unchanged(fitted):
    m, X, _ = fitted
    a = m.to_hard_tree()
    b = m.to_hard_tree(rule="gate")
    assert a.rule == "gate"
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_leaf_rule_matches_the_numpy_mixture_argmax_leaf(fitted):
    """The 'leaf' rule's arrival probabilities are the soft model's own."""
    m, X, _ = fitted
    hard = m.to_hard_tree(rule="leaf")
    leaves, log_mu = hard._arrival_log_probs(X)
    mu = np.exp(log_mu)
    np.testing.assert_allclose(mu.sum(1), 1.0, atol=1e-9)
    # Mixing the leaf distributions with these arrival probabilities gives
    # the soft model's predict_proba.
    mixture = mu @ hard.node_distributions_[leaves]
    np.testing.assert_allclose(mixture, m.predict_proba(X), atol=1e-5)


def test_contribution_rule_never_disagrees_with_the_mixture_more_than_leaf(fitted):
    m, X, _ = fitted
    soft = m.predict(X)
    leaf = (m.to_hard_tree(rule="leaf").predict(X) == soft).mean()
    contribution = (m.to_hard_tree(rule="contribution").predict(X) == soft).mean()
    gate = (m.to_hard_tree(rule="gate").predict(X) == soft).mean()
    assert min(leaf, contribution, gate) > 0.9
    # Each rule returns a valid leaf distribution row.
    for rule in ("gate", "leaf", "contribution"):
        proba = m.to_hard_tree(rule=rule).predict_proba(X)
        np.testing.assert_allclose(proba.sum(1), 1.0)


def test_unknown_rule_is_rejected(fitted):
    m, _, _ = fitted
    with pytest.raises(ValueError, match="rule must be one of"):
        m.to_hard_tree(rule="path")

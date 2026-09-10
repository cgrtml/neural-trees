"""Tests for exporting a trained soft tree as a hard one."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_trees import HardDecisionTree, SoftDecisionTree


@pytest.fixture(scope="module")
def wine_fit():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    soft = SoftDecisionTree(depth=4, max_epochs=60, random_state=0).fit(X_train, y_train)
    return soft, X_train, X_test, y_train, y_test


def test_export_has_the_shape_of_the_soft_tree(wine_fit):
    soft, _, _, _, _ = wine_fit
    hard = soft.to_hard_tree()

    assert isinstance(hard, HardDecisionTree)
    assert hard.depth == soft.depth
    assert hard.weights_.shape == (2 ** soft.depth - 1, soft.n_features_in_)
    assert hard.biases_.shape == (2 ** soft.depth - 1,)
    assert hard.leaf_distributions_.shape == (2 ** soft.depth, 3)
    np.testing.assert_array_equal(hard.classes_, soft.classes_)


def test_routing_follows_the_sign_of_each_gate(wine_fit):
    """The hard tree's path must be the one the soft gates would take."""
    soft, _, X_test, _, _ = wine_fit
    hard = soft.to_hard_tree()

    expected = []
    for x in X_test:
        node = 0
        for _ in range(hard.depth):
            score = hard.weights_[node] @ x + hard.biases_[node]
            node = 2 * node + 1 + int(score > 0)
        expected.append(node - len(hard.weights_))

    np.testing.assert_array_equal(hard._leaf_index(X_test), np.array(expected))


def test_predictions_are_the_reached_leaf_distribution(wine_fit):
    soft, _, X_test, _, _ = wine_fit
    hard = soft.to_hard_tree()

    proba = hard.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        hard.predict(X_test), hard.classes_[proba.argmax(axis=1)]
    )
    leaves = hard.leaf_distributions_[hard._leaf_index(X_test)]
    np.testing.assert_allclose(proba, leaves)


def test_agrees_with_the_soft_tree_on_most_samples(wine_fit):
    """
    A mixture over leaves is not a single path, so the two models are not
    identical. They should still agree on the large majority of samples.
    """
    soft, _, X_test, _, y_test = wine_fit
    hard = soft.to_hard_tree()

    agreement = (hard.predict(X_test) == soft.predict(X_test)).mean()
    assert agreement > 0.9
    assert hard.score(X_test, y_test) > soft.score(X_test, y_test) - 0.1


def test_prediction_needs_no_torch(wine_fit):
    """The export exists to take PyTorch out of the inference path."""
    import sys

    soft, _, X_test, _, _ = wine_fit
    hard = soft.to_hard_tree()

    for array in (hard.weights_, hard.biases_, hard.leaf_distributions_):
        assert isinstance(array, np.ndarray)
    assert "torch" not in type(hard).__module__
    assert sys.modules["neural_trees.decision_trees.hard_tree"].__dict__.get("torch") is None


def test_export_text_is_readable(wine_fit):
    soft, _, _, _, _ = wine_fit
    hard = soft.to_hard_tree()
    names = [f"feature_{i}" for i in range(hard.n_features_in_)]

    text = hard.export_text(feature_names=names, max_features=2)
    lines = text.splitlines()

    assert len(lines) == 2 ** (hard.depth + 1) - 1
    assert lines[0].startswith("if ")
    assert any("predict" in line for line in lines)
    assert "feature_0" in text or "feature_1" in text
    # Only the requested number of terms per split, plus a count of the rest.
    assert f"(+{hard.n_features_in_ - 2} more)" in text


def test_export_text_defaults_and_validates_feature_names(wine_fit):
    soft, _, _, _, _ = wine_fit
    hard = soft.to_hard_tree()

    assert "x0" in hard.export_text() or "x1" in hard.export_text()
    with pytest.raises(ValueError, match="feature_names has"):
        hard.export_text(feature_names=["only", "two"])


def test_wrong_feature_count_is_rejected(wine_fit):
    soft, _, X_test, _, _ = wine_fit
    hard = soft.to_hard_tree()

    with pytest.raises(ValueError, match="features, but"):
        hard.predict(X_test[:, :3])


def test_depth_one_tree_exports():
    X, y = load_iris(return_X_y=True)
    soft = SoftDecisionTree(depth=1, max_epochs=20, random_state=0).fit(X, y)
    hard = soft.to_hard_tree()

    assert hard.weights_.shape[0] == 1
    assert hard.leaf_distributions_.shape[0] == 2
    assert set(hard.predict(X)).issubset(set(hard.classes_))

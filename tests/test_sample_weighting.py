"""
Weighting behaves the same way across the torch-backed estimators.

SoftDecisionTree has its own weighting tests; these cover the two that gained
sample_weight and class_weight later, and assert the properties that make a
weighting implementation trustworthy rather than merely present.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

from neural_trees import GALNetwork, HierarchicalMixtureOfExperts

ESTIMATORS = [
    lambda **kw: HierarchicalMixtureOfExperts(depth=1, max_epochs=20, random_state=0, **kw),
    lambda **kw: GALNetwork(max_epochs=40, random_state=0, **kw),
]
IDS = ["HierarchicalMixtureOfExperts", "GALNetwork"]


@pytest.fixture(scope="module")
def imbalanced():
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=4, n_redundant=0,
        weights=[0.94, 0.06], flip_y=0.02, class_sep=0.7, random_state=0,
    )
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_uniform_weights_match_the_unweighted_fit(make_estimator):
    X, y = load_iris(return_X_y=True)

    plain = make_estimator().fit(X, y)
    weighted = make_estimator().fit(X, y, sample_weight=np.full(len(X), 4.0))

    np.testing.assert_allclose(
        plain.predict_proba(X), weighted.predict_proba(X), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_zero_weight_removes_a_class(make_estimator):
    X, y = load_iris(return_X_y=True)
    weights = np.where(y == 2, 0.0, 1.0)

    estimator = make_estimator().fit(X, y, sample_weight=weights)

    assert list(estimator.classes_) == [0, 1, 2]
    assert 2 not in set(estimator.predict(X))


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_sample_weight_length_is_validated(make_estimator):
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError):
        make_estimator().fit(X, y, sample_weight=np.ones(7))


def test_balanced_class_weight_rescues_the_minority_in_hmoe(imbalanced):
    X, y = imbalanced

    plain = HierarchicalMixtureOfExperts(depth=2, max_epochs=80, random_state=0).fit(X, y)
    balanced = HierarchicalMixtureOfExperts(
        depth=2, max_epochs=80, random_state=0, class_weight="balanced"
    ).fit(X, y)

    assert recall_score(y, balanced.predict(X), pos_label=1) > recall_score(
        y, plain.predict(X), pos_label=1
    ) + 0.3
    assert balanced.score(X, y) > plain.score(X, y) - 0.05


def test_balanced_class_weight_rescues_the_minority_in_gal(imbalanced):
    """
    Unweighted, GAL predicted the rare class **zero** times on this problem
    while reporting 0.930 accuracy, which is exactly the failure class
    weighting exists to catch.
    """
    X, y = imbalanced

    plain = GALNetwork(max_epochs=120, random_state=0).fit(X, y)
    balanced = GALNetwork(
        max_epochs=120, random_state=0, class_weight="balanced"
    ).fit(X, y)

    assert recall_score(y, plain.predict(X), pos_label=1) < 0.2
    assert recall_score(y, balanced.predict(X), pos_label=1) > 0.5


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_explicit_class_weight_dict_is_accepted(make_estimator, imbalanced):
    X, y = imbalanced
    estimator = make_estimator(class_weight={0: 1.0, 1: 10.0}).fit(X, y)
    assert estimator.predict(X).shape == y.shape


def test_gal_growth_criterion_sees_the_weights():
    """
    Weighting the loss but judging the architecture on unweighted accuracy
    would let sample_weight change what the network fits while leaving what it
    builds untouched. With extreme weights the model must follow them.
    """
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    X, y = make_blobs(centers=2, random_state=0, cluster_std=20)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=0)

    gal = GALNetwork(
        max_epochs=100, random_state=0, class_weight={0: 1000, 1: 0.0001}
    ).fit(X_train, y_train)

    # Class 1 is worth ten million times less; nothing should land there.
    assert set(gal.predict(X_test)) == {0}


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_predictions_do_not_depend_on_row_order(make_estimator):
    """
    Both estimators predict in float64 on the CPU for this reason: BLAS blocks
    differently for different memory layouts, and in float32 a borderline
    argmax could flip depending on where the sample sat in the batch.
    """
    X, y = load_iris(return_X_y=True)
    estimator = make_estimator().fit(X, y)

    forward = estimator.predict_proba(X[:40])
    reversed_back = estimator.predict_proba(X[:40][::-1])[::-1]
    np.testing.assert_allclose(forward, reversed_back, rtol=1e-9, atol=1e-12)

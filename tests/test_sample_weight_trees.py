"""sample_weight and class_weight in the two multivariate trees (#95)."""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from neural_trees import MultivariateDecisionTree, OmnivariateDecisionTree
from neural_trees.decision_trees.omnivariate_tree import _cv_score_or_none

MODELS = {
    "multivariate": lambda **kw: MultivariateDecisionTree(max_depth=3, random_state=0, **kw),
    "omnivariate": lambda **kw: OmnivariateDecisionTree(max_depth=3, random_state=0, **kw),
}


@pytest.fixture(scope="module")
def wine():
    X, y = load_wine(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


def _two_blobs(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(200, 2), rng.randn(200, 2) + [2.0, 0.0]])
    y = np.repeat([0, 1], 200)
    return X, y


def test_unweighted_scoring_is_cross_val_score(wine):
    """Pins the None path of the omnivariate selection to what it replaced."""
    X, y = wine
    y_bin = (y == 1).astype(int)
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)
    ours = _cv_score_or_none(stump, X, y_bin, 3)
    theirs = cross_val_score(stump, X, y_bin, cv=3, scoring="accuracy").mean()
    assert ours == theirs


@pytest.mark.parametrize("name", MODELS)
def test_none_and_unit_weights_agree(wine, name):
    X, y = wine
    plain = MODELS[name]().fit(X, y)
    explicit_none = MODELS[name]().fit(X, y, sample_weight=None)
    ones = MODELS[name]().fit(X, y, sample_weight=np.ones(len(y)))
    assert np.array_equal(plain.predict_proba(X), explicit_none.predict_proba(X))
    # Unit weights take the weighted path; the multivariate discriminant is
    # then solved directly rather than by scikit-learn, so labels are
    # compared rather than bits.
    assert (plain.predict(X) == ones.predict(X)).mean() > 0.98


@pytest.mark.parametrize("name", MODELS)
def test_weighting_a_class_up_moves_the_split(name):
    """
    Would fail if the weights were dropped anywhere between fit and the leaf:
    with class 1 weighted ten to one, points between the two blobs must be
    called class 1, which needs the weights in the discriminant (or stump)
    and in the leaf distribution.
    """
    X, y = _two_blobs()
    between = np.array([[0.6, 0.0], [0.8, 0.0], [0.9, 0.0]])
    kw = {"max_depth": 1} if name == "multivariate" else {"max_depth": 1, "selection": "accuracy"}
    plain = MODELS[name]().set_params(**kw).fit(X, y)
    heavy = MODELS[name]().set_params(**kw).fit(X, y, sample_weight=np.where(y == 1, 10.0, 1.0))
    assert (plain.predict(between) == 0).all()
    assert (heavy.predict(between) == 1).all()


@pytest.mark.parametrize("name", MODELS)
def test_balanced_class_weight_raises_minority_recall(name):
    X, y = _two_blobs()
    keep = np.r_[np.arange(200), np.arange(200, 220)]  # class 1 at 10%
    Xi, yi = X[keep], y[keep]
    kw = {"max_depth": 1} if name == "multivariate" else {"max_depth": 1, "selection": "accuracy"}
    plain = MODELS[name]().set_params(**kw).fit(Xi, yi)
    balanced = MODELS[name]().set_params(**kw, class_weight="balanced").fit(Xi, yi)
    Xt, yt = _two_blobs(seed=1)
    minority = yt == 1
    assert (balanced.predict(Xt[minority]) == 1).mean() > (plain.predict(Xt[minority]) == 1).mean()


def _failures(est):
    return [r["check_name"] for r in check_estimator(est, on_fail=None) if r["status"] == "failed"]


def test_estimator_checks_with_sample_weight():
    # Both trees fail exactly one check, the one that requires weighting a
    # row to be identical to repeating it. The check's data has 15 rows and
    # 30 features, so every node discriminant is under-determined; the
    # weighted path solves it with a weighted pseudo-inverse and the
    # unweighted path with scikit-learn's SVD solver, which agree only up
    # to floating point, and on under-determined data that is not enough
    # for identical predictions. The omnivariate tree fails it because two of
    # its three candidate split types (LDA, MLP) take no sample_weight. The
    # class_weight check passes for both because min_weight_fraction_leaf,
    # which the check sets to 0.01, refuses leaves of negligible weight.
    for est in (
        MultivariateDecisionTree(max_depth=2, random_state=0),
        OmnivariateDecisionTree(max_depth=2, random_state=0),
    ):
        assert _failures(est) == ["check_sample_weight_equivalence_on_dense_data"]

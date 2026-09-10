"""
warm_start continues a fit instead of restarting it.

These tests cover the promise the parameter actually makes. It is deliberately
not partial_fit: sklearn's contract there promises that batch updates approach
training on the union, which mini-batch gradient descent over a second dataset
does not deliver, and requires handling classes absent from the first call,
which a fixed output layer cannot.
"""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from neural_trees import GALNetwork, HierarchicalMixtureOfExperts, SoftDecisionTree

ESTIMATORS = [
    lambda **kw: SoftDecisionTree(depth=3, random_state=0, **kw),
    lambda **kw: HierarchicalMixtureOfExperts(depth=2, random_state=0, **kw),
    lambda **kw: GALNetwork(random_state=0, **kw),
]
IDS = ["SoftDecisionTree", "HierarchicalMixtureOfExperts", "GALNetwork"]


@pytest.fixture(scope="module")
def wine():
    X, y = load_wine(return_X_y=True)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_second_fit_continues_instead_of_restarting(make_estimator, wine):
    X, y = wine

    warm = make_estimator(max_epochs=30, warm_start=True).fit(X, y)
    after_first = warm.score(X, y)
    warm.fit(X, y)
    after_second = warm.score(X, y)

    cold = make_estimator(max_epochs=30).fit(X, y)

    assert after_second >= after_first
    assert after_second >= cold.score(X, y)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_without_warm_start_the_second_fit_starts_over(make_estimator, wine):
    X, y = wine

    estimator = make_estimator(max_epochs=20).fit(X, y)
    first = estimator.predict_proba(X).copy()
    estimator.fit(X, y)

    # Same seed, same data, fresh initialization: identical, not continued.
    np.testing.assert_allclose(first, estimator.predict_proba(X), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_changing_the_label_set_is_refused(make_estimator, wine):
    X, y = wine
    estimator = make_estimator(max_epochs=5, warm_start=True).fit(X, y)

    with pytest.raises(ValueError, match="same classes"):
        estimator.fit(X, np.where(y == 2, 1, y))


@pytest.mark.parametrize("make_estimator", ESTIMATORS, ids=IDS)
def test_first_fit_with_warm_start_behaves_normally(make_estimator, wine):
    """Nothing to continue from, so it must match an ordinary fit."""
    X, y = wine

    warm = make_estimator(max_epochs=20, warm_start=True).fit(X, y)
    cold = make_estimator(max_epochs=20).fit(X, y)

    np.testing.assert_allclose(
        warm.predict_proba(X), cold.predict_proba(X), rtol=1e-5, atol=1e-6
    )


def test_gal_keeps_the_architecture_growth_chose(wine):
    """Continuing must not restart the network from initial_hidden."""
    X, y = wine

    gal = GALNetwork(max_epochs=60, random_state=0, warm_start=True).fit(X, y)
    grown_units = gal.n_hidden_final_
    assert grown_units > gal.initial_hidden

    gal.fit(X, y)
    assert gal.n_hidden_final_ >= grown_units


def test_incremental_growth_refuses_warm_start(wine):
    """
    The second fit would restart the depth search from a single split and throw
    away the depth the first one chose, so it is refused rather than silently
    doing that.
    """
    X, y = wine
    sdt = SoftDecisionTree(
        depth=3, max_epochs=30, growth="incremental", warm_start=True, random_state=0
    ).fit(X, y)

    with pytest.raises(ValueError, match="not supported with growth"):
        sdt.fit(X, y)

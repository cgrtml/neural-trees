import numpy as np
import pytest
from sklearn.datasets import load_diabetes, make_regression
from sklearn.utils.estimator_checks import check_estimator

from neural_trees import SoftDecisionTreeRegressor


def _diabetes():
    X, y = load_diabetes(return_X_y=True)
    return (X - X.mean(0)) / X.std(0), y


def test_fits_and_beats_the_mean_predictor():
    X, y = _diabetes()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=80, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.3          # the mean predictor scores 0
    assert m.predict(X).shape == (len(y),)
    assert m.n_iter_ == 80 and len(m.training_history_) == 80


def test_untrained_tree_predicts_the_mean():
    X, y = _diabetes()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=0, random_state=0).fit(X, y)
    assert np.allclose(m.predict(X), y.mean(), atol=1e-4)


def test_multi_output_and_sample_weight():
    X, Y = make_regression(n_samples=300, n_features=6, n_targets=2, noise=5.0, random_state=0)
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=60, random_state=0).fit(X, Y)
    assert m.predict(X).shape == (300, 2) and m.n_outputs_ == 2
    assert m.score(X, Y) > 0.5
    w = np.ones(300)
    w[:150] = 0.0                         # ignore half the data
    m2 = SoftDecisionTreeRegressor(depth=2, max_epochs=20, random_state=0).fit(X, Y, sample_weight=w)
    m3 = SoftDecisionTreeRegressor(depth=2, max_epochs=20, random_state=0).fit(X[150:], Y[150:])
    assert np.allclose(m2.y_mean_, m3.y_mean_)


def test_early_stopping_restores_the_best_epoch():
    X, y = _diabetes()
    m = SoftDecisionTreeRegressor(depth=3, max_epochs=200, early_stopping=True,
                                  n_iter_no_change=5, random_state=0).fit(X, y)
    assert m.n_iter_ < 200
    assert "val_loss" in m.training_history_[0]


def test_leaf_values_are_in_target_units():
    X, y = _diabetes()
    m = SoftDecisionTreeRegressor(depth=2, max_epochs=30, random_state=0).fit(X, y)
    v = m.get_leaf_values()
    assert v.shape == (4, 1) and y.min() - 50 < v.mean() < y.max() + 50
    assert len(m.get_split_weights()) == 3


def test_validation():
    X, y = _diabetes()
    with pytest.raises(ValueError, match="depth"):
        SoftDecisionTreeRegressor(depth=0).fit(X, y)
    with pytest.raises(ValueError, match="validation_fraction"):
        SoftDecisionTreeRegressor(early_stopping=True, validation_fraction=1.5).fit(X, y)


def test_passes_the_scikit_learn_regressor_checks():
    """
    Every check but one. The exception is the same one the classifier fails:
    weighting a sample is not bit-identical to repeating it for a mini-batch
    learner, because the repeated dataset is batched differently. The
    configuration is trained enough to fit the checks' small regression
    problems; with fifteen epochs check_regressors_train fails for lack of
    training, not for lack of compliance.
    """
    est = SoftDecisionTreeRegressor(depth=3, max_epochs=100, learning_rate=0.05, random_state=0)
    failures = [r["check_name"] for r in check_estimator(est, on_fail=None) if r["status"] != "passed"]
    assert failures == ["check_sample_weight_equivalence_on_dense_data"]

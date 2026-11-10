"""Threshold inputs for distilled students (DistilledTree(thresholds=k), ThresholdBasis) and combine="mean"."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from neural_trees import DistilledForest, DistilledTree, SoftDecisionTree
from neural_trees.distillation import ThresholdBasis, _teacher_splits


def _data():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def _small_student():
    return SoftDecisionTree(depth=3, max_epochs=20, random_state=0)


def _teachers():
    out = [HistGradientBoostingClassifier(max_iter=20, random_state=0),
           GradientBoostingClassifier(n_estimators=20, random_state=0),
           RandomForestClassifier(20, max_depth=4, random_state=0),
           DecisionTreeClassifier(max_depth=4, random_state=0)]
    try:
        import xgboost as xgb
        out.append(xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0))
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        # n_jobs=1: LightGBM's OpenMP crashes on macOS once torch is loaded (a libomp clash, not ours)
        out.append(lgb.LGBMClassifier(n_estimators=20, num_leaves=8, verbose=-1, n_jobs=1))
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("teacher", _teachers(), ids=lambda t: type(t).__name__)
def test_basis_inputs_follow_the_teachers_split_direction(teacher):
    Xtr, _, ytr, _ = _data()
    teacher.fit(Xtr, ytr)
    splits = _teacher_splits(teacher, Xtr.shape[1])
    assert splits and all(0 <= j < Xtr.shape[1] for j, *_ in splits)
    basis = ThresholdBasis(teacher, Xtr.shape[1], k=8)
    assert 1 <= len(basis.pairs) <= 8
    Z = basis.transform(Xtr)
    assert Z.shape == (len(Xtr), Xtr.shape[1] + len(basis.pairs))
    np.testing.assert_allclose(Z[:, :Xtr.shape[1]], Xtr.astype(np.float32))
    assert set(np.unique(Z[:, Xtr.shape[1]:])) <= {0.0, 1.0}
    j, t, right_eq = basis.pairs[0]
    row = np.zeros((1, Xtr.shape[1]))
    row[0, j] = t                                             # exactly at the threshold
    assert basis.transform(row)[0, Xtr.shape[1]] == float(right_eq)
    names = basis.feature_names()
    assert len(names) == Z.shape[1] and names[Xtr.shape[1]].startswith(f"x{j} ")


def test_the_cart_teacher_split_is_read_exactly():
    Xtr, _, ytr, _ = _data()
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr, ytr)
    basis = ThresholdBasis(stump, Xtr.shape[1], k=4)
    assert len(basis.pairs) == 1
    goes_right = stump.apply(Xtr) == stump.tree_.children_right[0]
    np.testing.assert_array_equal(basis.transform(Xtr)[:, -1].astype(bool), goes_right)


def test_basis_keeps_missing_values_missing():
    Xtr, _, ytr, _ = _data()
    basis = ThresholdBasis(HistGradientBoostingClassifier(max_iter=5).fit(Xtr, ytr), Xtr.shape[1], k=4)
    row = Xtr[:1].copy()
    row[0, basis.pairs[0][0]] = np.nan
    assert np.isnan(basis.transform(row)[0, Xtr.shape[1]])


def test_distilled_tree_with_thresholds_predicts_and_explains_on_original_features():
    Xtr, Xte, ytr, yte = _data()
    m = DistilledTree(teacher=HistGradientBoostingClassifier(max_iter=30, random_state=0),
                      student=_small_student(), synthetic=1, thresholds=16, random_state=0).fit(Xtr, ytr)
    assert m.basis_ is not None and len(m.basis_.pairs) == 16
    assert m.student_inputs(Xte).shape == (len(Xte), Xte.shape[1] + 16)
    assert m.student_.n_features_in_ == Xte.shape[1] + 16
    P = m.predict_proba(Xte)
    assert P.shape == (len(Xte), 2) and np.allclose(P.sum(1), 1)
    assert (m.predict(Xte) == yte).mean() > 0.85
    assert 0.8 < m.fidelity(Xte) <= 1.0
    names = [f"f{j}" for j in range(Xte.shape[1])]
    e = m.explain(Xte[0], feature_names=names,
                  feature_bounds=(Xtr.min(0), Xtr.max(0)))
    assert isinstance(e.to_text(), str)
    assert len(m.to_numpy().predict_proba(m.student_inputs(Xte).astype(np.float32))) == len(Xte)


def test_default_is_unchanged_without_thresholds():
    Xtr, Xte, ytr, _ = _data()
    m = DistilledTree(teacher=HistGradientBoostingClassifier(max_iter=10, random_state=0),
                      student=_small_student(), synthetic=0, random_state=0).fit(Xtr, ytr)
    assert m.basis_ is None
    assert m.student_inputs(Xte) is Xte
    assert m.student_.n_features_in_ == Xte.shape[1]


def test_forest_mean_combine_averages_probabilities():
    Xtr, Xte, ytr, _ = _data()
    kw = dict(teacher=HistGradientBoostingClassifier(max_iter=20, random_state=0), student=_small_student(),
              n_trees=2, synthetic=0, thresholds=8, random_state=0)
    log = DistilledForest(**kw).fit(Xtr, ytr)
    mean = DistilledForest(combine="mean", **kw).fit(Xtr, ytr)
    Z = mean.student_inputs(Xte)
    expect = np.mean([t.predict_proba(Z) for t in mean.estimators_], axis=0)
    np.testing.assert_allclose(mean.predict_proba(Xte), expect)
    # the log-sum is sharper than the average of the same trees
    assert np.abs(log.predict_proba(Xte) - 0.5).mean() >= np.abs(mean.predict_proba(Xte) - 0.5).mean() - 1e-9


@pytest.mark.parametrize("kw", [{"thresholds": -1}, {"thresholds": 2.5}, {"thresholds": True}])
def test_rejects_bad_thresholds(kw):
    Xtr, _, ytr, _ = _data()
    with pytest.raises(ValueError, match="thresholds"):
        DistilledTree(student=_small_student(), synthetic=0, **kw).fit(Xtr, ytr)


def test_rejects_bad_combine_and_non_tree_teacher():
    from sklearn.linear_model import LogisticRegression
    Xtr, _, ytr, _ = _data()
    with pytest.raises(ValueError, match="combine"):
        DistilledForest(student=_small_student(), synthetic=0, combine="max").fit(Xtr, ytr)
    with pytest.raises(ValueError, match="tree-ensemble"):
        DistilledTree(teacher=LogisticRegression(), student=_small_student(), synthetic=0,
                      thresholds=4).fit(Xtr, ytr)


def test_pruning_folds_keep_the_class_copies_of_a_row_together():
    from neural_trees.decision_trees.soft_decision_tree import _pruning_folds
    X = np.random.default_rng(0).normal(size=(200, 3))
    Xe, ye = np.vstack([X, X]), np.r_[np.zeros(200, int), np.ones(200, int)]   # each row once per class
    for tr, te in _pruning_folds(Xe, ye, 0):
        assert not set(map(tuple, Xe[tr])) & set(map(tuple, Xe[te]))
    # rows that never repeat keep the stratified folds of before
    from sklearn.model_selection import StratifiedKFold
    y = (X[:, 0] > 0).astype(int)
    old = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    for (a, b), (c, d) in zip(_pruning_folds(X, y, 0), old):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_cart_init_on_class_expanded_rows_is_not_pruned_to_the_root():
    # an imbalanced teacher whose argmax is one split (x0 > 1.2) and whose ranking elsewhere is
    # smooth: with folds over the class copies the pruning picked that one split alone (the
    # credit-default collapse, 5 Nov 2026; this case gave 1 split before the fix, 126 after).
    # The choice still depends on the fold layout: over 12 seeded group-fold layouts it gave
    # 1 to 126 splits, one layout of 12 collapsing to 1. Folds are seeded and the same on every
    # CPU since 0.9.1, so this checks the layout of random_state=0.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6))
    p = np.where(X[:, 0] > 1.2, 0.7, 1 / (1 + np.exp(-(-2.0 + 0.5 * X[:, 1:4].sum(1)))))
    k0, k1 = 1 - p >= 0.01, p >= 0.01
    Xe = np.vstack([X[k0], X[k1]])
    ye = np.r_[np.zeros(k0.sum(), int), np.ones(k1.sum(), int)]
    we = np.r_[(1 - p)[k0], p[k1]]
    m = SoftDecisionTree(depth=8, init="cart", max_epochs=1, random_state=0).fit(Xe, ye, sample_weight=we)
    assert int(m.model_.is_split.sum()) > 10


def test_one_se_rule_picks_a_tree_no_larger_than_the_best_scoring_one():
    from sklearn.tree import DecisionTreeClassifier

    from neural_trees.decision_trees.soft_decision_tree import (
        _pruned_cart,
        _pruning_folds,
    )
    from neural_trees.distillation import _brier, _one_se_cart
    X, y = load_breast_cancer(return_X_y=True)
    w = np.ones(len(y))
    folds = _pruning_folds(X, y, 0)

    def make(a):
        return DecisionTreeClassifier(max_depth=8, ccp_alpha=a, random_state=0)
    best, _ = _pruned_cart(make, X, y, w, True, 0, _brier, folds)
    one_se = _one_se_cart(make, X, y, w, _brier, folds)
    assert 1 <= one_se.get_n_leaves() <= best.get_n_leaves()
    assert _brier(one_se, X, y, w) <= 0


def test_pruning_folds_keep_row_copies_together_and_repeat_exactly():
    from neural_trees.decision_trees.soft_decision_tree import _pruning_folds
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    Xe = np.vstack([X, X]); ye = np.r_[np.zeros(200, int), np.ones(200, int)]
    a, b = _pruning_folds(Xe, ye, 7), _pruning_folds(Xe, ye, 7)
    assert all(np.array_equal(p[1], q[1]) for p, q in zip(a, b))
    fold = np.empty(400, int)
    for k, (_, te) in enumerate(a):
        fold[te] = k
    assert np.array_equal(fold[:200], fold[200:])
    assert sorted(np.concatenate([te for _, te in a]).tolist()) == list(range(400))

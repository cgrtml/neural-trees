"""The two-group construction must not close a node that still holds every class (#104)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from neural_trees import MultivariateDecisionTree, OmnivariateDecisionTree
from neural_trees.decision_trees._grouping import two_group_candidates


def _digits_fold():
    X, y = load_digits(return_X_y=True)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te]


def test_multivariate_tree_grows_on_digits():
    # Before #104 this fold gave 0.35 with three internal nodes: a node of
    # 1 074 samples and ten classes became a leaf at depth 1.
    Xtr, ytr, Xte, yte = _digits_fold()
    m = MultivariateDecisionTree(max_depth=6, random_state=0).fit(Xtr, ytr)
    assert m.n_nodes_ >= 8  # three before the fix; 11 to 17 across platforms since
    # 0.35 with the defect; 0.78 to 0.93 across CI runners since, because the
    # node discriminants on pixel data depend on the BLAS build.
    assert m.score(Xte, yte) > 0.7


def test_omnivariate_tree_grows_on_digits():
    Xtr, ytr, Xte, yte = _digits_fold()
    m = OmnivariateDecisionTree(max_depth=6, random_state=0).fit(Xtr, ytr)
    assert sum(m.get_split_type_distribution().values()) >= 10
    assert m.score(Xte, yte) > 0.85


def test_a_singleton_class_does_not_found_a_group():
    # Nine well-populated classes and one class with a single far-away
    # sample: unweighted clustering isolated it; weighted, it joins a group
    # and both groups carry real mass.
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(50, 4) + 3 * np.eye(4)[k % 4] * (k // 4 + 1) for k in range(9)] + [[[40.0, 40.0, 40.0, 40.0]]])
    y = np.r_[np.repeat(np.arange(9), 50), 9]
    w = np.ones(len(y))
    for y_bin in two_group_candidates(X, y, w, random_state=0, min_rows=3):
        assert np.bincount(y_bin).min() >= 3


def test_balanced_cut_is_offered_when_it_differs():
    rng = np.random.RandomState(1)
    X = np.vstack([rng.randn(30, 3) + [k, 0, 0] for k in range(6)])
    y = np.repeat(np.arange(6), 30)
    cands = two_group_candidates(X, y, np.ones(len(y)), random_state=0, min_rows=1)
    assert 1 <= len(cands) <= 2
    for y_bin in cands:
        assert set(np.unique(y_bin)) == {0, 1}

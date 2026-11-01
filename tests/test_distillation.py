"""neural_trees.distill: a soft tree fitted to a teacher's probabilities."""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_trees import SoftDecisionTree, distill, fidelity


def _data(load=load_wine):
    X, y = load(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def test_distilled_tree_follows_the_teacher_on_held_out_rows():
    Xtr, Xte, ytr, _ = _data()
    teacher = RandomForestClassifier(100, random_state=0).fit(Xtr, ytr)
    d = distill(teacher, Xtr, X_eval=Xte, student=SoftDecisionTree(depth=3, max_epochs=40, random_state=0))
    assert d.evaluated_on == "held-out"
    assert d.fidelity > 0.85 and d.fidelity_rules > 0.8
    assert 0.0 <= d.tv_distance < 0.5
    assert d.n_splits >= 1 and d.features_per_split >= 1
    assert d.fidelity == pytest.approx(fidelity(teacher, d.student, Xte))
    assert "fidelity" in str(d)


def test_string_labels_and_training_row_report():
    Xtr, _, ytr, _ = _data(load_iris)
    names = np.array(["setosa", "versicolor", "virginica"])[ytr]
    teacher = RandomForestClassifier(50, random_state=0).fit(Xtr, names)
    d = distill(teacher, Xtr, student=SoftDecisionTree(depth=2, max_epochs=30, random_state=0))
    assert d.evaluated_on == "training rows"
    assert set(d.student.classes_) <= set(names)
    assert set(d.student.predict(Xtr)) <= set(names)


def test_rejects_unfitted_teacher_and_bad_threshold():
    from sklearn.exceptions import NotFittedError

    Xtr, _, ytr, _ = _data()
    with pytest.raises(NotFittedError):
        distill(RandomForestClassifier(), Xtr)
    teacher = RandomForestClassifier(10, random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="threshold"):
        distill(teacher, Xtr, threshold=1.0)


# ── DistilledTree ───────────────────────────────────────────────────────────
from neural_trees import DistilledTree  # noqa: E402


def test_distilled_tree_trains_teacher_then_one_tree():
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, Xte, ytr, yte = _data()
    m = DistilledTree(student=SoftDecisionTree(depth=3, max_epochs=40, random_state=0)).fit(Xtr, ytr)
    assert isinstance(m.teacher_, HistGradientBoostingClassifier)
    assert isinstance(m.student_, SoftDecisionTree)
    assert (m.predict(Xte) == yte).mean() > 0.85
    assert m.predict_proba(Xte).shape == (len(Xte), 3)
    assert 0.8 < m.fidelity(Xte) <= 1.0
    assert m.to_numpy().predict(Xte).shape == (len(Xte),)
    assert m.explain(Xte[:1])[0].predicted_class in set(ytr)


def test_alpha_mixes_true_labels_and_validation():
    Xtr, _, ytr, _ = _data(load_iris)
    teacher = RandomForestClassifier(30, random_state=0)
    m = DistilledTree(teacher=teacher, student=SoftDecisionTree(depth=2, max_epochs=10, random_state=0),
                      alpha=0.5).fit(Xtr, ytr)
    assert m.teacher_ is not teacher                    # cloned, the argument stays unfitted
    with pytest.raises(ValueError, match="alpha"):
        DistilledTree(alpha=1.5).fit(Xtr, ytr)


def test_distilled_tree_accepts_missing_values():
    Xtr, Xte, ytr, _ = _data()
    Xtr = Xtr.copy()
    Xtr[::5, 2] = np.nan
    m = DistilledTree(student=SoftDecisionTree(depth=3, max_epochs=20, random_state=0)).fit(Xtr, ytr)
    assert list(m.student_.missing_columns_) == [2]
    assert m.predict(Xte).shape == (len(Xte),)


def test_unlabeled_rows_reach_the_student_not_the_teacher():
    Xtr, Xte, ytr, yte = _data()
    Xl, Xu, yl, _ = train_test_split(Xtr, ytr, test_size=0.6, random_state=0, stratify=ytr)
    m = DistilledTree(student=SoftDecisionTree(depth=3, max_epochs=40, random_state=0))
    m.fit(Xl, yl, X_unlabeled=Xu)
    assert m.teacher_.n_features_in_ == Xl.shape[1]
    assert (m.predict(Xte) == yte).mean() > 0.8
    with pytest.raises(ValueError, match="X_unlabeled"):
        DistilledTree().fit(Xl, yl, X_unlabeled=Xu[:, :3])


def test_munge_rows_stay_near_the_data():
    from neural_trees.distillation import munge

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(100, 30, 300), rng.integers(0, 3, 300), rng.normal(size=300)])
    X[::7, 2] = np.nan
    M = munge(X, passes=2, random_state=0)
    assert M.shape == (600, 3)
    assert np.array_equal(M, munge(X, passes=2, random_state=0), equal_nan=True)
    assert set(np.unique(M[:, 1])) <= {0.0, 1.0, 2.0}          # discrete feature copies values
    assert abs(np.nanmean(M[:, 0]) - 100) < 10 and np.nanstd(M[:, 0]) < 40
    assert np.isnan(M[:, 2]).any() and not np.isnan(M[:, :2]).any()
    assert munge(X, passes=0).shape == (0, 3)
    with pytest.raises(ValueError):
        munge(X, p=1.5)


def test_synthetic_rows_reach_the_student_only():
    Xtr, Xte, ytr, yte = _data()
    m = DistilledTree(student=SoftDecisionTree(depth=3, max_epochs=20, random_state=0), synthetic=1,
                      random_state=0).fit(Xtr, ytr)
    assert m.teacher_.n_features_in_ == Xtr.shape[1]
    assert (m.predict(Xte) == yte).mean() > 0.8
    m0 = DistilledTree(student=SoftDecisionTree(depth=3, max_epochs=20, random_state=0), synthetic=0).fit(Xtr, ytr)
    assert m0.predict(Xte).shape == (len(Xte),)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="synthetic"):
            DistilledTree(synthetic=bad).fit(Xtr, ytr)


@pytest.mark.parametrize("kind", ["tree", "forest"])
def test_scikit_learn_estimator_checks(kind):
    from sklearn.utils.estimator_checks import check_estimator

    from neural_trees import DistilledForest
    student = SoftDecisionTree(depth=2, max_epochs=5, random_state=0)
    est = (DistilledTree(student=student, synthetic=1, random_state=0) if kind == "tree"
           else DistilledForest(student=student, n_trees=2, synthetic=1, random_state=0))
    failed = {r["check_name"] for r in check_estimator(est, on_fail=None) if r["status"] == "failed"}
    assert failed <= {"check_sample_weight_equivalence_on_dense_data"}


def test_distilled_forest_fits_a_group_and_predicts_with_it():
    from neural_trees import DistilledForest
    Xtr, Xte, ytr, yte = _data()
    m = DistilledForest(student=SoftDecisionTree(depth=3, max_epochs=10, random_state=0), n_trees=3,
                        synthetic=1, random_state=0).fit(Xtr, ytr)
    assert len(m.estimators_) == 3
    P = m.predict_proba(Xte)
    assert P.shape == (len(Xte), len(m.classes_)) and np.allclose(P.sum(axis=1), 1)
    # the group's prediction is the argmax of the summed log probabilities of its trees
    L = sum(np.log(t.predict_proba(Xte)) for t in m.estimators_)
    assert (m.predict(Xte) == m.classes_[L.argmax(axis=1)]).all()
    # the same through the torch-free exports
    Ln = sum(t.predict_log_proba(Xte) for t in m.to_numpy())
    assert (m.classes_[Ln.argmax(axis=1)] == m.predict(Xte)).mean() > 0.99
    assert 0.0 <= m.fidelity(Xte) <= 1.0
    # bootstrap samples and seeds differ, so the trees do
    W = [t.model_.gates.weight.detach().numpy() for t in m.estimators_]
    assert not np.allclose(W[0], W[1])
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="n_trees"):
            DistilledForest(n_trees=bad).fit(Xtr, ytr)
    with pytest.raises(NotImplementedError):
        m.explain(Xte)

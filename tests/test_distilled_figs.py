"""neural_trees.DistilledFIGS: FIGS fitted to a teacher's probabilities."""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

pytest.importorskip("imodels")

from neural_trees import DistilledFIGS  # noqa: E402


def _split(load):
    X, y = load(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


def test_binary_probabilities_size_and_fidelity():
    Xtr, Xte, ytr, yte = _split(load_breast_cancer)
    m = DistilledFIGS(max_rules=16, random_state=0).fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    assert P.shape == (len(Xte), 2)
    assert np.allclose(P.sum(1), 1.0) and (P >= 0).all() and (P <= 1).all()
    assert 1 <= m.complexity_ <= 16 and len(m.figs_) == 1
    assert (m.predict(Xte) == yte).mean() > 0.9
    assert m.fidelity(Xte) > 0.9


def test_multiclass_string_labels():
    Xtr, Xte, ytr, _ = _split(load_iris)
    names = np.array(["setosa", "versicolor", "virginica"])
    m = DistilledFIGS(max_rules=8, random_state=0).fit(Xtr, names[ytr])
    assert len(m.figs_) == 3 and set(m.predict(Xte)) <= set(names)
    assert np.allclose(m.predict_proba(Xte).sum(1), 1.0)


def test_synthetic_rows_and_custom_teacher():
    from sklearn.ensemble import RandomForestClassifier
    Xtr, Xte, ytr, _ = _split(load_breast_cancer)
    m = DistilledFIGS(teacher=RandomForestClassifier(50, random_state=0), max_rules=8, synthetic=1,
                      random_state=0).fit(Xtr, ytr)
    assert isinstance(m.teacher_, RandomForestClassifier) and m.complexity_ <= 8


@pytest.mark.parametrize("kw", [{"max_rules": 0}, {"max_rules": 2.5}, {"synthetic": -1}, {"synthetic": True}])
def test_rejects_bad_parameters(kw):
    Xtr, _, ytr, _ = _split(load_breast_cancer)
    with pytest.raises(ValueError):
        DistilledFIGS(**kw).fit(Xtr, ytr)


def test_rejects_wrong_feature_count():
    Xtr, Xte, ytr, _ = _split(load_breast_cancer)
    m = DistilledFIGS(max_rules=4, random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError):
        m.predict(Xte[:, :5])

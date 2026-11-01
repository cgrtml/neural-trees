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

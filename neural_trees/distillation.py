"""
Distil a fitted classifier (a tree ensemble, typically) into one soft tree,
and report how faithfully the tree reproduces it.

The student learns the teacher's class probabilities, not its labels: each
row is repeated once per class the teacher gives at least `threshold`
probability, weighted by that probability, which is exactly the soft-target
cross-entropy. Measured on 24 datasets with XGBoost, LightGBM and random
forest teachers (`benchmarks/distill.py`): agreement with the teacher 0.934
for this student against 0.901 for CART of depth 6, with about 11 splits
against 38 leaves; the lead comes from splits that use many features and is
gone once each split is limited to three or fewer (`split_features`).
Perturbed copies of the rows made no measurable difference there, so the
transfer set is the data passed in.

Fidelity on the rows the student was fitted on overstates it. Pass held-out
rows as `X_eval`; without them the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["Distilled", "distill", "fidelity"]


@dataclass
class Distilled:
    """
    The student and how closely it follows the teacher.

    Attributes
    ----------
    student : SoftDecisionTree
        The fitted soft tree.
    fidelity : float
        Share of `evaluated_on` rows where the student predicts the teacher's
        class.
    fidelity_rules : float
        The same for the student read as hard rules (`to_hard_tree()`), which
        is the model a person reads.
    tv_distance : float
        Mean total-variation distance between the two probability vectors,
        0 for identical, 1 for disjoint.
    n_splits : int
        Splits in the student.
    features_per_split : float
        Mean number of features carrying 90 % of a split's absolute weight.
    evaluated_on : str
        ``"held-out"`` when `X_eval` was given, ``"training rows"`` otherwise.
    """

    student: Any
    fidelity: float
    fidelity_rules: float
    tv_distance: float
    n_splits: int
    features_per_split: float
    evaluated_on: str

    def __str__(self) -> str:
        return (f"fidelity {self.fidelity:.3f} (rules {self.fidelity_rules:.3f}) on {self.evaluated_on}, "
                f"TV {self.tv_distance:.3f}, {self.n_splits} splits, "
                f"{self.features_per_split:.1f} features per split")


def _proba_over(model, X, classes) -> np.ndarray:
    """Probabilities of `model` laid out over `classes`, zero where it has none."""
    P = np.asarray(model.predict_proba(X))
    out = np.zeros((len(X), len(classes)))
    pos = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, pos[c]] = P[:, j]
    return out


def fidelity(teacher, student, X) -> float:
    """Share of rows of `X` where `student` predicts the class `teacher` predicts."""
    X = check_array(X, dtype=np.float64)
    return float(np.mean(np.asarray(student.predict(X)) == np.asarray(teacher.predict(X))))


def _features_per_split(student) -> float:
    W = np.abs(student.model_.gates.weight.detach().cpu().numpy())
    W = W[student.model_.is_split.cpu().numpy()]
    counts = []
    for w in W:
        if w.sum() > 0:
            share = np.sort(w)[::-1].cumsum() / w.sum()
            counts.append(int(np.searchsorted(share, 0.9) + 1))
    return float(np.mean(counts)) if counts else 0.0


def distill(teacher, X, *, student=None, X_eval=None, threshold: float = 0.01,
            random_state: Optional[int] = None) -> Distilled:
    """
    Fit a soft tree to the probabilities of a fitted classifier.

    Parameters
    ----------
    teacher : fitted classifier with `predict_proba` and `classes_`
        XGBoost, LightGBM, a random forest, or any scikit-learn classifier.
    X : array-like of shape (n_samples, n_features)
        Rows to transfer on, usually the teacher's training rows, on the
        scale the teacher expects. Soft trees learn best on standardised
        features: put a `StandardScaler` in front of both if the teacher was
        fitted on raw ones, or pass a student inside a pipeline.
    student : unfitted SoftDecisionTree, optional
        Defaults to ``SoftDecisionTree(depth=6, growth="per_leaf",
        growth_init="residual", max_epochs=180)``, the configuration
        measured above. Pass ``split_features=k`` in it for rules that read
        more easily, at a measured cost in fidelity.
    X_eval : array-like, optional
        Held-out rows for the fidelity report.
    threshold : float, default=0.01
        A class enters a row's soft target only with at least this teacher
        probability; it keeps the repeated rows few.
    random_state : int, optional
        Seed for the default student.

    Returns
    -------
    Distilled
    """
    from neural_trees import SoftDecisionTree

    check_is_fitted(teacher)
    if not hasattr(teacher, "predict_proba"):
        raise ValueError("the teacher must have predict_proba")
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold!r}")
    X = check_array(X, dtype=np.float64)
    classes = np.asarray(teacher.classes_)
    P = np.asarray(teacher.predict_proba(X), dtype=np.float64)
    if P.shape != (len(X), len(classes)):
        raise ValueError(f"teacher.predict_proba returned shape {P.shape}, expected {(len(X), len(classes))}")
    rows, cols = np.nonzero(P >= max(threshold, 1e-12))

    if student is None:
        student = SoftDecisionTree(depth=6, growth="per_leaf", growth_init="residual", max_epochs=180,
                                   random_state=random_state)
    student.fit(X[rows], classes[cols], sample_weight=P[rows, cols])

    evaluated_on = "held-out" if X_eval is not None else "training rows"
    Xe = check_array(X_eval, dtype=np.float64) if X_eval is not None else X
    Pt = np.asarray(teacher.predict_proba(Xe), dtype=np.float64)
    Ps = _proba_over(student, Xe, classes)
    t_pred = classes[Pt.argmax(axis=1)]
    rules = student.to_hard_tree().predict(Xe)
    return Distilled(
        student=student,
        fidelity=float(np.mean(classes[Ps.argmax(axis=1)] == t_pred)),
        fidelity_rules=float(np.mean(np.asarray(rules) == t_pred)),
        tv_distance=float(0.5 * np.abs(Pt - Ps).sum(axis=1).mean()),
        n_splits=int(student.model_.is_split.sum()),
        features_per_split=_features_per_split(student),
        evaluated_on=evaluated_on,
    )

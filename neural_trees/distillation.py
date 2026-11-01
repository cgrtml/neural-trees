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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["Distilled", "DistilledTree", "distill", "fidelity"]


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


class DistilledTree(ClassifierMixin, BaseEstimator):
    """
    One soft tree trained on what a strong model learned: `fit` trains the
    `teacher` on (X, y), then the `student` tree on the teacher's class
    probabilities on the same rows (mixed with the true labels by `alpha`).
    Prediction, `explain` and every export go through the single student
    tree, so the deployed model is one readable tree; the teacher is kept
    only to measure agreement.

    Why: on large tables a single soft tree trained on the labels trails
    gradient boosting, and the probabilities of a boosted model are a
    better training signal than the labels. On 20 000-row training sets
    (one split, depth 8 to 10, XGBoost teacher; benchmarks/gelistir.py) this
    moved MiniBooNE from 0.914 to 0.926, jannis from 0.737 to 0.764 and
    MagicTelescope (10 000 rows) from 0.833 to 0.848, against 0.939, 0.783
    and 0.859 for the teacher; it did not help where the tree is short of
    capacity (electricity). See the documentation for the full measurement.

    Parameters
    ----------
    teacher : classifier with predict_proba, default=None
        Cloned and fitted on (X, y). None uses scikit-learn's
        `HistGradientBoostingClassifier()`; XGBoost or LightGBM work the same.
    student : SoftDecisionTree, default=None
        Cloned and fitted on the teacher's probabilities. None uses
        ``SoftDecisionTree(depth=8, init="cart", learn_temperature=True,
        max_epochs=60, batch_size=256)``.
    alpha : float, default=0.0
        Weight of the true labels against the teacher's probabilities in the
        student's loss; 0 learns the teacher only.
    threshold : float, default=0.01
        A class enters a row's soft target only with at least this teacher
        probability.

    Attributes
    ----------
    teacher_, student_ : the fitted models
    classes_, n_features_in_
    """

    def __init__(self, teacher=None, student=None, alpha: float = 0.0, threshold: float = 0.01):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None, X_unlabeled=None) -> DistilledTree:
        """
        Fit the teacher on (X, y), then the student on the teacher's
        probabilities on X and, if given, on `X_unlabeled`: rows without a
        label, from the same population, that the teacher labels for the
        student. Labels are often scarcer than rows (every customer is in
        the table, few have an outcome), and the teacher can pass on what it
        learned from the labelled rows through the unlabelled ones; on
        20 000-row training sets with twice as many unlabelled rows this put
        a depth-8 tree level with its XGBoost teacher on jannis (0.786
        against 0.783). The teacher itself sees only (X, y).
        """
        from sklearn.base import clone
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.multiclass import check_classification_targets

        from neural_trees import SoftDecisionTree
        from neural_trees._missing import allow_nan

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha!r}")
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {self.threshold!r}")
        X, y = check_X_y(X, y, **allow_nan())
        check_classification_targets(y)
        teacher = clone(self.teacher) if self.teacher is not None else HistGradientBoostingClassifier()
        student = clone(self.student) if self.student is not None else SoftDecisionTree(
            depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256)
        if sample_weight is None:
            teacher.fit(X, y)
            w = np.ones(len(y))
        else:
            teacher.fit(X, y, sample_weight=sample_weight)
            w = np.asarray(sample_weight, dtype=np.float64)
        classes = np.asarray(teacher.classes_)
        P = np.asarray(teacher.predict_proba(X), dtype=np.float64)
        Xt, wt = X, w
        if X_unlabeled is not None:
            Xu = check_array(X_unlabeled, **allow_nan())
            if Xu.shape[1] != X.shape[1]:
                raise ValueError(f"X_unlabeled has {Xu.shape[1]} features, X has {X.shape[1]}")
            Xt = np.vstack([X, Xu])
            P = np.vstack([P, np.asarray(teacher.predict_proba(Xu), dtype=np.float64)])
            wt = np.concatenate([w, np.full(len(Xu), float(np.mean(w)))])
        rows, cols = np.nonzero(P >= max(self.threshold, 1e-12))
        Xs, ys, ws = [Xt[rows]], [classes[cols]], [(1.0 - self.alpha) * P[rows, cols] * wt[rows]]
        if self.alpha > 0:
            Xs.append(X)
            ys.append(y)
            ws.append(self.alpha * w)
        student.fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
        self.teacher_, self.student_ = teacher, student
        self.classes_ = student.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.student_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self)
        return self.student_.predict(X)

    def fidelity(self, X) -> float:
        """Share of rows of `X` where the student predicts the teacher's class."""
        check_is_fitted(self)
        return fidelity(self.teacher_, self.student_, X)

    def explain(self, X, **kwargs):
        """The student tree's explanation; see `SoftDecisionTree.explain`."""
        check_is_fitted(self)
        return self.student_.explain(X, **kwargs)

    def to_numpy(self, **kwargs):
        return self.student_.to_numpy(**kwargs)

    def to_hard_tree(self, **kwargs):
        return self.student_.to_hard_tree(**kwargs)

    def to_onnx(self, **kwargs):
        return self.student_.to_onnx(**kwargs)

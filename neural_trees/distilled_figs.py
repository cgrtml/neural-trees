"""
FIGS fitted to a teacher's class probabilities instead of the labels.

FIGS (Tan et al., PNAS 2025; the `imodels` package) grows a small sum of
axis-aligned trees, one split at a time, under a total split budget. Fitted
on the labels it overfits as the budget grows. Fitted by regression on a
strong teacher's probabilities it keeps improving instead. Distilling into
FIGS is not new (FIGS-BD, arXiv 2503.06730; `imodels.DistilledRegressor`);
this class is the recipe measured in the equal-size audit of the
neural-trees paper.

Measured there on 9 OpenML tables x 3 seeds, 20 000 training rows each,
tuned XGBoost teacher, scored on held-out test rows that no choice was made
on (protocol and hypotheses fixed in advance):

    splits   FIGS on labels      DistilledFIGS
             acc gap   AUC       acc gap   AUC
    128      -2.86     86.44     -2.28     87.41
    256      -3.06     85.79     -1.93     87.80
    512      -3.58     84.31     -1.84     87.95

(acc gap = accuracy minus the teacher's, in points; teacher AUC 89.88.)
At 512 splits it is ahead of FIGS on labels at its best size (128) by
+1.02 accuracy and +1.52 AUC, on every one of the 9 tables, and about level
with an EBM without interactions (+0.68 accuracy, 95% interval +0.16 to
+1.25). A full EBM with interactions stays ahead by about one point.
Synthetic rows (MUNGE) gave the same result at 13 times the fit time, so
`synthetic` defaults to 0.

Needs `imodels` (``pip install neural-trees[figs]``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["DistilledFIGS"]


class DistilledFIGS(ClassifierMixin, BaseEstimator):
    """
    A FIGS model trained on what a strong model learned: `fit` trains the
    `teacher` on (X, y), then a `FIGSRegressor` with `max_rules` splits on
    the teacher's probability of each class (one regressor for two classes,
    one per class otherwise). Prediction uses only the FIGS model(s); the
    teacher is kept to measure agreement.

    Parameters
    ----------
    teacher : classifier with predict_proba, default=None
        Cloned and fitted on (X, y). None uses scikit-learn's
        `HistGradientBoostingClassifier()`. The measurements in the module
        docstring used a tuned XGBoost teacher.
    max_rules : int, default=256
        Split budget of each FIGS regressor. 256 to 512 is where the gain
        over FIGS on labels was measured; with several classes the total
        size is ``n_classes * max_rules``.
    synthetic : int, default=0
        MUNGE passes over the rows (see `neural_trees.distillation.munge`),
        labelled by the teacher for the student. Did not help in the
        measurements; kept for small tables.
    random_state : int, optional
        Seeds FIGS and the synthetic rows.

    Attributes
    ----------
    teacher_ : the fitted teacher
    figs_ : list of fitted `imodels.FIGSRegressor`, one per modelled class
    classes_, n_features_in_
    complexity_ : int, total number of splits over `figs_`
    """

    def __init__(self, teacher=None, max_rules: int = 256, synthetic: int = 0,
                 random_state: Optional[int] = None):
        self.teacher = teacher
        self.max_rules = max_rules
        self.synthetic = synthetic
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None) -> DistilledFIGS:
        try:
            from imodels import FIGSRegressor
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("DistilledFIGS needs imodels: pip install neural-trees[figs]") from e
        from sklearn.ensemble import HistGradientBoostingClassifier

        from neural_trees.distillation import munge

        if not isinstance(self.max_rules, (int, np.integer)) or isinstance(self.max_rules, bool) or self.max_rules < 1:
            raise ValueError(f"max_rules must be a positive int, got {self.max_rules!r}")
        if not isinstance(self.synthetic, (int, np.integer)) or isinstance(self.synthetic, bool) or self.synthetic < 0:
            raise ValueError(f"synthetic must be a non-negative int, got {self.synthetic!r}")
        X, y = check_X_y(X, y, dtype=np.float64)
        check_classification_targets(y)
        teacher = clone(self.teacher) if self.teacher is not None else HistGradientBoostingClassifier()
        if sample_weight is None:
            teacher.fit(X, y)
        else:
            teacher.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(teacher.classes_)
        if len(self.classes_) < 2:
            raise ValueError("DistilledFIGS needs at least two classes in y")
        Xt = X
        if self.synthetic:
            Xt = np.vstack([X, munge(X, passes=int(self.synthetic), random_state=self.random_state)])
        P = np.asarray(teacher.predict_proba(Xt), dtype=np.float64)
        targets = [P[:, 1]] if len(self.classes_) == 2 else [P[:, k] for k in range(len(self.classes_))]
        self.figs_ = [FIGSRegressor(max_rules=int(self.max_rules), random_state=self.random_state).fit(Xt, t)
                      for t in targets]
        self.teacher_ = teacher
        self.n_features_in_ = X.shape[1]
        self.complexity_ = int(sum(m.complexity_ for m in self.figs_))
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was fitted on {self.n_features_in_}")
        S = np.column_stack([np.asarray(m.predict(X), dtype=np.float64) for m in self.figs_])
        S = np.clip(S, 0.0, 1.0)
        if len(self.classes_) == 2:
            return np.column_stack([1.0 - S[:, 0], S[:, 0]])
        tot = S.sum(1, keepdims=True)
        return np.where(tot > 0, S / np.where(tot > 0, tot, 1.0), 1.0 / len(self.classes_))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fidelity(self, X) -> float:
        """Share of rows of `X` where this model predicts the teacher's class."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        return float(np.mean(self.predict(X) == np.asarray(self.teacher_.predict(X))))

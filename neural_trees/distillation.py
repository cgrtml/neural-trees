"""
Distil a fitted classifier (a tree ensemble, typically) into one soft tree,
and report how faithfully the tree reproduces it.

The student learns the teacher's class probabilities, not its labels: each
row is repeated once per class the teacher gives at least `threshold`
probability, weighted by that probability, which is exactly the soft-target
cross-entropy. Measured on 24 datasets with XGBoost, LightGBM and random
forest teachers (`benchmarks/distill.py`): agreement with the teacher 0.934
for this student against 0.901 for CART of depth 6, with about 11 splits
against 38 leaves. The lead comes from splits that use many features; to
limit each split to a few (`split_features`), start the student from CART
(`init="cart"`), which keeps it ahead of CART even at one feature per split.
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

__all__ = ["Distilled", "DistilledForest", "DistilledTree", "distill", "fidelity", "munge"]


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


def munge(X, passes: int = 2, p: float = 0.5, s: float = 5.0, random_state=None) -> np.ndarray:
    """
    Synthetic rows near the rows of `X`, for a teacher to label (MUNGE,
    Bucila, Caruana and Niculescu-Mizil, "Model compression", KDD 2006).

    Each pass makes one new row per row of `X`: feature by feature, with
    probability `p`, the value is replaced by one drawn around the row's
    nearest neighbour's (normal, sd = ``abs(difference) / s``); a feature with at
    most ten distinct values copies the neighbour's value instead. Neighbours
    are found on standardised features, so the scale of `X` does not matter.
    A missing value stays missing unless it is replaced by the neighbour's.

    Returns an array of shape (passes * len(X), n_features).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    if passes < 0 or not 0.0 <= p <= 1.0 or s <= 0:
        raise ValueError(f"need passes >= 0, p in [0, 1], s > 0; got {passes!r}, {p!r}, {s!r}")
    if passes == 0 or len(X) < 2:
        return np.empty((0, X.shape[1]))
    rng = np.random.default_rng(random_state)
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    Z = np.nan_to_num((X - np.nan_to_num(mu)) / np.where(np.nan_to_num(sd) > 0, sd, 1.0))
    nn = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z, return_distance=False)[:, 1]
    discrete = np.array([len(np.unique(c[~np.isnan(c)])) <= 10 for c in X.T])
    N = X[nn]
    spread = np.nan_to_num(np.abs(X - N) / s)
    out = []
    for _ in range(passes):
        drawn = np.where(discrete, N, rng.normal(np.nan_to_num(N), spread))
        drawn = np.where(np.isnan(N), np.nan, drawn)
        out.append(np.where(rng.random(X.shape) < p, drawn, X))
    return np.vstack(out)


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
    probabilities on the same rows, on synthetic rows near them (`synthetic`)
    and on unlabelled rows if given (mixed with the true labels by `alpha`).
    Prediction, `explain` and every export go through the single student
    tree, so the deployed model is one readable tree; the teacher is kept
    only to measure agreement.

    Why: on large tables a single soft tree trained on the labels trails
    gradient boosting, and the probabilities of a boosted model are a
    better training signal than the labels where the tree overfits them.
    On six OpenML tables at 5 000 to 100 000 rows (XGBoost teacher, depth-8
    student; benchmarks/damitma_olcek.py) this raised Higgs by 0.9 to 2.1
    points and jannis at 20 000 rows by 2.7, and did nothing where the tree
    is short of capacity (electricity, covertype); +0.6 points on average
    over 13 cells, not significant. What moves it is more rows for the
    teacher to label. Synthetic ones (`synthetic=2`, the default: two
    MUNGE passes, see `munge`) raised the same tree by 1.5 points over
    the tree on labels and 0.9 over plain distillation, higher in 12 of 13
    cells each (Wilcoxon p=0.0002 and 0.0005; benchmarks/munge.py), and
    cut the mean gap to XGBoost from 3.3 to 1.8 points, at about three
    times the fit time. With the default teacher the gain was smaller
    (+0.5 points, 8 of 12 fits higher, none lower; benchmarks/munge_lib.py),
    with a LightGBM teacher about the same (+0.9 over plain distillation, 9
    of 11 cells higher, none lower; benchmarks/munge_lgbm.py).
    Real unlabelled rows (`X_unlabeled` in `fit`) do a
    little better again. Below about 1 000 rows use
    ``SoftDecisionTree(growth="per_leaf")`` instead: on the 24 datasets of
    benchmarks/rakipler.py it was 1.2 points better there (7 of 8,
    p=0.016; benchmarks/damitma_kucuk.py). The documentation page "Large
    tables" has the full tables.

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
    synthetic : int, default=2
        MUNGE passes over the labelled rows; each adds len(X) synthetic rows
        that the teacher labels for the student. 0 turns it off.
    random_state : int, optional
        Seeds the synthetic rows and the default student.

    Attributes
    ----------
    teacher_, student_ : the fitted models
    classes_, n_features_in_
    """

    def __init__(self, teacher=None, student=None, alpha: float = 0.0, threshold: float = 0.01,
                 synthetic: int = 2, random_state: Optional[int] = None):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.threshold = threshold
        self.synthetic = synthetic
        self.random_state = random_state

    # NaN is accepted (the student handles it); scikit-learn >= 1.6 reads
    # __sklearn_tags__, older versions _more_tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {"allow_nan": True}

    def fit(self, X, y, sample_weight=None, X_unlabeled=None) -> DistilledTree:
        """
        Fit the teacher on (X, y), then the student on the teacher's
        probabilities on X and, if given, on `X_unlabeled`: rows without a
        label, from the same population, that the teacher labels for the
        student. Labels are often scarcer than rows (every customer is in
        the table, few have an outcome), and the teacher can pass on what it
        learned from the labelled rows through the unlabelled ones. With
        twice as many unlabelled rows as labelled ones the depth-8 tree
        gained 1.85 points over the same tree on labels, in 13 of 13 cells,
        and halved its mean gap to the XGBoost teacher (3.3 to 1.45
        points); on Higgs and jannis it reached the teacher. The teacher
        itself sees only (X, y).

        Like `SoftDecisionTree`, this passes scikit-learn's estimator checks
        except `check_sample_weight_equivalence_on_dense_data`, which no
        stochastic mini-batch learner can satisfy.
        """
        teacher, Xt, P, wt, X, y, w = self._teacher_rows(X, y, sample_weight, X_unlabeled)
        student = self._new_student(self.random_state)
        student.fit(*self._student_rows(teacher, Xt, P, wt, X, y, w))
        self.teacher_, self.student_ = teacher, student
        self.classes_ = student.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def _new_student(self, random_state):
        from sklearn.base import clone

        from neural_trees import SoftDecisionTree
        if self.student is not None:
            return clone(self.student)
        return SoftDecisionTree(depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                                random_state=random_state)

    def _teacher_rows(self, X, y, sample_weight, X_unlabeled):
        """Fit the teacher; return it with the transfer rows, their probabilities and weights."""
        from sklearn.base import clone
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.multiclass import check_classification_targets

        from neural_trees._missing import allow_nan

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha!r}")
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {self.threshold!r}")
        if not isinstance(self.synthetic, (int, np.integer)) or isinstance(self.synthetic, bool) \
                or self.synthetic < 0:
            raise ValueError(f"synthetic must be a non-negative int, got {self.synthetic!r}")
        X, y = check_X_y(X, y, **allow_nan())
        check_classification_targets(y)
        teacher = clone(self.teacher) if self.teacher is not None else HistGradientBoostingClassifier()
        if sample_weight is None:
            teacher.fit(X, y)
            w = np.ones(len(y))
        else:
            teacher.fit(X, y, sample_weight=sample_weight)
            w = np.asarray(sample_weight, dtype=np.float64)
        P = np.asarray(teacher.predict_proba(X), dtype=np.float64)
        extra = [munge(X, passes=int(self.synthetic), random_state=self.random_state)]
        if X_unlabeled is not None:
            Xu = check_array(X_unlabeled, **allow_nan())
            if Xu.shape[1] != X.shape[1]:
                raise ValueError(f"X_unlabeled has {Xu.shape[1]} features, X has {X.shape[1]}")
            extra.append(Xu)
        Xe = np.vstack(extra)
        Xt, wt = X, w
        if len(Xe):
            Xt = np.vstack([X, Xe])
            P = np.vstack([P, np.asarray(teacher.predict_proba(Xe), dtype=np.float64)])
            wt = np.concatenate([w, np.full(len(Xe), float(np.mean(w)))])
        return teacher, Xt, P, wt, X, y, w

    def _student_rows(self, teacher, Xt, P, wt, X, y, w):
        """(X, y, sample_weight) for the student: one row per class above `threshold`, plus labels by `alpha`."""
        classes = np.asarray(teacher.classes_)
        rows, cols = np.nonzero(P >= max(self.threshold, 1e-12))
        Xs, ys, ws = [Xt[rows]], [classes[cols]], [(1.0 - self.alpha) * P[rows, cols] * wt[rows]]
        if self.alpha > 0:
            Xs.append(X)
            ys.append(y)
            ws.append(self.alpha * w)
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)

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


class DistilledForest(DistilledTree):
    """
    A few soft trees trained on what a strong model learned: the same
    teacher and transfer rows as `DistilledTree`, then `n_trees` students,
    each on its own bootstrap sample of those rows. The group predicts with
    the sum of the trees' log probabilities.

    Use it when accuracy matters more than having one tree to read. Each
    tree is still a `SoftDecisionTree` and can be explained and exported on
    its own (`estimators_`), but the prediction is the group's.

    Parameters are those of `DistilledTree`, plus

    n_trees : int, default=3
        Number of student trees.

    Attributes
    ----------
    teacher_ : the fitted teacher
    estimators_ : list of the fitted student trees
    classes_, n_features_in_
    """

    def __init__(self, teacher=None, student=None, n_trees: int = 3, alpha: float = 0.0,
                 threshold: float = 0.01, synthetic: int = 2, random_state: Optional[int] = None):
        super().__init__(teacher=teacher, student=student, alpha=alpha, threshold=threshold,
                         synthetic=synthetic, random_state=random_state)
        self.n_trees = n_trees

    def fit(self, X, y, sample_weight=None, X_unlabeled=None) -> DistilledForest:
        """Fit the teacher, then `n_trees` students on bootstrap samples of the transfer rows."""
        if not isinstance(self.n_trees, (int, np.integer)) or isinstance(self.n_trees, bool) \
                or self.n_trees < 1:
            raise ValueError(f"n_trees must be a positive int, got {self.n_trees!r}")
        teacher, Xt, P, wt, X, y, w = self._teacher_rows(X, y, sample_weight, X_unlabeled)
        rng = np.random.default_rng(self.random_state)
        estimators = []
        for k in range(int(self.n_trees)):
            b = rng.integers(0, len(Xt), len(Xt))
            seed = None if self.random_state is None else int(self.random_state) + k
            student = self._new_student(seed)
            if self.student is not None and seed is not None and "random_state" in student.get_params():
                student.set_params(random_state=seed)
            Xs, ys, ws = self._student_rows(teacher, Xt[b], P[b], wt[b], X, y, w)
            estimators.append(student.fit(Xs, ys, sample_weight=ws))
        self.teacher_, self.estimators_ = teacher, estimators
        self.classes_ = estimators[0].classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        with np.errstate(divide="ignore"):
            L = sum(np.log(t.predict_proba(X)) for t in self.estimators_)
        L = L - L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fidelity(self, X) -> float:
        """Share of rows of `X` where the group predicts the teacher's class."""
        check_is_fitted(self)
        return float(np.mean(self.predict(X) == self.teacher_.predict(X)))

    def explain(self, X, **kwargs):
        raise NotImplementedError("a group has no single explanation; explain each tree in estimators_")

    def to_numpy(self, **kwargs):
        """One `NumpySoftTree` per tree; sum their `predict_log_proba` to predict like the group."""
        check_is_fitted(self)
        return [t.to_numpy(**kwargs) for t in self.estimators_]

    def to_hard_tree(self, **kwargs):
        raise NotImplementedError("export each tree in estimators_")

    def to_onnx(self, **kwargs):
        raise NotImplementedError("export each tree in estimators_")

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

from neural_trees._missing import allow_nan

__all__ = ["Distilled", "DistilledForest", "DistilledTree", "ThresholdBasis", "distill", "fidelity", "munge"]


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


def _teacher_splits(teacher, n_features):
    """
    (feature, threshold, right_if_equal, gain) for every split of a fitted
    tree ensemble: XGBoost, LightGBM, scikit-learn histogram gradient
    boosting, gradient boosting, random forest, extra trees or one decision
    tree. `right_if_equal` is True where a row equal to the threshold goes
    right (XGBoost: ``x < t`` goes left), False where it goes left
    (``x <= t``, the others).
    """
    out = []
    est = teacher.steps[-1][1] if hasattr(teacher, "steps") else teacher
    if hasattr(est, "get_booster"):                                    # XGBoost
        # the JSON dump, not trees_to_dataframe(), which needs pandas
        import json
        names = est.get_booster().feature_names
        pos = {n: i for i, n in enumerate(names)} if names else None
        stack = [json.loads(t) for t in est.get_booster().get_dump(dump_format="json", with_stats=True)]
        while stack:
            nd = stack.pop()
            if "split" in nd:
                f = nd["split"]
                out.append((pos[f] if pos else int(str(f)[1:]), float(nd["split_condition"]), True, float(nd["gain"])))
                stack.extend(nd["children"])
        return out
    if hasattr(est, "booster_"):                                       # LightGBM
        stack = [t["tree_structure"] for t in est.booster_.dump_model()["tree_info"]]
        while stack:
            nd = stack.pop()
            if "split_feature" in nd:
                if nd["decision_type"] == "<=":
                    out.append((int(nd["split_feature"]), float(nd["threshold"]), False, float(nd["split_gain"])))
                stack.extend((nd["left_child"], nd["right_child"]))
        return out
    if hasattr(est, "_predictors"):                                    # HistGradientBoosting
        for it in est._predictors:
            for pred in it:
                n = pred.nodes
                for r in n[~n["is_leaf"].astype(bool)]:
                    if not r["is_categorical"]:
                        out.append((int(r["feature_idx"]), float(r["num_threshold"]), False, float(r["gain"])))
        return out
    trees = getattr(est, "estimators_", None)
    if trees is None and hasattr(est, "tree_"):
        trees = [est]
    if trees is not None:                                              # sklearn trees and forests
        for t in np.asarray(trees, dtype=object).ravel():
            tr = t.tree_
            for i in np.where(tr.children_left >= 0)[0]:
                left, right = tr.children_left[i], tr.children_right[i]
                g = (tr.weighted_n_node_samples[i] * tr.impurity[i]
                     - tr.weighted_n_node_samples[left] * tr.impurity[left]
                     - tr.weighted_n_node_samples[right] * tr.impurity[right])
                out.append((int(tr.feature[i]), float(tr.threshold[i]), False, float(g)))
        return out
    raise ValueError(f"thresholds needs a tree-ensemble teacher (XGBoost, LightGBM or a scikit-learn tree "
                     f"model); got {type(est).__name__}")


class ThresholdBasis:
    """
    The inputs a threshold-basis student sees: the original features, then
    one 0/1 input per (feature, threshold) pair among the teacher's `k`
    splits with the largest total gain, 1 where the row goes right at that
    split.

    A depth-8 soft tree learns its own thresholds slowly and coarsely; the
    teacher has already found where each feature matters. With these inputs
    a gate can pick up a boosted model's cut points directly. On the nine
    tables of benchmarks/kanit.py (three seeds, k = 512, the tuned XGBoost
    as teacher; benchmarks/esik.py) one tree was 0.70 points under tuned
    XGBoost at 20 000 rows with 14 to 99 splits, against XGBoost's 4 031 to
    115 351; with 500 to 5 000 rows (rivals re-tuned at each size) the
    three-tree group was level with tuned XGBoost and LightGBM, and 0.40
    points above XGBoost at 500 rows (18 of 27 cells higher, Wilcoxon
    p = 0.012). These numbers were measured with the development script
    this class was taken from; the library code gave the same splits and
    accuracy within 0.1 point on the cells checked.
    """

    def __init__(self, teacher, n_features: int, k: int = 512):
        splits = _teacher_splits(teacher, n_features)
        gain = {}
        for j, t, right_eq, g in splits:
            if 0 <= j < n_features and np.isfinite(t):
                gain[(j, t, right_eq)] = gain.get((j, t, right_eq), 0.0) + g
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:int(k)]
        self.n_features = int(n_features)
        self.pairs = [key for key, _ in top]

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.pairs:
            return X.astype(np.float32)
        j = np.array([p[0] for p in self.pairs])
        t = np.array([p[1] for p in self.pairs])
        eq = np.array([p[2] for p in self.pairs])
        V = X[:, j]
        with np.errstate(invalid="ignore"):
            B = np.where(eq, V >= t, V > t).astype(np.float32)
        B[np.isnan(V)] = np.nan                       # missing stays missing (the student handles NaN)
        return np.hstack([X.astype(np.float32), B])

    def feature_names(self, names=None) -> list:
        names = list(names) if names is not None else [f"x{j}" for j in range(self.n_features)]
        return names + [f"{names[j]} {'>=' if eq else '>'} {t:.6g}" for j, t, eq in self.pairs]


def _brier(model, X, y, w):
    """Negative weighted Brier score of a fitted classifier (higher is better)."""
    P = model.predict_proba(X)
    Y = (np.asarray(y)[:, None] == model.classes_[None, :]).astype(float)
    return -float((((P - Y) ** 2).sum(1) * w).sum() / w.sum())


def _one_se_cart(make_tree, X, Y, w, score, folds):
    """
    CART pruned by Breiman's one-standard-error rule: among up to 12 strengths
    along the cost-complexity path, the largest whose mean fold score is within
    one standard error (over the folds) of the best.
    """
    path = make_tree(0.0).cost_complexity_pruning_path(X, Y, sample_weight=w)
    alphas = np.unique(np.clip(path.ccp_alphas[:-1], 0.0, None))
    if len(alphas) > 12:
        alphas = np.unique(np.quantile(alphas, np.linspace(0, 1, 12)))
    if len(alphas) <= 1:
        return make_tree(0.0).fit(X, Y, sample_weight=w)
    S = np.array([[score(make_tree(a).fit(X[tr], Y[tr], sample_weight=w[tr]), X[te], Y[te], w[te])
                   for tr, te in folds] for a in alphas])
    mean, se = S.mean(1), S.std(1, ddof=1) / np.sqrt(S.shape[1])
    best = int(mean.argmax())
    alpha = float(max(a for a, m in zip(alphas, mean) if m >= mean[best] - se[best]))
    return make_tree(alpha).fit(X, Y, sample_weight=w)


def _pruned_student(Xs, ys, ws, random_state, rows: int = 40_000):
    """
    The default student for threshold inputs: CART of depth 8 pruned on at
    most `rows` transfer rows (pruning on every row costs more than the fit),
    copied into a soft tree and fine-tuned on all rows. The pruning strength
    is chosen by 3-fold cross-validation with folds that keep the class copies
    of a row together, scored by the Brier score, with the one-standard-error
    rule. Weighted accuracy is flat on imbalanced transfer rows (a teacher
    whose argmax is one split scores the same for every larger tree), and
    taking the best mean accuracy grew trees 1.4 to 7 times larger at the same
    test accuracy on the nine tables of `benchmarks/kanit.py` at 500 to 5 000 rows.
    """
    from sklearn.tree import DecisionTreeClassifier

    from neural_trees import SoftDecisionTree
    from neural_trees.decision_trees.soft_decision_tree import _pruning_folds
    kw = dict(learn_temperature=True, max_epochs=60, batch_size=256, random_state=random_state)
    seed = random_state or 0
    sub = np.random.default_rng(seed).choice(len(ys), min(rows, len(ys)), replace=False)
    _, yi = np.unique(ys, return_inverse=True)
    folds = _pruning_folds(Xs[sub], yi[sub], seed)
    cart = _one_se_cart(lambda a: DecisionTreeClassifier(max_depth=8, ccp_alpha=a, random_state=random_state),
                        np.nan_to_num(Xs[sub]), ys[sub], ws[sub], _brier, folds)
    return SoftDecisionTree.from_hard_tree(cart, X=Xs, sharpness=10.0, **kw).fit(Xs, ys, sample_weight=ws)


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
    teacher to label. Synthetic ones (MUNGE passes, see `munge`) raised
    the same tree, with two passes, by 1.5 points over the tree on labels
    and 0.9 over plain distillation, higher in 12 of 13 cells each
    (Wilcoxon p=0.0002 and 0.0005; benchmarks/munge.py), and cut the mean
    gap to XGBoost from 3.3 to 1.8 points. Ten passes (`synthetic=10`,
    the default) added 0.5 points over two on the 11 cells up to 20 000
    rows (7 higher, none lower, p=0.001; benchmarks/munge.py with
    MUNGE_PASSES=10), leaving the tree 1.3 points behind XGBoost with 300
    trees and level with XGBoost with 30 trees, about ten times its size.
    With the default teacher ten passes gave +0.3 over two (8 of 12 fits
    higher, none lower, p=0.009; benchmarks/munge_lib.py). Thirty passes
    were no better than ten. The cost is fit time: ten passes fit in four
    to six times the time of two (about 15 minutes at 20 000 rows on one
    CPU thread); `synthetic=2` is the faster setting. With a LightGBM
    teacher two passes gained about as much as with XGBoost (+0.9 over
    plain distillation, 9 of 11 cells higher, none lower;
    benchmarks/munge_lgbm.py).
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
    synthetic : int, default=10
        MUNGE passes over the labelled rows; each adds len(X) synthetic rows
        that the teacher labels for the student. 0 turns it off; fewer
        passes fit faster and score a little lower.
    thresholds : int, default=0
        Give the student, next to the features, one 0/1 input for each of the
        teacher's `thresholds` highest-gain (feature, threshold) splits (see
        `ThresholdBasis`; the teacher must be a tree ensemble). 512 is the
        measured setting. Prediction, `explain` and `fidelity` take the
        original features; the exports (`to_numpy`, `to_hard_tree`,
        `to_onnx`) take ``student_inputs(X)``.
    random_state : int, optional
        Seeds the synthetic rows and the default student.

    Attributes
    ----------
    teacher_, student_ : the fitted models
    basis_ : ThresholdBasis or None
    classes_, n_features_in_
    """

    def __init__(self, teacher=None, student=None, alpha: float = 0.0, threshold: float = 0.01,
                 synthetic: int = 10, thresholds: int = 0, random_state: Optional[int] = None):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.threshold = threshold
        self.synthetic = synthetic
        self.thresholds = thresholds
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
        Xt, Xb = self._basis_rows(teacher, Xt, X)
        student = self._fit_student(self.random_state, *self._student_rows(teacher, Xt, P, wt, Xb, y, w))
        self.teacher_, self.student_ = teacher, student
        self.classes_ = student.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def _basis_rows(self, teacher, Xt, X):
        """Set `basis_` and return the transfer rows and labelled rows as the student sees them."""
        if not isinstance(self.thresholds, (int, np.integer)) or isinstance(self.thresholds, bool) \
                or self.thresholds < 0:
            raise ValueError(f"thresholds must be a non-negative int, got {self.thresholds!r}")
        self.basis_ = ThresholdBasis(teacher, X.shape[1], int(self.thresholds)) if self.thresholds else None
        if self.basis_ is None:
            return Xt, X
        return self.basis_.transform(Xt), self.basis_.transform(X)

    def student_inputs(self, X):
        """The student's inputs for rows of `X`: `X` itself, or with the threshold inputs appended."""
        check_is_fitted(self)
        if self.basis_ is None:
            return X
        return self.basis_.transform(check_array(X, dtype=np.float64, **allow_nan()))

    def _fit_student(self, random_state, Xs, ys, ws):
        if self.student is None and self.thresholds:
            return _pruned_student(Xs, ys, ws, random_state)
        student = self._new_student(random_state)
        if self.student is not None and random_state is not None and "random_state" in student.get_params():
            student.set_params(random_state=random_state)
        return student.fit(Xs, ys, sample_weight=ws)

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
        return self.student_.predict_proba(self.student_inputs(X))

    def predict(self, X):
        check_is_fitted(self)
        return self.student_.predict(self.student_inputs(X))

    def fidelity(self, X) -> float:
        """Share of rows of `X` where the student predicts the teacher's class."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, **allow_nan())
        return float(np.mean(np.asarray(self.predict(X)) == np.asarray(self.teacher_.predict(X))))

    def explain(self, X, **kwargs):
        """
        The student tree's explanation; see `SoftDecisionTree.explain`. With
        `thresholds`, `X` is in the original features; `feature_names` and
        `feature_bounds` may be given for them and are extended to the
        threshold inputs (named like ``"bill_sep > 1500"``, bounds 0 and 1).
        """
        check_is_fitted(self)
        if self.basis_ is None:
            return self.student_.explain(X, **kwargs)
        Z = self.student_inputs(np.asarray(X, dtype=np.float64).reshape(-1, self.n_features_in_))
        kwargs["feature_names"] = self.basis_.feature_names(kwargs.get("feature_names"))
        if kwargs.get("feature_bounds") is not None:
            lo, hi = (np.asarray(b, dtype=np.float64) for b in kwargs["feature_bounds"])
            m = len(self.basis_.pairs)
            kwargs["feature_bounds"] = (np.concatenate([lo, np.zeros(m)]), np.concatenate([hi, np.ones(m)]))
        return self.student_.explain(Z[0] if np.ndim(X) == 1 else Z, **kwargs)

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

    Measured with an XGBoost teacher (300 trees of depth 6), ten MUNGE
    passes and depth-8 students, on the 11 cells of six OpenML tables up
    to 20 000 rows (benchmarks/grup.py with MUNGE_PASSES=10): three trees
    were 0.38 points above one distilled tree (10 cells higher, none lower,
    p=0.001) and 0.9 below the teacher (p=0.12). With about 450 splits in
    all, the group was 1.2 points above XGBoost with 10 trees (521 splits;
    10 of 11 cells, p=0.003) and level with XGBoost with 100 trees (4 159
    splits; -0.4, p=0.58). Five trees were no better than three, and
    depth-6 students no better than one depth-8 tree. It fits in about
    three times the time of `DistilledTree`.

    Parameters are those of `DistilledTree`, plus

    n_trees : int, default=3
        Number of student trees.
    combine : {"log", "mean"}, default="log"
        "log" predicts with the sum of the trees' log probabilities (the
        setting measured above); "mean" averages their probabilities. The
        sum is overconfident: on the credit-default case
        (cases/credit-default/esik.py; three trees, threshold inputs, 1 000
        to 24 000 rows, 3 seeds) its calibration slope was 0.35 to 0.38 and ECE
        0.13 at every size, the mean's 1.05 to 1.15 and 0.016 to 0.021, at
        the same AUC. Use "mean" when the probability itself is used, as a
        probability of default is.

    Attributes
    ----------
    teacher_ : the fitted teacher
    estimators_ : list of the fitted student trees
    classes_, n_features_in_
    """

    def __init__(self, teacher=None, student=None, n_trees: int = 3, alpha: float = 0.0,
                 threshold: float = 0.01, synthetic: int = 10, thresholds: int = 0,
                 combine: str = "log", random_state: Optional[int] = None):
        super().__init__(teacher=teacher, student=student, alpha=alpha, threshold=threshold,
                         synthetic=synthetic, thresholds=thresholds, random_state=random_state)
        self.n_trees = n_trees
        self.combine = combine

    def fit(self, X, y, sample_weight=None, X_unlabeled=None) -> DistilledForest:
        """Fit the teacher, then `n_trees` students on bootstrap samples of the transfer rows."""
        if not isinstance(self.n_trees, (int, np.integer)) or isinstance(self.n_trees, bool) \
                or self.n_trees < 1:
            raise ValueError(f"n_trees must be a positive int, got {self.n_trees!r}")
        if self.combine not in ("log", "mean"):
            raise ValueError(f'combine must be "log" or "mean", got {self.combine!r}')
        teacher, Xt, P, wt, X, y, w = self._teacher_rows(X, y, sample_weight, X_unlabeled)
        Xt, Xb = self._basis_rows(teacher, Xt, X)
        rng = np.random.default_rng(self.random_state)
        estimators = []
        for k in range(int(self.n_trees)):
            b = rng.integers(0, len(Xt), len(Xt))
            seed = None if self.random_state is None else int(self.random_state) + k
            estimators.append(self._fit_student(seed, *self._student_rows(teacher, Xt[b], P[b], wt[b], Xb, y, w)))
        self.teacher_, self.estimators_ = teacher, estimators
        self.classes_ = estimators[0].classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        Z = self.student_inputs(X)
        if self.combine == "mean":
            return np.mean([t.predict_proba(Z) for t in self.estimators_], axis=0)
        with np.errstate(divide="ignore"):
            L = sum(np.log(t.predict_proba(Z)) for t in self.estimators_)
        L = L - L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fidelity(self, X) -> float:
        """Share of rows of `X` where the group predicts the teacher's class."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, **allow_nan())
        return float(np.mean(self.predict(X) == self.teacher_.predict(X)))

    def explain(self, X, **kwargs):
        raise NotImplementedError("a group has no single explanation; explain each tree in estimators_")

    def to_numpy(self, **kwargs):
        """One `NumpySoftTree` per tree, on ``student_inputs(X)``; sum their `predict_log_proba` (combine="log")
        or average their `predict_proba` (combine="mean") to predict like the group."""
        check_is_fitted(self)
        return [t.to_numpy(**kwargs) for t in self.estimators_]

    def to_hard_tree(self, **kwargs):
        raise NotImplementedError("export each tree in estimators_")

    def to_onnx(self, **kwargs):
        raise NotImplementedError("export each tree in estimators_")

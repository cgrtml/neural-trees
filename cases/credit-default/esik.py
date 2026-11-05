"""
The credit-default case with threshold inputs, at small and full training sizes, with
out-of-time, stability and calibration checks.

Data and split as run.py: OpenML 42477, sex, marriage and age kept out of the inputs, the same
6 000 test rows. Training sizes N in {1000, 2000, 5000, 24000}: a stratified subsample of the
24 000 training rows, drawn per seed (so the seeds also measure resampling stability).

Models: tuned XGBoost (the bank model; 20 configurations, validation AUC, early stopping, as
run.py), tuned CatBoost (20 configurations, same rule), logistic regression, tuned CART, and
`DistilledTree(thresholds=512)` and `DistilledForest(thresholds=512)` with that XGBoost as
teacher, behind a StandardScaler (Thr_tek, Thr_grup; Thr_grup_ort is the same group with
combine="mean").

Out of time. The data has no date column; each row is one card holder with six months of
history (April to September 2005). The out-of-time task uses that history: inputs are the
credit limit, education and four months of repayment status, bill and payment; the target is
"payment delayed one month or more" in the following month. Fit on window A (April to July ->
August) of the training holders; score on window A of the test holders (in time) and on window
B (May to August -> September) of the same test holders (one month later, out of time).

Per model and cell: accuracy at 0.5, AUC, KS, Brier, ECE (10 equal-count bins), calibration
intercept and slope (logistic recalibration of the logit), decline rate, agreement with the
bank model, splits. Stability across seeds: pairwise decision disagreement on the test rows,
mean absolute PD difference, Spearman rank correlation of the scores.

    python cases/credit-default/esik.py kos|ozet
    env: NS=1000,2000,5000,24000  SEEDS=0,1,2  JOBS=1  OOT=1 (also run the out-of-time cells)
"""

import itertools
import json
import os
import pathlib
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "_esik_cache"
OUT.mkdir(exist_ok=True)
SEEDS = tuple(int(v) for v in os.environ.get("SEEDS", "0,1,2").split(","))
THR = 0.5
TRIALS = int(os.environ.get("TRIALS", "20"))
K = 512
MONTHS = ("sep", "aug", "jul", "jun", "may", "apr")
NAMES = (
    ["credit_limit", "sex", "education", "marriage", "age"]
    + [f"repay_status_{m}" for m in MONTHS]
    + [f"bill_{m}" for m in MONTHS]
    + [f"paid_{m}" for m in MONTHS]
)
PROTECTED = ("sex", "marriage", "age")


def load():
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    X = d.data.copy()
    X.columns = NAMES
    return X, np.asarray(d.target).astype(int)


def base_split(y):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        np.arange(len(y)), test_size=6000, random_state=0, stratify=y
    )


def subsample(tr, y, n, seed):
    if n >= len(tr):
        return tr
    from sklearn.model_selection import train_test_split

    keep, _ = train_test_split(
        tr, train_size=n, random_state=100 + seed, stratify=y[tr]
    )
    return np.sort(keep)


def window(X, last):
    """Inputs for the out-of-time task: four months ending at `last`, target = delay in the month after."""
    order = list(reversed(MONTHS))  # apr .. sep
    i = order.index(last)
    ms = order[i - 3 : i + 1]
    cols = ["credit_limit", "education"] + [
        f"{k}_{m}" for k in ("repay_status", "bill", "paid") for m in ms
    ]
    F = X[cols].to_numpy(np.float32)
    y = (X[f"repay_status_{order[i + 1]}"].to_numpy() >= 1).astype(int)
    return F, y


# ── models ─────────────────────────────────────────────────────────────
def _val(Xtr, ytr, seed):
    from sklearn.model_selection import train_test_split

    return train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)


def tune_xgb(Xtr, ytr, seed):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    Xa, Xv, ya, yv = _val(Xtr, ytr, seed)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(TRIALS):
        p = {
            "learning_rate": float(10 ** rng.uniform(-2, -0.5)),
            "max_depth": int(rng.integers(3, 11)),
            "min_child_weight": float(10 ** rng.uniform(-1, 1.5)),
            "subsample": float(rng.uniform(0.5, 1)),
            "colsample_bytree": float(rng.uniform(0.5, 1)),
            "reg_lambda": float(10 ** rng.uniform(-2, 1.5)),
        }
        m = xgb.XGBClassifier(
            n_estimators=2000,
            early_stopping_rounds=50,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            **p,
        ).fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, {**p, "n_estimators": int(m.best_iteration) + 1})
    return xgb.XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, **best[1]).fit(
        Xtr, ytr
    ), best[1]


def tune_cat(Xtr, ytr, seed):
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    Xa, Xv, ya, yv = _val(Xtr, ytr, seed)
    rng = np.random.default_rng(seed + 7)
    best = None
    for _ in range(TRIALS):
        p = {
            "learning_rate": float(10 ** rng.uniform(-2, -0.5)),
            "depth": int(rng.integers(4, 11)),
            "l2_leaf_reg": float(10 ** rng.uniform(0, 1)),
            "random_strength": float(rng.uniform(0, 2)),
        }
        m = CatBoostClassifier(
            iterations=2000,
            od_type="Iter",
            od_wait=50,
            random_seed=seed,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
            **p,
        ).fit(Xa, ya, eval_set=(Xv, yv))
        auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, {**p, "iterations": int(m.get_best_iteration()) + 1})
    return CatBoostClassifier(
        random_seed=seed,
        thread_count=1,
        verbose=0,
        allow_writing_files=False,
        **best[1],
    ).fit(Xtr, ytr), best[1]


def tune_cart(Xtr, ytr, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    Xa, Xv, ya, yv = _val(Xtr, ytr, seed)
    best = None
    for depth in range(2, 13):
        for leaf in (1, 10, 50, 100, 200):
            m = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf, random_state=seed
            ).fit(Xa, ya)
            auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, {"max_depth": depth, "min_samples_leaf": leaf})
    return DecisionTreeClassifier(random_state=seed, **best[1]).fit(Xtr, ytr)


class _Mean:
    """The fitted group of a pipeline, predicting with combine="mean"."""

    def __init__(self, pipe):
        self.pipe = pipe

    def predict_proba(self, X):
        g = self.pipe.steps[-1][1]
        Z = g.student_inputs(self.pipe[:-1].transform(X))
        return np.mean([t.predict_proba(Z) for t in g.estimators_], axis=0)


# ── metrics ────────────────────────────────────────────────────────────
def metrics(y, p):
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    bins = np.array_split(np.argsort(p), 10)
    ece = float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(p))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    d = p > THR
    return {
        "accuracy": float((d == (y == 1)).mean()),
        "auc": float(roc_auc_score(y, p)),
        "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece,
        "cal_intercept": float(lr.intercept_[0]),
        "cal_slope": float(lr.coef_[0, 0]),
        "decline_rate": float(d.mean()),
        "mean_pd": float(p.mean()),
        "base_rate": float(y.mean()),
    }


def fit_all(Xtr, ytr, seed):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(1)
    t0 = time.time()
    xg, xp = tune_xgb(Xtr, ytr, seed)
    ms = {"XGBoost": (xg, time.time() - t0)}
    t0 = time.time()
    ms["CatBoost"] = (tune_cat(Xtr, ytr, seed)[0], time.time() - t0)
    t0 = time.time()
    ms["Logistic"] = (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
            Xtr, ytr
        ),
        time.time() - t0,
    )
    t0 = time.time()
    ms["CART"] = (tune_cart(Xtr, ytr, seed), time.time() - t0)
    import xgboost as xgb

    from neural_trees import DistilledForest, DistilledTree

    T = xgb.XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, **xp)
    t0 = time.time()
    ms["Thr_tek"] = (
        make_pipeline(
            StandardScaler(), DistilledTree(teacher=T, thresholds=K, random_state=seed)
        ).fit(Xtr, ytr),
        time.time() - t0,
    )
    t0 = time.time()
    ms["Thr_grup"] = (
        make_pipeline(
            StandardScaler(),
            DistilledForest(teacher=T, thresholds=K, n_trees=3, random_state=seed),
        ).fit(Xtr, ytr),
        time.time() - t0,
    )
    ms["Thr_grup_ort"] = (_Mean(ms["Thr_grup"][0]), ms["Thr_grup"][1])
    return ms, xp


def size(nm, m):
    if nm == "XGBoost":
        df = m.get_booster().trees_to_dataframe()
        return int((df.Feature != "Leaf").sum())
    if nm == "CatBoost":
        return int(m.tree_count_ * (2 ** m.get_all_params()["depth"] - 1))
    if nm == "CART":
        return int(m.tree_.node_count - m.tree_.n_leaves)
    if nm.startswith("Thr"):
        g = (m.pipe if isinstance(m, _Mean) else m).steps[-1][1]
        trees = g.estimators_ if hasattr(g, "estimators_") else [g.student_]
        return int(sum(int(t.model_.is_split.sum()) for t in trees))
    return 0


def cell(kind, n, seed):
    o = OUT / f"{kind}__{n}__{seed}.json"
    if o.exists():
        return json.loads(o.read_text())
    X, y = load()
    tr, te = base_split(y)
    tr = subsample(tr, y, n, seed)
    if kind == "pd":
        cols = [c for c in X.columns if c not in PROTECTED]
        F = X[cols].to_numpy(np.float32)
        sets = {"test": (F[te], y[te])}
        Xtr, ytr = F[tr], y[tr]
    else:  # oot
        FA, yA = window(X, "jul")
        FB, yB = window(X, "aug")
        sets = {"in_time": (FA[te], yA[te]), "out_of_time": (FB[te], yB[te])}
        Xtr, ytr = FA[tr], yA[tr]
    ms, xp = fit_all(Xtr, ytr, seed)
    r = {
        "kind": kind,
        "n": int(len(tr)),
        "seed": seed,
        "train_rate": float(ytr.mean()),
        "xgb_params": xp,
        "models": {},
    }
    for s, (Xs, ys) in sets.items():
        bank = ms["XGBoost"][0].predict_proba(Xs)[:, 1] > THR
        for nm, (m, sec) in ms.items():
            p = m.predict_proba(Xs)[:, 1]
            e = r["models"].setdefault(
                nm, {"splits": size(nm, m), "fit_seconds": round(sec, 1)}
            )
            e[s] = {
                **metrics(ys, p),
                "same_decision_as_xgb": float(((p > THR) == bank).mean()),
            }
            e[s + "_proba"] = np.round(p, 6).tolist()
    o.write_text(json.dumps(r))
    print(
        kind,
        n,
        seed,
        {k: round(v[list(sets)[0]]["auc"], 4) for k, v in r["models"].items()},
        flush=True,
    )
    return r


def ozet():
    from scipy.stats import spearmanr

    rows = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("ozet"):
            continue
        r = json.loads(f.read_text())
        rows.setdefault((r["kind"], r["n"]), []).append(r)
    summ = []
    for (kind, n), rs in sorted(rows.items()):
        sets = [
            k
            for k in rs[0]["models"]["XGBoost"]
            if isinstance(rs[0]["models"]["XGBoost"][k], dict)
        ]
        for nm in rs[0]["models"]:
            line = {
                "kind": kind,
                "n": n,
                "model": nm,
                "seeds": len(rs),
                "splits": float(np.mean([r["models"][nm]["splits"] for r in rs])),
            }
            for s in sets:
                for k in (
                    "accuracy",
                    "auc",
                    "ks",
                    "brier",
                    "ece",
                    "cal_intercept",
                    "cal_slope",
                    "same_decision_as_xgb",
                ):
                    line[f"{s}.{k}"] = float(
                        np.mean([r["models"][nm][s][k] for r in rs])
                    )
                P = [np.asarray(r["models"][nm][s + "_proba"]) for r in rs]
                pr = list(itertools.combinations(range(len(P)), 2))
                if pr:
                    line[f"{s}.stab_disagree"] = float(
                        np.mean([((P[a] > THR) != (P[b] > THR)).mean() for a, b in pr])
                    )
                    line[f"{s}.stab_mad_pd"] = float(
                        np.mean([np.abs(P[a] - P[b]).mean() for a, b in pr])
                    )
                    line[f"{s}.stab_spearman"] = float(
                        np.mean([spearmanr(P[a], P[b])[0] for a, b in pr])
                    )
            summ.append(line)
    (OUT / "ozet.json").write_text(json.dumps(summ, indent=1))
    for r in summ:
        s = "test" if r["kind"] == "pd" else "out_of_time"
        extra = f" in-time auc {r['in_time.auc']:.4f}" if r["kind"] == "oot" else ""
        print(
            f"{r['kind']:3s} {r['n']:5d} {r['model']:12s} auc {r[s + '.auc']:.4f} ks {r[s + '.ks']:.3f} "
            f"acc {r[s + '.accuracy']:.4f} brier {r[s + '.brier']:.4f} ece {r[s + '.ece']:.4f} "
            f"slope {r[s + '.cal_slope']:.2f} flip {r.get(s + '.stab_disagree', float('nan')):.4f} "
            f"rho {r.get(s + '.stab_spearman', float('nan')):.3f} splits {r['splits']:.0f}{extra}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "ozet":
        ozet()
    else:
        from joblib import Parallel, delayed

        ns = [int(v) for v in os.environ.get("NS", "1000,2000,5000,24000").split(",")]
        jobs = [("pd", n, s) for n in ns for s in SEEDS]
        if os.environ.get("OOT"):
            jobs += [("oot", n, s) for n in ns for s in SEEDS]
        Parallel(n_jobs=int(os.environ.get("JOBS", "1")))(
            delayed(cell)(*j) for j in jobs
        )

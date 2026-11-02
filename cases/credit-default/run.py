"""
Credit card default case study: a tuned XGBoost model, distilled into one
readable soft tree, with what a model validator would ask of the pair.

Data: "default of credit card clients" (Yeh and Lien, 2009; UCI, OpenML id
42477, CC0): 30 000 card holders of a bank in Taiwan, 2005, 23 attributes,
6 636 defaults the following month. Sex, marital status and age are kept out
of the model inputs, as a US lender would keep them out (ECOA, Regulation B),
and are used only to check the decisions by group.

The bank's model is XGBoost tuned on a validation split. The challenger is
`DistilledTree` with that XGBoost as teacher (the deployed model is one soft
tree), next to `DistilledForest` (three trees), a logistic regression (the
scorecard baseline) and a tuned CART. Every model sees the same 24 000
training rows and is scored on the same 6 000 test rows; three seeds.

What is written to results.json: performance (accuracy, AUC, KS, Brier) and
size, agreement with the bank's model, three declined or approved applicants
explained from the tree's own path with principal reasons and reachable
single changes that flip the decision, decisions by sex, marital status and
age, stability across seeds, and the exported model files.

Run:  python cases/credit-default/run.py      (one to two hours on a laptop)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SMOKE = bool(os.environ.get("SMOKE"))   # a quick end-to-end check: 2 000 rows, fewer trials and passes
CACHE = HERE / ("_smoke" if SMOKE else "_cache")
OUT = CACHE if SMOKE else HERE           # where results.json and the model files go
SEEDS = (0, 1, 2)
THRESHOLD = 0.5                    # decline when P(default) exceeds this
TRIALS = 2 if SMOKE else 20
FAST = {"synthetic": 1} if SMOKE else {}

# UCI column meanings (Yeh and Lien, 2009). Repayment status: -1 paid duly,
# 1 to 9 months of delay; -2 and 0 occur in the data and are not documented
# by the source (commonly read as "no use" and "revolving credit").
MONTHS = ("sep", "aug", "jul", "jun", "may", "apr")
NAMES = (["credit_limit", "sex", "education", "marriage", "age"]
         + [f"repay_status_{m}" for m in MONTHS]
         + [f"bill_{m}" for m in MONTHS]
         + [f"paid_{m}" for m in MONTHS])
PROTECTED = ("sex", "marriage", "age")


def load():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    X = d.data.copy()
    X.columns = NAMES
    y = np.asarray(d.target).astype(int)
    return X, y


def split(X, y):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=6000, random_state=0, stratify=y)
    return (tr[:2000], te) if SMOKE else (tr, te)


def inputs(X):
    return [c for c in X.columns if c not in PROTECTED]


# ── models ─────────────────────────────────────────────────────────────
def tune_xgboost(Xtr, ytr, seed):
    """20 random configurations scored on a 20% validation split with early stopping; returns params and trees."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(TRIALS):
        p = {"learning_rate": float(10 ** rng.uniform(-2, -0.5)), "max_depth": int(rng.integers(3, 11)),
             "min_child_weight": float(10 ** rng.uniform(-1, 1.5)), "subsample": float(rng.uniform(0.5, 1)),
             "colsample_bytree": float(rng.uniform(0.5, 1)), "reg_lambda": float(10 ** rng.uniform(-2, 1.5))}
        m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, eval_metric="logloss",
                              random_state=seed, n_jobs=1, verbosity=0, **p).fit(Xa, ya, eval_set=[(Xv, yv)],
                                                                                 verbose=False)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, p, int(m.best_iteration) + 1)
    return {**best[1], "n_estimators": best[2]}, float(best[0])


def xgboost(params, seed):
    import xgboost as xgb
    return xgb.XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, **params)


def tune_cart(Xtr, ytr, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    best = None
    for depth in range(2, 13):
        for leaf in (1, 10, 50, 100, 200):
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed).fit(Xa, ya)
            auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, {"max_depth": depth, "min_samples_leaf": leaf})
    return DecisionTreeClassifier(random_state=seed, **best[1])


def size(name, model):
    """Split nodes over all trees of the fitted model (the scaler of a pipeline is not counted)."""
    m = model.steps[-1][1] if hasattr(model, "steps") else model
    if name == "XGBoost (bank model)":
        df = m.get_booster().trees_to_dataframe()
        return int((df["Feature"] != "Leaf").sum())
    if name == "CART":
        return int(m.tree_.node_count - m.tree_.n_leaves)
    if name == "Logistic regression":
        return 0
    trees = m.estimators_ if hasattr(m, "estimators_") else [m.student_]
    return int(sum(int(np.asarray(t.to_numpy().is_split_).sum()) for t in trees))


def ks(y, p):
    from scipy.stats import ks_2samp
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def metrics(y, p):
    from sklearn.metrics import brier_score_loss, roc_auc_score
    decline = p > THRESHOLD
    return {"accuracy": float((decline == (y == 1)).mean()), "auc": float(roc_auc_score(y, p)),
            "ks": ks(y, p), "brier": float(brier_score_loss(y, p)), "decline_rate": float(decline.mean())}


def one_seed(seed):
    """Fit every model for one seed; cache the scores and test probabilities."""
    path = CACHE / f"seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    import torch
    torch.set_num_threads(1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from neural_trees import DistilledForest, DistilledTree
    X, y = load()
    tr, te = split(X, y)
    cols = inputs(X)
    Xtr, Xte = X.iloc[tr][cols].to_numpy(np.float32), X.iloc[te][cols].to_numpy(np.float32)
    ytr, yte = y[tr], y[te]
    params, val_auc = tune_xgboost(Xtr, ytr, seed)
    models = {
        "XGBoost (bank model)": xgboost(params, seed),
        "DistilledTree": make_pipeline(StandardScaler(), DistilledTree(teacher=xgboost(params, seed), random_state=seed, **FAST)),
        "DistilledForest": make_pipeline(StandardScaler(),
                                         DistilledForest(teacher=xgboost(params, seed), n_trees=3,
                                                         random_state=seed, **FAST)),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "CART": tune_cart(Xtr, ytr, seed),
    }
    out = {"seed": seed, "xgboost_params": params, "xgboost_val_auc": val_auc, "models": {}, "proba": {}}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_s = time.time() - t0
        p = m.predict_proba(Xte)[:, 1]
        t0 = time.time()
        for row in Xte[:200]:
            m.predict_proba(row[None])
        out["models"][name] = {**metrics(yte, p), "splits": size(name, m), "fit_seconds": fit_s,
                               "ms_per_row": (time.time() - t0) / 200 * 1e3}
        out["proba"][name] = p.tolist()
        print(f"seed {seed} {name:22s} auc {out['models'][name]['auc']:.4f} acc {out['models'][name]['accuracy']:.4f} "
              f"splits {out['models'][name]['splits']} {fit_s:.0f}s", flush=True)
    bank = np.asarray(out["proba"]["XGBoost (bank model)"]) > THRESHOLD
    for name in models:
        if name != "XGBoost (bank model)":
            out["models"][name]["same_decision_as_bank_model"] = float(
                ((np.asarray(out["proba"][name]) > THRESHOLD) == bank).mean())
    if seed == 0:                                   # the tree a reviewer reads, kept for sections below
        import pickle
        (CACHE / "tree_seed0.pkl").write_bytes(pickle.dumps(models["DistilledTree"]))
    path.write_text(json.dumps(out))
    return out


# ── explanations ───────────────────────────────────────────────────────
def principal_reasons(model, row, medians, cols, k=4):
    """
    Reasons in the sense of an adverse action notice, read from the deployed
    model itself: each input is set alone to its training median and the
    model re-scored; the inputs whose neutral value lowers P(default) most
    are the principal reasons for the risk the model sees.
    """
    base = float(model.predict_proba(row[None])[0, 1])
    trial = np.repeat(row[None], len(cols), axis=0)
    for j in range(len(cols)):
        trial[j, j] = medians[j]
    p = model.predict_proba(trial)[:, 1]
    order = np.argsort(p - base)
    return [{"input": cols[j], "value": float(row[j]), "median": float(medians[j]),
             "p_default_if_median": float(p[j]), "drop": float(base - p[j])}
            for j in order[:k] if base - p[j] > 0.005]


def reachable_flips(model, row, Xtr, cols, declined, k=5):
    """Single inputs moved to a training decile that flip the decision, smallest move (in sd) first."""
    trials = []
    for j, c in enumerate(cols):
        for q in np.unique(np.quantile(Xtr[:, j], np.linspace(0.1, 0.9, 9))):
            if q != row[j]:
                trials.append((j, float(q), abs(q - row[j]) / (Xtr[:, j].std() + 1e-9)))
    batch = np.repeat(row[None], len(trials), axis=0)
    for i, (j, q, _) in enumerate(trials):
        batch[i, j] = q
    p = model.predict_proba(batch)[:, 1]
    flips = [{"input": cols[j], "from": float(row[j]), "to": q, "p_default": float(pb), "size_sd": float(s)}
             for (j, q, s), pb in zip(trials, p) if (pb > THRESHOLD) != declined]
    flips.sort(key=lambda f: f["size_sd"])
    return flips[:k], len(trials)


def explanations(seed0):
    import pickle
    X, y = load()
    tr, te = split(X, y)
    cols = inputs(X)
    Xtr, Xte = X.iloc[tr][cols].to_numpy(np.float32), X.iloc[te][cols].to_numpy(np.float32)
    model = pickle.loads((CACHE / "tree_seed0.pkl").read_bytes())
    scaler, dist = model.steps[0][1], model.steps[1][1]
    p = model.predict_proba(Xte)[:, 1]
    bank = np.asarray(seed0["proba"]["XGBoost (bank model)"])
    medians = np.median(Xtr, axis=0)
    above = np.where(p > THRESHOLD)[0]
    picks = {
        "declined, highest risk": int(np.argmax(p)),
        "declined, nearest the cut-off": int(above[np.argmin(p[above] - THRESHOLD)]),
        "approved, typical applicant": int(np.argmin(np.abs(p - np.median(p)))),
    }
    Z = scaler.transform(Xtr)
    out = []
    for label, i in picks.items():
        row = Xte[i]
        e = dist.explain(scaler.transform(row[None])[0], feature_names=cols, max_terms=3,
                         feature_bounds=(Z.min(0).astype(np.float64), Z.max(0).astype(np.float64)))
        declined = bool(p[i] > THRESHOLD)
        flips, n_tried = reachable_flips(model, row, Xtr, cols, declined)
        cf = e.counterfactual
        cf_raw = None if cf is None else {
            "input": cf.feature, "from": float(row[cf.index]),
            "to": float(cf.to_value * scaler.scale_[cf.index] + scaler.mean_[cf.index]),
            "p_new_class": float(cf.new_probability), "new_class": int(cf.new_class)}
        out.append({"case": label, "test_row": int(te[i]), "actual_default": int(y[te[i]]),
                    "gate_counterfactual_raw_units": cf_raw,
                    "p_default": float(p[i]), "bank_model_p_default": float(bank[i]),
                    "declined": declined, "bank_model_declined": bool(bank[i] > THRESHOLD),
                    "applicant": {c: float(v) for c, v in zip(cols, row)},
                    "path_text": e.to_text(),
                    "reasons": principal_reasons(model, row, medians, cols),
                    "reachable_flips": flips, "reachable_tried": n_tried})
    return out


# ── decisions by group ─────────────────────────────────────────────────
def slices(seed0):
    from sklearn.metrics import roc_auc_score
    X, y = load()
    _, te = split(X, y)
    Xt, yt = X.iloc[te], y[te]
    age = Xt["age"].to_numpy()
    groups = {
        "sex: male": Xt["sex"].to_numpy() == 1, "sex: female": Xt["sex"].to_numpy() == 2,
        "married": Xt["marriage"].to_numpy() == 1, "single": Xt["marriage"].to_numpy() == 2,
        "age under 30": age < 30, "age 30 to 44": (age >= 30) & (age < 45), "age 45 and over": age >= 45,
    }
    out = {}
    for name in ("DistilledTree", "XGBoost (bank model)"):
        p = np.asarray(seed0["proba"][name])
        rows = {}
        for g, m in groups.items():
            d = p[m] > THRESHOLD
            rows[g] = {"n": int(m.sum()), "default_rate": float(yt[m].mean()), "decline_rate": float(d.mean()),
                       "auc": float(roc_auc_score(yt[m], p[m])),
                       "defaults_approved": float((~d[yt[m] == 1]).mean()),
                       "repayers_declined": float(d[yt[m] == 0].mean())}
        for fam in ("sex", "marital", "age"):
            keys = [g for g in groups if g.startswith(fam[:3]) or (fam == "marital" and g in ("married", "single"))]
            approve = {g: 1 - rows[g]["decline_rate"] for g in keys}
            top = max(approve.values())
            for g in keys:
                rows[g]["approval_ratio_to_highest"] = float(approve[g] / top)
        out[name] = rows
    return out


# ── the final model and its files ──────────────────────────────────────
def artefacts():
    import pickle
    X, _ = load()
    cols = inputs(X)
    model = pickle.loads((CACHE / "tree_seed0.pkl").read_bytes())
    scaler, dist = model.steps[0][1], model.steps[1][1]
    npt = dist.to_numpy(feature_names=cols)
    npt.to_json(str(OUT / "model.json"))
    hard = dist.to_hard_tree()
    rules = hard.export_text(feature_names=cols, max_features=4)
    (OUT / "rules.txt").write_text(rules, encoding="utf-8")
    (OUT / "preprocessing.json").write_text(json.dumps({
        "input_order": cols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "note": "Standardise each input as (x - mean) / scale in this order, then call the model."}, indent=1))
    _, te = split(*load())
    Xte = X.iloc[te][cols].to_numpy(np.float32)
    Z = scaler.transform(Xte).astype(np.float32)
    return {"n_splits": int(np.asarray(npt.is_split_).sum()), "json_bytes": (OUT / "model.json").stat().st_size,
            "numpy_matches_model": float(np.abs(npt.predict_proba(Z) - dist.predict_proba(Z)).max()),
            "rules_lines": rules.count("\n") + 1,
            "hard_rules_same_decision": float((hard.predict(Z) == dist.predict(Z)).mean())}


def main():
    from joblib import Parallel, delayed

    import neural_trees
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()
    runs = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 3)(delayed(one_seed)(s) for s in SEEDS)
    names = list(runs[0]["models"])
    summary = {n: {k: {"mean": float(np.mean([r["models"][n][k] for r in runs])),
                       "sd": float(np.std([r["models"][n][k] for r in runs], ddof=1))}
                   for k in runs[0]["models"][n]} for n in names}
    X, y = load()
    results = {
        "library": neural_trees.__version__,
        "data": {"source": "OpenML 42477 (UCI default of credit card clients, Yeh and Lien 2009), CC0",
                 "n": int(len(y)), "n_default": int(y.sum()), "inputs": inputs(X),
                 "kept_out_of_inputs": list(PROTECTED), "train_rows": 24000, "test_rows": 6000},
        "protocol": {"seeds": list(SEEDS), "threshold": THRESHOLD, "xgboost_trials": TRIALS},
        "summary": summary,
        "per_seed": [{k: v for k, v in r.items() if k != "proba"} for r in runs],
        "explanations": explanations(runs[0]),
        "slices_seed0": slices(runs[0]),
        "final_model": artefacts(),
        "total_seconds": time.time() - t0,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

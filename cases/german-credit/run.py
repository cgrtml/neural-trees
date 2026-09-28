"""
German Credit case study: the first "auditable model package" example.

Data: UCI Statlog German Credit (OpenML id 31), 1 000 applicants, 20 attributes,
700 good and 300 bad. The dataset ships a cost matrix: calling a bad applicant
good costs 5, calling a good one bad costs 1, so the cost-optimal decision is to
refuse when P(bad) exceeds 1/6.

The script measures a soft decision tree against the models a credit team would
actually consider, on identical folds, and writes everything a reviewer would
ask for: the fold-level numbers, the calibration table, the rules, three
explained decisions with counterfactuals, slices by age, sex and residency, the
model file in JSON and ONNX with a check that they predict the same thing, and
a model document that reads the numbers from results.json.

Run:  python cases/german-credit/run.py        (about ten minutes on a laptop)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import neural_trees
from neural_trees import SoftDecisionTree

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
FOLDS = 5
COST_FN, COST_FP = 5.0, 1.0            # bad called good; good called bad
BAYES_THRESHOLD = COST_FP / (COST_FP + COST_FN)


# ── data ───────────────────────────────────────────────────────────────
def load():
    d = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X = d.data.copy()
    y = d.target.astype(str).to_numpy()   # 'good' / 'bad'
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    for c in categorical:
        X[c] = X[c].astype(str)
    return X, y, numeric, categorical


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ], sparse_threshold=0.0)


def feature_names_after(pre):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names += list(cols)
        elif name == "cat":
            enc = trans.steps[-1][1]
            for col, cats in zip(cols, enc.categories_):
                names += [f"{col}={c}" for c in cats]
    return names


# ── models: one fixed configuration each, nothing tuned on this data ───
def models(seed):
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "Soft tree, per-leaf": SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf",
                                                growth_init="residual", random_state=seed),
        "Soft tree, depth 4": SoftDecisionTree(depth=4, max_epochs=150, random_state=seed),
        "Logistic regression": LogisticRegression(max_iter=5000, random_state=seed),
        "CART": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     random_state=seed, n_jobs=1, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                       random_state=seed, n_jobs=1, verbose=-1),
    }


NEEDS_INT = {"XGBoost", "LightGBM"}


def fit_predict(name, model, Xtr, ytr, Xte):
    """Fit, return P(bad) on the test rows. Boosters take 0/1 labels."""
    if name in NEEDS_INT:
        model.fit(Xtr, (ytr == "bad").astype(int))
        return model.predict_proba(Xte)[:, 1]
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, list(model.classes_).index("bad")]


def cost_per_applicant(y_true, p_bad, threshold):
    refuse = p_bad > threshold
    bad = y_true == "bad"
    return float((COST_FN * (bad & ~refuse).sum() + COST_FP * (~bad & refuse).sum()) / len(y_true))


def metrics(y_true, p_bad):
    bad = (y_true == "bad").astype(int)
    pred = (p_bad > 0.5).astype(int)
    return {
        "accuracy": float((pred == bad).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(bad, pred)),
        "auc": float(roc_auc_score(bad, p_bad)),
        "brier": float(brier_score_loss(bad, p_bad)),
        "log_loss": float(log_loss(bad, np.clip(p_bad, 1e-6, 1 - 1e-6))),
        "cost_at_0.5": cost_per_applicant(y_true, p_bad, 0.5),
        "cost_at_bayes": cost_per_applicant(y_true, p_bad, BAYES_THRESHOLD),
        "refusal_rate_at_bayes": float((p_bad > BAYES_THRESHOLD).mean()),
    }


# ── cross-validation ───────────────────────────────────────────────────
def cross_validate(X, y, numeric, categorical):
    per_fold = {}          # model -> list of metric dicts (15 entries)
    oof = {}               # (model, seed) -> P(bad) for every row, out of fold
    structure = []         # soft per-leaf: splits, root features, hard agreement
    seconds = {}
    for seed in SEEDS:
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            pre = preprocessor(numeric, categorical)
            Xtr = pre.fit_transform(X.iloc[tr]).astype(np.float32)
            Xte = pre.transform(X.iloc[te]).astype(np.float32)
            names = feature_names_after(pre)
            for name, model in models(seed).items():
                t0 = time.time()
                p = fit_predict(name, model, Xtr, y[tr], Xte)
                seconds.setdefault(name, []).append(time.time() - t0)
                per_fold.setdefault(name, []).append(metrics(y[te], p))
                oof.setdefault((name, seed), np.full(len(y), np.nan))[te] = p
                if name == "Soft tree, per-leaf":
                    hard = model.to_hard_tree()
                    w = model.get_split_weights()
                    # rank by mean |weight * value| on the training rows, so a 0/1
                    # one-hot column and a standardised numeric compete fairly
                    root = np.argsort(-np.abs(w[0] * Xtr).mean(0))[:3]
                    structure.append({
                        "seed": seed, "fold": k, "n_splits": int(len(w)),
                        "root_top_features": [names[i] for i in root],
                        "hard_agreement": float((hard.predict(Xte) == model.predict(Xte)).mean()),
                    })
                print(f"seed {seed} fold {k} {name:22s} acc {per_fold[name][-1]['accuracy']:.3f} "
                      f"auc {per_fold[name][-1]['auc']:.3f} cost {per_fold[name][-1]['cost_at_bayes']:.3f} "
                      f"{seconds[name][-1]:.1f}s", flush=True)
    return per_fold, oof, structure, seconds


def summarise(per_fold):
    out = {}
    for name, rows in per_fold.items():
        df = pd.DataFrame(rows)
        out[name] = {c: {"mean": float(df[c].mean()), "sd": float(df[c].std(ddof=1))} for c in df.columns}
    return out


def paired_tests(per_fold, reference="Soft tree, per-leaf"):
    from scipy.stats import ttest_rel, wilcoxon
    ref = pd.DataFrame(per_fold[reference])
    out = {}
    for name, rows in per_fold.items():
        if name == reference:
            continue
        df = pd.DataFrame(rows)
        out[name] = {}
        for c in ("accuracy", "auc", "cost_at_bayes"):
            d = df[c] - ref[c]
            t = float(ttest_rel(df[c], ref[c]).pvalue) if np.ptp(d) > 0 else 1.0
            w = float(wilcoxon(d).pvalue) if np.ptp(d) > 0 else 1.0
            out[name][c] = {"diff_mean": float(d.mean()), "p_t": t, "p_wilcoxon": w}
    return out


# ── calibration ────────────────────────────────────────────────────────
def calibration(y, p, bins=10):
    bad = (y == "bad").astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()), "observed_bad_rate": float(bad[m].mean())})
        ece += m.mean() * abs(p[m].mean() - bad[m].mean())
    return {"bins": rows, "ece": float(ece)}


# ── slices ─────────────────────────────────────────────────────────────
def slices(X, y, p, threshold):
    sex = X["personal_status"].str.startswith("female").map({True: "female", False: "male"})
    groups = {
        "age < 25": X["age"] < 25, "age >= 25": X["age"] >= 25,
        "female": sex == "female", "male": sex == "male",
        "foreign worker": X["foreign_worker"] == "yes", "not a foreign worker": X["foreign_worker"] == "no",
    }
    out = {}
    for g, m in groups.items():
        m = m.to_numpy()
        yt, pt = y[m], p[m]
        bad = yt == "bad"
        refuse = pt > threshold
        out[g] = {
            "n": int(m.sum()), "actual_bad_rate": float(bad.mean()),
            "refusal_rate": float(refuse.mean()),
            "accuracy": float(((pt > 0.5) == bad).mean()),
            "bad_missed_rate": float((~refuse[bad]).mean()) if bad.any() else float("nan"),
            "good_refused_rate": float(refuse[~bad].mean()) if (~bad).any() else float("nan"),
        }
    return out


# ── explanations ───────────────────────────────────────────────────────
def in_words(feature, from_value, to_value, X_row_raw):
    """A counterfactual on a one-hot or standardised column, said in the table's own terms."""
    if "=" in feature:
        col, val = feature.split("=", 1)
        now = X_row_raw[col]
        if to_value > from_value:
            return f"if {col} were '{val}' instead of '{now}'"
        return f"if {col} were anything other than '{val}'"
    return f"if {feature} moved from {from_value:.2f} to {to_value:.2f} (in standardised units)"


def reachable_counterfactuals(model, pre, X_train_raw, row, numeric, categorical, current):
    """
    Single changes a real applicant could make, tried on the raw table and
    re-encoded: every other value of each categorical column, and each numeric
    column moved to the training deciles. A flip is a change of the decision
    at the cost-optimal threshold. Returns the flips, smallest first.
    """
    p_bad_col = list(model.classes_).index("bad")
    trials = []
    for c in categorical:
        for v in sorted(X_train_raw[c].unique()):
            if v != row[c]:
                trials.append((c, v, 0.0))
    for c in numeric:
        for q in np.quantile(X_train_raw[c], np.linspace(0.1, 0.9, 9)):
            if q != row[c]:
                trials.append((c, float(q), abs(q - row[c]) / (X_train_raw[c].std() + 1e-9)))
    frame = pd.DataFrame([row.to_dict() for _ in trials])
    frame[numeric] = frame[numeric].astype(float)
    for i, (c, v, _) in enumerate(trials):
        frame.at[i, c] = v
    p = model.predict_proba(pre.transform(frame).astype(np.float32))[:, p_bad_col]
    flips = [{"column": c, "from": row[c].item() if hasattr(row[c], "item") else row[c], "to": v,
              "p_bad": float(pb), "size": float(size)}
             for (c, v, size), pb in zip(trials, p) if (pb > BAYES_THRESHOLD) != current]
    flips.sort(key=lambda f: (f["size"], -abs(f["p_bad"] - 0.5)))
    return flips[:5], len(trials)


def explanations(X, y, numeric, categorical):
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    tr, te = next(skf.split(X, y))
    pre = preprocessor(numeric, categorical)
    Xtr = pre.fit_transform(X.iloc[tr]).astype(np.float32)
    Xte = pre.transform(X.iloc[te]).astype(np.float32)
    names = feature_names_after(pre)
    model = SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf", growth_init="residual", random_state=0)
    model.fit(Xtr, y[tr])
    p = model.predict_proba(Xte)[:, list(model.classes_).index("bad")]
    lower, upper = Xtr.min(0).astype(np.float64), Xtr.max(0).astype(np.float64)
    picks = {
        "clearly refused (highest P(bad))": int(np.argmax(p)),
        "clearly accepted (lowest P(bad))": int(np.argmin(p)),
        "borderline (P(bad) nearest the 1/6 threshold)": int(np.argmin(np.abs(p - BAYES_THRESHOLD))),
    }
    out = []
    for label, i in picks.items():
        row = X.iloc[te[i]]
        e = model.explain(Xte[i], feature_names=names, max_terms=3, feature_bounds=(lower, upper))
        cf = e.counterfactual
        flips, n_tried = reachable_counterfactuals(model, pre, X.iloc[tr], row, numeric, categorical, current=p[i] > BAYES_THRESHOLD)
        out.append({
            "case": label, "row_index": int(te[i]), "actual": str(y[te[i]]), "p_bad": float(p[i]),
            "applicant": {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()},
            "text": e.to_text(),
            "counterfactual_in_words": None if cf is None else
                f"{in_words(cf.feature, cf.from_value, cf.to_value, row)}, the prediction would become "
                f"'{cf.new_class}' (P = {cf.new_probability:.3f})",
            "reachable_flips": flips, "reachable_tried": n_tried,
        })
    return out


# ── the final model, its rules and its files ───────────────────────────
def final_model(X, y, numeric, categorical):
    import onnx
    import onnxruntime as ort

    pre = preprocessor(numeric, categorical)
    Xa = pre.fit_transform(X).astype(np.float32)
    names = feature_names_after(pre)
    model = SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf", growth_init="residual", random_state=0)
    model.fit(Xa, y)
    npt = model.to_numpy(feature_names=names)
    npt.to_json(str(HERE / "model.json"))
    npt.save_onnx(str(HERE / "model.onnx"))
    sess = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])
    torch_p = model.predict_proba(Xa)
    onnx_p = sess.run(["probabilities"], {"X": Xa})[0]
    t0 = time.time()
    for row in Xa[:200]:
        sess.run(["probabilities"], {"X": row[None]})
    us_per_row = (time.time() - t0) / 200 * 1e6
    hard = model.to_hard_tree()
    rules = hard.export_text(feature_names=names, max_features=5)
    # what actually decides each gate: mean |weight * value| over the training rows
    W = model.get_split_weights()
    gates = []
    for g in range(len(W)):
        contrib = np.abs(W[g] * Xa).mean(0)
        top = np.argsort(-contrib)[:5]
        gates.append({"gate": g, "inputs": [(names[j], float(contrib[j] / contrib.sum())) for j in top]})
    (HERE / "rules.txt").write_text(rules, encoding="utf-8")
    # the preprocessing a consumer of model.onnx must reproduce
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    prep = {
        "input_order": names,
        "numeric": {c: {"impute_median": float(m), "mean": float(mu), "scale": float(s)}
                    for c, m, mu, s in zip(numeric, num.steps[0][1].statistics_, num.steps[1][1].mean_, num.steps[1][1].scale_)},
        "categorical": {c: [str(v) for v in cats] for c, cats in zip(categorical, cat.steps[-1][1].categories_)},
        "note": "Standardise each numeric column as (x - mean) / scale after median imputation; one-hot each categorical column in the listed order, unknown values as all zeros; concatenate numeric then categorical.",
    }
    (HERE / "preprocessing.json").write_text(json.dumps(prep, indent=1), encoding="utf-8")
    return {
        "n_splits": int(len(model.get_split_weights())),
        "n_leaves": int(len(model.get_leaf_distributions())),
        "n_features_after_encoding": int(Xa.shape[1]),
        "onnx_max_abs_diff": float(np.abs(torch_p - onnx_p).max()),
        "onnx_label_agreement": float((onnx_p.argmax(1) == torch_p.argmax(1)).mean()),
        "onnx_microseconds_per_row": float(us_per_row),
        "onnx_bytes": int((HERE / "model.onnx").stat().st_size),
        "json_bytes": int((HERE / "model.json").stat().st_size),
        "hard_agreement_in_sample": float((hard.predict(Xa) == model.predict(Xa)).mean()),
        "training_accuracy": float((model.predict(Xa) == y).mean()),
        "rules_lines": int(rules.count("\n") + 1),
        "gate_contributions": gates,
    }


def main():
    t_all = time.time()
    X, y, numeric, categorical = load()
    per_fold, oof, structure, seconds = cross_validate(X, y, numeric, categorical)
    summary = summarise(per_fold)
    tests = paired_tests(per_fold)
    cal = {name: calibration(y, oof[(name, 0)]) for name in per_fold}
    sl = slices(X, y, oof[("Soft tree, per-leaf", 0)], BAYES_THRESHOLD)
    sl_lr = slices(X, y, oof[("Logistic regression", 0)], BAYES_THRESHOLD)
    ex = explanations(X, y, numeric, categorical)
    fm = final_model(X, y, numeric, categorical)
    results = {
        "library": neural_trees.__version__,
        "data": {"source": "OpenML 31 (UCI Statlog German Credit)", "n": int(len(y)),
                 "n_bad": int((y == "bad").sum()), "n_good": int((y == "good").sum()),
                 "numeric": numeric, "categorical": categorical,
                 "cost_matrix": {"bad_called_good": COST_FN, "good_called_bad": COST_FP},
                 "bayes_threshold_on_p_bad": BAYES_THRESHOLD},
        "protocol": {"seeds": list(SEEDS), "folds": FOLDS, "fits_per_model": len(SEEDS) * FOLDS,
                     "preprocessing": "median impute + standardise numerics, one-hot categoricals, fitted inside each training fold"},
        "summary": summary,
        "tests_vs_soft_per_leaf": tests,
        "seconds_per_fit": {k: float(np.mean(v)) for k, v in seconds.items()},
        "calibration_seed0_oof": cal,
        "slices_soft_per_leaf_seed0_oof": sl,
        "slices_logistic_seed0_oof": sl_lr,
        "structure_per_fit": structure,
        "explanations": ex,
        "final_model": fm,
        "total_seconds": float(time.time() - t_all),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    pd.DataFrame({name: {c: f"{v['mean']:.3f} ± {v['sd']:.3f}" for c, v in s.items()} for name, s in summary.items()}).T.to_csv(HERE / "summary.csv")
    print(f"done in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()

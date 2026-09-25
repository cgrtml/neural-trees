"""Try it on your data: the whole flow on a table you upload, ending with a model you can take away."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ttest_rel
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from playground import (
    BASELINE_MODELS,
    HOSTED_SLOWDOWN,
    LIBRARY_MODELS,
    MODELS,
    SECONDS_PER_FOLD,
    build,
    defaults,
    glossary,
    ui,
)
from playground.yourdata import (
    MAX_COLS,
    MAX_ROWS,
    REGRESSORS,
    check_target,
    code_snippet,
    describe,
    feature_names_after,
    guess_target,
    prepare,
    preprocessor,
    read_table,
    task_for,
)

ui.title(
    "Try it on your data",
    lead="Upload a table (CSV or Excel), pick the column to predict, and run the same flow as "
         "Compare on it: who scores highest, whether the gap is real, what the soft tree learned, "
         "and a model file you can take away. Nothing is stored; the table lives in this browser "
         "session only.",
    eyebrow="Your table, the same steps",
)

# ── step 1: the table ────────────────────────────────────────────────
ui.step(1, "Get a table", f"A CSV or Excel file with one row per example and one column to predict: a class (classification) or a number (regression). Files up to 25 MB; a larger table is subsampled to {MAX_ROWS} rows and at most {MAX_COLS} columns are used. Comma, semicolon and tab separators are detected.")
u1, u2 = st.columns([2, 1])
with u1:
    up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
with u2:
    st.markdown("**No file at hand?**")
    s1, s2, s3 = st.columns(3)
    if s1.button("Breast Cancer", width="stretch", help="569 rows, 30 measurements, malignant or benign: classification"):
        d = load_breast_cancer(as_frame=True)
        df = d.data.copy()
        df["diagnosis"] = d.target.map({0: "malignant", 1: "benign"})
        st.session_state.yd = ("Breast Cancer sample", df)
        st.session_state.pop("yd_results", None)
    if s2.button("Wine", width="stretch", help="178 rows, 13 measurements, three cultivars: classification"):
        d = load_wine(as_frame=True)
        df = d.data.copy()
        df["cultivar"] = d.target.map(dict(enumerate(d.target_names)))
        st.session_state.yd = ("Wine sample", df)
        st.session_state.pop("yd_results", None)
    if s3.button("Diabetes", width="stretch", help="442 rows, 10 measurements, disease progression a year later: regression"):
        d = load_diabetes(as_frame=True)
        df = d.data.copy()
        df["progression"] = d.target
        st.session_state.yd = ("Diabetes sample", df)
        st.session_state.pop("yd_results", None)
if up is not None:
    try:
        with st.spinner(f"Reading {up.name} ({up.size / 1e6:.1f} MB)..."):
            df = read_table(up.getvalue(), up.name)
        if st.session_state.get("yd", ("",))[0] != up.name:
            st.session_state.pop("yd_results", None)
        st.session_state.yd = (up.name, df)
    except Exception as e:  # noqa: BLE001 - the user needs the reason
        st.error(f"Could not read that file: {e}")

if "yd" not in st.session_state:
    st.info("Upload a file or load a sample to continue.")
    st.stop()
name, df = st.session_state.yd
st.markdown(f"**{name}**: {len(df)} rows, {df.shape[1]} columns.")
st.dataframe(df.head(5), width="stretch", height=220)

t1, t2 = st.columns([1, 2])
with t1:
    target = st.selectbox("Column to predict", list(df.columns), index=list(df.columns).index(guess_target(df)), key="yd_target",
                          help="Guessed from the column names, then the last column. Change it if the guess is wrong.")
task = task_for(df, target)
ok, msg = check_target(df, target)
with t2:
    st.markdown(f"**{'Classification' if task == 'classification' else 'Regression'}**: predicting **{target}**.")
    (st.success if ok else st.error)(msg)
with st.expander("What happens to each column"):
    st.dataframe(describe(df, target), width="stretch")
if not ok:
    st.stop()

X, y, classes, numeric, categorical, notes = prepare(df, target)
for n in notes:
    st.caption(n)
if X.shape[1] == 0:
    st.error("No usable feature columns are left.")
    st.stop()

# ── step 2: models ───────────────────────────────────────────────────
if task == "classification":
    ui.step(2, "Choose the models", "Same models as Compare; the soft decision tree is the one that explains itself afterwards.")
    catalogue = {m: dict(group=MODELS[m]["group"], tag=MODELS[m]["tag"]) for m in MODELS}
    defaults_here = ["Soft Decision Tree", "Multivariate Tree", "CART (sklearn)", "Random Forest"]
    lib_names, base_names = LIBRARY_MODELS, BASELINE_MODELS
else:
    ui.step(2, "Choose the models", "Regression models: the soft tree with a value per leaf against three baselines.")
    catalogue = {m: dict(group=r["group"], tag=r["tag"]) for m, r in REGRESSORS.items()}
    defaults_here = list(REGRESSORS)
    lib_names = [m for m, r in REGRESSORS.items() if r["group"] == "neural-trees"]
    base_names = [m for m, r in REGRESSORS.items() if r["group"] == "baseline"]
for m in catalogue:
    st.session_state.setdefault(f"yd_{m}", m in defaults_here)
m1, m2 = st.columns(2)
with m1:
    st.markdown("**From neural-trees**")
    for m in lib_names:
        st.checkbox(f"{m}: {catalogue[m]['tag']}", key=f"yd_{m}")
with m2:
    st.markdown("**Baselines**")
    for m in base_names:
        st.checkbox(f"{m}: {catalogue[m]['tag']}", key=f"yd_{m}")
selected = [m for m in catalogue if st.session_state.get(f"yd_{m}")]
if len(selected) < 2:
    st.info("Pick at least two models.")
    st.stop()


def _make(model_name):
    if model_name in REGRESSORS:
        return REGRESSORS[model_name]["build"]()
    return build(model_name, defaults(model_name), 0)


# ── step 3: run ──────────────────────────────────────────────────────
metric = "accuracy" if task == "classification" else "R^2"
ui.step(3, "Run", f"Every model sees the same folds and is scored by {metric}; features are imputed, standardised and one-hot encoded inside each fold, never on the held-out part.")
folds = st.slider("Folds", 3, 10, 5, key="yd_folds")


@st.cache_data(show_spinner=False, max_entries=4096, ttl=6 * 3600)
def _fold(df, target, model_name, folds, seed, k):
    X, y, _, numeric, categorical, _ = prepare(df, target)
    if task_for(df, target) == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train, test = list(splitter.split(X, y))[k]
    pipe = make_pipeline(preprocessor(numeric, categorical), _make(model_name))
    pipe.fit(X.iloc[train], y[train])
    return float(pipe.score(X.iloc[test], y[test]))


@st.cache_resource(show_spinner=False, max_entries=32, ttl=6 * 3600)
def _fit_all(df, target, model_name):
    X, y, _, numeric, categorical, _ = prepare(df, target)
    pipe = make_pipeline(preprocessor(numeric, categorical), _make(model_name))
    return pipe.fit(X, y)


scale = max(len(y) / 500.0, 0.3) * (1 + X.shape[1] / 250.0)  # a real run: 1 460 rows x 79 columns, 4 models, 8 folds took 82 s hosted
hosted_s = sum(SECONDS_PER_FOLD.get(m, 0.4) for m in selected) * folds * scale * HOSTED_SLOWDOWN
with st.container(border=True):
    what = f"{len(classes)} classes" if task == "classification" else "a number to predict"
    st.markdown(f"**{len(selected)} models, {folds} folds, {len(y)} rows, {X.shape[1]} columns, {what}.** About **{max(1, round(hosted_s))} s** on the hosted app; a repeat of the same run is instant.")
    if st.button("Run on my data", type="primary", width="stretch"):
        results, total, done, t_start = {}, len(selected) * folds, 0, time.time()
        bar = st.progress(0, text="Starting...")
        with st.status(f"Training {len(selected)} models on {folds} folds", expanded=True) as status:
            for m in selected:
                t0, scores = time.time(), []
                try:
                    for k in range(folds):
                        bar.progress(done / total, text=f"{m}: fold {k + 1} of {folds} · {done} of {total} fits · {time.time() - t_start:.0f} s")
                        scores.append(_fold(df, target, m, folds, 0, k))
                        done += 1
                    results[m] = np.array(scores)
                    st.write(f"✓ {m}: {metric} {np.mean(scores):.3f} in {time.time() - t0:.1f} s")
                except Exception as e:  # noqa: BLE001 - shown to the user, not hidden
                    results[m] = str(e)
                    done += folds - len(scores)
                    st.write(f"✗ {m} failed: {str(e)[:100]}")
            status.update(label=f"Done: {len(selected)} models, {total} fits, {time.time() - t_start:.0f} s", state="complete", expanded=False)
        bar.empty()
        st.session_state.yd_results = (name, target, tuple(selected), folds, results)

if "yd_results" not in st.session_state:
    st.stop()
r_name, r_target, r_models, r_folds, results = st.session_state.yd_results
if (r_name, r_target) != (name, target):
    st.warning("The table or the target changed since the last run; press **Run on my data** again.")
    st.stop()
ok_res = {n: s for n, s in results.items() if not isinstance(s, str)}
failed = {n: s for n, s in results.items() if isinstance(s, str)}
if not ok_res:
    st.error("No model produced a result: " + "; ".join(f"{n}: {e[:80]}" for n, e in failed.items()))
    st.stop()
order = sorted(ok_res, key=lambda n: -ok_res[n].mean())
best = order[0]
rows, within, behind = [], [], []
for rank, n in enumerate(order, 1):
    s = ok_res[n]
    if n == best:
        p, reading = np.nan, "highest on these folds"
    else:
        d = ok_res[best] - s
        p = float(ttest_rel(ok_res[best], s).pvalue) if np.ptp(d) > 0 else 1.0
        reading = "within fold noise of the best" if p >= 0.05 else "behind the best"
        (within if p >= 0.05 else behind).append(n)
    rows.append({"rank": rank, "model": n, metric: f"{s.mean():.3f} ± {s.std():.3f}", "gap": f"{(s.mean() - ok_res[best].mean()):+.3f}", "p vs best": "" if np.isnan(p) else f"{p:.3f}", "reading": reading})
if task == "classification":
    floor, floor_text = float(np.bincount(y).max() / len(y)), "Always predicting the commonest class would score {:.3f}; that is the floor every model has to clear."
else:
    floor, floor_text = 0.0, "Always predicting the mean scores R^2 = 0; below that a model is worse than no model."
ui.verdict(
    f"Result on {r_name}, predicting {r_target}, {r_folds} folds",
    f"{best} scored highest: {metric} {ok_res[best].mean():.3f} ± {ok_res[best].std():.3f}",
    (f"Within fold noise of it: <b>{', '.join(within)}</b>. " if within else "")
    + (f"Measurably behind: <b>{', '.join(behind)}</b>. " if behind else "")
    + floor_text.format(floor),
)
if failed:
    st.warning("Did not run: " + ", ".join(f"**{n}** ({e[:80]})" for n, e in failed.items()))
st.dataframe(pd.DataFrame(rows).set_index("rank"), width="stretch", height=48 + 37 * len(rows))
fig = go.Figure(go.Bar(
    x=[ok_res[n].mean() for n in order], y=order, orientation="h",
    error_x=dict(type="data", array=[ok_res[n].std() for n in order], visible=True, thickness=1, width=4, color="rgba(21,32,43,0.55)"),
    marker_color=[ui.LIBRARY if catalogue[n]["group"] == "neural-trees" else ui.BASELINE for n in order],
    text=[f"{ok_res[n].mean():.3f}" for n in order], textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
))
fig.add_vline(x=floor, line=dict(color="rgba(21,32,43,0.4)", dash="dot"), annotation_text="majority class" if task == "classification" else "predicting the mean", annotation_position="top")
lo = max(-0.5, min(min(ok_res[n].mean() - ok_res[n].std() for n in order), floor) - 0.05)
fig.update_layout(height=50 + 34 * len(order), margin=dict(t=30, b=10, l=10, r=20), xaxis=dict(range=[lo, 1.02], title=f"{metric}, mean ± sd over folds"), yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, config={"displayModeBar": False})

# ── step 4: look inside ──────────────────────────────────────────────
soft_name = "Soft Decision Tree" if task == "classification" else "Soft Tree Regressor"
ui.step(4, "Look inside the soft tree", "Fitted once on all the rows: its rules, and for classification one row explained.")
if soft_name not in r_models:
    st.info(f"Tick **{soft_name}** in step 2 and run again to see its rules.")
else:
    with st.spinner("Fitting the soft tree on all rows..."):
        pipe = _fit_all(df, target, soft_name)
    pre = pipe.steps[0][1]
    tree = pipe.steps[-1][1]
    fnames = feature_names_after(pre)
    Xt = pre.transform(X)
    hard = tree.to_hard_tree()
    rules = hard.export_text(feature_names=fnames)
    if task == "classification":
        agree = float((hard.predict(Xt) == tree.predict(Xt)).mean())
        label = f"The tree as rules ({rules.count(chr(10)) + 1} lines; class numbers are {', '.join(f'{i} = {c}' for i, c in enumerate(classes))})"
        note = f"The hard reading agrees with the soft tree on {agree:.1%} of the rows."
    else:
        agree = float(np.corrcoef(hard.predict(Xt), tree.predict(Xt))[0, 1])
        label = f"The tree as rules ({rules.count(chr(10)) + 1} lines; one value of {r_target} per leaf)"
        note = f"The hard reading's predictions correlate {agree:.2f} with the soft tree's; the soft tree blends leaves, the rules pick one."
    with st.expander(label):
        st.code(rules, language=None)
    st.caption(note)
    if task == "classification":
        i = st.number_input("Row to explain", 0, len(X) - 1, 0, key="yd_row")
        ex = tree.explain(Xt[i:i + 1], feature_names=fnames)[0]
        st.markdown(f"Row {i}: actual **{r_target} = {classes[int(y[i])]}**, predicted **{classes[int(ex.predicted_class)]}** with p = {ex.probabilities[ex.predicted_class]:.3f}.")
        with st.expander("The explanation, gate by gate", expanded=True):
            st.code(ex.to_text(), language=None)
        if ex.counterfactual is not None:
            cf = ex.counterfactual
            if "=" in cf.feature:  # a one-hot column: the change is turning that category on or off
                col, val = cf.feature.split("=", 1)
                change = f"if **{col}** were {'not ' if cf.to_value < cf.from_value else ''}**{val}**"
            else:
                change = f"if **{cf.feature}** moved from {cf.from_value:.2f} to {cf.to_value:.2f} (standardised units)"
            st.markdown(f"**What would flip it:** {change}, the prediction would become **{r_target} = {classes[int(cf.new_class)]}** (p = {cf.new_probability:.3f}), verified by re-predicting.")
    else:
        st.caption("Per-row explanations exist for classification only, for now.")

    # ── step 5: take it away ─────────────────────────────────────────
    ui.step(5, "Take it with you", "Two things: the fitted tree as a file that predicts with numpy alone, and a Python script that repeats everything this page did, on your machine, so the result is yours and not this site's.")
    d1, d2 = st.columns([1, 2])
    with d1:
        st.download_button("Download model.json", tree.to_numpy(feature_names=fnames).to_json(), file_name="neural-trees-model.json", mime="application/json", width="stretch")
        st.caption("Load it with `NumpySoftTree.from_json(...)` and call `predict` on preprocessed rows (the same imputation, scaling and one-hot encoding). The script on the right shows that preprocessing exactly.")
    with d2:
        st.markdown("**The script**: reads your file, builds the same preprocessing, cross-validates the same model, prints the rules or an explanation, and saves `model.json`. Paste it into a notebook.")
        line = "SoftDecisionTree(depth=4, max_epochs=40, random_state=0)" if task == "classification" else REGRESSORS[soft_name]["line"]
        st.code(code_snippet(target, numeric, categorical, line, task=task, filename=r_name if "." in r_name else "your_file.csv"), language="python")

glossary(["fold", "accuracy ± sd", "p-value", "5x2cv F test", "standardised"])
ui.footer()

"""Compare: choose data, choose models, see what will run, run it, read the verdict."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ttest_rel

from playground import (
    BASELINE_MODELS,
    DATASETS,
    DEFAULT_DATASET,
    DEFAULT_MODELS,
    LIBRARY_MODELS,
    MODELS,
    dataset,
    dataset_figure,
    defaults,
    estimate_seconds,
    f_test,
    fold_score,
    glossary,
    params_key,
    ui,
)

ui.title(
    "Compare",
    lead="Pick the data, pick the models, run. Every model is trained and tested on the same "
         "cross-validation folds, so a difference is a difference between models, not between "
         "splits; the result says whether it is real or fold noise.",
    eyebrow="Three steps",
)


def _pick(names):
    for m in MODELS:
        st.session_state[f"cmp_{m}"] = m in names


# state handed over by another page
st.session_state.setdefault("cmp_data", DEFAULT_DATASET)
if "cmp_reset" in st.session_state:
    _pick(st.session_state.pop("cmp_reset"))
    st.session_state.pop("cmp_last", None)
for m in MODELS:
    st.session_state.setdefault(f"cmp_{m}", m in DEFAULT_MODELS)

# ── step 1: data ─────────────────────────────────────────────────────
ui.step(1, "Choose the data")
dataset_name = st.pills("Dataset", list(DATASETS), selection_mode="single", key="cmp_data", label_visibility="collapsed") or DEFAULT_DATASET
X, y, _, _ = dataset(dataset_name)
d1, d2 = st.columns([1, 3])
with d1:
    st.plotly_chart(dataset_figure(dataset_name, height=170), config={"displayModeBar": False}, key="cmp_ds_fig")
with d2:
    ui.card_text(dataset_name, DATASETS[dataset_name]["blurb"], stat=f"{len(y)} samples · {X.shape[1]} features · {len(np.unique(y))} classes"
                 + ("" if DATASETS[dataset_name]["two_d"] else " · drawn through its first two principal components"))

# ── step 2: models ───────────────────────────────────────────────────
ui.step(2, "Choose the models", "Green are this library's models, blue the baselines they are measured against. What each one is: the model page.")
m1, m2 = st.columns([3, 2])
with m1:
    st.markdown("**From neural-trees**")
    for name in LIBRARY_MODELS:
        st.checkbox(f"{name}: {MODELS[name]['tag']}", key=f"cmp_{name}")
with m2:
    st.markdown("**Baselines to beat**")
    for name in BASELINE_MODELS:
        st.checkbox(f"{name}: {MODELS[name]['tag']}", key=f"cmp_{name}")
    st.markdown("**Quick select**")
    q1, q2, q3 = st.columns(3)
    q1.button("All", on_click=_pick, args=(list(MODELS),), width="stretch")
    q2.button("Library", on_click=_pick, args=(LIBRARY_MODELS,), width="stretch")
    q3.button("Default", on_click=_pick, args=(DEFAULT_MODELS,), width="stretch")
selected = [m for m in MODELS if st.session_state.get(f"cmp_{m}")]

# ── step 3: run ──────────────────────────────────────────────────────
ui.step(3, "Run")
if len(selected) < 2:
    st.info("Pick at least two models.")
    st.stop()
folds = st.slider("Folds", 3, 10, 5, help="How many parts the data is cut into; each is held out once while the model trains on the rest.")
st.session_state.setdefault("split_seed", 0)
laptop_s, hosted_s = estimate_seconds(selected, folds, len(y))
with st.container(border=True):
    st.markdown(
        f"**{len(selected)} models, {folds} folds, {dataset_name}.** Each model is trained {folds} times "
        f"on {len(y)} samples ({X.shape[1]} features, standardised on each training fold, "
        f"{len(np.unique(y))} classes), once per fold, with its default settings. "
        f"About **{max(1, round(hosted_s))} s** on the hosted app, {max(1, round(laptop_s))} s on a laptop; "
        "a repeat of the same run is instant."
    )
    r1, r2, _ = st.columns([1, 1, 2])
    go_ = r1.button("Run the comparison", type="primary", width="stretch")
    if r2.button("Reshuffle the folds", width="stretch", help="Same data, a different fold assignment; the numbers move, and that movement is fold noise."):
        st.session_state.split_seed += 1
        go_ = True
    run_key = (dataset_name, tuple(selected), folds, st.session_state.split_seed)
    if go_ or "cmp_last" not in st.session_state:
        results = {}
        total, done, t_start = len(selected) * folds, 0, time.time()
        bar = st.progress(0, text="Starting...")
        with st.status(f"Training {len(selected)} models on {folds} folds", expanded=True) as status:
            for name in selected:
                t0, scores = time.time(), []
                try:
                    for k in range(folds):
                        bar.progress(done / total, text=f"{name}: fold {k + 1} of {folds} · {done} of {total} fits done · {time.time() - t_start:.0f} s elapsed")
                        scores.append(fold_score(dataset_name, name, params_key(defaults(name)), folds, st.session_state.split_seed, k))
                        done += 1
                    results[name] = np.array(scores)
                    st.write(f"✓ {name}: {np.mean(scores):.3f} in {time.time() - t0:.1f} s")
                except Exception as e:  # noqa: BLE001 - shown to the user, not hidden
                    results[name] = str(e)
                    done += folds - len(scores)
                    st.write(f"✗ {name} failed: {str(e)[:80]}")
            status.update(label=f"Done: {len(selected)} models, {total} fits, {time.time() - t_start:.0f} s", state="complete", expanded=False)
        bar.empty()
        st.session_state.cmp_last = (run_key, results)
    last_key, results = st.session_state.cmp_last
    if last_key != run_key:
        st.warning(f"The selection changed. The results below are for {last_key[0]} with {len(last_key[1])} models and {last_key[2]} folds; press **Run the comparison** to refresh them.")

# ── results ──────────────────────────────────────────────────────────
ok = {n: s for n, s in results.items() if not isinstance(s, str)}
failed = {n: s for n, s in results.items() if isinstance(s, str)}
if not ok:
    st.error("No model produced a result.")
    st.stop()
res_data, res_folds = last_key[0], last_key[2]
order = sorted(ok, key=lambda n: -ok[n].mean())
best = order[0]
rows = []
for rank, name in enumerate(order, 1):
    s = ok[name]
    if name == best:
        p, reading = np.nan, "highest on these folds"
    else:
        diff = ok[best] - s
        p = float(ttest_rel(ok[best], s).pvalue) if np.ptp(diff) > 0 else 1.0
        reading = "within fold noise of the best" if p >= 0.05 else "behind the best"
    rows.append({
        "rank": rank, "model": name, "group": MODELS[name]["group"],
        "accuracy": f"{s.mean():.3f} ± {s.std():.3f}",
        "gap (points)": f"{(s.mean() - ok[best].mean()) * 100:+.1f}",
        "p vs best": "" if np.isnan(p) else f"{p:.3f}",
        "reading": reading,
    })
within = [r["model"] for r in rows if r["reading"].startswith("within")]
behind = [r["model"] for r in rows if r["reading"].startswith("behind")]

ui.verdict(
    f"Result on {res_data}, {res_folds} folds",
    f"{best} scored highest: {ok[best].mean():.3f} ± {ok[best].std():.3f}",
    (f"Within fold noise of it: <b>{', '.join(within)}</b>; a paired t-test over the folds cannot separate them at 5%. " if within else "")
    + (f"Measurably behind: <b>{', '.join(behind)}</b>. " if behind else "")
    + "The t-test is the quick reading and it is optimistic; the F test in step 4 is the one to trust.",
)
if failed:
    st.warning("Did not run: " + ", ".join(f"**{n}** ({e[:60]})" for n, e in failed.items()))

st.dataframe(
    pd.DataFrame(rows).set_index("rank"), width="stretch", height=48 + 37 * len(rows),
    column_config={
        "model": st.column_config.TextColumn("model", width="medium"),
        "group": st.column_config.TextColumn("group", width="small"),
        "accuracy": st.column_config.TextColumn("accuracy ± sd", width="small"),
        "gap (points)": st.column_config.TextColumn("gap to best", width="small"),
        "p vs best": st.column_config.TextColumn("p vs best", width="small"),
        "reading": st.column_config.TextColumn("reading", width="medium"),
    },
)
fig = go.Figure(go.Bar(
    x=[ok[n].mean() for n in order], y=order, orientation="h",
    error_x=dict(type="data", array=[ok[n].std() for n in order], visible=True, thickness=1, width=4, color="rgba(21,32,43,0.55)"),
    marker_color=[ui.LIBRARY if MODELS[n]["group"] == "neural-trees" else ui.BASELINE for n in order],
    text=[f"{ok[n].mean():.3f}" for n in order], textposition="inside", insidetextanchor="start",
    textfont=dict(color="white", family="IBM Plex Mono, Menlo, monospace"),
))
lo = max(0.0, min(ok[n].mean() - ok[n].std() for n in order) - 0.05)
fig.update_layout(height=50 + 34 * len(order), margin=dict(t=10, b=10, l=10, r=20), xaxis=dict(range=[lo, 1.02], title="accuracy, mean ± sd over folds", gridcolor="rgba(21,32,43,0.08)"),
                  yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, config={"displayModeBar": False})
with st.expander("Why these results, model by model"):
    for name in order:
        st.markdown(f"**{name}** ({ok[name].mean():.3f}): {MODELS[name]['what']} {MODELS[name]['when']}")

# ── step 4: the test ─────────────────────────────────────────────────
ui.step(4, "Is the gap real?", "Optional. The combined 5x2cv F test (Alpaydin, 1999) refits two models on five different 2-fold splits and asks whether the difference survives all of them.")
a_col, b_col = st.columns(2)
a = a_col.selectbox("Model A", order, index=0, key="fa")
b = b_col.selectbox("Model B", [n for n in order if n != a], index=0, key="fb")
if st.button(f"Run the 5x2cv F test: {a} vs {b}", type="primary", width="stretch"):
    with st.spinner("Ten fits of each model..."):
        F, p, reject = f_test(res_data, a, params_key(defaults(a)), b, params_key(defaults(b)))
    m1, m2, m3 = st.columns(3)
    m1.metric("F", f"{F:.3f}")
    m2.metric("p", f"{p:.4f}")
    m3.metric("Different at 5%?", "Yes" if reject else "No")
    if reject:
        st.success(f"**{a}** and **{b}** differ on {res_data}; the gap is not fold noise (p = {p:.4f}).")
    else:
        st.info(f"No evidence that **{a}** and **{b}** differ on {res_data} (p = {p:.4f}). The one that scored higher today may not tomorrow.")

ui.next_step(
    f"See why {best} came out on top: what it does, what its settings change, and what it learned on {res_data}.",
    f"Look inside {best}",
    "views/model.py",
    model_pick=best, model_data=res_data,
)
st.page_link("views/yourdata.py", label="Or run the same comparison on your own CSV", icon="📄")
glossary()
ui.footer()

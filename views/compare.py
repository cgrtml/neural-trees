"""Compare: choose data, choose models, see what will run, run it, read the verdict."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ttest_rel

from playground import (
    BASELINE_MODELS,
    COLORS,
    DATASETS,
    DEFAULT_DATASET,
    DEFAULT_MODELS,
    LIBRARY_MODELS,
    MODELS,
    cv_scores,
    dataset,
    dataset_figure,
    defaults,
    f_test,
    glossary,
    params_key,
)

st.title("Compare")
st.markdown(
    "Three steps. Pick the data, pick the models, run. Every model is trained and "
    "tested on the **same cross-validation folds**, so what you get is a difference "
    "between models, not between splits, and the page says whether that difference "
    "is real or fold noise."
)

# ── step 1: data ─────────────────────────────────────────────────────
st.subheader("Step 1 · Choose the data")
st.session_state.setdefault("cmp_data", DEFAULT_DATASET)
cols = st.columns(len(DATASETS))
for col, (name, info) in zip(cols, DATASETS.items()):
    with col, st.container(border=True):
        X, y, _, _ = dataset(name)
        st.plotly_chart(dataset_figure(name), config={"displayModeBar": False}, key=f"ds_{name}")
        st.markdown(f"**{name}**")
        st.caption(f"{len(y)} samples · {X.shape[1]} features · {len(np.unique(y))} classes")
        st.caption(info["blurb"])
        chosen = st.session_state.cmp_data == name
        if st.button("Selected" if chosen else "Choose", key=f"pick_ds_{name}", type="primary" if chosen else "secondary", width="stretch", disabled=chosen):
            st.session_state.cmp_data = name
            st.rerun()
dataset_name = st.session_state.cmp_data

# ── step 2: models ───────────────────────────────────────────────────
st.subheader("Step 2 · Choose the models")


def _pick(names):
    for m in MODELS:
        st.session_state[f"cmp_{m}"] = m in names


for m in MODELS:
    st.session_state.setdefault(f"cmp_{m}", m in DEFAULT_MODELS)
b1, b2, b3, _ = st.columns([1, 1, 1, 3])
b1.button("All models", on_click=_pick, args=(list(MODELS),), width="stretch")
b2.button("Library only", on_click=_pick, args=(LIBRARY_MODELS,), width="stretch")
b3.button("Default four", on_click=_pick, args=(DEFAULT_MODELS,), width="stretch")

for label, names in (("From neural-trees", LIBRARY_MODELS), ("Baselines to beat", BASELINE_MODELS)):
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    for i, m in enumerate(names):
        with cols[i % 4], st.container(border=True):
            st.checkbox(f"**{m}**", key=f"cmp_{m}")
            st.caption(MODELS[m]["what"])
            st.caption(f"On the model page you can see {MODELS[m]['shows']}.")
selected = [m for m in MODELS if st.session_state.get(f"cmp_{m}")]

# ── step 3: run ──────────────────────────────────────────────────────
st.subheader("Step 3 · Run")
folds = st.slider("Folds", 3, 10, 5, help="How many parts the data is cut into; each is held out once.")
st.session_state.setdefault("split_seed", 0)
X, y, _, _ = dataset(dataset_name)
if len(selected) < 2:
    st.info("Pick at least two models.")
    st.stop()
st.markdown(
    f"**What will happen:** {len(selected)} models ({', '.join(selected)}) will each be trained "
    f"{folds} times on **{dataset_name}** ({len(y)} samples, {X.shape[1]} features standardised "
    f"on each training fold, {len(np.unique(y))} classes), once per fold, with their default "
    "settings. You get: who scored highest, who is within fold noise of them, who is measurably "
    "behind, and a test for any pair."
)
run_key = (dataset_name, tuple(selected), folds, st.session_state.split_seed)
r1, r2, _ = st.columns([1, 1, 3])
go_ = r1.button("Run the comparison", type="primary", width="stretch")
if r2.button("Reshuffle the folds", width="stretch", help="Same data, a different fold assignment; run again and watch the numbers move."):
    st.session_state.split_seed += 1
    run_key = (dataset_name, tuple(selected), folds, st.session_state.split_seed)
    go_ = True
first_visit = "cmp_last" not in st.session_state
if go_ or first_visit:
    results = {}
    bar = st.progress(0, text="Training...")
    for i, name in enumerate(selected):
        bar.progress((i + 1) / len(selected), text=f"Training {name} on {folds} folds...")
        try:
            results[name] = cv_scores(dataset_name, name, params_key(defaults(name)), folds, st.session_state.split_seed)
        except Exception as e:  # noqa: BLE001 - shown to the user, not hidden
            results[name] = str(e)
    bar.empty()
    st.session_state.cmp_last = (run_key, results)

last_key, results = st.session_state.cmp_last
if last_key != run_key:
    st.warning("The selection changed since the last run. Press **Run the comparison** to refresh the results below, which are for "
               f"{last_key[0]} with {len(last_key[1])} models and {last_key[2]} folds.")

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
table = pd.DataFrame(rows).set_index("rank")
within = [r["model"] for r in rows if r["reading"].startswith("within")]
behind = [r["model"] for r in rows if r["reading"].startswith("behind")]

st.divider()
st.subheader(f"Result · {best} scored highest on {res_data}")
st.markdown(
    f"{ok[best].mean():.3f} ± {ok[best].std():.3f} over {res_folds} folds. "
    + (f"Within fold noise of it: **{', '.join(within)}** (a paired t-test over the folds cannot separate them at 5%). " if within else "")
    + (f"Measurably behind: **{', '.join(behind)}**. " if behind else "")
    + "The t-test is the quick reading and it is optimistic, because the folds of one split are not independent; "
    "the F test below is the one to trust."
)
if failed:
    st.warning("Did not run: " + ", ".join(f"**{n}** ({e[:60]})" for n, e in failed.items()))

t1, t2 = st.columns([3, 2])
with t1:
    st.dataframe(table, width="stretch", height=48 + 37 * len(rows))
with t2:
    fig = go.Figure(go.Bar(
        x=[ok[n].mean() for n in order], y=order, orientation="h",
        error_x=dict(type="data", array=[ok[n].std() for n in order], visible=True),
        marker_color=[COLORS[n] for n in order], text=[f"{ok[n].mean():.3f}" for n in order], textposition="outside",
    ))
    lo = max(0.0, min(ok[n].mean() - ok[n].std() for n in order) - 0.05)
    fig.update_layout(height=60 + 40 * len(order), margin=dict(t=10, b=10, l=10, r=40), xaxis=dict(range=[lo, 1.02], title="accuracy, mean ± sd over folds"),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, config={"displayModeBar": False})

with st.expander("Why these results, model by model"):
    for name in order:
        st.markdown(f"**{name}** ({ok[name].mean():.3f}): {MODELS[name]['what']} {MODELS[name]['when']}")

st.subheader("Is the gap real?")
if len(order) >= 2:
    st.markdown(
        "The combined 5x2cv F test (Alpaydin, 1999) refits two models on five different 2-fold "
        "splits and asks whether the difference survives all of them. Slower than the table, and "
        "the test to trust."
    )
    a_col, b_col, go_col = st.columns([2, 2, 1])
    a = a_col.selectbox("Model A", order, index=0, key="fa")
    b = b_col.selectbox("Model B", [n for n in order if n != a], index=0, key="fb")
    go_col.markdown("&nbsp;")
    if go_col.button("Run the F test", width="stretch"):
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

glossary()

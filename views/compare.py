"""Compare: pick a dataset and models, get a verdict, then check whether the gap is real."""

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
    defaults,
    f_test,
    params_key,
)

st.title("Compare")
st.markdown(
    "Every selected model is trained and tested on the **same cross-validation folds** "
    "of one dataset, features standardised on the training fold, default settings. "
    "The table says who scored highest and whether anyone else is close enough that "
    "the difference could be fold noise."
)

left, right = st.columns([1, 2])
with left:
    dataset_name = st.selectbox("Dataset", list(DATASETS), index=list(DATASETS).index(DEFAULT_DATASET))
    st.caption(DATASETS[dataset_name]["blurb"])
    folds = st.slider("Folds", 3, 10, 5)
    st.session_state.setdefault("split_seed", 0)
    if st.button("Reshuffle the folds", help="Same data, a different fold assignment. Watch how much the numbers move."):
        st.session_state.split_seed += 1


def _pick(lib_names, base_names):
    """Button callback: runs before the pills render, so their state may be set."""
    st.session_state.cmp_lib, st.session_state.cmp_base = list(lib_names), list(base_names)


with right:
    st.session_state.setdefault("cmp_lib", [m for m in DEFAULT_MODELS if m in LIBRARY_MODELS])
    st.session_state.setdefault("cmp_base", [m for m in DEFAULT_MODELS if m in BASELINE_MODELS])
    lib = st.pills("From neural-trees", LIBRARY_MODELS, selection_mode="multi", key="cmp_lib")
    base = st.pills("Baselines", BASELINE_MODELS, selection_mode="multi", key="cmp_base")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button("All models", on_click=_pick, args=(LIBRARY_MODELS, BASELINE_MODELS))
    b2.button("Library only", on_click=_pick, args=(LIBRARY_MODELS, []))
    b3.button("Default four", on_click=_pick, args=([m for m in DEFAULT_MODELS if m in LIBRARY_MODELS], [m for m in DEFAULT_MODELS if m in BASELINE_MODELS]))
selected = [n for n in MODELS if n in (lib or []) + (base or [])]

if len(selected) < 2:
    st.info("Pick at least two models to compare.")
    st.stop()

results = {}
bar = st.progress(0, text="Training...")
for i, name in enumerate(selected):
    bar.progress((i + 1) / len(selected), text=f"Training {name}...")
    try:
        results[name] = cv_scores(dataset_name, name, params_key(defaults(name)), folds, st.session_state.split_seed)
    except Exception as e:  # noqa: BLE001 - shown to the user, not hidden
        results[name] = str(e)
bar.empty()

ok = {n: s for n, s in results.items() if not isinstance(s, str)}
failed = {n: s for n, s in results.items() if isinstance(s, str)}
if not ok:
    st.error("No model produced a result.")
    st.stop()

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
st.subheader(f"{best} scored highest on {dataset_name}")
st.markdown(
    f"{ok[best].mean():.3f} ± {ok[best].std():.3f} over {folds} folds. "
    + (f"**{', '.join(within)}**: within fold noise, a paired t-test over the folds cannot separate them at 5%. " if within else "")
    + (f"**{', '.join(behind)}**: measurably behind. " if behind else "")
    + "That t-test is the quick reading; the folds of one split are not independent, so it is optimistic. "
    "The button below runs the test this library recommends."
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
    fig.update_layout(height=60 + 40 * len(order), margin=dict(t=10, b=10, l=10, r=40), xaxis=dict(range=[lo, 1.02], title="accuracy"),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, config={"displayModeBar": False})

st.divider()
st.subheader("Is the gap real?")
if len(order) >= 2:
    runner_up = order[1]
    st.markdown(
        f"The combined 5x2cv F test (Alpaydin, 1999) refits **{best}** and **{runner_up}** on five "
        "different 2-fold splits and asks whether the difference survives all of them. It is slower "
        "than the table, and it is the test to trust."
    )
    a = st.selectbox("Model A", order, index=0, key="fa")
    b = st.selectbox("Model B", [n for n in order if n != a], index=0, key="fb")
    if st.button("Run the 5x2cv F test"):
        with st.spinner("Ten fits of each model..."):
            F, p, reject = f_test(dataset_name, a, params_key(defaults(a)), b, params_key(defaults(b)))
        m1, m2, m3 = st.columns(3)
        m1.metric("F", f"{F:.3f}")
        m2.metric("p", f"{p:.4f}")
        m3.metric("Different at 5%?", "Yes" if reject else "No")
        if reject:
            st.success(f"**{a}** and **{b}** differ on {dataset_name}; the gap is not fold noise (p = {p:.4f}).")
        else:
            st.info(f"No evidence that **{a}** and **{b}** differ on {dataset_name} (p = {p:.4f}). The one that scored higher today may not tomorrow.")

with st.expander("What exactly is being compared"):
    st.markdown(
        "- One dataset, standardised on each training fold, never on the test fold.\n"
        "- Stratified k-fold cross-validation; **every model sees the same folds**, so a difference is a difference between models, not between splits. *Reshuffle the folds* changes the assignment for all of them at once.\n"
        "- Default settings for every model, listed on *How a model works*. Nothing is tuned here, for any model.\n"
        "- The paired t-test in the table compares two models fold by fold. The 5x2cv F test repeats a 2-fold split five times, which is what makes its variance estimate honest."
    )

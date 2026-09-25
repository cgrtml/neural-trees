"""Start here: what the library is, and what each model does, seen rather than described."""

import streamlit as st

from playground import (
    BASELINE_MODELS,
    DATASETS,
    LIBRARY_MODELS,
    MODELS,
    boundary_figure,
    glossary,
)

st.title("neural-trees")
st.markdown(
    "A scikit-learn compatible library of **tree-shaped models that train by gradient "
    "descent**: soft decision trees, multivariate and omnivariate trees, a hierarchical "
    "mixture of experts and a network that grows while it learns, plus the statistical "
    "test that says whether one classifier is really better than another. The algorithms "
    "come from Ethem Alpaydin's research group; the implementations, the fixes and the "
    "measurements are this project's."
)

st.markdown("**How to use this site**")
c1, c2, c3, c4 = st.columns(4)
with c1, st.container(border=True):
    st.markdown("**1 · See what each model does**")
    st.caption("Below on this page: the boundary every model learns on the same data. The shape is the model.")
with c2, st.container(border=True):
    st.markdown("**2 · Compare them**")
    st.caption("Choose data and models, run, and read who wins and whether the gap is fold noise.")
    st.page_link("views/compare.py", label="Go to Compare", icon="🏁")
with c3, st.container(border=True):
    st.markdown("**3 · Look inside one**")
    st.caption("One model: what it does, when to use it, its knobs on a live boundary, what it learned.")
    st.page_link("views/model.py", label="Go to How a model works", icon="🔍")
with c4, st.container(border=True):
    st.markdown("**4 · Check the claims**")
    st.caption("What this library fixed and verified, and how it stands against XGBoost and friends.")
    st.page_link("views/verified.py", label="What was fixed", icon="✅")
    st.page_link("views/field.py", label="Against the field", icon="🏟️")

st.divider()
st.subheader("The same problem, ten models")
picked = st.radio("Data", [n for n in DATASETS if DATASETS[n]["two_d"]], horizontal=True, key="start_data", label_visibility="collapsed")
st.caption(DATASETS[picked]["blurb"] + " Every model below is fitted on it with its default settings; the number is training accuracy.")

for label, names in (("From neural-trees", LIBRARY_MODELS), ("Baselines they are measured against", BASELINE_MODELS)):
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    for i, name in enumerate(names):
        with cols[i % 4], st.container(border=True):
            with st.spinner(f"Fitting {name}..."):
                fig, acc = boundary_figure(picked, name, height=190)
            st.markdown(f"**{name}**")
            st.plotly_chart(fig, config={"displayModeBar": False}, key=f"thumb_{name}")
            st.caption(f"train accuracy {acc:.2f}")
            st.write(MODELS[name]["what"])

st.divider()
st.markdown(
    "Everything on these pages is computed live from the data shown; nothing is typed "
    "in. The larger comparison with XGBoost, LightGBM, GRANDE and NODE on 24 datasets, "
    "which takes hours, is on *Against the field*."
)
glossary(["decision boundary", "standardised"])

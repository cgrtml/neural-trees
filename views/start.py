"""Start here: what the library is, and what each model does, seen rather than described."""

import streamlit as st

from playground import (
    BASELINE_MODELS,
    DATASETS,
    LIBRARY_MODELS,
    MODELS,
    boundary_figure,
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

c1, c2, c3 = st.columns(3)
c1.markdown("**1. See what each model does**\n\nBelow: the boundary every model learns on the same two-dimensional data. The shape is the model.")
c2.markdown("**2. Compare them on real data**\n\n*Compare* runs your selection on one dataset with the same folds for every model and says whether the gaps are real or fold noise.")
c3.markdown("**3. Look inside one**\n\n*How a model works* explains one model, lets you turn its knobs, and shows what it learned: rules, gates, growth.")

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

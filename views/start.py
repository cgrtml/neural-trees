"""Start here: a short lesson for someone who has never seen a decision tree, ending at the models."""

import streamlit as st

from playground import (
    BASELINE_MODELS,
    DATASETS,
    LIBRARY_MODELS,
    MODELS,
    boundary_figure,
    glossary,
    ui,
)

ui.title(
    "neural-trees",
    lead="Decision trees you can read, trained the way neural networks are. This site shows what "
         "that means, lets you compare the models on real data, and shows what each one learned.",
    eyebrow="A scikit-learn compatible library",
)

# ── the lesson ───────────────────────────────────────────────────────
st.subheader("Three pictures, then you know what the library is about")
st.markdown(
    "A classifier draws a boundary between classes. Below, the same two-class data "
    "(two crescents, 500 points) and the boundary three models learn on it. The shape "
    "of the boundary is the model."
)
lesson = [
    ("CART (sklearn)", "1 · A classic decision tree (CART)",
     "Asks one question at a time: *is x1 above 0.3?* Each answer is a horizontal or vertical cut, so the boundary is a staircase."),
    ("Multivariate Tree", "2 · A tree that cuts diagonally (multivariate)",
     "Each question is about a weighted sum of the features, so one cut can be a diagonal line. Fewer cuts, same idea."),
    ("Soft Decision Tree", "3 · A tree whose cuts are soft (soft decision tree)",
     "Instead of yes or no, each question answers with a probability. The boundary bends, and the whole tree can be trained by gradient descent, like a neural network."),
]
for col, (model, heading, text) in zip(st.columns(3), lesson):
    with col, st.container(border=True):
        fig, acc = boundary_figure("Moons", model, height=230)
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"lesson_{model}")
        st.markdown(f"**{heading}**")
        st.write(text)
        st.caption(f"Training accuracy {acc:.2f}")

st.markdown(
    "That third idea is what this library is built around. The soft tree keeps what a tree "
    "gives you, a structure you can print as rules and a path you can follow for any single "
    "prediction, while gaining what neural networks have: a smooth boundary and training by "
    "gradient descent. The library also implements the trees in between, a mixture of "
    "specialist networks arranged as a tree, a network that grows while it trains, and the "
    "statistical test that says whether one model is really better than another."
)

ui.next_step(
    "Now test that on real data: run these three trees against a random forest on Wine, "
    "the same folds for all four, and see whether the soft tree's smooth boundary buys "
    "anything there.",
    "Try it: compare the three trees on Wine",
    "views/compare.py",
    cmp_data="Wine", cmp_reset=["CART (sklearn)", "Multivariate Tree", "Soft Decision Tree", "Random Forest"],
)

# ── what to do next ──────────────────────────────────────────────────
st.subheader("What you can do here")
c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("**Compare models on a dataset**")
    st.write("Pick data and models, press run. You get who scored highest, who is within noise of them, and a test for whether a gap is real.")
    st.page_link("views/compare.py", label="Compare", icon="🏁")
with c2, st.container(border=True):
    st.markdown("**See how one model works**")
    st.write("What it does, when to use it, how it works, its settings on a live boundary, and what it learned: rules, a prediction explained, its growth.")
    st.page_link("views/model.py", label="How a model works", icon="🔍")
with c3, st.container(border=True):
    st.markdown("**Check the claims**")
    st.write("Four models that did not work and were fixed, the scikit-learn checks every model passes, and a 24-dataset comparison with XGBoost, LightGBM, GRANDE and NODE.")
    st.page_link("views/verified.py", label="What was fixed and verified", icon="✅")
    st.page_link("views/field.py", label="Against the field", icon="🏟️")

# ── all the models ───────────────────────────────────────────────────
st.subheader("Every model in the box, on the same data")
picked = st.radio("Data", [n for n in DATASETS if DATASETS[n]["two_d"]], horizontal=True, key="start_data", label_visibility="collapsed")
st.caption(DATASETS[picked]["blurb"] + " Default settings for every model; the number is training accuracy.")
for label, names in (("From neural-trees", LIBRARY_MODELS), ("Baselines they are measured against", BASELINE_MODELS)):
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    for i, name in enumerate(names):
        with cols[i % 4], st.container(border=True):
            with st.spinner(f"Fitting {name}..."):
                fig, acc = boundary_figure(picked, name, height=180)
            st.markdown(f"**{name}** &nbsp;{ui.badge(MODELS[name]['group'])}", unsafe_allow_html=True)
            st.plotly_chart(fig, config={"displayModeBar": False}, key=f"thumb_{name}")
            st.caption(f"train accuracy {acc:.2f} · {MODELS[name]['what']}")

ui.next_step(
    "Pick any model above and look inside it: what it does, its settings on a live boundary, "
    "and what it learned.",
    "Look inside the soft decision tree",
    "views/model.py",
    model_pick="Soft Decision Tree",
)

st.divider()
st.markdown(
    "Everything on these pages is computed live from the data shown; nothing is typed in. "
    "The algorithms come from Ethem Alpaydin's research group; the implementations, the fixes "
    "and the measurements are this project's."
)
glossary(["decision boundary", "standardised"])

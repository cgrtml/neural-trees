"""Start here: a short lesson for someone who has never seen a decision tree, ending at the models."""

import streamlit as st

from playground import DATASETS, MODELS, boundary_figure, glossary, ui

ui.title(
    "neural-trees",
    lead="Decision trees you can read, trained the way neural networks are. This site shows what "
         "that means, lets you compare the models on real data, and shows what each one learned.",
    eyebrow="A scikit-learn compatible library",
)

# ── the lesson: three uniform cards ──────────────────────────────────
st.subheader("Three pictures, then you know what the library is about")
st.markdown(
    "A classifier draws a boundary between classes. Below, the same two-class data "
    "(two crescents, 500 points) and the boundary three models learn on it. The shape "
    "of the boundary is the model."
)
lesson = [
    ("CART (sklearn)", "1 · A classic decision tree",
     "One question at a time, one feature per question. Each answer is a horizontal or vertical cut, so the boundary is a staircase."),
    ("Multivariate Tree", "2 · A tree that cuts diagonally",
     "Each question weighs several features at once, so one cut can be a diagonal line. Fewer cuts for the same idea, and a smaller tree."),
    ("Soft Decision Tree", "3 · A tree whose cuts are soft",
     "Each question answers with a probability rather than yes or no. The boundary bends, and the whole tree trains like a neural network."),
]
for col, (model, heading, text) in zip(st.columns(3), lesson):
    with col, st.container(border=True):
        fig, acc = boundary_figure("Moons", model, height=240)
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"lesson_{model}")
        ui.card_text(heading, text, stat=f"training accuracy {acc:.2f}", lines=4, badge_line=False)

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

# ── what to do next: three uniform cards ─────────────────────────────
st.subheader("What you can do here")
todo = [
    ("Compare models on a dataset", "Pick data and models, press run. Who scored highest, who is within noise of them, and whether a gap is real.", "views/compare.py", "Compare", "🏁"),
    ("Try it on your own data", "Upload a CSV, pick the column to predict, run the same flow, read the soft tree's rules and take the model away as a file.", "views/yourdata.py", "Try it on your data", "📄"),
    ("See how one model works", "What it does, when to use it, its settings on a live boundary, and what it learned: rules, growth, an explained prediction.", "views/model.py", "How a model works", "🔍"),
    ("Check the claims", "The four models that were fixed, the scikit-learn checks every model passes, and the 24-dataset benchmark against XGBoost.", "views/verified.py", "What was fixed and verified", "✅"),
]
for col, (title, text, page, label, icon) in zip(st.columns(4), todo):
    with col, st.container(border=True):
        ui.card_text(title, text, lines=4, badge_line=False)
        st.page_link(page, label=label, icon=icon)

# ── every model, one grid ────────────────────────────────────────────
st.subheader("Every model in the box, on the same data")
picked = st.radio("Data", [n for n in DATASETS if DATASETS[n]["two_d"]], horizontal=True, key="start_data", label_visibility="collapsed")
st.caption(DATASETS[picked]["blurb"] + " Default settings for every model; green is this library, blue a baseline.")
names = list(MODELS)
for row in range(0, len(names), 5):
    for col, name in zip(st.columns(5), names[row:row + 5]):
        with col, st.container(border=True):
            with st.spinner(f"Fitting {name}..."):
                fig, acc = boundary_figure(picked, name, height=170)
            st.plotly_chart(fig, config={"displayModeBar": False}, key=f"thumb_{name}")
            ui.card_text(name, MODELS[name]["tag"], stat=f"accuracy {acc:.2f}", group=MODELS[name]["group"])

ui.next_step(
    "Pick any model and look inside it: what it does, its settings on a live boundary, and what it learned.",
    "Look inside the soft decision tree",
    "views/model.py",
    model_pick="Soft Decision Tree",
)
glossary(["decision boundary", "standardised"])
ui.footer()

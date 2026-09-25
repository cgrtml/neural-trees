"""
The neural-trees playground.
Run: streamlit run app.py

Five pages, in the order a visitor asks the questions: what each model does
(seen on the same data), how they compare on a dataset and whether a gap is
real, how one model works inside, what this library fixed and verified, and
how the models stand against XGBoost, LightGBM, GRANDE and NODE.

The registry of models and every cached fit live in `playground/`; the pages
in `views/` only render.
"""

import streamlit as st

from playground import ui

st.set_page_config(page_title="neural-trees playground", page_icon="🌳", layout="wide")

pages = [
    st.Page("views/start.py", title="Start here", icon="🌳", default=True),
    st.Page("views/compare.py", title="Compare", icon="🏁"),
    st.Page("views/model.py", title="How a model works", icon="🔍"),
    st.Page("views/verified.py", title="What was fixed and verified", icon="✅"),
    st.Page("views/field.py", title="Against the field", icon="🏟️"),
]
nav = st.navigation(pages, position="top")
ui.inject()
with st.sidebar:
    st.markdown("**neural-trees**")
    st.caption(
        "Tree-shaped models that train by gradient descent, scikit-learn compatible. "
        "[GitHub](https://github.com/cgrtml/neural-trees) · "
        "[Docs](https://cagritemel.com/neural-trees/) · "
        "[PyPI](https://pypi.org/project/neural-trees/)"
    )
    st.caption("Built by [Cagri Temel](https://github.com/cgrtml)")
nav.run()

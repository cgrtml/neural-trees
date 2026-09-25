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

import os

# One thread each. The hosted app has one or two vCPUs, and torch and
# scikit-learn's OpenMP pools fighting over them made a 15-second run take
# minutes; the models here are small enough that one thread is the fast path.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import streamlit as st  # noqa: E402

from playground import ui  # noqa: E402

st.set_page_config(page_title="neural-trees playground", page_icon="🌳", layout="wide", initial_sidebar_state="collapsed")

pages = [
    st.Page("views/start.py", title="Start here", icon="🌳", default=True),
    st.Page("views/compare.py", title="Compare", icon="🏁"),
    st.Page("views/yourdata.py", title="Try it on your data", icon="📄"),
    st.Page("views/model.py", title="How a model works", icon="🔍"),
    st.Page("views/verified.py", title="What was fixed and verified", icon="✅"),
    st.Page("views/field.py", title="Against the field", icon="🏟️"),
]
nav = st.navigation(pages, position="top")
ui.inject()
nav.run()

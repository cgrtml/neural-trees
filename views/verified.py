"""What was fixed and verified: the part of the library that is this project's own work."""

import pandas as pd
import streamlit as st

from playground import ui

ui.title(
    "What was fixed, and how it was checked",
    lead="The algorithms come from published papers. What this library adds is the part that makes "
         "them trustworthy: four models that did not work were found and fixed, every estimator is "
         "held to scikit-learn's contract, and every design choice the papers leave open was measured.",
    eyebrow="The library's own contribution",
)

st.subheader("Four models that did not work")
st.markdown(
    "An early release shipped these, and none of them announced it. They were found "
    "by writing tests, not by reading the code; each has a regression test that fails "
    "against the old code."
)
st.dataframe(pd.DataFrame([
    ("Hierarchical MoE", "Could not take a single gradient step: child weights were written in place into a tensor autograd had saved, so every backward() raised.", "Trains; 0.977 on Breast Cancer."),
    ("Omnivariate Tree", "Computed each node's decision, discarded it and always went right. Every sample reached the same leaf: 0.000 on Iris.", "Routes on the decision; 0.971 on Breast Cancer at depth 3."),
    ("GAL Network", "Took one full-batch step per epoch, so growth and pruning decided on a network that had barely moved. Chance (0.333) on three separable blobs.", "Mini-batch training; 0.952 on Iris, 0.978 on Breast Cancer."),
    ("Condensed nearest neighbour", "A single pass over the training set, leaving a prototype store that did not classify that set correctly.", "Sweeps until consistent."),
    ("Multivariate and Omnivariate trees (0.7.0)", "Closed a node that still held every class whenever one class at the node was rare: a one-sample class founded its own group and the split was refused.", "Digits at depth 6: 0.35 to 0.93 and 0.49 to 0.95."),
], columns=["model", "what was wrong", "after"]).set_index("model"), width="stretch")

st.subheader("Held to scikit-learn's contract")
st.markdown(
    "Every estimator runs `sklearn.utils.estimator_checks.check_estimator`: 55 checks, "
    "63 for the ones that accept `sample_weight`. The one check some fail requires "
    "weighting a row to be *identical* to repeating it, which is false for a "
    "mini-batch learner (the repeated dataset is batched differently) and for a "
    "k-nearest rule (a repeated row fills several of the k slots). Each docstring says "
    "which check and why. 373 tests in total, on Python 3.9 to 3.13, in CI."
)
st.dataframe(pd.DataFrame([
    ("Soft Decision Tree", "62 / 63", "row repetition (mini-batch)"),
    ("Soft Decision Tree Regressor", "regressor checks, one exception", "row repetition (mini-batch)"),
    ("Multivariate Tree", "63 / 63", ""),
    ("Omnivariate Tree", "62 / 63", "row repetition (two candidate splitters take no weights)"),
    ("Hierarchical MoE", "62 / 63", "row repetition (mini-batch)"),
    ("GAL Network", "62 / 63", "row repetition (mini-batch)"),
    ("Weighted KNN", "61 / 63", "row repetition, dense and sparse (k-nearest rule)"),
    ("Naive Bayes", "63 / 63", ""),
], columns=["estimator", "checks passed", "the exception"]).set_index("estimator"), width="stretch")

st.subheader("Design choices that were measured, not assumed")
st.markdown(
    "- **Growing a soft tree.** The obvious way, giving both new children the parent's "
    "distribution so the function is unchanged, leaves the new gate's gradient exactly "
    "zero: the added level can never learn. Proved, measured (up to 40 points lost on "
    "multi-class data), fixed with any perturbation of the children. The same defect "
    "was then found in the per-leaf growth and fixed there too.\n"
    "- **New units in GAL** are fitted to the residual error before they are installed: "
    "smaller networks on every dataset tried, not more accurate ones, and on Digits "
    "less accurate. Documented as a parsimony choice.\n"
    "- **Requiring statistical significance** before a node gets a more expressive split "
    "makes the omnivariate tree larger for the same or lower accuracy. Off by default.\n"
    "- **Stopping GAL on validation loss** under-grows: 2 to 7 points lost on six "
    "datasets. Off by default.\n"
    "- **Three rules for reading a soft tree as a hard one** were measured for agreement "
    "and cost; none won on both, so the default stayed.\n"
    "- **Calibration.** The soft tree's probabilities are as calibrated as logistic "
    "regression's and far better than a random forest's (expected calibration error "
    "0.023 / 0.052 / 0.030 on Iris / Wine / Breast Cancer against 0.038 / 0.098 / 0.201)."
)
st.markdown(
    "The numbers, the scripts that produced them and the protocol are on the "
    "[design decisions](https://cagritemel.com/neural-trees/design_decisions.html) page; "
    "the growth result is the subject of the accompanying paper. The README benchmark "
    "table is re-run in CI and fails when a cell drifts, which happened once."
)

ui.footer()

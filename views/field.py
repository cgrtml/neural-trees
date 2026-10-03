"""Against the field: the offline benchmark against XGBoost, LightGBM, GRANDE and NODE."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from playground import ui

ui.title(
    "Against the field",
    lead="Nine models on 24 datasets, three seeds of five-fold cross-validation, nothing tuned: "
         "one fixed configuration per model everywhere. Measured offline, because it takes hours; "
         "this page reads the output.",
    eyebrow="XGBoost, LightGBM, GRANDE, NODE",
)


@st.cache_data(show_spinner=False)
def _load():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "rakipler-sonuc.json"
    S = json.loads(path.read_text(encoding="utf-8"))
    models = S["_modeller"]
    rows = []
    for name, R in S.items():
        if name.startswith("_"):
            continue
        row = {"dataset": name, "n": R["n"], "K": R["K"]}
        for m in models:
            row[m] = R[m]["acc"] if "acc" in R[m] else np.nan
        rows.append(row)
    return models, pd.DataFrame(rows)


models, field = _load()
subset = st.radio("Datasets", ["all", "n ≤ 1000", "n > 1000", "binary", "multi-class"], horizontal=True, key="field_subset")
mask = {"all": field["n"] > 0, "n ≤ 1000": field["n"] <= 1000, "n > 1000": field["n"] > 1000,
        "binary": field["K"] == 2, "multi-class": field["K"] > 2}[subset]
sub = field[mask].reset_index(drop=True)
acc = sub[models]
ranks = acc.rank(axis=1, ascending=False)
diffs = {m: (acc[m] - acc["XGBoost"]) * 100 for m in models}
summary = pd.DataFrame({
    "mean accuracy": acc.mean().round(3),
    "mean rank": ranks.mean().round(2),
    "vs XGBoost (points)": pd.Series({m: d.mean() for m, d in diffs.items()}).round(2),
    "wins / ties / losses vs XGBoost": pd.Series({m: f"{int((d > 0.5).sum())} / {int((d.abs() <= 0.5).sum())} / {int((d < -0.5).sum())}" for m, d in diffs.items()}),
}).sort_values("mean rank")

st.subheader(f"{len(sub)} datasets, best rank first")
st.dataframe(summary, width="stretch")

if subset == "n ≤ 1000":
    st.success(
        "On small tables the order reverses: **GAL beats untuned XGBoost on every one of "
        "the eight datasets and the per-leaf soft tree on seven**. The MLP wins there too, "
        "so the finding is that smooth gradient-trained models beat untuned boosting with "
        "a few hundred rows; what the soft tree adds is that it can be read and explained."
    )
elif subset == "n > 1000":
    st.info("Above 1 000 rows XGBoost, LightGBM and Random Forest win almost every comparison against the soft models. With thousands of rows and accuracy as the only criterion, use LightGBM.")
else:
    st.info(
        "Over everything, the boosted ensembles, Random Forest and an MLP lead and the soft "
        "models sit a point or two behind; NODE, at these defaults, is last. Switch to "
        "**n ≤ 1000** for the part that is interesting."
    )

st.markdown("**Accuracy per dataset** (best in bold)")
shown = sub.set_index("dataset")
st.dataframe(
    shown.style.highlight_max(axis=1, subset=models, props="font-weight: bold; background-color: #fff6d5;").format({m: "{:.3f}" for m in models}),
    width="stretch", height=min(60 + 36 * len(shown), 900),
)
st.caption(
    "The 1 000-row split was chosen after seeing the data and the summary means are not a "
    "hypothesis test; the Wilcoxon tests and the exact configurations are on the "
    "[documentation page](https://cagritemel.com/neural-trees/benchmarks.html)."
)

st.subheader("Does averaging soft trees close the gap?")
st.markdown(
    "The fair objection to the large-data result is that it pits one soft tree against 300 boosted ones. "
    "`benchmarks/soft_forest.py` bags **25 per-leaf soft trees** on the sixteen datasets above 1 000 rows, same folds and seeds, nothing tuned."
)
sf_path = Path(__file__).resolve().parent.parent / "benchmarks" / "soft_forest-sonuc.json"
if sf_path.exists():
    SF = json.loads(sf_path.read_text(encoding="utf-8"))
    sf_rows = [{"dataset": k, "n": v["n"], "K": v["K"], "soft forest (25 trees)": v["acc"], "per-leaf soft tree": v["SoftTree per-leaf"], "XGBoost": v["XGBoost"], "forest s/fit": v["fit_sn"]}
               for k, v in SF.items() if not k.startswith("_")]
    sf = pd.DataFrame(sf_rows).sort_values(["K", "dataset"]).set_index("dataset")
    st.dataframe(sf.style.highlight_max(axis=1, subset=["soft forest (25 trees)", "per-leaf soft tree", "XGBoost"], props="font-weight: bold; background-color: #fff6d5;")
                 .format({c: "{:.3f}" for c in ["soft forest (25 trees)", "per-leaf soft tree", "XGBoost"]} | {"forest s/fit": "{:.0f}"}),
                 width="stretch", height=60 + 36 * len(sf))
    gap_t = 100 * (sf["per-leaf soft tree"] - sf["XGBoost"]).mean()
    gap_f = 100 * (sf["soft forest (25 trees)"] - sf["XGBoost"]).mean()
    st.info(
        f"Averaging closes about half the gap: the single tree is {abs(gap_t):.1f} points behind XGBoost on average, the forest {abs(gap_f):.1f} "
        f"(wins, ties, losses 4/6/6, Wilcoxon p = 0.32). The forest beats the single tree on nine datasets and loses on none. "
        "The price is the point of a soft tree: no single path to read, no rule list, no counterfactual, and about 160 times XGBoost's fit time. "
        "If explanation does not matter, use LightGBM; if it does, use one tree and accept the two points."
    )

ui.footer()

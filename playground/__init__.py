"""
Shared registry and cached work for the Streamlit playground.

One place says what every model is, when to use it, how it works and how
to build it; the views only render. Every fit is cached by its inputs and
shared across sessions, so the default pages are computed once for everyone
rather than once per visitor.
"""

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from neural_trees import (
    GALNetwork,
    HierarchicalMixtureOfExperts,
    MultivariateDecisionTree,
    NaiveBayesClassifier,
    OmnivariateDecisionTree,
    SoftDecisionTree,
    WeightedKNN,
    combined_5x2cv_f_test,
)

# ── datasets ─────────────────────────────────────────────────────────


def _synthetic(func, **kwargs):
    X, y = func(n_samples=500, random_state=42, **kwargs)
    return X, y, ["x1", "x2"], [str(c) for c in np.unique(y)]


def _sklearn(loader):
    d = loader()
    return d.data, d.target, [str(f) for f in d.feature_names], [str(t) for t in d.target_names]


DATASETS = {
    "Moons": dict(load=lambda: _synthetic(make_moons, noise=0.25), blurb="Two interleaving crescents. 2 features, 500 samples. The boundary has to bend.", two_d=True),
    "Circles": dict(load=lambda: _synthetic(make_circles, noise=0.15, factor=0.5), blurb="A ring around a disc. 2 features, 500 samples. No single line separates them.", two_d=True),
    "Iris": dict(load=lambda: _sklearn(load_iris), blurb="Three flower species from four measurements. 150 samples; tiny, so numbers move between folds.", two_d=False),
    "Wine": dict(load=lambda: _sklearn(load_wine), blurb="Three cultivars from 13 chemical measurements. 178 samples, nearly separable.", two_d=False),
    "Breast Cancer": dict(load=lambda: _sklearn(load_breast_cancer), blurb="Malignant or benign from 30 cell measurements. 569 samples, two classes.", two_d=False),
}
DEFAULT_DATASET = "Iris"

# ── models ───────────────────────────────────────────────────────────

MODELS = {
    "Soft Decision Tree": dict(
        group="neural-trees",
        what="A decision tree whose splits are smooth. Every sample reaches every leaf with some probability, so the boundary bends instead of stepping.",
        when="You want a tree you can read but a boundary that is not a staircase, and you want to explain single predictions.",
        how="Each internal node is a sigmoid gate on a weighted sum of the features; a sample's prediction is the mixture of the leaves weighted by the probability of reaching them. Because everything is differentiable, the whole tree trains by gradient descent, like a network.",
        ref="Irsoy, Yildiz and Alpaydin, ICPR 2012",
        verified="62 of 63 scikit-learn checks; the exception is the one no mini-batch learner passes.",
        knobs=[dict(param="depth", label="Depth", kind="slider", lo=1, hi=6, default=4),
               dict(param="max_epochs", label="Epochs", kind="slider", lo=10, hi=120, default=40, step=10)],
        build=lambda p, s: SoftDecisionTree(depth=p.get("depth", 4), max_epochs=p.get("max_epochs", 40), random_state=s),
    ),
    "Multivariate Tree": dict(
        group="neural-trees",
        what="A hard decision tree whose splits are lines through many features at once, not thresholds on one.",
        when="Features are correlated and a single oblique cut does what CART needs a staircase of cuts to do.",
        how="At each node the classes are grouped into two sides and a linear discriminant is fitted; its hyperplane is the split. Children are built the same way until the node is pure or too small.",
        ref="Alpaydin and Cetin 1995; Yildiz and Alpaydin, IEEE TNN 2001",
        verified="63 of 63 scikit-learn checks.",
        knobs=[dict(param="max_depth", label="Max depth", kind="slider", lo=1, hi=8, default=3)],
        build=lambda p, s: MultivariateDecisionTree(max_depth=p.get("max_depth", 3), random_state=s),
    ),
    "Omnivariate Tree": dict(
        group="neural-trees",
        what="A tree that picks, at every node, whether a single-feature split, a linear split or a small network split fits that node's data best.",
        when="You do not know the shape of the boundary in advance and want the tree to decide, node by node.",
        how="Three candidate splitters are cross-validated on the node's data and the most accurate wins (or, with selection='test', the simplest that is not significantly worse). The chosen splitter routes the samples and the children repeat the process.",
        ref="Yildiz and Alpaydin, IEEE TNN 2001",
        verified="62 of 63 scikit-learn checks; two of the three candidate splitters take no sample_weight.",
        knobs=[dict(param="max_depth", label="Max depth", kind="slider", lo=1, hi=6, default=3)],
        build=lambda p, s: OmnivariateDecisionTree(max_depth=p.get("max_depth", 3), random_state=s),
    ),
    "Hierarchical MoE": dict(
        group="neural-trees",
        what="A tree of gates that route regions of the input to specialist networks, trained end to end.",
        when="The data has distinct regimes, each better served by its own model, and a single global rule underfits.",
        how="Internal nodes are gating networks and leaves are expert networks; the output is the gate-weighted mixture of the experts. Subtree dropout switches off whole branches during training so no expert can be leaned on entirely.",
        ref="Irsoy and Alpaydin, Neurocomputing 2021",
        verified="62 of 63 scikit-learn checks; the mini-batch exception.",
        knobs=[dict(param="depth", label="Depth", kind="slider", lo=1, hi=3, default=2),
               dict(param="max_epochs", label="Epochs", kind="slider", lo=20, hi=100, default=50, step=10)],
        build=lambda p, s: HierarchicalMixtureOfExperts(depth=p.get("depth", 2), max_epochs=p.get("max_epochs", 50), random_state=s),
    ),
    "GAL Network": dict(
        group="neural-trees",
        what="A neural network that starts tiny, adds hidden units while it trains and prunes the ones that stop earning their place.",
        when="You do not know how big the network should be and would rather the data decide than a grid search.",
        how="Training proceeds in rounds; when the error stops falling a unit is added, fitted to what the network still gets wrong before it is installed; a unit whose removal does not hurt is pruned.",
        ref="Alpaydin, IJPRAI 1994",
        verified="62 of 63 scikit-learn checks; the mini-batch exception.",
        knobs=[dict(param="max_epochs", label="Epochs", kind="slider", lo=20, hi=200, default=80, step=20),
               dict(param="max_hidden", label="Max hidden units", kind="slider", lo=5, hi=60, default=30, step=5)],
        build=lambda p, s: GALNetwork(max_epochs=p.get("max_epochs", 80), max_hidden=p.get("max_hidden", 30), random_state=s),
    ),
    "Weighted KNN": dict(
        group="neural-trees",
        what="No training: a prediction is the vote of the nearest stored examples, closer ones counting more.",
        when="Small, clean data where similar inputs really do have similar labels.",
        how="Distances to every stored sample are computed, the k nearest vote with weight 1/distance^p, and an optional condensing step keeps only the examples needed to classify the training set.",
        ref="Cover and Hart 1967; Hart 1968 (condensing)",
        verified="61 of 63 scikit-learn checks; weighting a row is not repeating it for a k-nearest rule, and the docstring says why.",
        knobs=[dict(param="k", label="Neighbours (k)", kind="slider", lo=1, hi=25, default=5)],
        build=lambda p, s: WeightedKNN(k=p.get("k", 5)),
    ),
    "Naive Bayes": dict(
        group="neural-trees",
        what="A probabilistic baseline that assumes the features are independent given the class.",
        when="A fast, hard-to-break reference point; often strong on high-dimensional counts such as text.",
        how="Each class gets a mean and variance per feature; a prediction multiplies the per-feature likelihoods with the class prior and picks the largest posterior.",
        ref="Textbook (Alpaydin, Introduction to Machine Learning, ch. 4 to 5)",
        verified="63 of 63 scikit-learn checks; sparse input accepted.",
        knobs=[],
        build=lambda p, s: NaiveBayesClassifier(),
    ),
    "CART (sklearn)": dict(
        group="baseline",
        what="The standard decision tree: one feature, one threshold per node.",
        when="The reference every tree above is trying to improve on.",
        how="Greedy splitting on the single feature and threshold that most reduces Gini impurity, repeated until a depth limit or pure leaves.",
        ref="Breiman et al. 1984; scikit-learn",
        verified="scikit-learn's own.",
        knobs=[dict(param="max_depth", label="Max depth", kind="slider", lo=1, hi=12, default=5)],
        build=lambda p, s: DecisionTreeClassifier(max_depth=p.get("max_depth", 5), random_state=s),
    ),
    "Random Forest": dict(
        group="baseline",
        what="Hundreds of decision trees, each on a bootstrap sample and a random subset of features, voting.",
        when="The accuracy bar a single model has to clear on tabular data.",
        how="Bagging plus feature subsampling makes the trees different; averaging their votes cancels their individual errors.",
        ref="Breiman 2001; scikit-learn",
        verified="scikit-learn's own.",
        knobs=[dict(param="n_estimators", label="Trees", kind="slider", lo=10, hi=300, default=100, step=10)],
        build=lambda p, s: RandomForestClassifier(n_estimators=p.get("n_estimators", 100), random_state=s),
    ),
    "SVM (RBF)": dict(
        group="baseline",
        what="A kernel method: the other kind of answer to the same question, not a tree at all.",
        when="Medium-sized data where a smooth, maximum-margin boundary is what you want.",
        how="Maps the data through a radial-basis kernel and finds the separating surface with the widest margin; C trades margin against training errors.",
        ref="Cortes and Vapnik 1995; scikit-learn",
        verified="scikit-learn's own.",
        knobs=[dict(param="C", label="C (regularisation)", kind="select", options=[0.1, 1.0, 10.0], default=1.0)],
        build=lambda p, s: SVC(kernel="rbf", C=p.get("C", 1.0), random_state=s),
    ),
}
LIBRARY_MODELS = [n for n, m in MODELS.items() if m["group"] == "neural-trees"]
BASELINE_MODELS = [n for n, m in MODELS.items() if m["group"] == "baseline"]
DEFAULT_MODELS = ["Soft Decision Tree", "Multivariate Tree", "CART (sklearn)", "Random Forest"]

COLORS = {
    "Soft Decision Tree": "#1f77b4", "Multivariate Tree": "#17a2b8", "Omnivariate Tree": "#ff7f0e",
    "Hierarchical MoE": "#2ca02c", "GAL Network": "#d62728", "Weighted KNN": "#17becf",
    "Naive Bayes": "#bcbd22", "CART (sklearn)": "#9467bd", "Random Forest": "#8c564b", "SVM (RBF)": "#e377c2",
}
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def defaults(model_name):
    return {k["param"]: k["default"] for k in MODELS[model_name]["knobs"]}


def params_key(params):
    return json.dumps(params, sort_keys=True, default=str)


def build(model_name, params=None, seed=0):
    return MODELS[model_name]["build"](params or defaults(model_name), seed)


# ── cached work ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, max_entries=64, ttl=24 * 3600)
def dataset(name):
    X, y, features, targets = DATASETS[name]["load"]()
    return StandardScaler().fit_transform(X), np.asarray(y), features, targets


@st.cache_data(show_spinner=False, max_entries=1024, ttl=24 * 3600)
def cv_scores(dataset_name, model_name, params_json, folds, split_seed):
    X, y, _, _ = dataset(dataset_name)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    return cross_val_score(build(model_name, json.loads(params_json), 0), X, y, cv=splitter, scoring="accuracy")


@st.cache_data(show_spinner=False, max_entries=64, ttl=24 * 3600)
def two_d(dataset_name):
    X, y, _, _ = dataset(dataset_name)
    X2 = X if X.shape[1] == 2 else PCA(n_components=2, random_state=42).fit_transform(X)
    extent = (X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5)
    return X2, y, extent


@st.cache_data(show_spinner=False, max_entries=512, ttl=24 * 3600)
def boundary(dataset_name, model_name, params_json, step=0.06):
    """Fit on the 2D view of the data and label a grid; also the training accuracy there."""
    X2, y, (x0, x1, y0, y1) = two_d(dataset_name)
    xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
    xx, yy = np.meshgrid(xs, ys)
    model = build(model_name, json.loads(params_json), 0).fit(X2, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xs, ys, Z, float((model.predict(X2) == y).mean())


@st.cache_resource(show_spinner=False, max_entries=64, ttl=24 * 3600)
def fitted(dataset_name, model_name, params_json):
    """One model fitted on the whole (scaled) dataset, for inspection."""
    X, y, _, _ = dataset(dataset_name)
    return build(model_name, json.loads(params_json), 0).fit(X, y)


@st.cache_data(show_spinner=False, max_entries=256, ttl=24 * 3600)
def f_test(dataset_name, model_a, params_a, model_b, params_b):
    X, y, _, _ = dataset(dataset_name)
    r = combined_5x2cv_f_test(build(model_a, json.loads(params_a), 0), build(model_b, json.loads(params_b), 0), X, y)
    return float(r.statistic), float(r.p_value), bool(r.reject_null)


# ── plotting ─────────────────────────────────────────────────────────


def boundary_figure(dataset_name, model_name, params=None, height=340, title=None, points=True):
    X2, y, _ = two_d(dataset_name)
    xs, ys, Z, acc = boundary(dataset_name, model_name, params_key(params or defaults(model_name)))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Z, x=xs, y=ys, showscale=False, opacity=0.35,
        colorscale=[[0, "#ffcccc"], [0.5, "#ccffcc"], [1, "#ccccff"]], hoverinfo="skip",
    ))
    if points:
        for c in np.unique(y):
            m = y == c
            fig.add_trace(go.Scatter(
                x=X2[m, 0], y=X2[m, 1], mode="markers", name=f"class {c}",
                marker=dict(size=5, color=CLASS_COLORS[int(c) % len(CLASS_COLORS)], line=dict(width=0.5, color="white")),
            ))
    fig.update_layout(
        height=height, margin=dict(t=30 if title else 5, b=5, l=5, r=5), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=title, x=0.02, font=dict(size=13)) if title else None,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig, acc

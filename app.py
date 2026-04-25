"""
ML Playground — Interactive model comparison dashboard.
Run: streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from neural_trees import (
    SoftDecisionTree,
    OmnivariateDecisionTree,
    HierarchicalMixtureOfExperts,
    combined_5x2cv_f_test,
)
from neural_trees.classical.k_nearest_neighbors import WeightedKNN
from neural_trees.classical.naive_bayes import NaiveBayesClassifier
from neural_trees.classical.multilayer_perceptron import GALNetwork

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="ML Playground", page_icon="🧠", layout="wide")

st.title("🧠 ML Playground")
st.caption("Interactive classifier comparison powered by [neural-trees](https://github.com/cgrtml/neural-trees)")

# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────
DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Moons": lambda: _make_synthetic(make_moons, noise=0.25),
    "Circles": lambda: _make_synthetic(make_circles, noise=0.15, factor=0.5),
}


def _make_synthetic(func, **kwargs):
    X, y = func(n_samples=500, random_state=42, **kwargs)

    class Bunch:
        pass

    b = Bunch()
    b.data = X
    b.target = y
    b.feature_names = ["x1", "x2"]
    b.target_names = [str(i) for i in np.unique(y)]
    return b


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
def get_model(name):
    models = {
        "Soft Decision Tree": SoftDecisionTree(depth=4, max_epochs=40, verbose=False),
        "Omnivariate Tree": OmnivariateDecisionTree(max_depth=5),
        "Hierarchical MoE": HierarchicalMixtureOfExperts(depth=2, max_epochs=50, verbose=False),
        "GAL Network": GALNetwork(max_epochs=80, verbose=False),
        "CART (sklearn)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "Weighted KNN": WeightedKNN(k=5),
        "Naive Bayes": NaiveBayesClassifier(likelihood="gaussian"),
    }
    return models[name]


MODEL_NAMES = list(get_model.__code__.co_consts[1].keys()) if False else [
    "Soft Decision Tree",
    "Omnivariate Tree",
    "Hierarchical MoE",
    "GAL Network",
    "CART (sklearn)",
    "Random Forest",
    "SVM (RBF)",
    "Weighted KNN",
    "Naive Bayes",
]

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.header("Configuration")

dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
selected_models = st.sidebar.multiselect(
    "Models to compare",
    MODEL_NAMES,
    default=["Soft Decision Tree", "CART (sklearn)", "Random Forest"],
)
cv_folds = st.sidebar.slider("Cross-validation folds", 3, 10, 5)
show_boundary = st.sidebar.checkbox("Show decision boundaries", value=True)
run_stat_test = st.sidebar.checkbox("Run statistical test (5x2cv F)", value=False)

if run_stat_test and len(selected_models) >= 2:
    stat_model_a = st.sidebar.selectbox("Model A", selected_models, index=0)
    stat_model_b = st.sidebar.selectbox("Model B", [m for m in selected_models if m != stat_model_a], index=0)

run_button = st.sidebar.button("Run", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if not run_button:
    st.info("Select models and a dataset, then press **Run**.")
    st.stop()

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# Load data
data = DATASETS[dataset_name]()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
st.header("Cross-Validation Results")

results = {}
progress = st.progress(0)

for i, name in enumerate(selected_models):
    model = get_model(name)
    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")
    results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    progress.progress((i + 1) / len(selected_models))

progress.empty()

# Results table
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Accuracy Table")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
    table_md = "| Rank | Model | Accuracy | Std |\n|:---:|---|:---:|:---:|\n"
    for rank, (name, r) in enumerate(sorted_results, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        table_md += f"| {medal} | {name} | {r['mean']:.4f} | {r['std']:.4f} |\n"
    st.markdown(table_md)

with col2:
    st.subheader("Accuracy Comparison")
    fig = go.Figure()
    names = [n for n, _ in sorted_results]
    means = [r["mean"] for _, r in sorted_results]
    stds = [r["std"] for _, r in sorted_results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    fig.add_trace(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors[:len(names)],
        text=[f"{m:.3f}" for m in means],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Accuracy",
        yaxis_range=[max(0, min(means) - 0.1), 1.02],
        template="plotly_dark",
        height=400,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Decision boundaries (2D PCA projection)
# ──────────────────────────────────────────────
if show_boundary:
    st.header("Decision Boundaries")
    st.caption("Projected to 2D using PCA" if X.shape[1] > 2 else "")

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
    else:
        X_2d = X_scaled

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    n_models = len(selected_models)
    cols_per_row = min(n_models, 3)
    rows = (n_models + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=selected_models,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    colorscale_bg = [[0, "#1a1a2e"], [0.5, "#16213e"], [1, "#0f3460"]]
    marker_colors = ["#e94560", "#00b4d8", "#06d6a0", "#ffd166", "#ef476f"]

    for idx, name in enumerate(selected_models):
        row = idx // cols_per_row + 1
        col = idx % cols_per_row + 1

        model = get_model(name)
        model.fit(X_2d, y)
        Z = model.predict(grid).reshape(xx.shape)

        fig.add_trace(
            go.Contour(
                z=Z, x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h),
                showscale=False, opacity=0.4,
                contours=dict(coloring="heatmap"),
                colorscale="Viridis",
            ),
            row=row, col=col,
        )

        for c in np.unique(y):
            mask = y == c
            fig.add_trace(
                go.Scatter(
                    x=X_2d[mask, 0], y=X_2d[mask, 1],
                    mode="markers",
                    marker=dict(size=4, color=marker_colors[c % len(marker_colors)]),
                    showlegend=False,
                ),
                row=row, col=col,
            )

    fig.update_layout(
        height=350 * rows,
        template="plotly_dark",
        margin=dict(t=40, b=20),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Statistical test
# ──────────────────────────────────────────────
if run_stat_test and len(selected_models) >= 2:
    st.header("Statistical Test")
    st.caption(f"Alpaydın's Combined 5x2cv F Test: **{stat_model_a}** vs **{stat_model_b}**")

    with st.spinner("Running 5x2cv F test..."):
        result = combined_5x2cv_f_test(
            get_model(stat_model_a),
            get_model(stat_model_b),
            X_scaled, y,
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("F Statistic", f"{result.statistic:.4f}")
    col2.metric("P Value", f"{result.p_value:.4f}")
    col3.metric("Reject H₀?", "Yes" if result.reject_null else "No")

    if result.reject_null:
        st.success(f"The difference is **statistically significant** (p = {result.p_value:.4f}). {result.interpretation}")
    else:
        st.info(f"No significant difference found (p = {result.p_value:.4f}). {result.interpretation}")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("Built by [Cagri Temel](https://github.com/cgrtml) | [neural-trees](https://pypi.org/project/neural-trees/)")

"""
ML Playground — Interactive model comparison dashboard.
Run: streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import cross_val_score
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

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f0f2f6; border-radius: 10px; padding: 15px; }
    h1 { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ML Playground")
st.caption("Interactive classifier comparison powered by [neural-trees](https://github.com/cgrtml/neural-trees)")

# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────
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


DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Moons": lambda: _make_synthetic(make_moons, noise=0.25),
    "Circles": lambda: _make_synthetic(make_circles, noise=0.15, factor=0.5),
}

DATASET_INFO = {
    "Iris": "3 classes, 4 features, 150 samples",
    "Wine": "3 classes, 13 features, 178 samples",
    "Breast Cancer": "2 classes, 30 features, 569 samples",
    "Moons": "2 classes, 2 features, 500 samples (synthetic)",
    "Circles": "2 classes, 2 features, 500 samples (synthetic)",
}

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
MODEL_NAMES = [
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

MODEL_DESCRIPTIONS = {
    "Soft Decision Tree": "Differentiable tree with sigmoid gates (Irsoy et al., ICPR 2012)",
    "Omnivariate Tree": "Auto-selects univariate/linear/nonlinear splits (Yildiz & Alpaydin, 2001)",
    "Hierarchical MoE": "Tree-structured mixture of experts with dropout (Irsoy & Alpaydin, 2021)",
    "GAL Network": "Grow-and-learn constructive neural network (Alpaydin, 1994)",
    "CART (sklearn)": "Classic decision tree with Gini impurity",
    "Random Forest": "Ensemble of 100 decision trees",
    "SVM (RBF)": "Support vector machine with RBF kernel",
    "Weighted KNN": "Inverse-distance weighted k-nearest neighbors (k=5)",
    "Naive Bayes": "Gaussian naive Bayes classifier",
}

COLORS = {
    "Soft Decision Tree": "#1f77b4",
    "Omnivariate Tree": "#ff7f0e",
    "Hierarchical MoE": "#2ca02c",
    "GAL Network": "#d62728",
    "CART (sklearn)": "#9467bd",
    "Random Forest": "#8c564b",
    "SVM (RBF)": "#e377c2",
    "Weighted KNN": "#17becf",
    "Naive Bayes": "#bcbd22",
}


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


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

dataset_name = st.sidebar.selectbox("📊 Dataset", list(DATASETS.keys()))
st.sidebar.caption(DATASET_INFO[dataset_name])

selected_models = st.sidebar.multiselect(
    "🤖 Models to compare",
    MODEL_NAMES,
    default=["Soft Decision Tree", "CART (sklearn)", "Random Forest", "SVM (RBF)"],
)

cv_folds = st.sidebar.slider("🔄 Cross-validation folds", 3, 10, 5)
show_boundary = st.sidebar.checkbox("🗺️ Show decision boundaries", value=True)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Comparison", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Built by [Cagri Temel](https://github.com/cgrtml)")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if not run_button:
    st.info("👈 Select models and a dataset from the sidebar, then press **Run Comparison**.")

    # Show model cards
    st.header("Available Models")
    cols = st.columns(3)
    for i, name in enumerate(MODEL_NAMES):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: #f8f9fa; border-left: 4px solid {COLORS[name]};
                        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;">
                <strong>{name}</strong><br>
                <span style="font-size: 13px; color: #555;">{MODEL_DESCRIPTIONS[name]}</span>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# Load data
data = DATASETS[dataset_name]()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_classes = len(np.unique(y))

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
st.header(f"📊 Results on {dataset_name}")

results = {}
status = st.empty()
progress = st.progress(0)

for i, name in enumerate(selected_models):
    status.text(f"Training {name}...")
    try:
        model = get_model(name)
        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    except Exception as e:
        results[name] = {"mean": 0.0, "std": 0.0, "scores": np.array([0.0]), "error": str(e)}
    progress.progress((i + 1) / len(selected_models))

progress.empty()
status.empty()

sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)

# ──────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────
best_name, best_r = sorted_results[0]
metric_cols = st.columns(4)
metric_cols[0].metric("🏆 Best Model", best_name)
metric_cols[1].metric("Accuracy", f"{best_r['mean']:.4f}")
metric_cols[2].metric("Dataset", dataset_name)
metric_cols[3].metric("Models Compared", len(selected_models))

st.divider()

# ──────────────────────────────────────────────
# Results: Table + Bar Chart
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Ranking", "📊 Charts", "⚔️ Head-to-Head"])

with tab1:
    # Ranking table
    for rank, (name, r) in enumerate(sorted_results, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        has_error = "error" in r
        acc_text = "ERROR" if has_error else f"{r['mean']:.4f} ± {r['std']:.4f}"

        col_medal, col_name, col_acc, col_bar = st.columns([0.5, 2, 1.5, 4])
        col_medal.markdown(f"### {medal}")
        col_name.markdown(f"**{name}**")
        col_name.caption(MODEL_DESCRIPTIONS[name])
        col_acc.markdown(f"### {acc_text}")
        if not has_error:
            col_bar.progress(r["mean"])

with tab2:
    col_bar, col_box = st.columns(2)

    with col_bar:
        st.subheader("Accuracy Comparison")
        names = [n for n, _ in sorted_results if "error" not in _]
        means = [r["mean"] for _, r in sorted_results if "error" not in r]
        stds = [r["std"] for _, r in sorted_results if "error" not in r]
        bar_colors = [COLORS.get(n, "#999") for n in names]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=names, y=means,
            error_y=dict(type="data", array=stds, visible=True, color="#333"),
            marker_color=bar_colors,
            text=[f"{m:.3f}" for m in means],
            textposition="outside",
            textfont=dict(size=14, color="#333"),
        ))
        fig_bar.update_layout(
            yaxis_title="Accuracy",
            yaxis_range=[max(0, min(means) - 0.15), 1.05],
            height=450,
            margin=dict(t=20, b=80),
            xaxis_tickangle=-30,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_box:
        st.subheader("Score Distribution")
        fig_box = go.Figure()
        for name, r in sorted_results:
            if "error" not in r:
                fig_box.add_trace(go.Box(
                    y=r["scores"], name=name,
                    marker_color=COLORS.get(name, "#999"),
                    boxmean=True,
                ))
        fig_box.update_layout(
            yaxis_title="Accuracy per fold",
            height=450,
            margin=dict(t=20, b=80),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("⚔️ Head-to-Head Comparison")
    if len(selected_models) < 2:
        st.info("Select at least 2 models to compare head-to-head.")
    else:
        valid_models = [n for n in selected_models if "error" not in results[n]]
        h2h_col1, h2h_col2 = st.columns(2)
        model_a = h2h_col1.selectbox("Model A", valid_models, index=0)
        model_b = h2h_col2.selectbox("Model B", [m for m in valid_models if m != model_a], index=0)

        ra, rb = results[model_a], results[model_b]

        # Comparison metrics
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(f"<h2 style='text-align:center; color:{COLORS[model_a]}'>{model_a}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center'>{ra['mean']:.4f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#888'>± {ra['std']:.4f}</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<h1 style='text-align:center; margin-top:30px'>⚔️</h1>", unsafe_allow_html=True)
            diff = ra["mean"] - rb["mean"]
            winner = model_a if diff > 0 else model_b
            st.markdown(f"<p style='text-align:center; font-size:12px; color:#888'>Δ = {abs(diff):.4f}</p>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<h2 style='text-align:center; color:{COLORS[model_b]}'>{model_b}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center'>{rb['mean']:.4f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#888'>± {rb['std']:.4f}</p>", unsafe_allow_html=True)

        # Fold-by-fold comparison
        fig_h2h = go.Figure()
        folds = list(range(1, cv_folds + 1))
        fig_h2h.add_trace(go.Scatter(
            x=folds, y=ra["scores"], mode="lines+markers",
            name=model_a, line=dict(color=COLORS[model_a], width=3),
            marker=dict(size=10),
        ))
        fig_h2h.add_trace(go.Scatter(
            x=folds, y=rb["scores"], mode="lines+markers",
            name=model_b, line=dict(color=COLORS[model_b], width=3),
            marker=dict(size=10),
        ))
        fig_h2h.update_layout(
            xaxis_title="Fold",
            yaxis_title="Accuracy",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        # Statistical test
        st.subheader("📐 Statistical Significance Test")
        st.caption("Alpaydin's Combined 5×2cv F Test")
        if st.button("Run 5×2cv F Test", type="secondary"):
            with st.spinner("Running statistical test..."):
                try:
                    test_result = combined_5x2cv_f_test(
                        get_model(model_a), get_model(model_b), X_scaled, y,
                    )
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("F Statistic", f"{test_result.statistic:.4f}")
                    tc2.metric("P Value", f"{test_result.p_value:.4f}")
                    tc3.metric("Reject H₀?", "Yes ✅" if test_result.reject_null else "No ❌")

                    if test_result.reject_null:
                        st.success(f"**Statistically significant** difference (p = {test_result.p_value:.4f}). Winner: **{winner}**")
                    else:
                        st.info(f"**No significant difference** found (p = {test_result.p_value:.4f}). Models perform similarly.")
                except Exception as e:
                    st.error(f"Test failed: {e}")

# ──────────────────────────────────────────────
# Decision boundaries (2D PCA projection)
# ──────────────────────────────────────────────
if show_boundary:
    st.divider()
    st.header("🗺️ Decision Boundaries")
    st.caption("Projected to 2D using PCA" if X.shape[1] > 2 else "Original 2D feature space")

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
    else:
        X_2d = X_scaled.copy()

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    h = 0.06
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    valid = [n for n in selected_models if "error" not in results[n]]
    n_models = len(valid)
    if n_models == 0:
        st.warning("No valid models to plot.")
    else:
        cols_per_row = min(n_models, 3)
        rows = (n_models + cols_per_row - 1) // cols_per_row

        fig = make_subplots(
            rows=rows, cols=cols_per_row,
            subplot_titles=[f"{n} ({results[n]['mean']:.3f})" for n in valid],
            horizontal_spacing=0.06,
            vertical_spacing=0.12,
        )

        class_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        bg_colorscales = [
            [[0, "#fee"], [1, "#eef"]],
            [[0, "#fee"], [0.5, "#efe"], [1, "#eef"]],
        ]
        bg_cs = bg_colorscales[1] if n_classes > 2 else bg_colorscales[0]

        boundary_status = st.empty()
        for idx, name in enumerate(valid):
            boundary_status.text(f"Computing boundary for {name}...")
            row = idx // cols_per_row + 1
            col = idx % cols_per_row + 1

            try:
                model = get_model(name)
                model.fit(X_2d, y)
                Z = model.predict(grid).reshape(xx.shape)

                fig.add_trace(
                    go.Heatmap(
                        z=Z,
                        x=np.arange(x_min, x_max, h),
                        y=np.arange(y_min, y_max, h),
                        showscale=False,
                        opacity=0.3,
                        colorscale=bg_cs,
                    ),
                    row=row, col=col,
                )

                for c_idx in range(n_classes):
                    mask = y == c_idx
                    fig.add_trace(
                        go.Scatter(
                            x=X_2d[mask, 0], y=X_2d[mask, 1],
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=class_colors[c_idx % len(class_colors)],
                                line=dict(width=0.5, color="white"),
                            ),
                            showlegend=(idx == 0),
                            name=f"Class {c_idx}" if idx == 0 else None,
                        ),
                        row=row, col=col,
                    )
            except Exception as e:
                st.warning(f"Could not plot boundary for {name}: {e}")

        boundary_status.empty()

        fig.update_layout(
            height=380 * rows,
            margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("Built by [Cagri Temel](https://github.com/cgrtml) | Powered by [neural-trees](https://pypi.org/project/neural-trees/)")

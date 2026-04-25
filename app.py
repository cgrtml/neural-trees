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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 32px; color: white; margin-bottom: 24px;
}
.hero-banner h1 { font-size: 2rem; margin: 0 0 8px 0; color: white; }
.hero-banner p  { font-size: 1rem; opacity: 0.8; margin: 0; color: white; }
.model-card {
    background: #f8f9fa; border-radius: 10px; padding: 16px;
    border-left: 4px solid #1f77b4; margin-bottom: 8px;
}
.vs-badge {
    font-size: 2.5rem; font-weight: 700; color: #e74c3c;
    text-align: center; line-height: 80px;
}
</style>
""", unsafe_allow_html=True)

# Hero banner
st.markdown("""
<div class="hero-banner">
    <h1>🧠 ML Playground</h1>
    <p>Choose algorithms, tune hyperparameters, watch decision boundaries update live.<br>
    Powered by <a href="https://github.com/cgrtml/neural-trees" style="color:#58a6ff">neural-trees</a> — sklearn-compatible implementations of classic ML research.</p>
</div>
""", unsafe_allow_html=True)

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
    "Iris": "3 classes · 4 features · 150 samples",
    "Wine": "3 classes · 13 features · 178 samples",
    "Breast Cancer": "2 classes · 30 features · 569 samples",
    "Moons": "2 classes · 2 features · 500 samples (synthetic)",
    "Circles": "2 classes · 2 features · 500 samples (synthetic)",
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
    "Random Forest": "Ensemble of decision trees with bagging",
    "SVM (RBF)": "Support vector machine with RBF kernel",
    "Weighted KNN": "Inverse-distance weighted k-nearest neighbors",
    "Naive Bayes": "Gaussian naive Bayes classifier",
}

MODEL_COLORS = {
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

# ──────────────────────────────────────────────
# Sidebar — Configuration
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

dataset_name = st.sidebar.selectbox("📊 Dataset", list(DATASETS.keys()))
st.sidebar.caption(DATASET_INFO[dataset_name])

selected_models = st.sidebar.multiselect(
    "🤖 Models",
    MODEL_NAMES,
    default=["Soft Decision Tree", "CART (sklearn)", "Random Forest", "SVM (RBF)"],
)

cv_folds = st.sidebar.slider("🔄 CV Folds", 3, 10, 5)

# ──────────────────────────────────────────────
# Sidebar — Hyperparameters
# ──────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🎛️ Hyperparameters")

hp = {}
for name in selected_models:
    with st.sidebar.expander(f"**{name}**", expanded=False):
        if name == "Soft Decision Tree":
            hp[name] = {
                "depth": st.slider("Tree depth", 2, 8, 4, key="sdt_depth"),
                "max_epochs": st.slider("Max epochs", 10, 100, 40, key="sdt_epochs"),
                "lr": st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.01, key="sdt_lr"),
                "batch_size": st.select_slider("Batch size", [16, 32, 64, 128], value=64, key="sdt_bs"),
            }
        elif name == "Omnivariate Tree":
            hp[name] = {
                "max_depth": st.slider("Max depth", 2, 10, 5, key="odt_depth"),
                "min_samples_split": st.slider("Min samples split", 2, 30, 10, key="odt_mss"),
            }
        elif name == "Hierarchical MoE":
            hp[name] = {
                "depth": st.slider("Tree depth", 1, 4, 2, key="hmoe_depth"),
                "max_epochs": st.slider("Max epochs", 20, 100, 50, key="hmoe_epochs"),
                "dropout_rate": st.slider("Dropout rate", 0.0, 0.5, 0.3, step=0.05, key="hmoe_drop"),
            }
        elif name == "GAL Network":
            hp[name] = {
                "initial_hidden": st.slider("Initial hidden units", 1, 10, 2, key="gal_init"),
                "max_hidden": st.slider("Max hidden units", 10, 100, 50, key="gal_max"),
                "max_epochs": st.slider("Max epochs", 20, 200, 80, key="gal_epochs"),
            }
        elif name == "CART (sklearn)":
            hp[name] = {
                "max_depth": st.slider("Max depth", 1, 20, 5, key="cart_depth"),
                "criterion": st.selectbox("Criterion", ["gini", "entropy"], key="cart_crit"),
            }
        elif name == "Random Forest":
            hp[name] = {
                "n_estimators": st.slider("Number of trees", 10, 300, 100, step=10, key="rf_n"),
                "max_depth": st.slider("Max depth", 2, 20, 5, key="rf_depth"),
            }
        elif name == "SVM (RBF)":
            hp[name] = {
                "C": st.select_slider("C (regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key="svm_c"),
                "gamma": st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma"),
            }
        elif name == "Weighted KNN":
            hp[name] = {
                "k": st.slider("k (neighbors)", 1, 20, 5, key="knn_k"),
                "weight_power": st.slider("Weight power", 0.0, 4.0, 2.0, step=0.5, key="knn_wp"),
            }
        elif name == "Naive Bayes":
            hp[name] = {
                "likelihood": st.selectbox("Likelihood", ["gaussian", "bernoulli"], key="nb_like"),
            }

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Comparison", type="primary", use_container_width=True)
st.sidebar.divider()
st.sidebar.caption("Built by [Cagri Temel](https://github.com/cgrtml)")


# ──────────────────────────────────────────────
# Build model with hyperparameters
# ──────────────────────────────────────────────
def build_model(name):
    p = hp.get(name, {})
    if name == "Soft Decision Tree":
        return SoftDecisionTree(depth=p.get("depth", 4), max_epochs=p.get("max_epochs", 40),
                                learning_rate=p.get("lr", 0.01), batch_size=p.get("batch_size", 64), verbose=False)
    elif name == "Omnivariate Tree":
        return OmnivariateDecisionTree(max_depth=p.get("max_depth", 5), min_samples_split=p.get("min_samples_split", 10))
    elif name == "Hierarchical MoE":
        return HierarchicalMixtureOfExperts(depth=p.get("depth", 2), max_epochs=p.get("max_epochs", 50),
                                            dropout_rate=p.get("dropout_rate", 0.3), verbose=False)
    elif name == "GAL Network":
        return GALNetwork(initial_hidden=p.get("initial_hidden", 2), max_hidden=p.get("max_hidden", 50),
                          max_epochs=p.get("max_epochs", 80), verbose=False)
    elif name == "CART (sklearn)":
        return DecisionTreeClassifier(max_depth=p.get("max_depth", 5), criterion=p.get("criterion", "gini"), random_state=42)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 5), random_state=42)
    elif name == "SVM (RBF)":
        return SVC(kernel="rbf", C=p.get("C", 1.0), gamma=p.get("gamma", "scale"), probability=True, random_state=42)
    elif name == "Weighted KNN":
        return WeightedKNN(k=p.get("k", 5), weight_power=p.get("weight_power", 2.0))
    elif name == "Naive Bayes":
        return NaiveBayesClassifier(likelihood=p.get("likelihood", "gaussian"))


# ──────────────────────────────────────────────
# Landing page
# ──────────────────────────────────────────────
if not run_button:
    st.info("👈 Select models, tune hyperparameters, then press **Run Comparison**.")

    cols = st.columns(3)
    for i, name in enumerate(MODEL_NAMES):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#f8f9fa; border-left:4px solid {MODEL_COLORS[name]};
                        padding:12px 16px; border-radius:8px; margin-bottom:10px;">
                <strong style="color:{MODEL_COLORS[name]}">{name}</strong><br>
                <span style="font-size:13px; color:#555;">{MODEL_DESCRIPTIONS[name]}</span>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
data = DATASETS[dataset_name]()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_classes = len(np.unique(y))

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
results = {}
status = st.empty()
progress = st.progress(0)

for i, name in enumerate(selected_models):
    status.text(f"Training {name}...")
    try:
        model = build_model(name)
        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    except Exception as e:
        results[name] = {"mean": 0.0, "std": 0.0, "scores": np.array([0.0]), "error": str(e)}
    progress.progress((i + 1) / len(selected_models))

progress.empty()
status.empty()

sorted_results = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
valid_results = [(n, r) for n, r in sorted_results if "error" not in r]

# ──────────────────────────────────────────────
# Top metrics
# ──────────────────────────────────────────────
st.header(f"📊 Results — {dataset_name}")

if valid_results:
    best_name, best_r = valid_results[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Best Model", best_name)
    m2.metric("Accuracy", f"{best_r['mean']:.4f}")
    m3.metric("Dataset", dataset_name)
    m4.metric("Models Tested", len(selected_models))

st.divider()

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_rank, tab_chart, tab_h2h, tab_boundary = st.tabs([
    "📋 Ranking", "📊 Charts", "⚔️ Head-to-Head", "🗺️ Decision Boundaries"
])

# ── TAB 1: Ranking ──
with tab_rank:
    best_acc = valid_results[0][1]["mean"] if valid_results else 1.0
    for rank, (name, r) in enumerate(sorted_results, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        has_error = "error" in r
        color = MODEL_COLORS.get(name, "#999")
        border_weight = "3px" if rank <= 3 else "2px"
        bg = "#fffdf5" if rank == 1 else "#f8f9fa"
        pct = int(r["mean"] / best_acc * 100) if not has_error and best_acc > 0 else 0

        if has_error:
            card_html = f"""
            <div style="background:#fff5f5; border-left:3px solid #e74c3c; border-radius:10px;
                        padding:16px 20px; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:28px">{medal}</span>
                    <div style="flex:1">
                        <div style="font-weight:600; font-size:16px; color:#333">{name}</div>
                        <div style="font-size:12px; color:#888">{MODEL_DESCRIPTIONS[name]}</div>
                    </div>
                    <div style="color:#e74c3c; font-weight:600">❌ Error</div>
                </div>
            </div>"""
        else:
            card_html = f"""
            <div style="background:{bg}; border-left:{border_weight} solid {color}; border-radius:10px;
                        padding:16px 20px; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:16px;">
                    <span style="font-size:32px">{medal}</span>
                    <div style="flex:1">
                        <div style="font-weight:600; font-size:16px; color:#333">{name}</div>
                        <div style="font-size:12px; color:#888; margin-top:2px">{MODEL_DESCRIPTIONS[name]}</div>
                        <div style="margin-top:8px; background:#e9ecef; border-radius:6px; height:8px; overflow:hidden;">
                            <div style="width:{pct}%; height:100%; background:{color}; border-radius:6px;"></div>
                        </div>
                    </div>
                    <div style="text-align:right; min-width:100px;">
                        <div style="font-size:28px; font-weight:700; color:{color}">{r['mean']:.4f}</div>
                        <div style="font-size:12px; color:#999">± {r['std']:.4f}</div>
                    </div>
                </div>
            </div>"""
        st.markdown(card_html, unsafe_allow_html=True)

# ── TAB 2: Charts ──
with tab_chart:
    if not valid_results:
        st.warning("No valid results to display.")
    else:
        col_bar, col_box = st.columns(2)

        with col_bar:
            st.subheader("Accuracy Comparison")
            names = [n for n, _ in valid_results]
            means = [r["mean"] for _, r in valid_results]
            stds = [r["std"] for _, r in valid_results]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=names, y=means,
                error_y=dict(type="data", array=stds, visible=True, color="#333"),
                marker_color=[MODEL_COLORS.get(n, "#999") for n in names],
                text=[f"{m:.3f}" for m in means],
                textposition="outside",
                textfont=dict(size=14, color="#333"),
            ))
            fig_bar.update_layout(
                yaxis_title="Accuracy", yaxis_range=[max(0, min(means) - 0.15), 1.05],
                height=420, margin=dict(t=20, b=80), xaxis_tickangle=-25,
                plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_box:
            st.subheader("Per-Fold Distribution")
            fig_box = go.Figure()
            for name, r in valid_results:
                fig_box.add_trace(go.Box(
                    y=r["scores"], name=name,
                    marker_color=MODEL_COLORS.get(name, "#999"),
                    boxmean=True,
                ))
            fig_box.update_layout(
                yaxis_title="Accuracy per fold", height=420,
                margin=dict(t=20, b=80), showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(gridcolor="#eee"),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Radar chart
        if len(valid_results) >= 3:
            st.subheader("Radar Overview")
            fig_radar = go.Figure()
            categories = [n for n, _ in valid_results]
            for name, r in valid_results:
                vals = [r["mean"]] + [results[n]["mean"] if "error" not in results[n] else 0 for n in categories[1:]]
                # Just show each model's accuracy as a point
            fig_radar = go.Figure()
            for name, r in valid_results:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[results[n]["mean"] if n == name else 0 for n in categories] + [r["mean"]],
                    theta=categories + [categories[0]],
                    name=name,
                    line=dict(color=MODEL_COLORS.get(name, "#999")),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=400, showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 3: Head-to-Head ──
with tab_h2h:
    valid_names = [n for n, r in results.items() if "error" not in r]
    if len(valid_names) < 2:
        st.info("Select at least 2 working models for head-to-head comparison.")
    else:
        h1, h2 = st.columns(2)
        model_a = h1.selectbox("Model A", valid_names, index=0)
        remaining = [m for m in valid_names if m != model_a]
        model_b = h2.selectbox("Model B", remaining, index=min(1, len(remaining) - 1) if len(remaining) > 1 else 0)

        ra, rb = results[model_a], results[model_b]
        diff = ra["mean"] - rb["mean"]
        winner = model_a if diff > 0 else model_b

        # VS display
        v1, vs, v2 = st.columns([5, 2, 5])
        with v1:
            st.markdown(f"<h2 style='text-align:center; color:{MODEL_COLORS[model_a]}'>{model_a}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center'>{ra['mean']:.4f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#888'>σ = {ra['std']:.4f}</p>", unsafe_allow_html=True)
            # Show hyperparams
            if model_a in hp:
                st.caption("Hyperparameters:")
                for k, v in hp[model_a].items():
                    st.markdown(f"<span style='font-size:12px'>`{k}` = **{v}**</span>", unsafe_allow_html=True)
        with vs:
            st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
            delta_color = MODEL_COLORS[winner]
            st.markdown(f"<p style='text-align:center; font-size:13px'>Δ = <b style=\"color:{delta_color}\">{abs(diff):.4f}</b></p>", unsafe_allow_html=True)
            if abs(diff) < 0.005:
                st.markdown("<p style='text-align:center; font-size:11px; color:#888'>⚖️ Too close to call</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align:center; font-size:11px; color:{delta_color}'>👑 {winner}</p>", unsafe_allow_html=True)
        with v2:
            st.markdown(f"<h2 style='text-align:center; color:{MODEL_COLORS[model_b]}'>{model_b}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align:center'>{rb['mean']:.4f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:#888'>σ = {rb['std']:.4f}</p>", unsafe_allow_html=True)
            if model_b in hp:
                st.caption("Hyperparameters:")
                for k, v in hp[model_b].items():
                    st.markdown(f"<span style='font-size:12px'>`{k}` = **{v}**</span>", unsafe_allow_html=True)

        st.divider()

        # Fold-by-fold line chart
        st.subheader("Fold-by-Fold Accuracy")
        fig_h2h = go.Figure()
        folds = list(range(1, cv_folds + 1))
        fig_h2h.add_trace(go.Scatter(
            x=folds, y=ra["scores"], mode="lines+markers",
            name=model_a, line=dict(color=MODEL_COLORS[model_a], width=3), marker=dict(size=10),
        ))
        fig_h2h.add_trace(go.Scatter(
            x=folds, y=rb["scores"], mode="lines+markers",
            name=model_b, line=dict(color=MODEL_COLORS[model_b], width=3), marker=dict(size=10),
        ))
        # Add shaded area between them
        fig_h2h.add_trace(go.Scatter(
            x=folds + folds[::-1],
            y=list(ra["scores"]) + list(rb["scores"][::-1]),
            fill="toself", fillcolor="rgba(200,200,200,0.15)",
            line=dict(width=0), showlegend=False,
        ))
        fig_h2h.update_layout(
            xaxis_title="Fold", yaxis_title="Accuracy", height=350,
            plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        # Statistical test
        st.subheader("📐 Statistical Significance")
        st.caption("Alpaydin's Combined 5×2cv F Test — the gold standard for classifier comparison")
        if st.button("Run 5×2cv F Test"):
            with st.spinner("Running 5 repetitions of 2-fold CV..."):
                try:
                    test_result = combined_5x2cv_f_test(build_model(model_a), build_model(model_b), X_scaled, y)
                    t1, t2, t3 = st.columns(3)
                    t1.metric("F Statistic", f"{test_result.statistic:.4f}")
                    t2.metric("P Value", f"{test_result.p_value:.4f}")
                    t3.metric("Significant?", "Yes ✅" if test_result.reject_null else "No ❌")

                    if test_result.reject_null:
                        st.success(f"**Statistically significant** difference (p = {test_result.p_value:.4f}). {test_result.interpretation}")
                    else:
                        st.info(f"**No significant difference** (p = {test_result.p_value:.4f}). {test_result.interpretation}")
                except Exception as e:
                    st.error(f"Test failed: {e}")

# ── TAB 4: Decision Boundaries ──
with tab_boundary:
    st.caption("Projected to 2D via PCA" if X.shape[1] > 2 else "Original 2D feature space")

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
    else:
        X_2d = X_scaled.copy()

    valid_models = [n for n in selected_models if "error" not in results.get(n, {"error": True})]
    if not valid_models:
        st.warning("No valid models to plot.")
    else:
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        h = 0.06
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]

        n_models = len(valid_models)
        cols_per_row = min(n_models, 3)
        rows = (n_models + cols_per_row - 1) // cols_per_row

        fig = make_subplots(
            rows=rows, cols=cols_per_row,
            subplot_titles=[f"{n} ({results[n]['mean']:.3f})" for n in valid_models],
            horizontal_spacing=0.06, vertical_spacing=0.12,
        )

        class_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        boundary_status = st.empty()
        for idx, name in enumerate(valid_models):
            boundary_status.text(f"Computing boundary: {name}...")
            row = idx // cols_per_row + 1
            col = idx % cols_per_row + 1

            try:
                model = build_model(name)
                model.fit(X_2d, y)
                Z = model.predict(grid).reshape(xx.shape)

                # Background heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=Z, x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h),
                        showscale=False, opacity=0.25,
                        colorscale=[[0, "#ffcccc"], [0.5, "#ccffcc"], [1, "#ccccff"]],
                    ),
                    row=row, col=col,
                )

                # Data points
                for c_idx in range(n_classes):
                    mask = y == c_idx
                    fig.add_trace(
                        go.Scatter(
                            x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers",
                            marker=dict(size=5, color=class_colors[c_idx % len(class_colors)],
                                        line=dict(width=0.5, color="white")),
                            showlegend=(idx == 0),
                            name=f"Class {c_idx}" if idx == 0 else None,
                        ),
                        row=row, col=col,
                    )
            except Exception as e:
                st.warning(f"Boundary failed for {name}: {e}")

        boundary_status.empty()

        fig.update_layout(
            height=380 * rows, margin=dict(t=40, b=20, l=20, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        )
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("Built by [Cagri Temel](https://github.com/cgrtml) · Powered by [neural-trees](https://pypi.org/project/neural-trees/)")

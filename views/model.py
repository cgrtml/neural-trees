"""How a model works: one model, its knobs, its boundary, and what it learned."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from playground import (
    BASELINE_MODELS,
    DATASETS,
    LIBRARY_MODELS,
    MODELS,
    boundary_figure,
    dataset,
    defaults,
    fitted,
    glossary,
    params_key,
    two_d,
    ui,
)

ui.title("How a model works", lead="One model at a time: what it does, when to use it, how it works, what its settings do to the boundary, and what it learned from the data.", eyebrow="Look inside")

st.session_state.setdefault("model_pick", "Soft Decision Tree")
st.session_state.setdefault("model_data", "Moons")
name = st.selectbox("Model", LIBRARY_MODELS + BASELINE_MODELS, key="model_pick", index=None)
m = MODELS[name]
st.markdown(f"## {name} &nbsp;{ui.badge(m['group'])}", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(f"**What it does**\n\n{m['what']}")
c2.markdown(f"**When to use it**\n\n{m['when']}")
c3.markdown(f"**How it works**\n\n{m['how']}")
st.caption(f"Source: {m['ref']}. Verification: {m['verified']}")

st.divider()
left, right = st.columns([1, 2])
with left:
    st.markdown("**Knobs**")
    params = {}
    for k in m["knobs"]:
        if k["kind"] == "slider":
            params[k["param"]] = st.slider(k["label"], k["lo"], k["hi"], k["default"], step=k.get("step", 1), key=f"knob_{name}_{k['param']}")
        else:
            params[k["param"]] = st.select_slider(k["label"], k["options"], value=k["default"], key=f"knob_{name}_{k['param']}")
    if not m["knobs"]:
        st.caption("No settings to tune; this model has none that matter here.")
    data_name = st.radio("Data", list(DATASETS), index=None, key="model_data")
    st.caption(DATASETS[data_name]["blurb"] + ("" if DATASETS[data_name]["two_d"] else " Shown through its first two principal components."))
with right:
    with st.spinner(f"Fitting {name}..."):
        fig, acc = boundary_figure(data_name, name, params, height=380, title=f"{name} on {data_name}")
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption(
        f"Every point of the plane coloured by the class {name} predicts there, with the data on top; "
        f"training accuracy on this view {acc:.3f}. Move a knob on the left and the boundary refits."
    )
    st.info(f"**Try this:** {m['try_this']}")

st.divider()
st.subheader("What it learned")
st.caption(f"Fitted once on all of {data_name}. Below: {m['shows']}.")
X, y, features, targets = dataset(data_name)
with st.spinner("Fitting on the full dataset..."):
    model = fitted(data_name, name, params_key(params))

if name == "Soft Decision Tree":
    st.markdown("**The tree as rules.** Each gate read as a hard decision (`to_hard_tree()`); the export reports how often it agrees with the soft model it came from.")
    hard = model.to_hard_tree()
    agree = float((hard.predict(X) == model.predict(X)).mean())
    st.code(hard.export_text(feature_names=features), language=None)
    st.caption(f"Hard export agrees with the soft tree on {agree:.1%} of the training samples.")
    st.markdown("**One prediction, explained** (`explain()`).")
    idx = st.number_input("Sample", 0, len(y) - 1, 0, key="sdt_idx")
    ex = model.explain(X[idx:idx + 1], feature_names=features)[0]
    st.write(f"True class **{targets[int(y[idx])]}**, predicted **{targets[int(ex.predicted_class)]}** with p = {ex.probabilities[ex.predicted_class]:.3f}; leaf {ex.leaf} received {ex.leaf_probability:.2f} of the mass.")
    st.code(ex.to_text(), language=None)
    st.markdown("**Watch it learn.** The same tree fitted in stages with `warm_start=True`; the boundary is captured after each stage.")
    if st.button("Render the animation"):
        from neural_trees import SoftDecisionTree

        X2, y2, (x0, x1, y0, y1) = two_d(data_name)
        xs, ys = np.arange(x0, x1, 0.06), np.arange(y0, y1, 0.06)
        xx, yy = np.meshgrid(xs, ys)
        animator = SoftDecisionTree(depth=params.get("depth", 3), max_epochs=5, warm_start=True, random_state=0)
        frames = []
        for step in range(8):
            animator.fit(X2, y2)
            Z = animator.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            frames.append(go.Frame(data=[go.Heatmap(z=Z, x=xs, y=ys, showscale=False, opacity=0.35, colorscale=[[0, "#ffcccc"], [0.5, "#ccffcc"], [1, "#ccccff"]])], name=str(step), layout=dict(title=f"epoch {(step + 1) * 5}, train accuracy {animator.score(X2, y2):.3f}")))
        afig = go.Figure(data=[frames[0].data[0]] + [go.Scatter(x=X2[y2 == c, 0], y=X2[y2 == c, 1], mode="markers", marker=dict(size=5), name=f"class {c}") for c in np.unique(y2)], frames=frames)
        afig.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="white", title=frames[0].layout.title,
                           updatemenus=[dict(type="buttons", x=0.02, y=1.12, xanchor="left", buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)])])],
                           sliders=[dict(active=0, steps=[dict(method="animate", label=str((i + 1) * 5), args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for i in range(len(frames))], currentvalue=dict(prefix="epoch "))])
        afig.update_xaxes(showticklabels=False, showgrid=False)
        afig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(afig)

elif name == "Multivariate Tree":
    st.markdown(f"**{model.n_nodes_} oblique splits, depth {model.tree_depth_}.** The root split's weights: which features the first cut combines, and how.")
    w, b = model.get_split_weights()[0]
    order = np.argsort(-np.abs(w))[:10]
    fig = go.Figure(go.Bar(x=[w[i] for i in order][::-1], y=[features[i] for i in order][::-1], orientation="h", marker_color=["#2ca02c" if w[i] > 0 else "#d62728" for i in order][::-1]))
    fig.update_layout(height=60 + 28 * len(order), margin=dict(t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", xaxis_title=f"weight (bias {b:+.3f})")
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("CART would need a staircase of single-feature thresholds to draw this one line.")

elif name == "Omnivariate Tree":
    dist = model.get_split_type_distribution()
    st.markdown("**Which split type each node chose.** The tree decided per node; this is the tally.")
    fig = go.Figure(go.Bar(x=list(dist), y=list(dist.values()), marker_color=["#9467bd", "#17a2b8", "#ff7f0e"]))
    fig.update_layout(height=260, margin=dict(t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", yaxis_title="nodes")
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("univariate: one feature and a threshold (a CART node); linear: a hyperplane; nonlinear: a small network. With `selection='test'` the simplest type that is not significantly worse wins instead, and the design-decisions page measures what that costs.")

elif name == "Hierarchical MoE":
    st.markdown("**The routing as rules** (`to_hard_router()`): each gate read as a hard decision, so you can see which region goes to which expert.")
    st.code(model.to_hard_router().export_text(feature_names=features), language=None)

elif name == "GAL Network":
    hist = pd.DataFrame(model.architecture_history_)
    st.markdown(f"**It ended with {model.n_hidden_final_} hidden units.** How it got there: every grow and prune decision during training.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["n_hidden"], mode="lines", line=dict(color="#d62728"), name="hidden units"))
    for action, sym, col in (("grow", "triangle-up", "#2ca02c"), ("prune", "triangle-down", "#9467bd")):
        h = hist[hist["action"] == action]
        if len(h):
            fig.add_trace(go.Scatter(x=h["epoch"], y=h["n_hidden"], mode="markers", marker=dict(symbol=sym, size=10, color=col), name=action))
    fig.update_layout(height=300, margin=dict(t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="epoch", yaxis_title="hidden units")
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("A new unit is fitted to the residual error before it is installed (`growth_init='residual'`), which measured smaller networks than random units; the design-decisions page has the numbers.")

elif name == "Weighted KNN":
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from neural_trees import WeightedKNN

    st.markdown("**How much k matters here.** Five-fold accuracy for a sweep of k on this dataset.")
    ks = [1, 3, 5, 7, 11, 15, 21]
    accs = [cross_val_score(WeightedKNN(k=k), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean() for k in ks]
    fig = go.Figure(go.Scatter(x=ks, y=accs, mode="lines+markers", line=dict(color="#17becf")))
    fig.update_layout(height=280, margin=dict(t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", xaxis_title="k", yaxis_title="accuracy")
    st.plotly_chart(fig, config={"displayModeBar": False})

elif name == "Naive Bayes":
    st.markdown("**What it stored:** one mean per class per feature (standardised units). The gaps between rows are what it classifies with.")
    means = pd.DataFrame([t["mean"] for t in model.theta_], index=[str(t) for t in targets], columns=features)
    st.dataframe(means.round(2), width="stretch")

elif name == "CART (sklearn)":
    from sklearn.tree import export_text

    st.markdown("**The tree as rules** (scikit-learn's `export_text`).")
    st.code(export_text(model, feature_names=list(features)), language=None)

elif name == "Random Forest":
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)[:10]
    st.markdown("**Feature importances**, averaged over the trees.")
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=list(imp.index)[::-1], orientation="h", marker_color="#8c564b"))
    fig.update_layout(height=60 + 28 * len(imp), margin=dict(t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, config={"displayModeBar": False})

elif name == "SVM (RBF)":
    st.markdown(f"**{int(model.n_support_.sum())} support vectors** out of {len(y)} samples define the boundary; the rest of the data could be deleted without changing it.")

ui.next_step(
    f"Put {name} up against the three baselines on {data_name}, same folds for all, and see whether it holds up.",
    f"Compare {name} with the baselines",
    "views/compare.py",
    cmp_data=data_name, cmp_reset=[name] + BASELINE_MODELS,
)
glossary(["decision boundary", "standardised", "accuracy ± sd"])

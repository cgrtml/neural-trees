"""
Structure diagrams for the model page, as Graphviz DOT.

Streamlit renders DOT in the browser, so nothing has to be installed on
the server. Each function draws the *fitted* model: for a soft tree every
gate says which features decide it and how much of the data reaches it,
every leaf says what it predicts; for the hard trees the split type per
node; for the mixture of experts the routing tree; for the GAL network the
layer it grew to.
"""

import numpy as np

LIB = "#2F7D5B"
BASE = "#4A6FA5"
INK = "#15202B"
MUTED = "#6B7680"
PAPER = "#FFFFFF"
LEAF = "#EEF3F0"
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

_HEAD = (
    'digraph G {{ rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.8; splines=true;\n'
    '  node [shape=box, style="rounded,filled", fillcolor="{paper}", color="{lib}", penwidth=1.2, margin="0.12,0.06", '
    'fontname="IBM Plex Sans, Helvetica", fontsize=12, fontcolor="{ink}"];\n'
    '  edge [color="{muted}", arrowsize=0.7, fontname="IBM Plex Mono, Menlo", fontsize=10, fontcolor="{muted}"];\n'
)


def _esc(s: str) -> str:
    return str(s).replace('"', "'").replace("\\", "/").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _head() -> str:
    return _HEAD.format(paper=PAPER, ink=INK, lib=LIB, muted=MUTED)


def _tint(hex_colour: str, amount: float = 0.82) -> str:
    """The colour mixed with white, for fills that keep text readable."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _class_colour(k: int) -> str:
    return CLASS_COLORS[int(k) % len(CLASS_COLORS)]


def _html(title, lines, small=None):
    """An HTML-like label: bold title, detail lines, an optional small grey line."""
    body = f'<B>{title}</B>'
    for ln in lines:
        body += f'<BR/>{ln}'
    if small:
        body += f'<BR/><FONT POINT-SIZE="9" COLOR="{MUTED}">{small}</FONT>'
    return f"<{body}>"


def soft_tree_dot(npt, X, feature_names, classes, max_terms=2, max_levels=3) -> str:
    """
    A fitted soft tree (its numpy export) on data X: gates show the largest
    weights and the share of X that reaches them, edges the mean probability
    of taking that branch, leaves the predicted class (or value) and the
    share of the data that arrives.
    """
    X = np.asarray(X, dtype=np.float64)
    n_internal = npt.n_internal_
    logits = np.exp(npt.log_beta_) * (X @ npt.weights_.T + npt.biases_)
    p_right = 1.0 / (1.0 + np.exp(-logits))
    n_nodes = n_internal + npt.n_leaves_
    mu = np.zeros((len(X), n_nodes))
    mu[:, 0] = 1.0
    for i in range(n_internal):
        if npt.is_split_[i]:
            mu[:, 2 * i + 1] = mu[:, i] * (1 - p_right[:, i])
            mu[:, 2 * i + 2] = mu[:, i] * p_right[:, i]
    share = mu.mean(axis=0)
    lines = [_head()]

    def leaf_parts(node):
        """(title, detail, class index or None) for a leaf."""
        row = npt.node_logits_[node]
        if npt.kind == "classifier":
            p = np.exp(row - row.max())
            p /= p.sum()
            k = int(np.argmax(p))
            name = classes[k] if k < len(classes) else k
            return _esc(name), f"p = {p[k]:.2f}", k
        v = row * npt.y_scale_ + npt.y_mean_
        return ("value " + (f"{v[0]:,.3g}" if npt.single_output_ else ", ".join(f"{x:,.3g}" for x in v))), "", None

    def width_for(sh):
        return 1.0 + 1.4 * float(sh) ** 0.5

    def level_of(node):
        return int(np.floor(np.log2(node + 1)))

    def subtree_summary(node):
        """The deeper part of the tree folded into one box: how much data it holds and what most of it predicts."""
        stack, leaves = [node], []
        while stack:
            n = stack.pop()
            if n >= n_internal or not npt.is_split_[n]:
                leaves.append(n)
            else:
                stack.extend([2 * n + 1, 2 * n + 2])
        total = float(share[leaves].sum())
        if npt.kind == "classifier":
            rows = npt.node_logits_[leaves]
            p = np.exp(rows - rows.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            mix = (share[leaves][:, None] * p).sum(axis=0) / max(total, 1e-12)
            k = int(np.argmax(mix))
            return _esc(classes[k] if k < len(classes) else k), f"{len(leaves)} leaves below, mostly this", total, k
        return "values", f"{len(leaves)} leaves below", total, None

    folded = set()
    for node in range(n_nodes):
        if share[node] < 1e-6 and node != 0:
            continue
        if level_of(node) > max_levels:
            continue
        acting_leaf = node >= n_internal or not npt.is_split_[node]
        if acting_leaf:
            title, detail, k = leaf_parts(node)
            fill = _tint(_class_colour(k)) if k is not None else LEAF
            border = _class_colour(k) if k is not None else MUTED
            lines.append(f'  n{node} [fillcolor="{fill}", color="{border}", width={width_for(share[node]):.2f}, '
                         f'label={_html(title, [detail] if detail else [], f"{share[node]:.0%} of the data")}];')
        elif level_of(node) == max_levels:
            title, detail, total, k = subtree_summary(node)
            fill = _tint(_class_colour(k), 0.9) if k is not None else LEAF
            border = _class_colour(k) if k is not None else MUTED
            lines.append(f'  n{node} [style="rounded,filled,dashed", fillcolor="{fill}", color="{border}", width={width_for(total):.2f}, '
                         f'label={_html(title, [detail], f"{total:.0%} of the data")}];')
            folded.add(node)
        else:
            w = npt.weights_[node]
            order = np.argsort(-np.abs(w))[:max_terms]
            terms = [f"{w[j]:+.2f} × {_esc(feature_names[j])}" for j in order]
            if len(w) > max_terms:
                terms.append(f"+ {len(w) - max_terms} more")
            lines.append(f'  n{node} [width={width_for(share[node]):.2f}, label={_html(f"gate {node}", terms, f"{share[node]:.0%} of the data")}];')
            for child, right in ((2 * node + 1, False), (2 * node + 2, True)):
                if share[child] < 1e-6 or level_of(child) > max_levels:
                    continue
                pr = (p_right[:, node] if right else 1 - p_right[:, node])
                weighted = float((mu[:, node] * pr).sum() / max(mu[:, node].sum(), 1e-12))
                side = "right" if right else "left"
                lines.append(f'  n{node} -> n{child} [penwidth={0.6 + 3.0 * weighted:.2f}, taillabel="{weighted:.2f}", '
                             f'labelangle={-35 if right else 35}, labeldistance=2.2, tooltip="{side}: mean probability {weighted:.2f}"];')
    lines.append("}")
    return "\n".join(lines)


def _walk_hard(node, feature_names, classes, lines, counter, describe, level=0, max_levels=3):
    """Shared recursion for the multivariate and omnivariate node trees; deeper levels fold into one box."""
    my = counter[0]
    counter[0] += 1
    if level >= max_levels and not (node.is_leaf or node.left is None):
        lines.append(f'  n{my} [style="rounded,filled,dashed", fillcolor="{LEAF}", label="subtree"];')
        return my
    if node.is_leaf or node.left is None:
        d = node.distribution
        k = int(np.argmax(d)) if d is not None else 0
        name = classes[k] if k < len(classes) else k
        p = float(d[k]) if d is not None else 0.0
        lines.append(f'  n{my} [fillcolor="{_tint(_class_colour(k))}", color="{_class_colour(k)}", label={_html(_esc(name), [f"p = {p:.2f}"])}];')
        return my
    lines.append(f'  n{my} [label={describe(node)}];')
    left = _walk_hard(node.left, feature_names, classes, lines, counter, describe, level + 1, max_levels)
    right = _walk_hard(node.right, feature_names, classes, lines, counter, describe, level + 1, max_levels)
    lines.append(f'  n{my} -> n{left} [taillabel="no", labelangle=35, labeldistance=2];')
    lines.append(f'  n{my} -> n{right} [taillabel="yes", labelangle=-35, labeldistance=2];')
    return my


def multivariate_dot(model, feature_names, classes, max_terms=2) -> str:
    def describe(node):
        w = node.weights
        order = np.argsort(-np.abs(w))[:max_terms]
        terms = [f"{w[j]:+.2f} × {_esc(feature_names[j])}" for j in order]
        if len(w) > max_terms:
            terms.append(f"+ {len(w) - max_terms} more")
        return _html("oblique cut", terms, f"{node.bias:+.2f} &gt; 0 ?")

    lines = [_head()]
    _walk_hard(model.root_, feature_names, classes, lines, [0], describe)
    lines.append("}")
    return "\n".join(lines)


def omnivariate_dot(model, feature_names, classes) -> str:
    colours = {"univariate": "#E9DFF5", "linear": "#DCEFF2", "nonlinear": "#FBE7D3"}

    def describe(node):
        kind = node.split_type or "?"
        detail = []
        clf = node.classifier
        if kind == "univariate" and hasattr(clf, "tree_"):
            j, t = int(clf.tree_.feature[0]), float(clf.tree_.threshold[0])
            detail = [f"{_esc(feature_names[j])} &gt; {t:.2f} ?"]
        elif kind == "linear" and hasattr(clf, "coef_"):
            w = np.ravel(clf.coef_)
            order = np.argsort(-np.abs(w))[:2]
            detail = [f"{w[j]:+.2f} × {_esc(feature_names[j])}" for j in order]
        elif kind == "nonlinear":
            detail = ["a small network decides"]
        return _html(f"{kind} split", detail)

    lines = [_head()]
    _walk_hard(model.root_, feature_names, classes, lines, [0], describe)
    out = []
    for ln in lines:
        for kind, col in colours.items():
            if f"<B>{kind} split</B>" in ln:
                ln = ln.replace("[label=", f'[fillcolor="{col}", color="{MUTED}", label=')
        out.append(ln)
    out.append("}")
    return "\n".join(out)


def hme_dot(model, X, classes) -> str:
    """The routing tree of a fitted mixture: gates and the experts they lead to, with the share of X routed to each expert."""
    router = model.to_hard_router()
    b, depth = router.branching_factor, router.depth
    routed = router._route(np.asarray(X, dtype=np.float64))
    n_experts = b ** depth
    counts = np.bincount(routed, minlength=n_experts) / max(len(X), 1)
    lines = [_head()]
    # gates level by level
    offset = 0
    for level in range(depth):
        for pos in range(b ** level):
            lines.append(f'  g{level}_{pos} [label={_html("gate", [], f"level {level}")}];')
        offset += b ** level
    for level in range(depth - 1):
        for pos in range(b ** level):
            for c in range(b):
                lines.append(f"  g{level}_{pos} -> g{level + 1}_{pos * b + c};")
    for pos in range(b ** (depth - 1)):
        for c in range(b):
            e = pos * b + c
            lines.append(f'  e{e} [fillcolor="{_tint(BASE, 0.85)}", color="{BASE}", width={1.0 + 1.4 * float(counts[e]) ** 0.5:.2f}, label={_html(f"expert {e}", [], f"{counts[e]:.0%} of the data")}];')
            lines.append(f"  g{depth - 1}_{pos} -> e{e};")
    lines.append("}")
    return "\n".join(lines)


def gal_dot(n_features, n_hidden, n_classes) -> str:
    """Input, the hidden layer it grew to, output; drawn as three boxes with the counts, not one node per unit."""
    return _head() + (
        f'  rankdir=LR; ranksep=1.2;\n'
        f'  i [label={_html("input", [f"{n_features} features"])}];\n'
        f'  h [fillcolor="{_tint(LIB, 0.85)}", label={_html("hidden layer", [f"{n_hidden} units"], "grown and pruned during training")}];\n'
        f'  o [label={_html("output", [f"{n_classes} classes"])}];\n'
        f'  i -> h [taillabel="every input to every unit", labelangle=0, labeldistance=3]; h -> o;\n'
        "}"
    )

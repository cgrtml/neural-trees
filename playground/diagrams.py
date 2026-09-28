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
PAPER = "#FBFCFB"
LEAF = "#EEF3F0"

_HEAD = (
    'digraph G {{ rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;\n'
    '  node [shape=box, style="rounded,filled", fillcolor="{paper}", color="{ink}", fontname="IBM Plex Sans, Helvetica", fontsize=10, fontcolor="{ink}"];\n'
    '  edge [color="{ink}", fontname="IBM Plex Mono, Menlo", fontsize=9, fontcolor="{ink}"];\n'
)


def _esc(s: str) -> str:
    return str(s).replace('"', "'").replace("\\", "/")


def _head() -> str:
    return _HEAD.format(paper=PAPER, ink=INK)


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

    def leaf_label(node):
        row = npt.node_logits_[node]
        if npt.kind == "classifier":
            p = np.exp(row - row.max())
            p /= p.sum()
            k = int(np.argmax(p))
            name = classes[k] if k < len(classes) else k
            return f"{_esc(name)}\\np = {p[k]:.2f}"
        v = row * npt.y_scale_ + npt.y_mean_
        return f"value {v[0]:,.3g}" if npt.single_output_ else "value " + ", ".join(f"{x:,.3g}" for x in v)

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
            what = f"mostly {_esc(classes[k] if k < len(classes) else k)}"
        else:
            what = "values"
        return f"subtree: {len(leaves)} leaves\\n{what}\\n{total:.0%} of data"

    folded = set()
    for node in range(n_nodes):
        if share[node] < 1e-6 and node != 0:
            continue
        if level_of(node) > max_levels:
            continue
        acting_leaf = node >= n_internal or not npt.is_split_[node]
        if acting_leaf:
            lines.append(f'  n{node} [fillcolor="{LEAF}", label="{leaf_label(node)}\\n{share[node]:.0%} of data"];')
        elif level_of(node) == max_levels:
            lines.append(f'  n{node} [style="rounded,filled,dashed", fillcolor="{LEAF}", label="{subtree_summary(node)}"];')
            folded.add(node)
        else:
            w = npt.weights_[node]
            order = np.argsort(-np.abs(w))[:max_terms]
            terms = "\\n".join(f"{w[j]:+.2f} {_esc(feature_names[j])}" for j in order)
            more = f"\\n(+{len(w) - max_terms} more)" if len(w) > max_terms else ""
            lines.append(f'  n{node} [label="gate {node}\\n{terms}{more}\\n{share[node]:.0%} of data"];')
            for child, right in ((2 * node + 1, False), (2 * node + 2, True)):
                if share[child] < 1e-6 or level_of(child) > max_levels:
                    continue
                pr = (p_right[:, node] if right else 1 - p_right[:, node])
                weighted = float((mu[:, node] * pr).sum() / max(mu[:, node].sum(), 1e-12))
                lines.append(f'  n{node} -> n{child} [label="{"right" if right else "left"} {weighted:.2f}"];')
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
        lines.append(f'  n{my} [fillcolor="{LEAF}", label="{_esc(name)}\\np = {p:.2f}"];')
        return my
    lines.append(f'  n{my} [label="{describe(node)}"];')
    left = _walk_hard(node.left, feature_names, classes, lines, counter, describe, level + 1, max_levels)
    right = _walk_hard(node.right, feature_names, classes, lines, counter, describe, level + 1, max_levels)
    lines.append(f'  n{my} -> n{left} [label="no"];')
    lines.append(f'  n{my} -> n{right} [label="yes"];')
    return my


def multivariate_dot(model, feature_names, classes, max_terms=2) -> str:
    def describe(node):
        w = node.weights
        order = np.argsort(-np.abs(w))[:max_terms]
        terms = "\\n".join(f"{w[j]:+.2f} {_esc(feature_names[j])}" for j in order)
        more = f"\\n(+{len(w) - max_terms} more)" if len(w) > max_terms else ""
        return f"{terms}{more}\\n{node.bias:+.2f} > 0 ?"

    lines = [_head()]
    _walk_hard(model.root_, feature_names, classes, lines, [0], describe)
    lines.append("}")
    return "\n".join(lines)


def omnivariate_dot(model, feature_names, classes) -> str:
    colours = {"univariate": "#E9DFF5", "linear": "#DCEFF2", "nonlinear": "#FBE7D3"}

    def describe(node):
        kind = node.split_type or "?"
        detail = ""
        clf = node.classifier
        if kind == "univariate" and hasattr(clf, "tree_"):
            j, t = int(clf.tree_.feature[0]), float(clf.tree_.threshold[0])
            detail = f"\\n{_esc(feature_names[j])} > {t:.2f} ?"
        elif kind == "linear" and hasattr(clf, "coef_"):
            w = np.ravel(clf.coef_)
            order = np.argsort(-np.abs(w))[:2]
            detail = "\\n" + "\\n".join(f"{w[j]:+.2f} {_esc(feature_names[j])}" for j in order)
        elif kind == "nonlinear":
            detail = "\\nsmall network"
        return f"{kind} split{detail}"

    lines = [_head()]
    # colour by split type: patch node lines after the walk
    _walk_hard(model.root_, feature_names, classes, lines, [0], describe)
    out = []
    for ln in lines:
        for kind, col in colours.items():
            if f'label="{kind} split' in ln:
                ln = ln.replace("[label=", f'[fillcolor="{col}", label=')
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
            lines.append(f'  g{level}_{pos} [label="gate\\nlevel {level}"];')
        offset += b ** level
    for level in range(depth - 1):
        for pos in range(b ** level):
            for c in range(b):
                lines.append(f"  g{level}_{pos} -> g{level + 1}_{pos * b + c};")
    for pos in range(b ** (depth - 1)):
        for c in range(b):
            e = pos * b + c
            lines.append(f'  e{e} [fillcolor="{LEAF}", label="expert {e}\\n{counts[e]:.0%} of data"];')
            lines.append(f"  g{depth - 1}_{pos} -> e{e};")
    lines.append("}")
    return "\n".join(lines)


def gal_dot(n_features, n_hidden, n_classes) -> str:
    """Input, the hidden layer it grew to, output; drawn as three boxes with the counts, not one node per unit."""
    return _head() + (
        f'  rankdir=LR;\n'
        f'  i [label="{n_features} inputs"];\n'
        f'  h [fillcolor="{LEAF}", label="{n_hidden} hidden units\\n(grown and pruned\\nduring training)"];\n'
        f'  o [label="{n_classes} outputs"];\n'
        f'  i -> h [label="full"]; h -> o [label="full"];\n'
        "}"
    )

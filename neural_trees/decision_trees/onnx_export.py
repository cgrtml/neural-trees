"""
A fitted soft tree as an ONNX graph.

Built from the numpy export, so it needs neither torch nor the training
code: the gates are one matrix product and a sigmoid, each acting leaf's
arrival probability is the product of the gate probabilities on its path,
and the prediction is that mixture applied to the leaf table. The graph
uses only standard operators (MatMul, Add, Sigmoid, Gather, ReduceProd,
Concat, ArgMax), so it runs in ONNX Runtime and in every other ONNX
consumer, on servers, in browsers and on phones, and predicts the same
thing as the torch model to float32 precision.

`onnx` is an optional dependency: `pip install neural-trees[onnx]`.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from neural_trees import __version__


def _paths(depth: int, is_split: np.ndarray) -> List[Tuple[int, List[Tuple[int, bool]]]]:
    """(node, [(gate, went_right), ...]) for every node that acts as a leaf, breadth-first."""
    n_internal = 2 ** depth - 1
    out: List[Tuple[int, List[Tuple[int, bool]]]] = []
    stack: List[Tuple[int, List[Tuple[int, bool]]]] = [(0, [])]
    while stack:
        node, path = stack.pop(0)
        if node >= n_internal or not is_split[node]:
            out.append((node, path))
            continue
        stack.append((2 * node + 1, path + [(node, False)]))
        stack.append((2 * node + 2, path + [(node, True)]))
    return out


def to_onnx(tree, opset: int = 17, name: str = "neural-trees soft tree"):
    """
    Convert a :class:`~neural_trees.NumpySoftTree` to an ``onnx.ModelProto``.

    Input ``X`` is float32 of shape ``(n, n_features)``, the preprocessed
    features the tree was trained on. A classifier outputs ``probabilities``
    ``(n, n_classes)`` and ``label`` ``(n,)`` holding the class values
    (int64 or string, as they were at fit); a regressor outputs ``value``
    ``(n,)`` or ``(n, n_outputs)`` in target units.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("ONNX export needs the onnx package: pip install neural-trees[onnx]") from exc

    beta = np.exp(tree.log_beta_)                                     # (n_internal,)
    W = (beta[:, None] * tree.weights_).T.astype(np.float32)          # (n_features, n_internal)
    b = (beta * tree.biases_).astype(np.float32)                      # (n_internal,)
    n_internal = tree.n_internal_
    leaves = _paths(tree.depth, tree.is_split_)

    inits = [
        numpy_helper.from_array(W, "gate_weights"),
        numpy_helper.from_array(b, "gate_biases"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "gate_weights"], ["gate_dot"]),
        helper.make_node("Add", ["gate_dot", "gate_biases"], ["gate_logits"]),
        helper.make_node("Sigmoid", ["gate_logits"], ["p_right"]),
        helper.make_node("Sub", ["one", "p_right"], ["p_left"]),
        # column i is the left probability of gate i, column n_internal + i the right one
        helper.make_node("Concat", ["p_left", "p_right"], ["pq"], axis=1),
    ]
    mu_names = []
    for k, (node, path) in enumerate(leaves):
        name_k = f"mu_{k}"
        if path:
            idx = np.array([g + (n_internal if right else 0) for g, right in path], dtype=np.int64)
            inits.append(numpy_helper.from_array(idx, f"path_{k}"))
            nodes.append(helper.make_node("Gather", ["pq", f"path_{k}"], [f"pick_{k}"], axis=1))
            nodes.append(helper.make_node("ReduceProd", [f"pick_{k}"], [name_k], axes=[1], keepdims=1))
        else:
            # A tree with no split at all: the root is the only leaf and receives everything.
            inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), f"path_{k}"))
            nodes.append(helper.make_node("Gather", ["pq", f"path_{k}"], [f"pick_{k}"], axis=1))
            nodes.append(helper.make_node("Sub", [f"pick_{k}", f"pick_{k}"], [f"zero_{k}"]))
            nodes.append(helper.make_node("Add", [f"zero_{k}", "one"], [name_k]))
        mu_names.append(name_k)
    if len(mu_names) == 1:
        nodes.append(helper.make_node("Identity", mu_names, ["mu"]))
    else:
        nodes.append(helper.make_node("Concat", mu_names, ["mu"], axis=1))
    leaf_index = np.array([node for node, _ in leaves])
    table = tree.node_logits_[leaf_index]                             # (L, K)

    metadata = {
        "library": f"neural-trees {__version__}",
        "kind": tree.kind,
        "depth": str(tree.depth),
        "feature_names": ",".join(tree.feature_names_ or []),
    }
    if tree.kind == "classifier":
        table = np.exp(table - table.max(axis=1, keepdims=True))
        table = (table / table.sum(axis=1, keepdims=True)).astype(np.float32)
        inits.append(numpy_helper.from_array(table, "leaf_distributions"))
        nodes.append(helper.make_node("MatMul", ["mu", "leaf_distributions"], ["probabilities"]))
        nodes.append(helper.make_node("ArgMax", ["probabilities"], ["label_index"], axis=1, keepdims=0))
        classes = tree.classes_
        if np.issubdtype(classes.dtype, np.integer) or np.issubdtype(classes.dtype, np.bool_):
            inits.append(numpy_helper.from_array(classes.astype(np.int64), "classes"))
            label_type = TensorProto.INT64
        else:
            inits.append(helper.make_tensor("classes", TensorProto.STRING, [len(classes)], [str(c).encode("utf-8") for c in classes]))  # type: ignore[misc]
            label_type = TensorProto.STRING
        nodes.append(helper.make_node("Gather", ["classes", "label_index"], ["label"], axis=0))
        outputs = [
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["n", len(classes)]),
            helper.make_tensor_value_info("label", label_type, ["n"]),
        ]
        metadata["classes"] = ",".join(str(c) for c in classes)
    else:
        values = (table * tree.y_scale_).astype(np.float32)           # (L, n_outputs), target units
        inits.append(numpy_helper.from_array(values, "leaf_values"))
        inits.append(numpy_helper.from_array(np.asarray(tree.y_mean_, dtype=np.float32), "target_mean"))
        nodes.append(helper.make_node("MatMul", ["mu", "leaf_values"], ["value_centred"]))
        if tree.single_output_:
            nodes.append(helper.make_node("Add", ["value_centred", "target_mean"], ["value_2d"]))
            inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat"))
            nodes.append(helper.make_node("Reshape", ["value_2d", "flat"], ["value"]))
            outputs = [helper.make_tensor_value_info("value", TensorProto.FLOAT, ["n"])]
        else:
            nodes.append(helper.make_node("Add", ["value_centred", "target_mean"], ["value"]))
            outputs = [helper.make_tensor_value_info("value", TensorProto.FLOAT, ["n", values.shape[1]])]

    graph = helper.make_graph(
        nodes, "soft_tree",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["n", tree.n_features_in_])],
        outputs, initializer=inits,
    )
    model = helper.make_model(graph, producer_name="neural-trees", opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def save_onnx(tree, path: str, **kwargs: Any) -> None:
    """Write :func:`to_onnx` to a file."""
    import onnx

    onnx.save(to_onnx(tree, **kwargs), path)

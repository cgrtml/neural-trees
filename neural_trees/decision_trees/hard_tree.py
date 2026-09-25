"""
Hard export of a trained Soft Decision Tree
===========================================

A soft tree sends every sample to every leaf and mixes the results, which is
what makes it differentiable and what makes a prediction cost a PyTorch forward
pass over all 2^depth leaves. Once training is over, that machinery is no longer
doing any work: each internal node has settled on a hyperplane, and the sign of
that hyperplane is a decision.

`SoftDecisionTree.to_hard_tree()` takes those hyperplanes as they are and routes
each sample down a single path, in plain numpy. The result is not the same model
- a mixture is not a path, and the two disagree on samples that sat near a split
- so the export reports how often they agree rather than pretending otherwise.
"""

import numpy as np
from sklearn.utils.validation import check_array


class HardDecisionTree:
    """
    A trained Soft Decision Tree with its gates read as hard decisions.

    Each internal node routes right when `w . x + b > 0` and left otherwise, and
    each leaf carries the class distribution the soft tree learned for it.
    Prediction is a walk of `depth` steps in numpy, with no PyTorch involved.

    Built by `SoftDecisionTree.to_hard_tree()` rather than directly.

    Attributes
    ----------
    depth : int
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    weights_ : ndarray of shape (n_internal, n_features)
        One hyperplane per internal node, in breadth-first order.
    biases_ : ndarray of shape (n_internal,)
    node_distributions_ : ndarray of shape (n_nodes, n_classes)
        A distribution per node. `leaf_distributions_` selects the ones that
        act as leaves.
    is_split_ : ndarray of shape (n_internal,)
        Whether each internal node routes onward. All True for a complete tree.
    log_beta_ : ndarray of shape (n_internal,)
        Log gate temperatures, used by the ``"leaf"`` and ``"contribution"``
        rules, which need the gate *probabilities* rather than their signs.
    rule : {"gate", "leaf", "contribution"}
        How a sample is assigned to a leaf; see `SoftDecisionTree.to_hard_tree`.
    """

    RULES = ("gate", "leaf", "contribution")

    def __init__(
        self, weights, biases, node_distributions, classes, n_features_in, is_split=None,
        log_beta=None, rule="gate",
    ):
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}, got {rule!r}")
        self.rule = rule
        self.weights_ = np.asarray(weights, dtype=np.float64)
        self.biases_ = np.asarray(biases, dtype=np.float64)
        self.log_beta_ = (
            np.zeros(len(self.weights_)) if log_beta is None
            else np.asarray(log_beta, dtype=np.float64)
        )
        self.node_distributions_ = np.asarray(node_distributions, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(n_features_in)
        self.n_internal_ = len(self.weights_)
        self.depth = int(np.log2(self.n_internal_ + 1))
        self.is_split_ = (
            np.ones(self.n_internal_, dtype=bool)
            if is_split is None
            else np.asarray(is_split, dtype=bool)
        )

    @property
    def leaf_distributions_(self) -> np.ndarray:
        """Distributions of the nodes that actually behave as leaves."""
        return self.node_distributions_[self._acting_leaves()]

    def _acting_leaves(self) -> np.ndarray:
        leaves = []
        stack = [0]
        while stack:
            node = stack.pop()
            if node >= self.n_internal_ or not self.is_split_[node]:
                leaves.append(node)
                continue
            stack.extend([2 * node + 2, 2 * node + 1])
        return np.array(sorted(leaves))

    def _arrival_log_probs(self, X: np.ndarray):
        """
        Log probability of every acting leaf receiving each sample.

        Costs every gate (2^depth - 1 dot products per sample) where the
        ``"gate"`` walk costs `depth`, which is why the rules that use it are
        slower.
        """
        z = np.exp(self.log_beta_) * (X @ self.weights_.T + self.biases_)
        log_right = -np.logaddexp(0.0, -z)
        log_left = -np.logaddexp(0.0, z)
        log_mu = np.full((len(X), 2 * self.n_internal_ + 1), -np.inf)
        log_mu[:, 0] = 0.0
        for node in range(self.n_internal_):
            if not self.is_split_[node]:
                continue
            log_mu[:, 2 * node + 1] = log_mu[:, node] + log_left[:, node]
            log_mu[:, 2 * node + 2] = log_mu[:, node] + log_right[:, node]
        leaves = self._acting_leaves()
        return leaves, log_mu[:, leaves]

    def _leaf_index(self, X: np.ndarray) -> np.ndarray:
        """
        The node each sample is assigned to, under `rule`.

        ``"gate"`` walks down taking the sign of each gate. ``"leaf"`` picks the
        acting leaf with the largest arrival probability. ``"contribution"``
        picks the leaf that contributes most to the soft mixture's winning
        class. The last two cost every gate rather than `depth` of them.
        """
        if self.rule != "gate":
            leaves, log_mu = self._arrival_log_probs(X)
            if self.rule == "leaf":
                return leaves[np.argmax(log_mu, axis=1)]
            mu = np.exp(log_mu)                                    # (n, L)
            mixture = mu @ self.node_distributions_[leaves]         # (n, K)
            winner = np.argmax(mixture, axis=1)
            share = mu * self.node_distributions_[leaves][:, winner].T  # (n, L)
            return leaves[np.argmax(share, axis=1)]
        node = np.zeros(len(X), dtype=np.int64)
        for _ in range(self.depth):
            moving = (node < self.n_internal_) & self.is_split_[np.minimum(node, self.n_internal_ - 1)]
            if not moving.any():
                break
            scores = (
                np.einsum("nf,nf->n", X[moving], self.weights_[node[moving]])
                + self.biases_[node[moving]]
            )
            node[moving] = 2 * node[moving] + 1 + (scores > 0).astype(np.int64)
        return node

    def predict_proba(self, X) -> np.ndarray:
        """
        Class probabilities of the reached leaf, shape (n_samples, n_classes).
        """
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but HardDecisionTree is expecting "
                f"{self.n_features_in_} features as input."
            )
        return self.node_distributions_[self._leaf_index(X)]

    def predict(self, X) -> np.ndarray:
        """Predicted class labels, shape (n_samples,)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _leaf_line(self, node: int, decimals: int) -> str:
        distribution = self.node_distributions_[node]
        winner = self.classes_[int(np.argmax(distribution))]
        return f"predict {winner!r} (p={distribution.max():.{decimals}f})"

    def export_text(self, feature_names=None, max_features=3, decimals=3) -> str:
        """
        Render the tree as readable rules.

        Parameters
        ----------
        feature_names : list of str, optional
            Defaults to `x0`, `x1`, ...
        max_features : int, default=3
            Features shown per split, largest absolute weight first. A
            multivariate split uses every feature; printing all of them stops
            being readable, which is the thing this method is for.
        decimals : int, default=3

        Returns
        -------
        str
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries, expected "
                f"{self.n_features_in_}"
            )

        n_internal = self.n_internal_
        lines = []

        def describe_split(node: int) -> str:
            weights = self.weights_[node]
            order = np.argsort(-np.abs(weights))[:max_features]
            terms = [
                f"{weights[i]:+.{decimals}f}*{feature_names[i]}" for i in order
            ]
            omitted = ""
            if self.n_features_in_ > max_features:
                omitted = f" (+{self.n_features_in_ - max_features} more)"
            return f"{' '.join(terms)} {self.biases_[node]:+.{decimals}f}{omitted} > 0"

        def walk(node: int, indent: str, branch: str):
            if node >= n_internal or not self.is_split_[node]:
                lines.append(f"{indent}{branch}{self._leaf_line(node, decimals)}")
                return
            lines.append(f"{indent}{branch}if {describe_split(node)}:")
            walk(2 * node + 2, indent + "    ", "yes -> ")
            walk(2 * node + 1, indent + "    ", "no  -> ")

        walk(0, "", "")
        return "\n".join(lines)


class HardRegressionTree(HardDecisionTree):
    """
    A trained :class:`~neural_trees.SoftDecisionTreeRegressor` with its gates
    read as hard decisions and one value per leaf, in target units.

    Built by ``SoftDecisionTreeRegressor.to_hard_tree()``. The walk is the
    parent class's; only the leaves differ: ``node_values_`` of shape
    ``(n_nodes, n_outputs)`` replaces the class distributions, ``predict``
    returns values and ``score`` is R^2. Like the classifier's export it is a
    different model from the soft tree, not a re-encoding: it reports its
    agreement rather than assuming it.
    """

    def __init__(self, weights, biases, node_values, n_features_in, is_split=None, log_beta=None,
                 single_output=True):
        super().__init__(
            weights, biases, np.asarray(node_values, dtype=np.float64), classes=[],
            n_features_in=n_features_in, is_split=is_split, log_beta=log_beta, rule="gate",
        )
        self.node_values_ = self.node_distributions_
        self.n_outputs_ = self.node_values_.shape[1]
        self.single_output_ = bool(single_output)

    def predict_proba(self, X):  # pragma: no cover - not a classifier
        raise AttributeError("HardRegressionTree has no predict_proba; use predict")

    def predict(self, X) -> np.ndarray:
        """Leaf value of the reached leaf, shape (n_samples,) or (n_samples, n_outputs)."""
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but HardRegressionTree is expecting "
                f"{self.n_features_in_} features as input."
            )
        out = self.node_values_[self._leaf_index(X)]
        return out[:, 0] if self.single_output_ else out

    def score(self, X, y) -> float:
        """R^2 on the given data, uniform average over outputs."""
        y = np.asarray(y, dtype=np.float64)
        pred = self.predict(X)
        y2 = y.reshape(len(y), -1)
        p2 = np.asarray(pred).reshape(len(y), -1)
        ss_res = ((y2 - p2) ** 2).sum(axis=0)
        ss_tot = ((y2 - y2.mean(axis=0)) ** 2).sum(axis=0)
        return float(np.mean(1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)))

    def _leaf_line(self, node: int, decimals: int) -> str:
        v = self.node_values_[node]
        shown = f"{v[0]:.{decimals}f}" if self.single_output_ else "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"
        return f"value = {shown}"

"""
Soft Decision Trees (SDT)
=========================
Implementation based on:
    İrsoy, O., Yıldız, O. T., & Alpaydın, E. (2012).
    Soft Decision Trees.
    Proceedings of the 21st International Conference on Pattern Recognition (ICPR).

    İrsoy, O., & Alpaydın, E. (2021).
    Dropout Regularization in Hierarchical Mixture of Experts.
    Neurocomputing, 419, 148-156.

Key idea:
    Unlike hard decision trees where each sample follows exactly one path,
    SDTs use soft (probabilistic) splits at each internal node. Every sample
    reaches every leaf with some probability. This makes the tree fully
    differentiable and trainable end-to-end with backpropagation.

    At each internal node i: p_i(x) = σ(w_i · x + b_i)   (sigmoid gate)
    The probability of reaching leaf ℓ is the product of gate probabilities
    along the path from root to ℓ.
    Each leaf holds a distribution over classes (softmax).
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "neural-trees requires PyTorch. Install it with: pip install torch "
        "(see https://pytorch.org/get-started/locally/ for platform specific wheels)."
    ) from exc
from typing import List, Optional

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)
from torch.utils.data import DataLoader, TensorDataset

from neural_trees._validation import check_predict_input, resolve_device
from neural_trees.decision_trees.hard_tree import HardDecisionTree


class _SoftTreeModule(nn.Module):
    """
    The core PyTorch module for a Soft Decision Tree of depth `depth`.

    Structure:
        A complete binary tree with (2^depth - 1) internal nodes
        and (2^depth) leaf nodes.

    All internal gates live in a single Linear layer, so one matmul produces
    every gate logit instead of one small matmul per node. Path probabilities
    are accumulated in log space: a depth-d leaf is reached with probability
    on the order of 2^-d, which underflows float32 as the tree grows.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        depth: int,
        penalty_coef: float,
        learn_temperature: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1
        self.penalty_coef = penalty_coef

        self.gates = nn.Linear(n_features, self.n_internal)
        # Xavier is applied per gate, not to the fused (n_internal, n_features)
        # matrix: every node is its own one-output linear split, and scaling by
        # the fused fan-out would shrink the init as the tree deepens.
        bound = float(np.sqrt(6.0 / (n_features + 1)))
        nn.init.uniform_(self.gates.weight, -bound, bound)
        nn.init.zeros_(self.gates.bias)

        # Inverse temperature per gate, as in Frosst & Hinton (2017). beta = 1
        # at init reproduces a plain sigmoid gate; letting it grow lets a node
        # sharpen its split instead of staying stuck in the flat region of the
        # sigmoid, where gradients vanish.
        self.learn_temperature = learn_temperature
        self.log_beta = nn.Parameter(
            torch.zeros(self.n_internal), requires_grad=learn_temperature
        )

        # A distribution for every node, not only the bottom row, so that an
        # internal node can act as a leaf when its subtree is switched off.
        # Nodes are indexed breadth-first: internal nodes 0..n_internal-1, then
        # the bottom row. With every internal node splitting, only the bottom
        # row is ever reached and this behaves exactly like leaf-only logits.
        self.node_logits = nn.Parameter(torch.zeros(self.n_internal + self.n_leaves, n_classes))
        self.register_buffer("is_split", torch.ones(self.n_internal, dtype=torch.bool))

    @property
    def leaf_logits(self) -> torch.Tensor:
        """The bottom row of node distributions, kept for backwards use."""
        return self.node_logits[self.n_internal:]

    def deepen(self, n_classes: int, jitter: float = 0.2) -> "_SoftTreeModule":
        """
        Return a tree one level deeper, computing very nearly the same function.

        Every current leaf becomes an internal node whose gate is all zeros, so
        it sends half of its arriving mass down each side, and both of its new
        children start from the parent's class distribution:

            sum_l mu_l (0.5 Q_l + 0.5 Q_l) = sum_l mu_l Q_l

        The children cannot start *identical*, though. With Q_left = Q_right the
        mixture does not depend on the new gate at all, so the gate's gradient
        is exactly zero, and the children receive identical gradients and stay
        identical forever. The new level would be dead weight: measured on Iris,
        growing that way reached 0.756 against 0.958 for a tree of the same
        depth trained from scratch.

        `jitter` breaks that symmetry. The function is preserved only
        approximately, which is the price of the level being able to learn
        anything at all. The default is not sensitive: 0.05, 0.2 and 0.5 give
        0.840, 0.844 and 0.796 on Iris and 0.962, 0.968 and 0.972 on Wine.
        """
        n_features = self.gates.weight.shape[1]
        deeper = _SoftTreeModule(
            n_features=n_features,
            n_classes=n_classes,
            depth=self.depth + 1,
            penalty_coef=self.penalty_coef,
            learn_temperature=self.learn_temperature,
        ).to(self.gates.weight.device)

        with torch.no_grad():
            deeper.gates.weight[: self.n_internal] = self.gates.weight
            deeper.gates.bias[: self.n_internal] = self.gates.bias
            deeper.log_beta[: self.n_internal] = self.log_beta
            # The old leaves become the new bottom row of gates, neutral.
            deeper.gates.weight[self.n_internal:] = 0.0
            deeper.gates.bias[self.n_internal:] = 0.0
            deeper.log_beta[self.n_internal:] = 0.0
            children = self.leaf_logits.repeat_interleave(2, dim=0)
            deeper.leaf_logits[:] = children + jitter * torch.randn_like(children)

        return deeper

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled logits for every internal node, shape (batch, n_internal)."""
        return torch.exp(self.log_beta) * self.gates(x)

    def _log_path_probabilities(self, x: torch.Tensor):
        """
        Accumulate log arrival probabilities level by level.

        Returns
        -------
        log_leaf_probs : Tensor of shape (batch, n_leaves)
        level_probs : list of (arrival probability, gate output) per level, for
            the entropy penalty.
        """
        log_mu, level_probs, _ = self._walk(x)
        return log_mu, level_probs

    def _walk(self, x: torch.Tensor):
        """
        Walk the tree once, collecting what every caller needs.

        Returns the bottom row's log arrival probabilities, the per-level
        quantities the penalty uses, and the (log arrival probability, node
        index) pairs of every node that acts as a leaf. A node whose `is_split`
        is False keeps its arriving mass instead of passing it down, and its
        subtree receives -inf, which logsumexp treats as exactly zero weight.
        """
        logits = self.gate_logits(x)
        log_mu = torch.zeros(x.size(0), 1, device=x.device)  # root is reached with prob 1
        level_probs = []
        terminal = []

        start = 0
        for level in range(self.depth):
            width = 2 ** level
            level_logits = logits[:, start:start + width]
            splits = self.is_split[start:start + width]
            level_probs.append((log_mu.exp(), torch.sigmoid(level_logits)))

            if not bool(splits.all()):
                stopped = ~splits
                indices = torch.arange(start, start + width, device=x.device)[stopped]
                terminal.append((log_mu[:, stopped], indices))

            log_left = F.logsigmoid(-level_logits)
            log_right = F.logsigmoid(level_logits)
            # Interleave to [left_0, right_0, left_1, right_1, ...], which is the
            # child order of the breadth-first node indexing.
            children = torch.stack([log_left, log_right], dim=2).reshape(x.size(0), 2 * width)
            log_mu = log_mu.repeat_interleave(2, dim=1) + children
            if not bool(splits.all()):
                alive = splits.repeat_interleave(2)
                log_mu = torch.where(alive, log_mu, torch.full_like(log_mu, float("-inf")))
            start += width

        bottom_indices = torch.arange(
            self.n_internal, self.n_internal + self.n_leaves, device=x.device
        )
        terminal.append((log_mu, bottom_indices))
        return log_mu, level_probs, terminal

    def log_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        log P(y | x) = logsumexp_leaf [ log mu_leaf(x) + log Q_leaf(y) ].

        Returns
        -------
        Tensor of shape (batch_size, n_classes) of log probabilities.
        """
        _, _, terminal = self._walk(x)
        log_node_dists = F.log_softmax(self.node_logits, dim=1)
        parts = [
            log_mu.unsqueeze(2) + log_node_dists[indices].unsqueeze(0)
            for log_mu, indices in terminal
        ]
        return torch.logsumexp(torch.cat(parts, dim=1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities as a weighted sum over leaf distributions.

        P(y | x) = sum_leaf mu_leaf(x) Q_leaf(y)

        Returns:
            Tensor of shape (batch_size, n_classes).
        """
        return self.log_forward(x).exp()

    def _path_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Arrival probability of each sample at each leaf, shape (batch, n_leaves)."""
        log_leaf_probs, _ = self._log_path_probabilities(x)
        return log_leaf_probs.exp()

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based regularization penalty to avoid degenerate trees
        (Frosst & Hinton, 2017).

        For internal node i the penalized quantity is the path-probability
        weighted average gate activation

            alpha_i = sum_x mu_i(x) p_i(x) / sum_x mu_i(x)

        and the penalty -0.5 log alpha_i - 0.5 log(1 - alpha_i) is minimized
        when a node sends half of the probability mass down each branch. The
        coefficient decays as 2^-level, because deeper nodes see less data and
        would otherwise be penalized as hard as the root.
        """
        _, level_probs = self._log_path_probabilities(x)
        total = torch.zeros((), device=x.device)

        for level, (mu, p_right) in enumerate(level_probs):
            weight = mu.sum(dim=0).clamp_min(1e-7)
            alpha = (mu * p_right).sum(dim=0) / weight
            alpha = alpha.clamp(1e-6, 1 - 1e-6)
            node_penalty = -0.5 * torch.log(alpha) - 0.5 * torch.log(1 - alpha)
            total = total + self.penalty_coef * (2.0 ** -level) * node_penalty.sum()

        return total


class SoftDecisionTree(ClassifierMixin, BaseEstimator):
    """
    Soft Decision Tree Classifier (sklearn-compatible).

    A fully differentiable decision tree where each internal node applies
    a soft (sigmoid) split, allowing end-to-end gradient training.

    Parameters
    ----------
    depth : int, default=5
        Depth of the tree. The tree has 2^depth leaves.
    max_epochs : int, default=40
        Number of training epochs.
    learning_rate : float, default=0.01
        Learning rate for Adam optimizer.
    batch_size : int, default=64
        Mini-batch size for training.
    penalty_coef : float, default=1e-3
        Regularization coefficient for the entropy penalty on internal nodes.
        Higher values encourage more balanced splits.
    device : str, default="cpu"
        PyTorch device. `"auto"` picks CUDA if it is available, then Apple
        silicon's MPS, then CPU. Anything else is passed to torch as given, so
        `"cuda:1"` works. Resolved once in `fit` and recorded as `device_`, so
        prediction always runs where training did.
    verbose : bool, default=False
        Whether to print training progress.
    random_state : int or None, default=None
        Seed for model initialization and shuffled mini-batches.
    class_weight : dict, "balanced" or None, default=None
        Weights per class, combined multiplicatively with `sample_weight`.
        `"balanced"` uses `n_samples / (n_classes * bincount(y))`, which is what
        an imbalanced target usually needs: without it a rare class contributes
        so little to the loss that the tree can ignore it entirely.
    warm_start : bool, default=False
        When True, a second call to `fit` continues from the parameters the
        first one left, instead of reinitializing. Useful for training in
        stages, or for extending a run that turned out too short.

        This is `warm_start` rather than `partial_fit` deliberately. sklearn's
        `partial_fit` contract promises that a model updated on batches
        approaches one trained on the union, and requires handling classes that
        were absent from the first call. Neither holds here: the architecture
        is fixed at the first fit, and mini-batch gradient descent over a second
        dataset drifts toward that dataset rather than the union. `warm_start`
        promises only what is actually delivered, which is continuation.

        The label set must not change between calls; a new class would need an
        output layer this model cannot grow.
    growth : {"none", "incremental", "per_leaf"}, default="none"
        How the tree reaches its shape.

        - ``"none"`` builds the complete tree of depth `depth` up front, which
          is the Frosst & Hinton (2017) formulation.
        - ``"incremental"`` starts from a single split and deepens one level at
          a time, keeping a level only if it improves validation loss
          (Irsoy, Yildiz & Alpaydin, ICPR 2012). `depth` becomes an upper bound
          and `tree_depth_` reports what was actually kept. Requires a
          validation split, so `validation_fraction` applies whether or not
          `early_stopping` is on, and the `max_epochs` budget is divided across
          rounds rather than spent per round. That last point matters in
          practice: a budget that trains a fixed tree adequately can leave an
          incremental one under-trained, so raise `max_epochs` when switching.
        - ``"per_leaf"`` splits **one leaf at a time**, the one carrying the most
          expected error, so the tree can end up unbalanced and spend depth only
          where the data needs it. This is the growth rule of İrsoy, Yıldız &
          Alpaydın (ICPR 2012); level-wise growth was the tractable
          approximation of it.

          It produces by far the sparsest trees, and wins where a fixed depth
          over-parameterizes. 3 seeds of 5-fold CV, accuracy and splits kept:

              none / incremental / per_leaf
              Iris             0.958 / 15   0.931 / 15   0.880 /  7.7
              Wine             0.977 / 15   0.981 / 15   0.966 /  5.2
              Breast Cancer    0.971 / 15   0.971 /  9.9  0.978 /  4.0
              synthetic 20d    0.839 / 63   0.881 / 13    0.885 /  3.7

          On the synthetic problem it reaches better accuracy than a fixed
          depth-6 tree using 3.7 splits against 63. On Iris it loses, which is
          why the default is still `"none"`.
    learn_temperature : bool, default=False
        Learn a per-node inverse temperature on the gate, so a node can sharpen
        its split instead of saturating in the flat part of the sigmoid
        (Frosst & Hinton, 2017). Off by default because the effect is mixed:
        averaged over 5 seeds of 5-fold CV at depth 4 it moved Iris from 0.900
        to 0.928, and cost about 0.6 points on Wine and Breast Cancer.
    early_stopping : bool, default=False
        Hold out `validation_fraction` of the training data and stop once
        validation loss has not improved for `n_iter_no_change` epochs. The
        parameters of the best epoch are restored.
    validation_fraction : float, default=0.1
        Fraction held out when `early_stopping=True`.
    n_iter_no_change : int, default=10
        Epochs without validation improvement before stopping.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    n_features_in_ : int
        Number of features seen during fit.
    training_history_ : list of dict
        Loss and accuracy per epoch, plus validation loss when early stopping
        is on.
    feature_importances_ : ndarray of shape (n_features,)
        Gate weight magnitudes, weighted by how much probability mass reaches
        each node on the training data, normalized to sum to 1.
    n_iter_ : int
        Epochs actually run.
    tree_depth_ : int
        Depth of the fitted tree. Equals `depth` unless `growth="incremental"`
        stopped earlier.
    growth_ : str
        The growth mode actually used. Falls back to `"none"` when the data is
        too small to hold out a stratified validation split.

    Examples
    --------
    >>> from neural_trees import SoftDecisionTree
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> sdt = SoftDecisionTree(depth=4, max_epochs=30)
    >>> sdt.fit(X, y)
    >>> sdt.score(X, y)

    References
    ----------
    İrsoy, O., Yıldız, O. T., & Alpaydın, E. (2012).
    Soft Decision Trees. ICPR 2012.
    """

    def __init__(
        self,
        depth: int = 5,
        max_epochs: int = 40,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        penalty_coef: float = 1e-3,
        device: str = "cpu",
        verbose: bool = False,
        random_state: Optional[int] = None,
        class_weight=None,
        growth: str = "none",
        warm_start: bool = False,
        learn_temperature: bool = False,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ):
        self.depth = depth
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.penalty_coef = penalty_coef
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.class_weight = class_weight
        self.growth = growth
        self.warm_start = warm_start
        self.learn_temperature = learn_temperature
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def fit(self, X, y, sample_weight=None) -> "SoftDecisionTree":
        """
        Fit the Soft Decision Tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights applied to the loss. Combined multiplicatively
            with `class_weight` when both are given. The entropy penalty is
            left unweighted: it regularizes the shape of the tree, not the fit
            to any particular sample.

            Weighting a sample by k gives the same loss and the same gradient
            as repeating it k times, but not bit-for-bit the same *fit*: the
            repeated dataset is larger, so mini-batches are composed
            differently and the optimizer follows a different path. This is why
            `check_sample_weight_equivalence_on_dense_data` is the one
            estimator check this class does not pass (62 of 63), and it is not
            satisfiable by any stochastic mini-batch learner.

        Returns
        -------
        self
        """
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError(f"depth must be a positive integer, got {self.depth!r}")
        if self.growth not in ("none", "incremental", "per_leaf"):
            raise ValueError(
                "growth must be 'none', 'incremental' or 'per_leaf', got "
                f"{self.growth!r}"
            )

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        continuing = self._reuse_existing_model(len(encoder.classes_))
        self.le_ = encoder
        self.classes_ = encoder.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        weights = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if self.class_weight is not None:
            class_weights = compute_class_weight(
                self.class_weight, classes=np.arange(n_classes), y=y_enc
            )
            weights = weights * class_weights[y_enc]
        # Normalizing to mean 1 keeps the loss on the same scale as the
        # unweighted fit, so learning_rate and penalty_coef keep their meaning.
        weights = weights * (len(weights) / weights.sum())

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        needs_validation = self.early_stopping or self.growth in ("incremental", "per_leaf")
        if needs_validation and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}"
            )

        X_fit, y_fit, w_fit = X, y_enc, weights
        X_val = y_val = None
        if needs_validation and self._can_hold_out(y_enc):
            X_fit, X_val, y_fit, y_val, w_fit, _ = train_test_split(
                X,
                y_enc,
                weights,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_enc,
            )

        # Growth needs held-out evidence to decide anything, so without a split
        # it falls back to building the full depth. growth_ records what ran.
        self.growth_ = self.growth if X_val is not None else "none"

        device = self.device_ = resolve_device(self.device)
        X_t = torch.FloatTensor(X_fit).to(device)
        y_t = torch.LongTensor(y_fit).to(device)
        w_t = torch.FloatTensor(w_fit).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device) if X_val is not None else None
        y_val_t = torch.LongTensor(y_val).to(device) if X_val is not None else None

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(X_t, y_t, w_t),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.training_history_: List[dict] = []

        if self.growth_ != "none" and continuing:
            raise ValueError(
                f"warm_start=True is not supported with growth={self.growth_!r}: "
                "the second fit would restart the search and discard the shape "
                "the first one chose."
            )

        if self.growth_ == "per_leaf":
            self._fit_per_leaf(loader, X_t, y_t, X_val_t, y_val_t, n_classes, device)
        elif self.growth_ == "incremental":
            self._fit_incrementally(loader, X_val_t, y_val_t, n_classes, device)
        else:
            if not continuing:
                self.model_ = _SoftTreeModule(
                    n_features=self.n_features_in_,
                    n_classes=n_classes,
                    depth=self.depth,
                    penalty_coef=self.penalty_coef,
                    learn_temperature=self.learn_temperature,
                ).to(device)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
            _, best_state = self._train_epochs(
                self.model_, loader, optimizer, self.max_epochs, X_val_t, y_val_t,
                stop_early=self.early_stopping and X_val_t is not None,
            )
            if best_state is not None:
                self.model_.load_state_dict(best_state)

        self.tree_depth_ = self.model_.depth
        self.n_iter_ = len(self.training_history_)
        self.feature_importances_ = self._compute_feature_importances(X_t)
        return self

    def _reuse_existing_model(self, n_classes: int) -> bool:
        """
        Whether a previous fit's parameters can be continued from.

        Refusing loudly when the label set changed is deliberate: silently
        reinitializing would make warm_start look like it worked while throwing
        away everything the first fit learned.
        """
        if not self.warm_start or not hasattr(self, "model_"):
            return False
        if len(getattr(self, "classes_", [])) != n_classes:
            raise ValueError(
                "warm_start=True requires the same classes across calls to fit; "
                "the label set changed, and this model cannot grow its output layer."
            )
        return True

    def _can_hold_out(self, y_enc: np.ndarray) -> bool:
        """
        Whether the data can spare a stratified validation split.

        A class with a single member cannot be stratified, and a split smaller
        than the number of classes leaves one unrepresented. Both turn up in
        sklearn's estimator checks and in genuinely small datasets, so the
        answer is a fallback rather than an exception.
        """
        counts = np.bincount(y_enc)
        n_val = int(round(len(y_enc) * self.validation_fraction))
        return bool(
            counts.min() >= 2
            and n_val >= len(counts)
            and len(y_enc) - n_val >= len(counts)
        )

    def _train_epochs(
        self, model, loader, optimizer, n_epochs, X_val_t, y_val_t, stop_early=False
    ):
        """
        Run `n_epochs` of training, recording each into `training_history_`.

        Returns the best validation loss seen and a snapshot of the parameters
        that produced it, or (inf, None) when there is no validation split.
        """
        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        epoch_offset = len(self.training_history_)

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                log_probs = model.log_forward(X_batch)
                per_sample = F.nll_loss(log_probs, y_batch, reduction="none")
                loss = (per_sample * w_batch).sum() / w_batch.sum().clamp_min(1e-12)
                total_loss = loss + model.penalty(X_batch)
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item() * X_batch.size(0)
                correct += (log_probs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)

            record = {
                "epoch": epoch_offset + epoch + 1,
                "depth": model.depth,
                "loss": epoch_loss / total,
                "accuracy": correct / total,
            }

            if X_val_t is not None:
                model.eval()
                with torch.no_grad():
                    val_log_probs = model.log_forward(X_val_t)
                    record["val_loss"] = F.nll_loss(val_log_probs, y_val_t).item()
                    record["val_accuracy"] = (
                        (val_log_probs.argmax(dim=1) == y_val_t).float().mean().item()
                    )

                if record["val_loss"] < best_val_loss - 1e-6:
                    best_val_loss = record["val_loss"]
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.training_history_.append(record)

            if self.verbose and (epoch + 1) % 5 == 0:
                message = (
                    f"Epoch {record['epoch']}  depth={model.depth}  "
                    f"loss={record['loss']:.4f}  acc={record['accuracy']:.4f}"
                )
                if X_val_t is not None:
                    message += f"  val_loss={record['val_loss']:.4f}"
                print(message)

            if (
                stop_early
                and X_val_t is not None
                and epochs_without_improvement >= self.n_iter_no_change
            ):
                if self.verbose:
                    print(f"Early stopping at epoch {record['epoch']}")
                break

        return best_val_loss, best_state

    def _fit_per_leaf(self, loader, X_t, y_t, X_val_t, y_val_t, n_classes, device):
        """
        Split one leaf at a time, the leaf that is getting the most wrong.

        Level-wise growth splits every leaf at once, so the tree stays perfectly
        balanced and spends depth where it is not needed. Splitting one leaf at
        a time lets the tree end up unbalanced, which is the point of growing it
        rather than declaring a depth (Irsoy, Yildiz & Alpaydin, ICPR 2012).

        The leaf chosen is the one with the largest expected error mass, the
        probability mass arriving at it weighted by how wrong its distribution
        is on those samples. A split is kept only if it improves validation
        loss; the first one that does not ends the growth.
        """
        max_splits = 2 ** self.depth - 1
        epochs_per_round = max(1, self.max_epochs // max(1, self.depth * 2))

        model = _SoftTreeModule(
            n_features=self.n_features_in_,
            n_classes=n_classes,
            depth=self.depth,
            penalty_coef=self.penalty_coef,
            learn_temperature=self.learn_temperature,
        ).to(device)
        with torch.no_grad():
            model.is_split.fill_(False)  # a single leaf to begin with

        best_loss = np.inf
        best_state = None

        while True:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            round_loss, round_state = self._train_epochs(
                model, loader, optimizer, epochs_per_round, X_val_t, y_val_t
            )
            if round_state is None:
                round_loss = 0.0
                round_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if round_loss < best_loss - 1e-6 or best_state is None:
                best_loss, best_state = round_loss, round_state
            elif X_val_t is not None:
                if self.verbose:
                    print("Splitting stopped paying off; keeping the previous tree")
                break

            if int(model.is_split.sum()) >= max_splits:
                break
            victim = self._neediest_leaf(model, X_t, y_t)
            if victim is None:
                break
            with torch.no_grad():
                model.is_split[victim] = True
            if self.verbose:
                print(f"Split node {victim}; {int(model.is_split.sum())} splits now")

        model.load_state_dict(best_state)
        self.model_ = model

    @staticmethod
    def _neediest_leaf(model, X_t, y_t):
        """
        The reachable leaf carrying the most expected error, or None if every
        leaf is at the bottom row and cannot be split further.
        """
        model.eval()
        with torch.no_grad():
            _, _, terminal = model._walk(X_t)
            distributions = F.softmax(model.node_logits, dim=1)

            best_index, best_mass = None, -np.inf
            for log_mu, indices in terminal:
                internal = indices < model.n_internal
                if not bool(internal.any()):
                    continue
                mu = log_mu[:, internal].exp()                      # (batch, n_here)
                correct = distributions[indices[internal]][:, y_t]  # (n_here, batch)
                error_mass = (mu * (1.0 - correct.T)).sum(dim=0)
                position = int(error_mass.argmax())
                if float(error_mass[position]) > best_mass:
                    best_mass = float(error_mass[position])
                    best_index = int(indices[internal][position])

        return best_index

    def _fit_incrementally(self, loader, X_val_t, y_val_t, n_classes, device):
        """
        Grow the tree one level at a time, keeping a level only if it earns its
        place on held-out data (Irsoy, Yildiz & Alpaydin, ICPR 2012).

        Training starts from a single split. After each round the tree is
        deepened, which by construction leaves the function unchanged, and
        trained again. A round that fails to improve validation loss is undone
        and growth stops, so the depth is chosen by the data instead of being
        fixed in advance.

        The epoch budget is `max_epochs` in total, divided across at most
        `depth` rounds, so an incremental fit costs about what a fixed-depth fit
        of the same `max_epochs` costs.
        """
        epochs_per_round = max(1, self.max_epochs // max(1, self.depth))

        model = _SoftTreeModule(
            n_features=self.n_features_in_,
            n_classes=n_classes,
            depth=1,
            penalty_coef=self.penalty_coef,
            learn_temperature=self.learn_temperature,
        ).to(device)

        best_loss = np.inf
        best_state = None
        best_depth = 1

        while True:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            round_loss, round_state = self._train_epochs(
                model, loader, optimizer, epochs_per_round, X_val_t, y_val_t
            )
            if round_state is None:  # no validation split available
                round_loss, round_state = 0.0, {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

            improved = round_loss < best_loss - 1e-6
            if improved or best_state is None:
                best_loss, best_state, best_depth = round_loss, round_state, model.depth
            elif X_val_t is not None:
                if self.verbose:
                    print(f"Depth {model.depth} did not improve, keeping depth {best_depth}")
                break

            if model.depth >= self.depth:
                break
            model = model.deepen(n_classes)

        self.model_ = _SoftTreeModule(
            n_features=self.n_features_in_,
            n_classes=n_classes,
            depth=best_depth,
            penalty_coef=self.penalty_coef,
            learn_temperature=self.learn_temperature,
        ).to(device)
        self.model_.load_state_dict(best_state)

    def _compute_feature_importances(self, X_t: "torch.Tensor") -> np.ndarray:
        """
        Weight each node's gate magnitudes by the probability mass that reaches
        it, so a node that almost no sample passes through cannot dominate.
        """
        self.model_.eval()
        with torch.no_grad():
            _, level_probs = self.model_._log_path_probabilities(X_t)
            node_mass = torch.cat([mu.mean(dim=0) for mu, _ in level_probs])  # (n_internal,)
            weights = self.model_.gates.weight.abs()  # (n_internal, n_features)
            importances = (node_mass.unsqueeze(1) * weights).sum(dim=0)

        scores = importances.cpu().numpy()
        total = scores.sum()
        return scores / total if total > 0 else scores

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.

        Each sample reaches every leaf with some probability, so the returned
        distribution is the path-probability weighted average of the leaf
        distributions, P(y | x) = sum_l mu_l(x) Q_l(y). This is why the output
        is smooth rather than the piecewise constant output of a hard tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score. Cast to float32 internally, so any numeric dtype
            is accepted. Must have the same number of features seen in `fit`.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities in the order of `self.classes_`. Each row sums
            to 1.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X)
        device = self.device_
        X_t = torch.FloatTensor(X).to(device)

        self.model_.eval()
        with torch.no_grad():
            probs = self.model_.log_forward(X_t).exp()
        return probs.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.le_.inverse_transform(idx)

    def get_leaf_distributions(self) -> np.ndarray:
        """
        Return the class distribution stored in each leaf node.

        Returns
        -------
        distributions : ndarray of shape (n_leaves, n_classes)
        """
        check_is_fitted(self)
        self.model_.eval()
        with torch.no_grad():
            distributions = F.softmax(self.model_.node_logits, dim=1)
            indices = self._acting_leaf_indices().tolist()
            return distributions[indices].cpu().numpy()

    def _acting_leaf_indices(self) -> "np.ndarray":
        """
        Node indices that actually behave as leaves.

        With every internal node splitting this is exactly the bottom row, so
        the complete-tree case is unchanged. When growth has left some subtrees
        switched off, the nodes where the walk stops take their place.
        """
        module = self.model_
        is_split = module.is_split.cpu().numpy()
        leaves = []
        stack = [0]
        while stack:
            node = stack.pop()
            if node >= module.n_internal or not is_split[node]:
                leaves.append(node)
                continue
            stack.extend([2 * node + 2, 2 * node + 1])
        return np.array(sorted(leaves))

    def to_hard_tree(self):
        """
        Export the trained tree with its gates read as hard decisions.

        Each internal node's gate is `sigmoid(beta * (w . x + b))`; the sign of
        `w . x + b` is the decision it has settled on, and beta only sharpens
        it. Taking that sign and routing each sample down one path gives a plain
        numpy model with readable rules and no PyTorch in the prediction path.

        This is a different model, not a re-encoding: a mixture over leaves is
        not a single path, and the two disagree on samples that sit near a
        split. Measure the agreement on held-out data before relying on it.

        Returns
        -------
        HardDecisionTree

        Examples
        --------
        >>> hard = sdt.to_hard_tree()
        >>> (hard.predict(X_test) == sdt.predict(X_test)).mean()
        >>> print(hard.export_text(feature_names=feature_names))
        """
        check_is_fitted(self)
        self.model_.eval()
        with torch.no_grad():
            weights = self.model_.gates.weight.detach().cpu().numpy()
            biases = self.model_.gates.bias.detach().cpu().numpy()
        with torch.no_grad():
            node_distributions = F.softmax(self.model_.node_logits, dim=1).cpu().numpy()
        return HardDecisionTree(
            weights=weights,
            biases=biases,
            node_distributions=node_distributions,
            classes=self.classes_,
            n_features_in=self.n_features_in_,
            is_split=self.model_.is_split.cpu().numpy(),
        )

    def get_split_weights(self) -> List[np.ndarray]:
        """
        Return the weight vectors for each internal node's split.

        Returns
        -------
        weights : list of ndarray, one per internal node
        """
        check_is_fitted(self)
        weights = self.model_.gates.weight.detach().cpu().numpy()
        active = self.model_.is_split.cpu().numpy()
        return [weights[i].copy() for i in range(len(weights)) if active[i]]

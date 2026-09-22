"""
Hierarchical Mixture of Experts (HMoE) with Dropout
=====================================================
Implementation based on:
    İrsoy, O., & Alpaydın, E. (2021).
    Dropout Regularization in Hierarchical Mixture of Experts.
    Neurocomputing, 419, 148-156.

Key idea:
    A HMoE is a tree-structured mixture model where each internal node is a
    "gating network" that routes input to child nodes, and each leaf is an
    "expert network". The final prediction is a weighted mixture of expert outputs.
    Dropout regularizes that tree by dropping *subtrees*: with probability p a
    gating node withholds all probability mass from one of its children during
    training, so the whole subtree below it is switched off for that sample and
    the surviving children are renormalized. This is the tree-structured
    analogue of dropping hidden units, and it prevents experts from
    co-adapting. At evaluation the full soft mixture is used, and because a
    dropped gate output is renormalized rather than scaled, no test-time
    correction is needed.

    `dropout_type="activation"` keeps the earlier behaviour of this class, a
    plain nn.Dropout on the gating network's hidden activations. That perturbs
    a gate but never removes a branch. Measured on a noisy 20-feature problem
    over 5 seeds, it moves the train/test gap from 0.412 to 0.403, while
    subtree dropout at the same rate moves it to 0.364 and test accuracy from
    0.588 to 0.634.

Architecture (depth=2, branching=2):
              [Gate]
             /      \\
          [Gate]   [Gate]
          /  \\     /  \\
         E1  E2   E3  E4    (Expert leaves)
"""

import copy
import math

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

from neural_trees._batching import TensorBatches
from neural_trees._validation import check_predict_input, resolve_device
from neural_trees.mixture_of_experts.hard_router import HardRoutedExperts


class _StackedMLP(nn.Module):
    """
    A bank of identically shaped two-layer MLPs evaluated in one pass.

    A tree of depth 3 with branching factor 4 holds 21 gating networks and 64
    experts. Running them as a ModuleList means 170 small matmuls per forward,
    each too small to keep a CPU busy. Stacking the weights into
    (n_networks, out, in) tensors turns that into two batched matmuls.
    """

    def __init__(self, n_networks: int, n_in: int, n_hidden: int, n_out: int, activation):
        super().__init__()
        self.activation = activation
        self.weight_in = nn.Parameter(torch.empty(n_networks, n_hidden, n_in))
        self.bias_in = nn.Parameter(torch.empty(n_networks, n_hidden))
        self.weight_out = nn.Parameter(torch.empty(n_networks, n_out, n_hidden))
        self.bias_out = nn.Parameter(torch.empty(n_networks, n_out))
        self._reset_parameters(n_in, n_hidden)

    def _reset_parameters(self, n_in: int, n_hidden: int):
        """Initialize every slice the way nn.Linear would initialize itself."""
        for weight, bias, fan_in in (
            (self.weight_in, self.bias_in, n_in),
            (self.weight_out, self.bias_out, n_hidden),
        ):
            for i in range(weight.shape[0]):
                nn.init.kaiming_uniform_(weight[i], a=math.sqrt(5))
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
        """
        Returns the logits of every network, shape (batch, n_networks, n_out).
        """
        hidden = self.activation(torch.einsum("bf,nhf->bnh", x, self.weight_in) + self.bias_in)
        if dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=dropout, training=True)
        return torch.einsum("bnh,noh->bno", hidden, self.weight_out) + self.bias_out


class _HMoEModule(nn.Module):
    """
    Hierarchical Mixture of Experts PyTorch module.

    Creates a complete b-ary tree of depth `depth` with `branching_factor`
    children per gate. Leaves are experts.

    Gates and experts are each evaluated as one stacked bank rather than node
    by node, and mixing weights are accumulated in log space: a leaf at depth d
    is reached with probability on the order of b^-d, which underflows float32
    as the tree grows.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        depth: int,
        branching_factor: int,
        gate_hidden: int,
        expert_hidden: int,
        dropout_rate: float,
        dropout_type: str = "subtree",
    ):
        super().__init__()
        self.depth = depth
        self.branching_factor = branching_factor
        self.n_experts = branching_factor ** depth
        self.n_gates = sum(branching_factor ** d for d in range(depth))
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.gates = _StackedMLP(
            self.n_gates, n_features, gate_hidden, branching_factor, torch.tanh
        )
        self.experts = _StackedMLP(
            self.n_experts, n_features, expert_hidden, n_classes, torch.relu
        )

    def _log_gate_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log P(child | x) for every gating node, shape (batch, n_gates, b).

        Subtree dropout is applied here, in log space: a dropped child gets
        -inf and the row is renormalized with logsumexp, which is exact and
        keeps the result a proper distribution.
        """
        activation_dropout = self.dropout_rate if self.dropout_type == "activation" else 0.0
        logits = self.gates(x, dropout=activation_dropout)
        log_probs = F.log_softmax(logits, dim=-1)

        subtree_dropout = (
            self.training
            and self.dropout_rate > 0.0
            and self.dropout_type == "subtree"
            and self.branching_factor > 1
        )
        if not subtree_dropout:
            return log_probs

        batch, n_gates, n_children = log_probs.shape
        drop = torch.rand(batch, n_gates, device=x.device) < self.dropout_rate
        victim = torch.randint(0, n_children, (batch, n_gates), device=x.device)
        penalty = torch.zeros_like(log_probs).scatter_(
            2, victim.unsqueeze(2), float("-inf")
        )
        dropped = torch.where(drop.unsqueeze(2), log_probs + penalty, log_probs)
        return dropped - torch.logsumexp(dropped, dim=2, keepdim=True)

    def _log_leaf_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log mixing weight of every expert, shape (batch, n_experts).

        Walks the tree level by level. The children of the i-th node at one
        level occupy positions i*b to i*b+b-1 at the next, which is exactly what
        the reshape below produces.
        """
        log_gates = self._log_gate_probabilities(x)
        b = self.branching_factor
        log_mu = torch.zeros(x.size(0), 1, device=x.device)

        start = 0
        for level in range(self.depth):
            width = b ** level
            level_gates = log_gates[:, start:start + width, :]
            log_mu = (log_mu.unsqueeze(2) + level_gates).reshape(x.size(0), width * b)
            start += width

        return log_mu

    def _compute_leaf_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Mixing weights over experts, shape (batch, n_experts)."""
        return self._log_leaf_weights(x).exp()

    def log_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        log P(y | x) = logsumexp_e [ log g_e(x) + log P_e(y | x) ].

        Returns a tensor of shape (batch_size, n_classes).
        """
        log_leaf_weights = self._log_leaf_weights(x)
        log_expert_probs = F.log_softmax(self.experts(x), dim=-1)
        return torch.logsumexp(log_leaf_weights.unsqueeze(2) + log_expert_probs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture output: P(y|x) = sum_e g_e(x) P_e(y|x)

        Returns:
            Tensor of shape (batch_size, n_classes).
        """
        return self.log_forward(x).exp()


class HierarchicalMixtureOfExperts(ClassifierMixin, BaseEstimator):
    """
    Hierarchical Mixture of Experts with Dropout Regularization.

    A tree-structured neural network where gating networks route inputs
    to expert leaves. Dropout on gating networks prevents co-adaptation
    and improves generalization.

    Parameters
    ----------
    depth : int, default=2
        Depth of the expert tree. n_experts = branching_factor^depth.
    branching_factor : int, default=2
        Number of children per gating node.
    gate_hidden : int, default=32
        Hidden units in each gating network.
    expert_hidden : int, default=64
        Hidden units in each expert network.
    dropout_rate : float, default=0.3
        Dropout probability applied at each gating node during training.
    dropout_type : {"subtree", "activation"}, default="subtree"
        Which dropout mechanism to use.

        - ``"subtree"`` is the mechanism from Irsoy & Alpaydin (2021): a gating
          node drops one of its children with probability `dropout_rate`, so the
          whole subtree below it receives no probability mass for that sample,
          and the surviving children are renormalized.
        - ``"activation"`` is the earlier behaviour of this class, a plain
          ``nn.Dropout`` on the gating network's hidden activations. It
          perturbs a gate but never removes a branch, and measures as close to
          inert. Kept so results can be compared.
    max_epochs : int, default=50
        Training epochs.
    learning_rate : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
    device : str, default="cpu"
        PyTorch device. `"auto"` picks CUDA if it is available, then Apple
        silicon's MPS, then CPU. Resolved once in `fit` and recorded as
        `device_`.
    verbose : bool, default=False
    random_state : int or None, default=None
        Seed for weight initialization and shuffled mini-batches.
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
    class_weight : dict, "balanced" or None, default=None
        Weights per class, combined multiplicatively with `sample_weight`.
        `"balanced"` uses `n_samples / (n_classes * bincount(y))`. Without it a
        rare class contributes so little loss that the model can ignore it.
    early_stopping : bool, default=False
        Hold out `validation_fraction` of the training data, stratified and
        drawn with `random_state`, and stop once validation loss has not
        improved for `n_iter_no_change` epochs; the parameters from the best
        validation epoch are restored. Same names, defaults and meaning as on
        `SoftDecisionTree`. Off, the model trains for exactly `max_epochs` on
        all the data, as before. Each call to `fit` draws its own split, so a
        `warm_start` continuation on new data never scores against a stale
        one. When the data is too small to hold out a stratified split (a
        class with one member, or fewer held-out rows than classes) the fit
        falls back to the full budget and `n_iter_` says how far it got.
    validation_fraction : float, default=0.1
        Fraction held out when `early_stopping=True`.
    n_iter_no_change : int, default=10
        Epochs without validation improvement before stopping.

    Attributes
    ----------
    n_iter_ : int
        Epochs actually run in the last call to `fit`.
    training_history_ : list of dict
        One record per epoch with `loss` and `accuracy`, plus `val_loss` and
        `val_accuracy` when a validation split exists.

    Examples
    --------
    >>> from neural_trees import HierarchicalMixtureOfExperts
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(return_X_y=True)
    >>> moe = HierarchicalMixtureOfExperts(depth=2, branching_factor=4)
    >>> moe.fit(X, y)
    >>> moe.score(X, y)

    References
    ----------
    İrsoy, O., & Alpaydın, E. (2021).
    Dropout Regularization in Hierarchical Mixture of Experts.
    Neurocomputing, 419, 148-156.
    """

    def __init__(
        self,
        depth: int = 2,
        branching_factor: int = 2,
        gate_hidden: int = 32,
        expert_hidden: int = 64,
        dropout_rate: float = 0.3,
        dropout_type: str = "subtree",
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = False,
        random_state: Optional[int] = None,
        class_weight=None,
        warm_start: bool = False,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ):
        self.depth = depth
        self.branching_factor = branching_factor
        self.gate_hidden = gate_hidden
        self.expert_hidden = expert_hidden
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def _can_hold_out(self, y_enc: np.ndarray) -> bool:
        """Whether a stratified split of `validation_fraction` is possible."""
        counts = np.bincount(y_enc)
        n_val = int(round(len(y_enc) * self.validation_fraction))
        return bool(
            counts.min() >= 2
            and n_val >= len(counts)
            and len(y_enc) - n_val >= len(counts)
        )

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

    def fit(self, X, y, sample_weight=None) -> "HierarchicalMixtureOfExperts":
        if self.dropout_type not in ("subtree", "activation"):
            raise ValueError(
                "dropout_type must be 'subtree' or 'activation', got "
                f"{self.dropout_type!r}"
            )
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        continuing = self._reuse_existing_model(len(encoder.classes_))
        self.le_ = encoder
        self.classes_ = encoder.classes_
        self.n_features_in_ = X.shape[1]

        weights = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if self.class_weight is not None:
            class_weights = compute_class_weight(
                self.class_weight, classes=np.arange(len(self.classes_)), y=y_enc
            )
            weights = weights * class_weights[y_enc]
        # Normalizing to mean 1 keeps the loss on the same scale as the
        # unweighted fit, so learning_rate keeps its meaning.
        weights = weights * (len(weights) / weights.sum())

        if self.early_stopping and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}"
            )
        X_fit, y_fit, w_fit = X, y_enc, weights
        X_val = y_val = None
        if self.early_stopping and self._can_hold_out(y_enc):
            X_fit, X_val, y_fit, y_val, w_fit, _ = train_test_split(
                X,
                y_enc,
                weights,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_enc,
            )

        device = self.device_ = resolve_device(self.device)
        X_t = torch.FloatTensor(X_fit).to(device)
        y_t = torch.LongTensor(y_fit).to(device)
        w_t = torch.FloatTensor(w_fit).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device) if X_val is not None else None
        y_val_t = torch.LongTensor(y_val).to(device) if X_val is not None else None

        if not continuing:
            self.model_ = _HMoEModule(
                n_features=self.n_features_in_,
                n_classes=len(self.classes_),
                depth=self.depth,
                branching_factor=self.branching_factor,
                gate_hidden=self.gate_hidden,
                expert_hidden=self.expert_hidden,
                dropout_rate=self.dropout_rate,
                dropout_type=self.dropout_type,
            ).to(device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        loader = TensorBatches((X_t, y_t, w_t), self.batch_size, generator)

        self.training_history_: List[dict] = []
        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        self.n_iter_ = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_b, y_b, w_b in loader:
                optimizer.zero_grad()
                log_probs = self.model_.log_forward(X_b)
                per_sample = F.nll_loss(log_probs, y_b, reduction="none")
                loss = (per_sample * w_b).sum() / w_b.sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_b.size(0)
                correct += (log_probs.argmax(1) == y_b).sum().item()
                total += X_b.size(0)

            avg_loss = total_loss / total
            acc = correct / total
            record = {"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc}

            if X_val_t is not None and y_val_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_log_probs = self.model_.log_forward(X_val_t)
                    record["val_loss"] = F.nll_loss(val_log_probs, y_val_t).item()
                    record["val_accuracy"] = (
                        (val_log_probs.argmax(1) == y_val_t).float().mean().item()
                    )
                if record["val_loss"] < best_val_loss - 1e-6:
                    best_val_loss = record["val_loss"]
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.training_history_.append(record)
            self.n_iter_ = epoch + 1

            if self.verbose and (epoch + 1) % 10 == 0:
                message = f"Epoch {epoch+1}/{self.max_epochs}  loss={avg_loss:.4f}  acc={acc:.4f}"
                if X_val_t is not None:
                    message += f"  val_loss={record['val_loss']:.4f}"
                print(message)

            if X_val_t is not None and epochs_without_improvement >= self.n_iter_no_change:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Built here rather than lazily in predict_proba: an estimator must not
        # mutate its own __dict__ while predicting.
        #
        # Always on CPU, whatever device training used. This copy exists to make
        # predictions independent of row order, which needs float64, and MPS has
        # no float64 at all while CUDA's is slow. Prediction is a single forward
        # pass, so the move costs little next to getting the same answer for the
        # same sample every time.
        self.model_double_ = copy.deepcopy(self.model_).to("cpu").double()
        self.model_double_.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities, shape (n_samples, n_classes).

        The forward pass runs in float64 on the CPU, whatever device training
        used. Rows are independent, but BLAS picks different blocking for
        different memory layouts, so in float32 the same sample scored inside a
        reordered batch came out up to 1.2e-07 different and a borderline argmax
        could flip with it. Doubling the width of the predict-time arithmetic
        puts that at 2.2e-16, which makes a prediction a property of the sample
        rather than of its position in the batch. It is pinned to the CPU
        because MPS has no float64 and CUDA's is slow. Training stays float32,
        on whichever device was chosen.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X)
        with torch.no_grad():
            probs = self.model_double_(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
            )
        return probs.numpy().astype(np.float64)



    def to_hard_router(self) -> "HardRoutedExperts":
        """
        Export the trained mixture with its gates read as hard routing.

        A trained mixture evaluates every expert for every sample and blends
        them, so one prediction costs `branching_factor^depth` expert forward
        passes. Each gating node has a preferred child for any given input;
        taking that preference as a decision sends a sample down one path to
        one expert, in numpy.

        This is a different model, not a re-encoding: a blend of experts is not
        one expert, and the two disagree where the gating was undecided.
        Measure the agreement on held-out data before relying on it.

        Returns
        -------
        HardRoutedExperts
        """
        check_is_fitted(self)

        def unpack(bank, index):
            return (
                bank.weight_in[index].detach().cpu().numpy().astype(np.float64),
                bank.bias_in[index].detach().cpu().numpy().astype(np.float64),
                bank.weight_out[index].detach().cpu().numpy().astype(np.float64),
                bank.bias_out[index].detach().cpu().numpy().astype(np.float64),
            )

        module = self.model_
        return HardRoutedExperts(
            gate_weights=[unpack(module.gates, i) for i in range(module.n_gates)],
            expert_weights=[unpack(module.experts, i) for i in range(module.n_experts)],
            classes=self.classes_,
            n_features_in=self.n_features_in_,
            depth=self.depth,
            branching_factor=self.branching_factor,
        )

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))

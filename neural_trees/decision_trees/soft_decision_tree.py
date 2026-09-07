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
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "neural-trees requires PyTorch. Install it with: pip install torch "
        "(see https://pytorch.org/get-started/locally/ for platform specific wheels)."
    ) from exc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.utils.multiclass import check_classification_targets

from neural_trees._validation import check_predict_input
from typing import Optional, List


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

        self.leaf_logits = nn.Parameter(torch.zeros(self.n_leaves, n_classes))

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled logits for every internal node, shape (batch, n_internal)."""
        return torch.exp(self.log_beta) * self.gates(x)

    def _log_path_probabilities(self, x: torch.Tensor):
        """
        Accumulate log arrival probabilities level by level.

        Returns
        -------
        log_leaf_probs : Tensor of shape (batch, n_leaves)
        level_probs : list of Tensor, arrival probabilities of the internal
            nodes at each level, used by the entropy penalty.
        """
        logits = self.gate_logits(x)
        log_mu = torch.zeros(x.size(0), 1, device=x.device)  # root is reached with prob 1
        level_probs = []

        start = 0
        for level in range(self.depth):
            width = 2 ** level
            level_logits = logits[:, start:start + width]
            level_probs.append((log_mu.exp(), torch.sigmoid(level_logits)))

            log_left = F.logsigmoid(-level_logits)
            log_right = F.logsigmoid(level_logits)
            # Interleave to [left_0, right_0, left_1, right_1, ...], which is the
            # child order of the breadth-first node indexing.
            children = torch.stack([log_left, log_right], dim=2).reshape(x.size(0), 2 * width)
            log_mu = log_mu.repeat_interleave(2, dim=1) + children
            start += width

        return log_mu, level_probs

    def log_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        log P(y | x) = logsumexp_leaf [ log mu_leaf(x) + log Q_leaf(y) ].

        Returns
        -------
        Tensor of shape (batch_size, n_classes) of log probabilities.
        """
        log_leaf_probs, _ = self._log_path_probabilities(x)
        log_leaf_dists = F.log_softmax(self.leaf_logits, dim=1)  # (n_leaves, n_classes)
        return torch.logsumexp(log_leaf_probs.unsqueeze(2) + log_leaf_dists.unsqueeze(0), dim=1)

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
        PyTorch device ("cpu" or "cuda").
    verbose : bool, default=False
        Whether to print training progress.
    random_state : int or None, default=None
        Seed for model initialization and shuffled mini-batches.
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
        self.learn_temperature = learn_temperature
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def fit(self, X, y):
        """
        Fit the Soft Decision Tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError(f"depth must be a positive integer, got {self.depth!r}")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_fit, y_fit = X, y_enc
        X_val = y_val = None
        if self.early_stopping:
            if not 0.0 < self.validation_fraction < 1.0:
                raise ValueError(
                    "validation_fraction must be in (0, 1), got "
                    f"{self.validation_fraction!r}"
                )
            stratify = y_enc if np.bincount(y_enc).min() >= 2 else None
            X_fit, X_val, y_fit, y_val = train_test_split(
                X,
                y_enc,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )

        device = torch.device(self.device)
        X_t = torch.FloatTensor(X_fit).to(device)
        y_t = torch.LongTensor(y_fit).to(device)

        self.model_ = _SoftTreeModule(
            n_features=self.n_features_in_,
            n_classes=n_classes,
            depth=self.depth,
            penalty_coef=self.penalty_coef,
            learn_temperature=self.learn_temperature,
        ).to(device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X_t, y_t)
        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.LongTensor(y_val).to(device)

        self.training_history_: List[dict] = []
        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                log_probs = self.model_.log_forward(X_batch)
                loss = F.nll_loss(log_probs, y_batch)
                penalty = self.model_.penalty(X_batch)
                total_loss = loss + penalty
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item() * X_batch.size(0)
                correct += (log_probs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)

            avg_loss = epoch_loss / total
            acc = correct / total
            record = {"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc}

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_log_probs = self.model_.log_forward(X_val_t)
                    val_loss = F.nll_loss(val_log_probs, y_val_t).item()
                    val_acc = (val_log_probs.argmax(dim=1) == y_val_t).float().mean().item()
                record["val_loss"] = val_loss
                record["val_accuracy"] = val_acc

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.training_history_.append(record)

            if self.verbose and (epoch + 1) % 5 == 0:
                message = f"Epoch {epoch+1}/{self.max_epochs}  loss={avg_loss:.4f}  acc={acc:.4f}"
                if X_val is not None:
                    message += f"  val_loss={record['val_loss']:.4f}"
                print(message)

            if X_val is not None and epochs_without_improvement >= self.n_iter_no_change:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.n_iter_ = len(self.training_history_)
        self.feature_importances_ = self._compute_feature_importances(X_t)
        return self

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

        importances = importances.cpu().numpy()
        total = importances.sum()
        return importances / total if total > 0 else importances

    def predict_proba(self, X):
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
        device = torch.device(self.device)
        X_t = torch.FloatTensor(X).to(device)

        self.model_.eval()
        with torch.no_grad():
            probs = self.model_.log_forward(X_t).exp()
        return probs.cpu().numpy()

    def predict(self, X):
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
            dists = F.softmax(self.model_.leaf_logits, dim=1)
        return dists.cpu().numpy()

    def get_split_weights(self) -> List[np.ndarray]:
        """
        Return the weight vectors for each internal node's split.

        Returns
        -------
        weights : list of ndarray, one per internal node
        """
        check_is_fitted(self)
        weights = self.model_.gates.weight.detach().cpu().numpy()
        return [row.copy() for row in weights]

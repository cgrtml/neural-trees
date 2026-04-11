"""
Soft Decision Trees (SDT)
=========================
Implementation based on:
    İrsoy, O., Yıldız, O. T., & Alpaydın, E. (2012).
    Soft Decision Trees.
    Proceedings of the 21st International Conference on Pattern Recognition (ICPR).

    İrsoy, O., & Alpaydın, E. (2021).
    Dropout Regularization in Hierarchical Mixture of Experts.
    Neurocomputing, 419, 148–156.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Optional, List


class _InternalNode(nn.Module):
    """A single internal (splitting) node with a learnable linear gate."""

    def __init__(self, n_features: int, penalty_coef: float = 1e-3):
        super().__init__()
        self.gate = nn.Linear(n_features, 1)
        self.penalty_coef = penalty_coef
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(go right | x) in [0, 1] for each sample."""
        return torch.sigmoid(self.gate(x).squeeze(-1))


class _LeafNode(nn.Module):
    """A leaf node holding a learnable class distribution."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.distribution = nn.Parameter(torch.zeros(n_classes))

    def forward(self) -> torch.Tensor:
        """Returns class probabilities (softmax over leaf parameters)."""
        return F.softmax(self.distribution, dim=0)


class _SoftTreeModule(nn.Module):
    """
    The core PyTorch module for a Soft Decision Tree of depth `depth`.

    Structure:
        A complete binary tree with (2^depth - 1) internal nodes
        and (2^depth) leaf nodes.
    """

    def __init__(self, n_features: int, n_classes: int, depth: int, penalty_coef: float):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = 2 ** depth - 1

        self.internal_nodes = nn.ModuleList(
            [_InternalNode(n_features, penalty_coef) for _ in range(self.n_internal)]
        )
        self.leaf_nodes = nn.ModuleList(
            [_LeafNode(n_classes) for _ in range(self.n_leaves)]
        )
        self.penalty_coef = penalty_coef

    def _path_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability of each sample reaching each leaf.

        Returns:
            Tensor of shape (batch_size, n_leaves) — the arrival probabilities μ_ℓ(x).
        """
        batch_size = x.size(0)
        # Store per-node probabilities as a list (avoids in-place ops that break autograd)
        node_probs = [None] * (self.n_internal + self.n_leaves)
        node_probs[0] = torch.ones(batch_size, device=x.device)

        for node_idx in range(self.n_internal):
            p_right = self.internal_nodes[node_idx](x)  # (batch,)
            p_left = 1.0 - p_right
            parent_prob = node_probs[node_idx]

            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2

            node_probs[left_child] = parent_prob * p_left
            node_probs[right_child] = parent_prob * p_right

        leaf_probs = torch.stack(node_probs[self.n_internal:], dim=1)  # (batch, n_leaves)
        return leaf_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities as a weighted sum over leaf distributions.

        P(y | x) = Σ_ℓ μ_ℓ(x) · Q_ℓ(y)

        Returns:
            Tensor of shape (batch_size, n_classes).
        """
        leaf_probs = self._path_probabilities(x)  # (batch, n_leaves)
        leaf_dists = torch.stack(
            [leaf.forward() for leaf in self.leaf_nodes], dim=0
        )  # (n_leaves, n_classes)

        output = torch.matmul(leaf_probs, leaf_dists)  # (batch, n_classes)
        return output

    def penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based regularization penalty to avoid degenerate trees.
        Encourages each internal node to use both branches roughly equally.
        """
        total_penalty = torch.zeros(1, device=x.device)
        # Use detached node probs for alpha weighting (no grad needed here)
        node_probs = [None] * self.n_internal
        node_probs[0] = torch.ones(x.size(0), device=x.device)

        for node_idx in range(self.n_internal):
            p_right = self.internal_nodes[node_idx](x)
            p_left = 1 - p_right

            alpha = node_probs[node_idx].mean().detach()
            h = -0.5 * torch.log(p_right.mean().clamp(1e-7)) \
                - 0.5 * torch.log(p_left.mean().clamp(1e-7))
            total_penalty = total_penalty + self.penalty_coef * alpha * h

            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2
            if left_child < self.n_internal:
                node_probs[left_child] = node_probs[node_idx] * p_left.detach()
                node_probs[right_child] = node_probs[node_idx] * p_right.detach()

        return total_penalty.squeeze()


class SoftDecisionTree(BaseEstimator, ClassifierMixin):
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

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    n_features_in_ : int
        Number of features seen during fit.
    training_history_ : list of dict
        Loss and accuracy per epoch.

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
    ):
        self.depth = depth
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.penalty_coef = penalty_coef
        self.device = device
        self.verbose = verbose

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
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        device = torch.device(self.device)
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.LongTensor(y_enc).to(device)

        self.model_ = _SoftTreeModule(
            n_features=self.n_features_in_,
            n_classes=n_classes,
            depth=self.depth,
            penalty_coef=self.penalty_coef,
        ).to(device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.training_history_: List[dict] = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                probs = self.model_(X_batch)
                loss = F.nll_loss(torch.log(probs.clamp(1e-7)), y_batch)
                penalty = self.model_.penalty(X_batch)
                total_loss = loss + penalty
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item() * X_batch.size(0)
                preds = probs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)

            avg_loss = epoch_loss / total
            acc = correct / total
            self.training_history_.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        X = check_array(X)
        device = torch.device(self.device)
        X_t = torch.FloatTensor(X).to(device)

        self.model_.eval()
        with torch.no_grad():
            probs = self.model_(X_t)
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
            dists = torch.stack(
                [leaf.forward() for leaf in self.model_.leaf_nodes], dim=0
            )
        return dists.cpu().numpy()

    def get_split_weights(self) -> List[np.ndarray]:
        """
        Return the weight vectors for each internal node's split.

        Returns
        -------
        weights : list of ndarray, one per internal node
        """
        check_is_fitted(self)
        return [
            node.gate.weight.detach().cpu().numpy().flatten()
            for node in self.model_.internal_nodes
        ]

"""
Hierarchical Mixture of Experts (HMoE) with Dropout
=====================================================
Implementation based on:
    İrsoy, O., & Alpaydın, E. (2021).
    Dropout Regularization in Hierarchical Mixture of Experts.
    Neurocomputing, 419, 148–156.

Key idea:
    A HMoE is a tree-structured mixture model where each internal node is a
    "gating network" that routes input to child nodes, and each leaf is an
    "expert network". The final prediction is a weighted mixture of expert outputs.
    Dropout on the gating network prevents co-adaptation of experts and acts as
    a regularizer similar to model ensembling.

Architecture (depth=2, branching=2):
              [Gate]
             /      \\
          [Gate]   [Gate]
          /  \\     /  \\
         E1  E2   E3  E4    (Expert leaves)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import List, Optional


class _GatingNetwork(nn.Module):
    """Gating network that outputs a soft probability distribution over children."""

    def __init__(self, n_features: int, n_children: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, n_children),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities over children."""
        return F.softmax(self.net(x), dim=-1)


class _ExpertNetwork(nn.Module):
    """Leaf expert network that maps input to class probabilities."""

    def __init__(self, n_features: int, n_classes: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class _HMoEModule(nn.Module):
    """
    Hierarchical Mixture of Experts PyTorch module.

    Creates a complete binary tree of depth `depth` with `branching_factor`
    children per gate. Leaves are experts.
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
    ):
        super().__init__()
        self.depth = depth
        self.branching_factor = branching_factor
        self.n_experts = branching_factor ** depth

        # Compute number of gating nodes (internal nodes in a complete b-ary tree)
        n_gates = sum(branching_factor ** d for d in range(depth))

        self.gates = nn.ModuleList([
            _GatingNetwork(n_features, branching_factor, gate_hidden, dropout_rate)
            for _ in range(n_gates)
        ])
        self.experts = nn.ModuleList([
            _ExpertNetwork(n_features, n_classes, expert_hidden)
            for _ in range(self.n_experts)
        ])

    def _compute_leaf_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mixing weights for each expert (leaf) using a top-down pass.

        Returns:
            Tensor of shape (batch_size, n_experts)
        """
        batch_size = x.size(0)
        b = self.branching_factor

        # Initialize gate node weights: root has weight 1
        n_gates = len(self.gates)
        gate_weights = torch.zeros(batch_size, n_gates + self.n_experts, device=x.device)
        gate_weights[:, 0] = 1.0

        for gate_idx in range(n_gates):
            gate_out = self.gates[gate_idx](x)  # (batch, b)
            for child in range(b):
                child_idx = b * gate_idx + child + 1
                if child_idx < n_gates:
                    gate_weights[:, child_idx] = gate_weights[:, gate_idx] * gate_out[:, child]
                else:
                    # This child is a leaf
                    leaf_idx = child_idx - n_gates
                    if leaf_idx < self.n_experts:
                        gate_weights[:, n_gates + leaf_idx] = (
                            gate_weights[:, gate_idx] * gate_out[:, child]
                        )

        return gate_weights[:, n_gates:]  # (batch, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture output: P(y|x) = Σ_e g_e(x) · P_e(y|x)

        Returns:
            Tensor of shape (batch_size, n_classes)
        """
        leaf_weights = self._compute_leaf_weights(x)  # (batch, n_experts)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (batch, n_experts, n_classes)

        output = (leaf_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return output  # (batch, n_classes)


class HierarchicalMixtureOfExperts(BaseEstimator, ClassifierMixin):
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
        Dropout probability on gating network activations.
    max_epochs : int, default=50
        Training epochs.
    learning_rate : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
    device : str, default="cpu"
    verbose : bool, default=False

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
    Neurocomputing, 419, 148–156.
    """

    def __init__(
        self,
        depth: int = 2,
        branching_factor: int = 2,
        gate_hidden: int = 32,
        expert_hidden: int = 64,
        dropout_rate: float = 0.3,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.depth = depth
        self.branching_factor = branching_factor
        self.gate_hidden = gate_hidden
        self.expert_hidden = expert_hidden
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        device = torch.device(self.device)
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.LongTensor(y_enc).to(device)

        self.model_ = _HMoEModule(
            n_features=self.n_features_in_,
            n_classes=len(self.classes_),
            depth=self.depth,
            branching_factor=self.branching_factor,
            gate_hidden=self.gate_hidden,
            expert_hidden=self.expert_hidden,
            dropout_rate=self.dropout_rate,
        ).to(device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.training_history_: List[dict] = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_b, y_b in loader:
                optimizer.zero_grad()
                probs = self.model_(X_b)
                loss = F.nll_loss(torch.log(probs.clamp(1e-7)), y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_b.size(0)
                correct += (probs.argmax(1) == y_b).sum().item()
                total += X_b.size(0)

            avg_loss = total_loss / total
            acc = correct / total
            self.training_history_.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        device = torch.device(self.device)
        self.model_.eval()
        with torch.no_grad():
            probs = self.model_(torch.FloatTensor(X).to(device))
        return probs.cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))

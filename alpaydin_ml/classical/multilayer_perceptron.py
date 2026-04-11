"""
GAL Network — Grow and Learn
=============================
Implementation based on:
    Alpaydın, E. (1994).
    GAL: Networks that Grow when they Learn and Shrink when they Forget.
    International Journal of Pattern Recognition and Artificial Intelligence, 8, 391–414.

Key idea:
    A constructive neural network that dynamically adds hidden units when the
    network is failing to learn (high error), and prunes units when they become
    redundant (low activation variance). This avoids the need to pre-specify
    network architecture.

    Growth criterion:  if error > θ_grow  → add a new hidden unit
    Pruning criterion: if Var(activation) < θ_prune  → remove the unit
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import List


class GALNetwork(BaseEstimator, ClassifierMixin):
    """
    GAL (Grow and Learn) Constructive Neural Network.

    Starts with a minimal network and adds hidden units automatically
    when learning stagnates, prunes them when they become redundant.

    Parameters
    ----------
    initial_hidden : int, default=2
        Initial number of hidden units.
    max_hidden : int, default=50
        Maximum hidden units before stopping growth.
    grow_threshold : float, default=0.1
        Error threshold above which a new unit is added.
    prune_threshold : float, default=1e-4
        Activation variance below which a unit is pruned.
    max_epochs : int, default=100
        Maximum training epochs.
    learning_rate : float, default=0.01
    check_interval : int, default=5
        How often (in epochs) to check growth/pruning conditions.
    device : str, default="cpu"
    verbose : bool, default=False

    References
    ----------
    Alpaydın, E. (1994). GAL: Networks that Grow when they Learn and Shrink
    when they Forget. IJPRAI, 8, 391–414.
    """

    def __init__(
        self,
        initial_hidden: int = 2,
        max_hidden: int = 50,
        grow_threshold: float = 0.1,
        prune_threshold: float = 1e-4,
        max_epochs: int = 100,
        learning_rate: float = 0.01,
        check_interval: int = 5,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.initial_hidden = initial_hidden
        self.max_hidden = max_hidden
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.check_interval = check_interval
        self.device = device
        self.verbose = verbose

    def _build_model(self, n_features: int, n_hidden: int, n_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_classes),
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)
        device = torch.device(self.device)

        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.LongTensor(y_enc).to(device)

        n_hidden = self.initial_hidden
        model = self._build_model(self.n_features_in_, n_hidden, n_classes).to(device)
        self.architecture_history_: List[dict] = []

        for epoch in range(self.max_epochs):
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            model.train()
            optimizer.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == y_t).float().mean().item()
            error = 1.0 - acc
            self.architecture_history_.append(
                {"epoch": epoch + 1, "n_hidden": n_hidden, "error": error}
            )

            if (epoch + 1) % self.check_interval == 0:
                with torch.no_grad():
                    hidden_acts = model[1](model[0](X_t))  # (N, n_hidden)
                    act_var = hidden_acts.var(dim=0)  # (n_hidden,)

                # Prune low-variance units
                keep_mask = act_var > self.prune_threshold
                if keep_mask.sum() < n_hidden and keep_mask.sum() >= 1:
                    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
                    W1 = model[0].weight.data[keep_idx]
                    b1 = model[0].bias.data[keep_idx]
                    W2 = model[2].weight.data[:, keep_idx]
                    b2 = model[2].bias.data

                    n_hidden = len(keep_idx)
                    model = self._build_model(self.n_features_in_, n_hidden, n_classes).to(device)
                    model[0].weight.data = W1
                    model[0].bias.data = b1
                    model[2].weight.data = W2
                    model[2].bias.data = b2

                    if self.verbose:
                        print(f"Epoch {epoch+1}: Pruned to {n_hidden} hidden units")

                # Grow if error is high
                elif error > self.grow_threshold and n_hidden < self.max_hidden:
                    W1_new = torch.cat([
                        model[0].weight.data,
                        torch.randn(1, self.n_features_in_, device=device) * 0.1
                    ], dim=0)
                    b1_new = torch.cat([model[0].bias.data, torch.zeros(1, device=device)])
                    W2_new = torch.cat([
                        model[2].weight.data,
                        torch.randn(n_classes, 1, device=device) * 0.1
                    ], dim=1)

                    n_hidden += 1
                    model = self._build_model(self.n_features_in_, n_hidden, n_classes).to(device)
                    model[0].weight.data = W1_new
                    model[0].bias.data = b1_new
                    model[2].weight.data = W2_new

                    if self.verbose:
                        print(f"Epoch {epoch+1}: Grew to {n_hidden} hidden units")

        self.model_ = model
        self.n_hidden_final_ = n_hidden
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        device = torch.device(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.FloatTensor(X).to(device))
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

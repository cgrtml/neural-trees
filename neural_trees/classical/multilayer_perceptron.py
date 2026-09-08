"""
GAL Network: Grow and Learn
=============================
Implementation based on:
    Alpaydın, E. (1994).
    GAL: Networks that Grow when they Learn and Shrink when they Forget.
    International Journal of Pattern Recognition and Artificial Intelligence, 8, 391-414.

Key idea:
    A constructive neural network that dynamically adds hidden units when the
    network is failing to learn (high error), and prunes units when they become
    redundant (low activation variance). This avoids the need to pre-specify
    network architecture.

    Growth criterion:  if error > θ_grow  → add a new hidden unit
    Pruning criterion: if Var(activation) < θ_prune  → remove the unit
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, TensorDataset

from neural_trees._validation import check_predict_input


class GALNetwork(ClassifierMixin, BaseEstimator):
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
    batch_size : int, default=32
        Mini-batch size. Training used to take a single full-batch step per
        epoch, which left the network barely moved from its initialization when
        the growth criterion was evaluated.
    momentum : float, default=0.9
        Momentum for the SGD optimizer.
    device : str, default="cpu"
    verbose : bool, default=False
    random_state : int or None, default=None
        Seed for weight initialization and for the units added during growth.
        Set it for reproducible architectures.

    References
    ----------
    Alpaydın, E. (1994). GAL: Networks that Grow when they Learn and Shrink
    when they Forget. IJPRAI, 8, 391-414.
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
        batch_size: int = 32,
        momentum: float = 0.9,
        device: str = "cpu",
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.initial_hidden = initial_hidden
        self.max_hidden = max_hidden
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.check_interval = check_interval
        self.batch_size = batch_size
        self.momentum = momentum
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def _make_optimizer(self, model: nn.Sequential) -> "torch.optim.Optimizer":
        """A fresh optimizer, needed whenever growth or pruning rebuilds the network."""
        return torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _build_model(self, n_features: int, n_hidden: int, n_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_classes),
        )

    @staticmethod
    def _hidden_activations(model: nn.Sequential, X_t: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            return model[1](model[0](X_t))

    def _contributions(self, model: nn.Sequential, X_t: "torch.Tensor") -> "torch.Tensor":
        """
        How much each hidden unit actually moves the output.

        Activation variance alone, which is what pruning used to look at, says
        nothing about whether the unit matters: a nearly constant unit with a
        large outgoing weight still shifts every logit. Weighting the spread of
        a unit's activation by the size of its outgoing weights measures the
        thing pruning is supposed to care about.
        """
        activations = self._hidden_activations(model, X_t)
        spread = activations.std(dim=0)
        outgoing = model[2].weight.data.abs().sum(dim=0)
        return spread * outgoing

    def _rebuild_with_units(
        self, model: nn.Sequential, keep_idx, n_classes: int, device
    ) -> nn.Sequential:
        """Return a copy of `model` holding only the hidden units in `keep_idx`."""
        rebuilt = self._build_model(self.n_features_in_, len(keep_idx), n_classes).to(device)
        rebuilt[0].weight.data = model[0].weight.data[keep_idx].clone()
        rebuilt[0].bias.data = model[0].bias.data[keep_idx].clone()
        rebuilt[2].weight.data = model[2].weight.data[:, keep_idx].clone()
        rebuilt[2].bias.data = model[2].bias.data.clone()
        return rebuilt

    def _grown(self, model: nn.Sequential, n_classes: int, device) -> nn.Sequential:
        """Return a copy of `model` with one more hidden unit."""
        n_hidden = model[0].weight.shape[0]
        grown = self._build_model(self.n_features_in_, n_hidden + 1, n_classes).to(device)
        grown[0].weight.data = torch.cat([
            model[0].weight.data,
            torch.randn(1, self.n_features_in_, device=device) * 0.1,
        ], dim=0)
        grown[0].bias.data = torch.cat([
            model[0].bias.data, torch.zeros(1, device=device)
        ])
        grown[2].weight.data = torch.cat([
            model[2].weight.data,
            torch.randn(n_classes, 1, device=device) * 0.1,
        ], dim=1)
        grown[2].bias.data = model[2].bias.data.clone()
        return grown

    @staticmethod
    def _evaluate(model: nn.Sequential, X_t: "torch.Tensor", y_t: "torch.Tensor"):
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t).item()
            error = 1.0 - (logits.argmax(1) == y_t).float().mean().item()
        return loss, error

    @staticmethod
    def _snapshot(model: nn.Sequential) -> dict:
        return {
            "n_hidden": model[0].weight.shape[0],
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        }

    def _restore(self, snapshot: dict, n_classes: int, device) -> nn.Sequential:
        model = self._build_model(self.n_features_in_, snapshot["n_hidden"], n_classes).to(device)
        model.load_state_dict(snapshot["state"])
        return model

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
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

        optimizer = self._make_optimizer(model)

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=min(self.batch_size, len(X_t)),
            shuffle=True,
            generator=generator,
        )

        for epoch in range(self.max_epochs):
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            # The growth and pruning criteria have to see the network as it
            # stands at the end of the epoch, not the logits from one stale
            # mini-batch.
            model.eval()
            with torch.no_grad():
                error = 1.0 - (model(X_t).argmax(1) == y_t).float().mean().item()
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

                    optimizer = self._make_optimizer(model)

                    if self.verbose:
                        print(f"Epoch {epoch+1}: Pruned to {n_hidden} hidden units")

                # Grow if error is high
                elif error > self.grow_threshold and n_hidden < self.max_hidden:
                    b2_old = model[2].bias.data.clone()
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
                    model[2].bias.data = b2_old

                    optimizer = self._make_optimizer(model)

                    if self.verbose:
                        print(f"Epoch {epoch+1}: Grew to {n_hidden} hidden units")

        self.model_ = model
        self.n_hidden_final_ = n_hidden
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_predict_input(self, X)
        device = torch.device(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.FloatTensor(X).to(device))
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

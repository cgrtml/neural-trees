"""
Soft Decision Tree Regressor
============================
The regression form of the soft decision tree of İrsoy, Yıldız & Alpaydın
(ICPR 2012). The tree is the same object as in `SoftDecisionTree`: sigmoid
gates route probability mass, every sample reaches every leaf with some
probability. What changes is what a leaf holds. A classifier's leaf holds a
class distribution; here a leaf holds a value (or a vector of values for
multi-output targets), and the prediction is the arrival-probability-weighted
average of the leaf values,

    y_hat(x) = sum_leaf mu_leaf(x) v_leaf .

Everything is trained by gradient descent on the weighted squared error, with
the same balanced-routing penalty the classifier uses.

Not yet supported, and said so rather than silently missing: growth during
training (`growth="incremental"` / `"per_leaf"`) and the hard-tree export.
Both are the classifier's machinery with the leaf type changed and are the
next things to add; see issue #103.
"""
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_X_y

from neural_trees._batching import TensorBatches
from neural_trees._validation import check_predict_input, resolve_device
from neural_trees.decision_trees.soft_decision_tree import _SoftTreeModule


class _SoftTreeRegModule(_SoftTreeModule):
    """
    The classifier's module with the leaves read as values.

    `node_logits` is reused as the (n_nodes, n_outputs) table of node values so
    that every piece of tree machinery (walking, splitting flags, gate
    penalty) is shared unchanged. Only the combination differs: a weighted
    sum of values instead of a log-mixture of distributions.
    """

    def predict_values(self, x: torch.Tensor) -> torch.Tensor:
        _, _, terminal = self._walk(x)
        out: Optional[torch.Tensor] = None
        for log_mu, indices in terminal:
            part = (log_mu.exp().unsqueeze(2) * self.node_logits[indices].unsqueeze(0)).sum(dim=1)
            out = part if out is None else out + part
        assert out is not None  # _walk always yields the bottom level
        return out


class SoftDecisionTreeRegressor(RegressorMixin, BaseEstimator):
    """
    Soft decision tree for regression, scikit-learn compatible.

    Parameters
    ----------
    depth : int, default=4
        Depth of the complete tree; ``2**depth`` leaves.
    max_epochs : int, default=100
    learning_rate : float, default=0.01
    batch_size : int, default=64
    penalty_coef : float, default=1e-3
        Weight of the balanced-routing penalty, the same regulariser the
        classifier uses; it discourages gates from sending everything one way.
    device : str, default="cpu"
        ``"auto"`` picks CUDA, then MPS, then CPU.
    random_state : int or None, default=None
    early_stopping : bool, default=False
        Hold out `validation_fraction` of the training data and stop when the
        validation loss has not improved for `n_iter_no_change` epochs,
        restoring the best parameters.
    validation_fraction : float, default=0.1
    n_iter_no_change : int, default=10
    verbose : bool, default=False

    Attributes
    ----------
    model_ : the fitted module
    n_features_in_ : int
    n_outputs_ : int
    n_iter_ : int
        Epochs actually run.
    training_history_ : list of dict
        Per-epoch training loss and, with early stopping, validation loss.

    Notes
    -----
    Targets are centred and scaled internally by their (weighted) mean and
    standard deviation for the optimiser's sake; predictions are returned in
    the original units. Leaves start at the target mean, so an untrained
    tree predicts the mean everywhere, which is the right place to start
    from.
    """

    def __init__(
        self,
        depth: int = 4,
        max_epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        penalty_coef: float = 1e-3,
        device: str = "cpu",
        random_state: Optional[int] = None,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        verbose: bool = False,
    ):
        self.depth = depth
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.penalty_coef = penalty_coef
        self.device = device
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose

    # scikit-learn >= 1.6 reads __sklearn_tags__; older versions read _more_tags.
    # Both say the same thing: multi-output targets are supported.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def _more_tags(self):
        return {"multioutput": True}

    def fit(self, X, y, sample_weight=None) -> "SoftDecisionTreeRegressor":
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError(f"depth must be a positive integer, got {self.depth!r}")
        if self.early_stopping and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}"
            )
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y, dtype=np.float64)
        single = y.ndim == 1
        Y = y.reshape(-1, 1) if single else y
        self.n_outputs_ = Y.shape[1]
        self._single_output = single

        weights = _check_sample_weight(sample_weight, X, dtype=np.float64)
        weights = weights * (len(weights) / weights.sum())

        # centre and scale targets; leaves start at the weighted mean
        self.y_mean_ = np.average(Y, axis=0, weights=weights)
        self.y_scale_ = np.sqrt(np.average((Y - self.y_mean_) ** 2, axis=0, weights=weights))
        self.y_scale_ = np.where(self.y_scale_ > 1e-12, self.y_scale_, 1.0)
        Ys = (Y - self.y_mean_) / self.y_scale_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_fit, Y_fit, w_fit, X_val, Y_val = X, Ys, weights, None, None
        if self.early_stopping and len(X) >= 10:
            X_fit, X_val, Y_fit, Y_val, w_fit, _ = train_test_split(
                X, Ys, weights, test_size=self.validation_fraction, random_state=self.random_state
            )

        device = self.device_ = resolve_device(self.device)
        X_t = torch.FloatTensor(X_fit).to(device)
        Y_t = torch.FloatTensor(Y_fit).to(device)
        w_t = torch.FloatTensor(w_fit).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device) if X_val is not None else None
        Y_val_t = torch.FloatTensor(Y_val).to(device) if Y_val is not None else None

        model = _SoftTreeRegModule(
            n_features=self.n_features_in_,
            n_classes=self.n_outputs_,
            depth=self.depth,
            penalty_coef=self.penalty_coef,
            learn_temperature=False,
        ).to(device)
        loader = TensorBatches(
            (X_t, Y_t, w_t), self.batch_size, torch.Generator().manual_seed(self.random_state or 0)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        self.training_history_ = []
        best_val, best_state, since_best = np.inf, None, 0
        for epoch in range(self.max_epochs):
            model.train()
            total, n = 0.0, 0
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                pred = model.predict_values(xb)
                per_sample = ((pred - yb) ** 2).mean(dim=1)
                loss = (per_sample * wb).sum() / wb.sum().clamp_min(1e-12) + model.penalty(xb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
                n += xb.size(0)
            record = {"epoch": epoch + 1, "loss": total / max(n, 1)}
            if X_val_t is not None:
                model.eval()
                with torch.no_grad():
                    record["val_loss"] = ((model.predict_values(X_val_t) - Y_val_t) ** 2).mean().item()
                if record["val_loss"] < best_val - 1e-8:
                    best_val, since_best = record["val_loss"], 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    since_best += 1
            self.training_history_.append(record)
            if self.verbose:
                print(record)
            if X_val_t is not None and since_best >= self.n_iter_no_change:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.n_iter_ = len(self.training_history_)
        self.model_ = model
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = check_predict_input(self, X)
        self.model_.eval()
        with torch.no_grad():
            out = self.model_.predict_values(torch.FloatTensor(X).to(self.device_)).cpu().numpy()
        out = out * self.y_scale_ + self.y_mean_
        return out[:, 0] if self._single_output else out

    def get_leaf_values(self) -> np.ndarray:
        """Leaf values in the original target units, shape (n_leaves, n_outputs)."""
        check_is_fitted(self)
        v = self.model_.leaf_logits.detach().cpu().numpy()
        return v * self.y_scale_ + self.y_mean_

    def get_split_weights(self):
        """Gate weight vectors, one per internal node, in feature units."""
        check_is_fitted(self)
        return [w for w in self.model_.gates.weight.detach().cpu().numpy()]

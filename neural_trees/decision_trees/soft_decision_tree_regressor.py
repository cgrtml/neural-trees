
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

Growth during training (`growth="incremental"` / `"per_leaf"`), the numpy
export and the hard-tree export work as on the classifier; `explain` does not
exist for regression yet.
Both are the classifier's machinery with the leaf type changed and are the
next things to add; see issue #103.
"""
import copy
import math
from typing import List, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, check_X_y

from neural_trees._batching import TensorBatches
from neural_trees._validation import check_predict_input, reject_sparse, resolve_device
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

    def split_directions(self, x: torch.Tensor, y: torch.Tensor, sample_weight=None):
        """
        The classifier's `split_directions` with the residual read as
        y - prediction instead of onehot(y) - p: for every bottom leaf, the
        unit direction its value should move in and a gate direction along the
        features of the samples pulling it that way.
        """
        with torch.no_grad():
            log_mu, _, _ = self._walk(x)
            mu = log_mu.exp()                                   # (n, n_leaves)
            r = y - self.predict_values(x)                      # (n, n_out)
            n_out = r.shape[1]
            w = torch.ones(x.size(0), device=x.device) if sample_weight is None else sample_weight
            wm = mu * w.unsqueeze(1)
            resid = wm.t() @ r                                  # (n_leaves, n_out)
            norms = resid.norm(dim=1, keepdim=True)
            direction = resid / norms.clamp_min(1e-12)
            weak = norms.squeeze(1) < 1e-8
            if bool(weak.any()):
                rnd = torch.randn(int(weak.sum()), n_out, device=x.device)
                direction[weak] = rnd / rnd.norm(dim=1, keepdim=True)
            align = (r @ direction.t()) * wm
            gate = align.t() @ x
            gate = gate / gate.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return direction, gate

    def split_direction_for(self, x: torch.Tensor, y: torch.Tensor, node: int, sample_weight=None):
        """`split_directions` for one node currently acting as a leaf; see the classifier."""
        with torch.no_grad():
            _, _, terminal = self._walk(x)
            mu = None
            for log_mu, indices in terminal:
                hit = (indices == node).nonzero()
                if hit.numel():
                    mu = log_mu[:, int(hit[0])].exp()
                    break
            if mu is None:
                return None, None
            r = y - self.predict_values(x)
            w = torch.ones(x.size(0), device=x.device) if sample_weight is None else sample_weight
            wm = mu * w
            resid = wm @ r
            if float(resid.norm()) < 1e-8:
                resid = torch.randn(r.shape[1], device=x.device)
            direction = resid / resid.norm()
            gate = ((r @ direction) * wm) @ x
            gate = gate / gate.norm().clamp_min(1e-12)
        return direction, gate


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
    growth : {"none", "incremental", "per_leaf"}, default="none"
        As on the classifier. ``"incremental"`` grows one level at a time and
        keeps a level only if held-out squared error improves; ``"per_leaf"``
        splits the leaf carrying the most weighted squared error, one at a
        time, and stops at the first split that does not pay off. Either one
        holds out ``validation_fraction`` of the training data to decide, so
        the fixed tree trains on more rows; the classifier's measurements of
        that cost apply here too. Without enough rows for a split (fewer than
        ten) the fit falls back to the full depth and ``growth_`` says so.
    growth_init : {"random", "residual", "residual_gate"}, default="residual"
        How a split's two children are told apart. Children start at the
        parent's value and are pushed apart by ``growth_jitter``: at random,
        or along the leaf's residual direction (the mean of y - prediction
        over the samples that reach it), optionally with the new gate pointed
        at the samples pulling towards the right child. Identical children
        would leave the new gate with a zero gradient, as the paper shows for
        the classifier; the same argument holds for a sum of values.
    growth_jitter : float, default=0.2
        Size of that push, in standardised target units.
    growth_budget : {"split", "full"}, default="split"
        Whether ``max_epochs`` is divided across growth rounds or given to
        every round.
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
    growth_ : str
        The growth that ran: the ``growth`` setting, or ``"none"`` when the
        data could not spare a validation split.
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
        growth: str = "none",
        growth_init: str = "residual",
        growth_jitter: float = 0.2,
        growth_budget: str = "split",
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
        self.growth = growth
        self.growth_init = growth_init
        self.growth_jitter = growth_jitter
        self.growth_budget = growth_budget

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
        if self.growth not in ("none", "incremental", "per_leaf"):
            raise ValueError(f"growth must be 'none', 'incremental' or 'per_leaf', got {self.growth!r}")
        if self.growth_init not in ("random", "residual", "residual_gate"):
            raise ValueError(
                f"growth_init must be 'random', 'residual' or 'residual_gate', got {self.growth_init!r}"
            )
        if self.growth_budget not in ("split", "full"):
            raise ValueError(f"growth_budget must be 'split' or 'full', got {self.growth_budget!r}")
        needs_validation = self.early_stopping or self.growth != "none"
        if needs_validation and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}"
            )
        reject_sparse(self, X)
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
        if needs_validation and len(X) >= 10:
            X_fit, X_val, Y_fit, Y_val, w_fit, _ = train_test_split(
                X, Ys, weights, test_size=self.validation_fraction, random_state=self.random_state
            )
        # Growth needs held-out evidence to decide anything; without a split
        # it falls back to the full depth. growth_ records what ran.
        self.growth_ = self.growth if X_val is not None else "none"

        device = self.device_ = resolve_device(self.device)
        X_t = torch.FloatTensor(X_fit).to(device)
        Y_t = torch.FloatTensor(Y_fit).to(device)
        w_t = torch.FloatTensor(w_fit).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device) if X_val is not None else None
        Y_val_t = torch.FloatTensor(Y_val).to(device) if Y_val is not None else None

        loader = TensorBatches(
            (X_t, Y_t, w_t), self.batch_size, torch.Generator().manual_seed(self.random_state or 0)
        )
        self.training_history_: List[dict] = []

        if self.growth_ == "per_leaf":
            model = self._fit_per_leaf(loader, X_t, Y_t, w_t, X_val_t, Y_val_t, device)
        elif self.growth_ == "incremental":
            model = self._fit_incrementally(loader, X_val_t, Y_val_t, device)
        else:
            model = self._new_module(self.depth, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            _, best_state = self._train_epochs(
                model, loader, optimizer, self.max_epochs, X_val_t, Y_val_t,
                stop_early=self.early_stopping and X_val_t is not None,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
        self.n_iter_ = len(self.training_history_)
        self.tree_depth_ = model.depth
        self.model_ = model
        # A float64 CPU copy for prediction, as the mixture of experts keeps:
        # in float32 the same row scored inside a different batch differs by
        # ~1e-7 from BLAS blocking, which target scaling amplifies past the
        # tolerance scikit-learn's subset-invariance check applies.
        self.model_double_ = copy.deepcopy(self.model_).cpu().double()
        self.model_double_.eval()
        return self

    # ── training and growth ──
    def _new_module(self, depth: int, device) -> _SoftTreeRegModule:
        return _SoftTreeRegModule(
            n_features=self.n_features_in_,
            n_classes=self.n_outputs_,
            depth=depth,
            penalty_coef=self.penalty_coef,
            learn_temperature=False,
        ).to(device)

    def _train_epochs(self, model, loader, optimizer, n_epochs, X_val_t, Y_val_t, stop_early=False):
        """
        Run `n_epochs`, recording each into `training_history_`. Returns the
        best validation loss and the parameters that produced it, or
        (inf, None) without a validation split.
        """
        best_val, best_state, since_best = np.inf, None, 0
        for _ in range(n_epochs):
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
            record = {"epoch": len(self.training_history_) + 1, "depth": model.depth, "loss": total / max(n, 1)}
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
            if stop_early and X_val_t is not None and since_best >= self.n_iter_no_change:
                break
        return best_val, best_state

    def _epochs_per_round(self, rounds: int) -> int:
        if self.growth_budget == "full":
            return max(1, self.max_epochs)
        return max(1, self.max_epochs // max(1, rounds))

    def _fit_incrementally(self, loader, X_val_t, Y_val_t, device):
        """One level at a time, kept only if held-out squared error improves."""
        epochs_per_round = self._epochs_per_round(self.depth)
        model = self._new_module(1, device)
        best_loss, best_state, best_depth = np.inf, None, 1
        while True:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            round_loss, round_state = self._train_epochs(model, loader, optimizer, epochs_per_round, X_val_t, Y_val_t)
            if round_state is None:
                round_loss, round_state = 0.0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if round_loss < best_loss - 1e-8 or best_state is None:
                best_loss, best_state, best_depth = round_loss, round_state, model.depth
            elif X_val_t is not None:
                if self.verbose:
                    print(f"Depth {model.depth} did not improve, keeping depth {best_depth}")
                break
            if model.depth >= self.depth:
                break
            if self.growth_init == "random":
                model = model.deepen(self.n_outputs_, jitter=self.growth_jitter)
            else:
                X_all, Y_all, w_all = loader.dataset.tensors
                direction, gate = model.split_directions(X_all, Y_all, w_all)
                model = model.deepen(
                    self.n_outputs_, jitter=self.growth_jitter, direction=direction,
                    gate=gate if self.growth_init == "residual_gate" else None,
                )
        final = self._new_module(best_depth, device)
        final.load_state_dict(best_state)
        return final

    def _fit_per_leaf(self, loader, X_t, Y_t, w_t, X_val_t, Y_val_t, device):
        """Split the leaf carrying the most weighted squared error, one at a time."""
        max_splits = 2 ** self.depth - 1
        epochs_per_round = self._epochs_per_round(self.depth * 2)
        model = self._new_module(self.depth, device)
        with torch.no_grad():
            model.is_split.fill_(False)
        best_loss, best_state = np.inf, None
        while True:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            round_loss, round_state = self._train_epochs(model, loader, optimizer, epochs_per_round, X_val_t, Y_val_t)
            if round_state is None:
                round_loss, round_state = 0.0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if round_loss < best_loss - 1e-8 or best_state is None:
                best_loss, best_state = round_loss, round_state
            elif X_val_t is not None:
                if self.verbose:
                    print("Splitting stopped paying off; keeping the previous tree")
                break
            if int(model.is_split.sum()) >= max_splits:
                break
            victim = self._neediest_leaf(model, X_t, Y_t, w_t)
            if victim is None:
                break
            self._initialise_split(model, victim, X_t, Y_t, w_t)
            if self.verbose:
                print(f"Split node {victim}; {int(model.is_split.sum())} splits now")
        model.load_state_dict(best_state)
        return model

    @staticmethod
    def _neediest_leaf(model, X_t, Y_t, w_t):
        """The acting leaf with the most weighted squared error mass, or None."""
        model.eval()
        with torch.no_grad():
            _, _, terminal = model._walk(X_t)
            err = ((model.predict_values(X_t) - Y_t) ** 2).mean(dim=1) * w_t   # (n,)
            best_index, best_mass = None, -np.inf
            for log_mu, indices in terminal:
                internal = indices < model.n_internal
                if not bool(internal.any()):
                    continue
                mass = (log_mu[:, internal].exp() * err.unsqueeze(1)).sum(dim=0)
                position = int(mass.argmax())
                if float(mass[position]) > best_mass:
                    best_mass = float(mass[position])
                    best_index = int(indices[internal][position])
        return best_index

    def _initialise_split(self, model, victim: int, X_t, Y_t, w_t):
        """Open `victim`: children inherit its value, pushed apart; gate neutral."""
        left, right = 2 * victim + 1, 2 * victim + 2
        with torch.no_grad():
            parent = model.node_logits[victim].clone()
            if self.growth_init == "random":
                step = self.growth_jitter * torch.randn_like(parent)
                gate = None
            else:
                direction, gate = model.split_direction_for(X_t, Y_t, victim, w_t)
                step = self.growth_jitter * math.sqrt(self.n_outputs_) * direction
                if self.growth_init != "residual_gate":
                    gate = None
            model.node_logits[left] = parent - step
            model.node_logits[right] = parent + step
            model.gates.weight[victim] = 0.0 if gate is None else 0.1 * gate
            model.gates.bias[victim] = 0.0
            model.log_beta[victim] = 0.0
            model.is_split[victim] = True

    # ── exports ──
    def to_numpy(self, feature_names=None):
        """
        The fitted tree as a :class:`~neural_trees.NumpySoftTree` of kind
        ``"regressor"``: same model, no torch in the prediction path, JSON
        round trip. Predictions agree with :meth:`predict` to float32 precision.
        """
        from neural_trees.decision_trees.numpy_soft_tree import NumpySoftTree

        check_is_fitted(self)
        m = self.model_
        return NumpySoftTree(
            depth=m.depth,
            weights=m.gates.weight.detach().cpu().numpy(),
            biases=m.gates.bias.detach().cpu().numpy(),
            log_beta=m.log_beta.detach().cpu().numpy(),
            node_logits=m.node_logits.detach().cpu().numpy(),
            is_split=m.is_split.detach().cpu().numpy(),
            classes=[],
            feature_names=feature_names,
            kind="regressor",
            y_mean=self.y_mean_,
            y_scale=self.y_scale_,
            single_output=self._single_output,
        )

    def to_hard_tree(self):
        """
        The gates read as hard decisions and one value per leaf, in target
        units: a :class:`~neural_trees.HardRegressionTree` that prints its
        rules. A different model from the soft tree, as for the classifier;
        measure its agreement before relying on it.
        """
        from neural_trees.decision_trees.hard_tree import HardRegressionTree

        check_is_fitted(self)
        m = self.model_
        values = m.node_logits.detach().cpu().numpy() * self.y_scale_ + self.y_mean_
        return HardRegressionTree(
            weights=m.gates.weight.detach().cpu().numpy(),
            biases=m.gates.bias.detach().cpu().numpy(),
            node_values=values,
            n_features_in=self.n_features_in_,
            is_split=m.is_split.detach().cpu().numpy(),
            log_beta=m.log_beta.detach().cpu().numpy(),
            single_output=self._single_output,
        )

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        X = check_predict_input(self, X)
        with torch.no_grad():
            out = self.model_double_.predict_values(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
            ).numpy()
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

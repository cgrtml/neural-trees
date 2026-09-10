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

    This implementation departs from the 1994 paper in how a unit is added.
    A new unit is fitted to the residual error of the frozen network, in the
    manner of Fahlman & Lebiere's cascade-correlation (1990), and enters with
    zero outgoing weights so the network's function is unchanged at the moment
    of growth. Measured on Iris over 3 seeds of 5-fold CV, that reaches 0.958
    accuracy with 7.1 hidden units where random initialization reached 0.931
    with 22.9. `growth_init="random"` restores the earlier behaviour.
"""

import copy

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
        Training-error threshold above which a new unit is added. Only used by
        the ``"error_threshold"`` policy.
    prune_threshold : float, default=1e-4
        Activation variance below which a unit is pruned. Only used by the
        ``"error_threshold"`` policy.
    max_epochs : int, default=100
        Maximum training epochs.
    learning_rate : float, default=0.01
    check_interval : int, default=5
        How often (in epochs) to reconsider the architecture.
    growth_policy : {"error_threshold", "validation"}, default="error_threshold"
        How growth and pruning decide.

        - ``"validation"`` holds out `validation_fraction` of the training data
          and changes the architecture only when validation loss has stopped
          improving. A unit is pruned when removing it does not hurt validation
          loss, and one is grown otherwise. Training stops after `patience`
          consecutive changes that fail to improve validation loss, and the
          parameters of the best epoch are restored.
        - ``"error_threshold"`` is the earlier behaviour: prune any unit whose
          activation variance falls below `prune_threshold`, otherwise grow
          whenever *training* error exceeds `grow_threshold`. That rule grows
          the network until it fits the training set, with nothing held out to
          say whether the extra capacity helped.

        Falls back to ``"error_threshold"`` when the data is too small to hold
        out a usable validation split; `growth_policy_` records what was used.

        ``"validation"`` is not the default yet. It reaches far smaller networks
        for the same accuracy where capacity is not the constraint (Breast
        Cancer: 0.977 with 3.9 units against 0.975 with 2.0), but it under-grows
        where capacity *is* the constraint (6 separable blobs: 0.620 with 4.1
        units against 0.967 with 15.1). The cause is measured, not guessed: a
        capacity-starved network keeps improving its validation loss slowly, so
        "loss is still falling" never signals that more units are what is
        missing. Deciding this properly needs a signal about what a new unit
        would buy, which a randomly initialized unit cannot provide.
    growth_init : {"residual", "random"}, default="residual"
        How a new hidden unit is initialized.

        - ``"residual"`` fits the unit to what the frozen network still gets
          wrong, maximizing the correlation between its activation and the
          residual error (Fahlman & Lebiere, 1990), and gives it zero outgoing
          weights so the network's function is unchanged at the moment of
          growth.
        - ``"random"`` is the earlier behaviour: random incoming and outgoing
          weights, which perturbs every logit on arrival.
    n_candidates : int, default=4
        Candidate units trained in parallel at each growth step; the one that
        correlates best with the residual is installed. Ignored when
        `growth_init="random"`.
    candidate_epochs : int, default=40
        Gradient steps spent fitting each candidate. Ignored when
        `growth_init="random"`.
    validation_fraction : float, default=0.2
        Fraction held out under the ``"validation"`` policy.
    tol : float, default=1e-2
        Relative improvement in validation loss that counts as progress. A loss
        still creeping down by a fraction of a percent per check is a network
        that has stopped learning anything useful with the capacity it has, and
        treating that as progress is what keeps it from ever growing.
    patience : int, default=5
        Consecutive architecture changes without a validation improvement
        before training stops.
    error_patience : int, default=2
        Checks without any improvement in validation *error* before the
        architecture is reconsidered, even while validation loss is still
        falling. A network that has run out of capacity keeps sharpening the
        same mistakes, which shows up as a falling loss over a flat error.
    batch_size : int, default=32
        Mini-batch size. Training used to take a single full-batch step per
        epoch, which left the network barely moved from its initialization when
        the growth criterion was evaluated.
    momentum : float, default=0.9
        Momentum for the SGD optimizer.
    device : str, default="cpu"
        PyTorch device. `"auto"` picks CUDA if it is available, then Apple
        silicon's MPS, then CPU. Resolved once in `fit` and recorded as
        `device_`.
    verbose : bool, default=False
    random_state : int or None, default=None
        Seed for weight initialization and for the units added during growth.
        Set it for reproducible architectures.
    warm_start : bool, default=False
        When True, a second call to `fit` continues from the network the first
        one left, keeping both its weights and the architecture growth chose,
        instead of restarting from `initial_hidden`.
    class_weight : dict, "balanced" or None, default=None
        Weights per class, combined multiplicatively with `sample_weight`.
        `"balanced"` uses `n_samples / (n_classes * bincount(y))`. Without it a
        rare class contributes so little loss that the model can ignore it.

    References
    ----------
    Alpaydın, E. (1994). GAL: Networks that Grow when they Learn and Shrink
    when they Forget. IJPRAI, 8, 391-414.
    """

    # Declared so the type is known where warm_start reads it back, before the
    # assignment at the end of fit. A bare annotation adds nothing to the
    # instance, so sklearn's "no attributes set in __init__" check is unaffected.
    model_: nn.Sequential

    def __init__(
        self,
        initial_hidden: int = 2,
        max_hidden: int = 50,
        grow_threshold: float = 0.1,
        prune_threshold: float = 1e-4,
        max_epochs: int = 100,
        learning_rate: float = 0.01,
        check_interval: int = 5,
        growth_policy: str = "error_threshold",
        growth_init: str = "residual",
        n_candidates: int = 4,
        candidate_epochs: int = 40,
        validation_fraction: float = 0.2,
        tol: float = 1e-2,
        patience: int = 5,
        error_patience: int = 2,
        batch_size: int = 32,
        momentum: float = 0.9,
        device: str = "cpu",
        verbose: bool = False,
        random_state: Optional[int] = None,
        class_weight=None,
        warm_start: bool = False,
    ):
        self.initial_hidden = initial_hidden
        self.max_hidden = max_hidden
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.check_interval = check_interval
        self.growth_policy = growth_policy
        self.growth_init = growth_init
        self.n_candidates = n_candidates
        self.candidate_epochs = candidate_epochs
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.patience = patience
        self.error_patience = error_patience
        self.batch_size = batch_size
        self.momentum = momentum
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.class_weight = class_weight
        self.warm_start = warm_start

    def _make_optimizer(self, model: nn.Sequential) -> "torch.optim.Optimizer":
        """A fresh optimizer, needed whenever growth or pruning rebuilds the network."""
        return torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _carry_optimizer(
        self,
        optimizer: "torch.optim.Optimizer",
        old_model: nn.Sequential,
        new_model: nn.Sequential,
        keep_idx=None,
    ) -> "torch.optim.Optimizer":
        """
        Rebuild the optimizer for a changed architecture, keeping the momentum
        of the units that survived.

        Growth and pruning replace the module, and a fresh optimizer starts
        every surviving unit from a standstill. Training then has to rebuild
        the momentum it had before it can make progress, which is why the
        network looked like it had stopped improving whenever the architecture
        moved.

        The buffers are reshaped exactly the way the weights are: pruning keeps
        `keep_idx`, growth appends a zero row and column for the new unit,
        which has no history to carry.
        """
        new_optimizer = self._make_optimizer(new_model)
        if self.momentum == 0:
            return new_optimizer

        old_params = list(old_model.parameters())
        new_params = list(new_model.parameters())
        for position, (old_param, new_param) in enumerate(zip(old_params, new_params)):
            buffer = optimizer.state.get(old_param, {}).get("momentum_buffer")
            if buffer is None:
                continue

            if keep_idx is not None:
                if position == 0:          # first layer weight, one row per unit
                    carried = buffer[keep_idx]
                elif position == 1:        # first layer bias
                    carried = buffer[keep_idx]
                elif position == 2:        # second layer weight, one column per unit
                    carried = buffer[:, keep_idx]
                else:                      # second layer bias, unit independent
                    carried = buffer
            else:
                carried = torch.zeros_like(new_param)
                if position == 2:
                    carried[:, : buffer.shape[1]] = buffer
                elif position == 3:
                    carried = buffer.clone()
                else:
                    carried[: buffer.shape[0]] = buffer

            new_optimizer.state[new_param]["momentum_buffer"] = carried.clone()

        return new_optimizer

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

    def _residual(self, model: nn.Sequential, X_t, y_t, n_classes: int):
        """
        What the current network is getting wrong, per class.

        For softmax with cross-entropy this is the gradient of the loss with
        respect to the logits, `p - y`, which is exactly the part of the target
        the existing units have failed to explain.
        """
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(X_t), dim=1)
            residual = probabilities - F.one_hot(y_t, n_classes).float()
            return residual - residual.mean(dim=0, keepdim=True)

    def _train_candidate(self, X_t, residual, device):
        """
        Fit a candidate hidden unit to the residual error, and report how well
        it managed (Fahlman & Lebiere, 1990).

        The candidate maximizes the correlation between its own activation and
        what the frozen network still gets wrong, normalized by the spread of
        that activation so the objective cannot be inflated by scaling alone.

        The returned score is the useful part: a unit that cannot correlate
        with the residual is telling you that capacity is not what is missing,
        which is a signal the network's own learning curve does not provide.
        """
        best_score = -np.inf
        best_weight = best_bias = None

        for _ in range(self.n_candidates):
            candidate = nn.Linear(self.n_features_in_, 1).to(device)
            optimizer = torch.optim.Adam(candidate.parameters(), lr=0.05)

            score = torch.zeros((), device=device)
            for _ in range(self.candidate_epochs):
                optimizer.zero_grad()
                activation = torch.sigmoid(candidate(X_t)).squeeze(-1)
                centered = activation - activation.mean()
                covariance = (centered.unsqueeze(1) * residual).sum(dim=0).abs().sum()
                # Clamp before the square root, not after: sqrt has an infinite
                # gradient at zero, so a candidate whose sigmoid saturates into a
                # constant activation sends NaN back through the optimizer and
                # every later score with it.
                score = covariance / centered.pow(2).sum().clamp_min(1e-12).sqrt()
                (-score).backward()
                optimizer.step()

            if np.isfinite(score.item()) and score.item() > best_score:
                best_score = score.item()
                best_weight = candidate.weight.detach().clone()
                best_bias = candidate.bias.detach().clone()

        if best_weight is None:
            # No candidate produced a usable score. Fall back to a small random
            # unit rather than refusing to grow.
            best_weight = torch.randn(1, self.n_features_in_, device=device) * 0.1
            best_bias = torch.zeros(1, device=device)
            best_score = 0.0

        return best_weight, best_bias, best_score

    def _grown(self, model: nn.Sequential, n_classes: int, device, X_t=None, y_t=None):
        """
        Return a copy of `model` with one more hidden unit.

        With `growth_init="residual"` the new unit is fitted to the residual
        error first and enters with **zero** outgoing weights, so the network
        computes exactly the same function the moment it grows. Growth cannot
        make the model worse; it can only give training something new to use.
        A random unit, by contrast, perturbs every logit on arrival and then has
        to be trained from noise.
        """
        n_hidden = model[0].weight.shape[0]
        grown = self._build_model(self.n_features_in_, n_hidden + 1, n_classes).to(device)

        use_residual = self.growth_init == "residual" and X_t is not None
        if use_residual:
            residual = self._residual(model, X_t, y_t, n_classes)
            new_weight, new_bias, score = self._train_candidate(X_t, residual, device)
            new_output = torch.zeros(n_classes, 1, device=device)
            self.last_candidate_score_ = float(score)
        else:
            new_weight = torch.randn(1, self.n_features_in_, device=device) * 0.1
            new_bias = torch.zeros(1, device=device)
            new_output = torch.randn(n_classes, 1, device=device) * 0.1

        grown[0].weight.data = torch.cat([model[0].weight.data, new_weight], dim=0)
        grown[0].bias.data = torch.cat([model[0].bias.data, new_bias])
        grown[2].weight.data = torch.cat([model[2].weight.data, new_output], dim=1)
        grown[2].bias.data = model[2].bias.data.clone()
        return grown

    @staticmethod
    def _evaluate(
        model: nn.Sequential,
        X_t: "torch.Tensor",
        y_t: "torch.Tensor",
        w_t: Optional["torch.Tensor"] = None,
    ):
        """
        Weighted loss and error, which is what growth and pruning have to see.

        Weighting the loss but judging the architecture on unweighted accuracy
        would let sample_weight change what the network fits while leaving what
        it *builds* untouched, so a rare class could be worth a lot to the loss
        and nothing to the decision about capacity.
        """
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            correct = (logits.argmax(1) == y_t).float()
            if w_t is None:
                loss = F.cross_entropy(logits, y_t).item()
                error = 1.0 - correct.mean().item()
            else:
                per_sample = F.cross_entropy(logits, y_t, reduction="none")
                total = w_t.sum().clamp_min(1e-12)
                loss = ((per_sample * w_t).sum() / total).item()
                error = 1.0 - ((correct * w_t).sum() / total).item()
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

    def _reuse_existing_model(self, n_classes: int) -> bool:
        """
        Whether a previous fit's network can be continued from.

        Refusing loudly when the label set changed is deliberate: silently
        reinitializing would make warm_start look like it worked while throwing
        away everything the first fit learned, architecture included.
        """
        if not self.warm_start or not hasattr(self, "model_"):
            return False
        if len(getattr(self, "classes_", [])) != n_classes:
            raise ValueError(
                "warm_start=True requires the same classes across calls to fit; "
                "the label set changed, and this model cannot grow its output layer."
            )
        return True

    def fit(self, X, y, sample_weight=None) -> "GALNetwork":
        """
        Fit the network, growing and pruning hidden units as it trains.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        if self.growth_policy not in ("validation", "error_threshold"):
            raise ValueError(
                "growth_policy must be 'validation' or 'error_threshold', got "
                f"{self.growth_policy!r}"
            )
        if self.growth_init not in ("residual", "random"):
            raise ValueError(
                f"growth_init must be 'residual' or 'random', got {self.growth_init!r}"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}"
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

        n_classes = len(self.classes_)
        device = self.device_ = resolve_device(self.device)

        X_fit, y_fit, w_fit, X_val, y_val, w_val = self._split_for_validation(X, y_enc, weights)
        self.growth_policy_ = "validation" if X_val is not None else "error_threshold"

        X_t = torch.FloatTensor(X_fit).to(device)
        y_t = torch.LongTensor(y_fit).to(device)
        w_t = torch.FloatTensor(w_fit).to(device)
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.LongTensor(y_val).to(device)
            w_val_t = torch.FloatTensor(w_val).to(device)
        else:
            X_val_t, y_val_t, w_val_t = X_t, y_t, w_t

        # Continuing keeps the architecture the previous fit grew, not just its
        # weights: restarting from initial_hidden would throw away the search.
        model: nn.Sequential = (
            self.model_
            if continuing
            else self._build_model(self.n_features_in_, self.initial_hidden, n_classes).to(device)
        )
        optimizer = self._make_optimizer(model)

        generator = None
        if self.random_state is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(X_t, y_t, w_t),
            batch_size=min(self.batch_size, len(X_t)),
            shuffle=True,
            generator=generator,
        )

        self.architecture_history_: List[dict] = []
        best_loss = np.inf
        best_snapshot = self._snapshot(model)
        # The architecture decision asks what the last stretch of training
        # bought, so it compares against the previous checkpoint. best_loss is
        # bookkeeping for restoring the best parameters at the end, and using it
        # as the reference made the network prune itself at the very first
        # check, before it had trained at all.
        reference_loss = np.inf
        best_error = np.inf
        checks_without_error_gain = 0
        changes_without_gain = 0

        for epoch in range(self.max_epochs):
            model.train()
            for X_batch, y_batch, w_batch in loader:
                optimizer.zero_grad()
                per_sample = F.cross_entropy(model(X_batch), y_batch, reduction="none")
                loss = (per_sample * w_batch).sum() / w_batch.sum().clamp_min(1e-12)
                loss.backward()
                optimizer.step()

            # Architecture decisions look at the network as it stands at the end
            # of the epoch, not at one stale mini-batch's logits.
            train_loss, train_error = self._evaluate(model, X_t, y_t, w_t)
            val_loss, val_error = (
                self._evaluate(model, X_val_t, y_val_t, w_val_t)
                if X_val is not None
                else (train_loss, train_error)
            )

            record = {
                "epoch": epoch + 1,
                "n_hidden": model[0].weight.shape[0],
                "error": train_error,
                "loss": train_loss,
                "val_error": val_error,
                "val_loss": val_loss,
                "action": "train",
            }

            if (epoch + 1) % self.check_interval == 0:
                if self.growth_policy_ == "validation":
                    if val_error < best_error - 1e-9:
                        best_error = val_error
                        checks_without_error_gain = 0
                    else:
                        checks_without_error_gain += 1

                    model, optimizer, record, changes_without_gain = self._validation_step(
                        model, optimizer, record, X_t, y_t, X_val_t, y_val_t, w_val_t,
                        n_classes, device, val_loss, reference_loss,
                        checks_without_error_gain, changes_without_gain,
                    )
                    if record["action"] in ("grow", "prune"):
                        checks_without_error_gain = 0
                    if val_loss < best_loss:
                        best_loss, best_snapshot = val_loss, self._snapshot(model)

                    # After a change, the next check has to judge the new
                    # architecture against its own starting point. Comparing it
                    # to the pre-change loss punishes growth for the dip a fresh
                    # random unit causes, which stopped the search after three
                    # attempts before any added unit had time to become useful.
                    reference_loss = (
                        self._evaluate(model, X_val_t, y_val_t, w_val_t)[0]
                        if record["action"] in ("grow", "prune")
                        else val_loss
                    )
                    if changes_without_gain >= self.patience:
                        if self.verbose:
                            print(f"Epoch {epoch+1}: architecture search exhausted, stopping")
                        self.architecture_history_.append(record)
                        break
                else:
                    model, optimizer, record = self._threshold_step(
                        model, optimizer, record, X_t, y_t, train_error, n_classes, device
                    )
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_snapshot = self._snapshot(model)

            elif val_loss < best_loss:
                best_loss = val_loss
                best_snapshot = self._snapshot(model)

            self.architecture_history_.append(record)

            if self.verbose and (epoch + 1) % self.check_interval == 0:
                print(
                    f"Epoch {epoch+1}/{self.max_epochs}  units={record['n_hidden']}  "
                    f"train_err={train_error:.4f}  val_loss={val_loss:.4f}  {record['action']}"
                )

        self.model_ = self._restore(best_snapshot, n_classes, device)
        # A float64 copy on the CPU, for the same reason HMoE keeps one: rows
        # are independent, but BLAS blocks differently for different memory
        # layouts, so in float32 a sample scored inside a reordered batch can
        # come out slightly different and a borderline argmax can flip. The
        # effect grows with the network, so it reappears exactly when growth
        # has done its job.
        self.model_double_ = copy.deepcopy(self.model_).to("cpu").double()
        self.model_double_.eval()
        self.n_hidden_final_ = best_snapshot["n_hidden"]
        self.best_val_loss_ = float(best_loss)
        self.n_iter_ = len(self.architecture_history_)
        return self

    def _split_for_validation(self, X: np.ndarray, y_enc: np.ndarray, weights: np.ndarray):
        """
        Hold out a validation split, or report that the data cannot support one.

        check_estimator and small real datasets both hand over sets where a
        stratified split would leave a class unrepresented, so the policy falls
        back rather than failing.
        """
        if self.growth_policy != "validation":
            return X, y_enc, weights, None, None, None

        counts = np.bincount(y_enc)
        n_classes = len(counts)
        n_val = int(round(len(X) * self.validation_fraction))
        if counts.min() < 2 or n_val < n_classes or len(X) - n_val < n_classes:
            return X, y_enc, weights, None, None, None

        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            X,
            y_enc,
            weights,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y_enc,
        )
        return X_fit, y_fit, w_fit, X_val, y_val, w_val

    def _validation_step(
        self, model, optimizer, record, X_t, y_t, X_val_t, y_val_t, w_val_t, n_classes, device,
        val_loss, reference_loss, checks_without_error_gain, changes_without_gain,
    ):
        """
        Change the architecture only when validation loss says the current one
        has stopped paying off.

        While validation loss is still falling, the network is learning and its
        size is not the constraint. Once it stalls, try removing the least
        useful unit first: a smaller network that validates just as well is
        strictly better. Only if that fails is a unit added.
        """
        # Falling loss alone does not mean the architecture is adequate. A
        # capacity-starved network keeps getting more confident about the same
        # mistakes: on six separable blobs the validation loss fell from 1.81 to
        # 1.17 while the error sat at 0.46 the whole time, so a loss-only rule
        # never grew past three units. Progress has to show up in the error too.
        loss_improving = val_loss < reference_loss * (1.0 - self.tol)
        error_stalled = checks_without_error_gain >= self.error_patience
        if loss_improving and not error_stalled:
            record["action"] = "keep"
            return model, optimizer, record, 0

        n_hidden = model[0].weight.shape[0]

        if n_hidden > 1:
            contributions = self._contributions(model, X_t)
            victim = int(contributions.argmin())
            keep_idx = [i for i in range(n_hidden) if i != victim]
            candidate = self._rebuild_with_units(model, keep_idx, n_classes, device)
            candidate_loss, _ = self._evaluate(candidate, X_val_t, y_val_t, w_val_t)
            # Removing a unit is worth it only if validation loss does not get
            # worse than it already is: a smaller network that validates the
            # same is strictly the better model.
            if candidate_loss <= val_loss + 1e-6:
                record["action"] = "prune"
                record["n_hidden"] = n_hidden - 1
                return (
                    candidate,
                    self._carry_optimizer(optimizer, model, candidate, keep_idx),
                    record,
                    changes_without_gain + 1,
                )

        if n_hidden < self.max_hidden:
            grown = self._grown(model, n_classes, device, X_t, y_t)
            record["action"] = "grow"
            record["n_hidden"] = n_hidden + 1
            return (
                grown,
                self._carry_optimizer(optimizer, model, grown),
                record,
                changes_without_gain + 1,
            )

        record["action"] = "capped"
        return model, optimizer, record, changes_without_gain + 1

    def _threshold_step(self, model, optimizer, record, X_t, y_t, train_error, n_classes, device):
        """The pre-0.4 rule: prune on activation variance, grow on training error."""
        n_hidden = model[0].weight.shape[0]
        activations = self._hidden_activations(model, X_t)
        keep_mask = activations.var(dim=0) > self.prune_threshold

        if 1 <= int(keep_mask.sum()) < n_hidden:
            keep_idx = keep_mask.nonzero(as_tuple=True)[0].tolist()
            pruned = self._rebuild_with_units(model, keep_idx, n_classes, device)
            record["action"] = "prune"
            record["n_hidden"] = len(keep_idx)
            return pruned, self._carry_optimizer(optimizer, model, pruned, keep_idx), record

        if train_error > self.grow_threshold and n_hidden < self.max_hidden:
            grown = self._grown(model, n_classes, device, X_t, y_t)
            record["action"] = "grow"
            record["n_hidden"] = n_hidden + 1
            return grown, self._carry_optimizer(optimizer, model, grown), record

        return model, optimizer, record

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities, shape (n_samples, n_classes).

        The forward pass runs in float64 on the CPU so that a prediction is a
        property of the sample rather than of its position in the batch.
        Training stays float32 on whichever device was chosen.
        """
        check_is_fitted(self)
        X = check_predict_input(self, X)
        with torch.no_grad():
            logits = self.model_double_(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
            )
            probs = F.softmax(logits, dim=1)
        return probs.numpy()

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        return self.le_.inverse_transform(self.predict_proba(X).argmax(axis=1))

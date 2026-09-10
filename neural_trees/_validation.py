"""Shared input validation helpers."""

import numpy as np
from sklearn.utils.validation import check_array


def check_predict_input(estimator, X):
    """
    Validate `X` at predict time and reject a feature count that does not match
    the one seen during `fit`.

    scikit-learn's estimator contract requires this: without it, passing a
    differently shaped `X` silently produces meaningless output instead of an
    error.
    """
    X = check_array(X)
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(estimator).__name__} "
            f"is expecting {n_features} features as input."
        )
    # torch cannot build a tensor from a negatively strided view, which is what
    # reversed or otherwise reordered input arrives as.
    return np.ascontiguousarray(X)


def resolve_device(device):
    """
    Turn a `device` parameter into a concrete torch device.

    `"auto"` picks the fastest backend that is actually present: CUDA, then
    Apple silicon's MPS, then CPU. Anything else is handed to torch as given,
    so an explicit `"cuda:1"` still works, and an unusable value fails here,
    with the parameter named, rather than deep inside a forward pass.

    Imported lazily so that the estimators that do not use torch keep working
    without it installed.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        return torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"device must be 'auto' or a torch device string, got {device!r}"
        ) from exc

"""Shared input validation helpers."""

import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_array


def reject_sparse(estimator, X):
    """
    Raise a model-specific error for sparse input to a model that needs dense.

    The torch models multiply dense weight matrices, so a sparse matrix would
    be densified on arrival anyway; saying so, and naming the fix, beats
    scikit-learn's generic "dense data is required".
    """
    if sp.issparse(X):
        raise TypeError(
            f"{type(estimator).__name__} requires dense input; it multiplies dense "
            "weight matrices, so a sparse matrix would be densified immediately. "
            "Pass X.toarray() (or X.todense()) instead."
        )
    return X


def check_predict_input(estimator, X, accept_sparse=False):
    """
    Validate `X` at predict time and reject a feature count that does not match
    the one seen during `fit`.

    scikit-learn's estimator contract requires this: without it, passing a
    differently shaped `X` silently produces meaningless output instead of an
    error. With `accept_sparse`, CSR input is validated and returned as CSR;
    without it, sparse input gets a model-specific error naming the fix.
    """
    if accept_sparse:
        X = check_array(X, accept_sparse="csr")
    else:
        X = check_array(reject_sparse(estimator, X))
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(estimator).__name__} "
            f"is expecting {n_features} features as input."
        )
    if sp.issparse(X):
        return X
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

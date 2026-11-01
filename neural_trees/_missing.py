"""
Missing values for the soft trees.

A missing cell is set to 0 and an indicator column, 1 where the cell was
missing, is appended for every feature that had missing values in training.
For a linear gate ``w . x + b`` this is exactly a learned per-gate default:
when feature j is missing its term ``w_j * x_j`` is replaced by the
indicator's weight ``c_j``, a free parameter of that gate, so each split
learns on its own which way, and how strongly, a missing value pushes. It is
the soft counterpart of the default direction a gradient-boosted tree learns
per split, and it keeps the model a plain linear gate, so the numpy, hard-rule
and ONNX exports carry it with one extra step on the input.

A feature with no missing values in training gets no indicator; a missing
value there at prediction time is set to 0 and nothing else, which on
standardised inputs is the training mean.
"""
from __future__ import annotations

import inspect
from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.utils.validation import check_array

__all__ = ["allow_nan", "augment", "augmented_names", "missing_columns"]

_FINITE_KW = ("ensure_all_finite" if "ensure_all_finite" in inspect.signature(check_array).parameters
              else "force_all_finite")


def allow_nan() -> dict:
    """Keyword for check_array / check_X_y that lets NaN through and still rejects inf."""
    return {_FINITE_KW: "allow-nan"}


def missing_columns(X: np.ndarray) -> np.ndarray:
    """Indices of the columns of `X` with at least one NaN."""
    return np.flatnonzero(np.isnan(X).any(axis=0)).astype(np.int64)


def augment(X: np.ndarray, columns: Optional[Union[Sequence[int], np.ndarray]]) -> np.ndarray:
    """`X` with NaN set to 0 and one indicator column per entry of `columns` appended."""
    X = np.asarray(X, dtype=np.float64)
    nan = np.isnan(X)
    if not nan.any() and (columns is None or len(columns) == 0):
        return X
    filled = np.where(nan, 0.0, X)
    if columns is None or len(columns) == 0:
        return filled
    return np.hstack([filled, nan[:, np.asarray(columns, dtype=np.int64)].astype(np.float64)])


def augmented_names(names: Sequence[str], columns: Optional[Union[Sequence[int], np.ndarray]]) -> List[str]:
    """Feature names for the augmented input: the originals, then '<name> is missing'."""
    out = list(names)
    for j in (columns if columns is not None else []):
        out.append(f"{names[int(j)]} is missing")
    return out

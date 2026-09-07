"""Shared input validation helpers."""

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
    return X

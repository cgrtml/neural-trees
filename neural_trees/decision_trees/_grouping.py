"""
Turn the classes at a tree node into the two groups a binary split separates.

Both multivariate trees cluster class centroids into two groups and hand the
resulting two-group problem to a discriminant. Two things go wrong when the
clustering ignores how many samples each class has (#104): a class with a
sample or two far from the rest founds a group of its own, and a group of one
or two samples cannot be split off under `min_samples_leaf`, so the node
closes with every class still inside it. The centroids are therefore
clustered with class mass as weight, and a mass-balanced cut along the
principal axis of the centroids is offered as the fallback grouping when the
first one yields a split the tree's leaf-size rule refuses.
"""
from typing import List

import numpy as np
from sklearn.cluster import KMeans


def two_group_candidates(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    random_state,
    min_rows: int = 1,
) -> List[np.ndarray]:
    """
    Binary labelings for the samples at a node, best first; empty if none.

    The first is the mass-weighted clustering of the centroids. The second is
    the mass-balanced cut along the centroids' principal axis, offered so the
    caller can fall back to it when the discriminant fitted to the first
    grouping produces a split its leaf-size rule refuses: a group of a few
    far-away samples is separable, so the discriminant overfits it and the
    sign of the hyperplane can leave fewer rows on one side than the group
    had. `min_rows` is the smallest number of rows either group may have.
    """
    present = np.unique(y)
    if len(present) < 2:
        return []
    if len(present) == 2:
        return [(y == present[1]).astype(int)]

    centroids = np.vstack([
        np.average(X[y == c], axis=0, weights=w[y == c]) if w[y == c].sum() > 0
        else X[y == c].mean(axis=0)
        for c in present
    ])
    mass = np.array([max(w[y == c].sum(), 1e-12) for c in present])
    rows = np.array([int((y == c).sum()) for c in present])

    candidates = []
    group = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
        centroids, sample_weight=mass
    )
    if len(np.unique(group)) == 2 and _rows_per_group(group, rows).min() >= min_rows:
        candidates.append(group)

    centre = np.average(centroids, axis=0, weights=mass)
    _, _, vt = np.linalg.svd((centroids - centre) * np.sqrt(mass)[:, None], full_matrices=False)
    order = np.argsort(centroids @ vt[0])
    cumulative = np.cumsum(mass[order])
    cut = int(np.argmin(np.abs(cumulative - cumulative[-1] / 2))) + 1
    cut = min(max(cut, 1), len(present) - 1)
    balanced = np.zeros(len(present), dtype=int)
    balanced[order[cut:]] = 1
    if _rows_per_group(balanced, rows).min() >= min_rows and not any(
        np.array_equal(balanced, g) or np.array_equal(1 - balanced, g) for g in candidates
    ):
        candidates.append(balanced)
    return [_expand(present, g, y) for g in candidates]


def _rows_per_group(group: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.array([rows[group == g].sum() for g in (0, 1)])


def _expand(present: np.ndarray, group: np.ndarray, y: np.ndarray) -> np.ndarray:
    mapping = {c: int(g) for c, g in zip(present, group)}
    return np.array([mapping[label] for label in y])

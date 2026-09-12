"""
Plotting the decision boundary
==============================

Plot the decision boundary a Soft Decision Tree learns on make_moons.

The boundary is smooth rather than axis-aligned and piecewise constant,
which is the visible difference between soft and hard splits.

Run with:
    python examples/04_decision_boundary.py

Writes examples/output/04_decision_boundary.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from neural_trees import SoftDecisionTree

X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

model = SoftDecisionTree(depth=4, max_epochs=40, random_state=42)
model.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
contour = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.8)
ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.2)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=25)
ax.set_title(f"SoftDecisionTree(depth=4) on make_moons, accuracy {model.score(X, y):.3f}")
fig.colorbar(contour, ax=ax, label="P(class 1)")

# Writing a file only makes sense when this runs as a script. Inside the
# documentation gallery there is no __file__ and the figure is captured directly.
if "__file__" in globals():
    out_path = Path(__file__).parent / "output" / "04_decision_boundary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
else:
    plt.show()

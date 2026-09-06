"""
Use SoftDecisionTree inside a scikit-learn Pipeline with StandardScaler,
and report 5-fold cross-validated accuracy on the Wine dataset.

Scaling matters here: the sigmoid gates saturate when raw feature scales
differ by orders of magnitude, as they do in Wine.

Run with:
    python examples/02_pipeline_with_scaler.py
"""
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import SoftDecisionTree

X, y = load_wine(return_X_y=True)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SoftDecisionTree(depth=4, max_epochs=60, random_state=42)),
])

scores = cross_val_score(pipe, X, y, cv=5)

print(f"Fold accuracies: {[round(s, 3) for s in scores]}")
print(f"Mean accuracy:   {scores.mean():.3f} (+/- {scores.std():.3f})")

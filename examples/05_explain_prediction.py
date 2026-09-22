"""
Explaining one prediction
=========================

Ask a soft decision tree why it predicted what it predicted for a single
sample: which leaf got the mass, how each gate on the way leaned and why, and
what single change would have flipped the class.

Run with:
    python examples/05_explain_prediction.py
"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_trees import SoftDecisionTree

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, stratify=data.target, random_state=0
)
scaler = StandardScaler().fit(X_train)
model = SoftDecisionTree(depth=3, max_epochs=60, random_state=0)
model.fit(scaler.transform(X_train), y_train)

# Explain the first three test samples. Values are in standardised units,
# because that is what the model was fitted on.
explanations = model.explain(scaler.transform(X_test[:3]), feature_names=list(data.feature_names))
for i, e in enumerate(explanations):
    print(f"--- test sample {i}, true class {y_test[i]} ---")
    print(e.to_text())
    print()

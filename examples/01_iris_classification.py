"""
Minimal end-to-end example: train a Soft Decision Tree on Iris.

Run with:
    python examples/01_iris_classification.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from neural_trees import SoftDecisionTree

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = SoftDecisionTree(depth=4, max_epochs=40, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

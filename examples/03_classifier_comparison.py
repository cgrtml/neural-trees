"""
Is the difference significant?
==============================

Compare a Soft Decision Tree against CART with Alpaydin's combined 5x2cv F test.

The test asks whether the accuracy gap between the two models is larger than
what the variance of resampling alone would explain.

Run with:
    python examples/03_classifier_comparison.py
"""
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from neural_trees import SoftDecisionTree, combined_5x2cv_f_test

X, y = load_breast_cancer(return_X_y=True)

soft_tree = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SoftDecisionTree(depth=4, max_epochs=40, random_state=42)),
])
cart = DecisionTreeClassifier(max_depth=4, random_state=42)

result = combined_5x2cv_f_test(soft_tree, cart, X, y)

print(f"F statistic: {result.statistic:.4f}")
print(f"p-value:     {result.p_value:.4f}")
if result.reject_null:
    print("Interpretation: the two classifiers differ significantly (p < 0.05).")
else:
    print("Interpretation: no significant difference; the gap is within resampling noise.")

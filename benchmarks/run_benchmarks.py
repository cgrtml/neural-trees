"""
Reproduce the benchmark table in the README.

Every number in the table comes from this script, so it can be re-run and
checked. Accuracy is 5-fold stratified cross-validation on a
StandardScaler + model pipeline, with a fixed seed.

Run with:
    python benchmarks/run_benchmarks.py
"""
import argparse
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from neural_trees import (
    GALNetwork,
    HierarchicalMixtureOfExperts,
    MultivariateDecisionTree,
    SoftDecisionTree,
)

DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
}

MODELS = {
    "Soft Decision Tree (depth=4)": lambda seed: SoftDecisionTree(
        depth=4, max_epochs=40, random_state=seed
    ),
    "Multivariate Tree (depth=3)": lambda seed: MultivariateDecisionTree(
        max_depth=3, random_state=seed
    ),
    "Hierarchical MoE (depth=2)": lambda seed: HierarchicalMixtureOfExperts(
        depth=2, max_epochs=60, random_state=seed
    ),
    "GAL Network": lambda seed: GALNetwork(max_epochs=100, random_state=seed),
    "CART (sklearn)": lambda seed: DecisionTreeClassifier(random_state=seed),
    "Random Forest": lambda seed: RandomForestClassifier(random_state=seed),
    "SVM (RBF)": lambda seed: SVC(kernel="rbf", random_state=seed),
}


def run(seeds):
    results = {name: {} for name in MODELS}
    for dataset_name, load in DATASETS.items():
        X, y = load(return_X_y=True)
        for model_name, make_model in MODELS.items():
            means = []
            for seed in seeds:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", make_model(seed)),
                ])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                means.append(cross_val_score(pipeline, X, y, cv=cv).mean())
            results[model_name][dataset_name] = (float(np.mean(means)), float(np.std(means)))
            print(
                f"{dataset_name:<14} {model_name:<30} "
                f"{np.mean(means):.3f} +/- {np.std(means):.3f}",
                flush=True,
            )
    return results


def print_markdown_table(results):
    header = "| Model | " + " | ".join(DATASETS) + " |"
    print("\n" + header)
    print("|-------|" + "|".join([":----:"] * len(DATASETS)) + "|")
    for model_name, scores in results.items():
        cells = " | ".join(f"{scores[d][0]:.3f}" for d in DATASETS)
        print(f"| {model_name} | {cells} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds to average")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    print_markdown_table(run(range(args.seeds)))

"""
Reproduce the benchmark table in the README.

Every number in the table comes from this script, so it can be re-run and
checked. Accuracy is 5-fold stratified cross-validation on a
StandardScaler + model pipeline, with a fixed seed.

Run with:
    python benchmarks/run_benchmarks.py

Check the README against a fresh run with:
    python benchmarks/run_benchmarks.py --check

CI runs that check (see .github/workflows/benchmark-table.yml). Cells are
compared with an explicit tolerance, `TOLERANCE` below; a cell that moves by
more than that fails loudly rather than being rounded away.
"""
import argparse
import json
import os
import re
import sys
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


# The README shows three decimals, so "matches" means the fresh mean rounds to
# the README cell: half a unit in the last digit. The drift that motivated the
# check was one unit (0.951 -> 0.952), so anything looser would have missed it.
# The reference environment is the CI runner; if another machine's torch build
# lands a cell elsewhere, that is reported rather than absorbed.
TOLERANCE = 0.0005
README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")


def read_readme_table(path=README):
    """The benchmark table as {model: {dataset: value}}, bold markers stripped."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.startswith("|")]
    header = next(ln for ln in lines if ln.startswith("| Model |"))
    datasets = [c.strip() for c in header.strip("|").split("|")][1:]
    table = {}
    for ln in lines[lines.index(header) + 2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(datasets) + 1:
            break
        model = re.sub(r"\*\*", "", cells[0])
        try:
            table[model] = {d: float(v) for d, v in zip(datasets, cells[1:])}
        except ValueError:
            break
    return table


def check_against_readme(results, tolerance=TOLERANCE):
    """Return the list of cells that disagree with the README; empty means it matches."""
    table = read_readme_table()
    problems = []
    for model, scores in results.items():
        if model not in table:
            problems.append(f"{model}: missing from README")
            continue
        for dataset, (mean, _) in scores.items():
            expected = table[model].get(dataset)
            if expected is None:
                problems.append(f"{model} / {dataset}: missing from README")
            elif abs(mean - expected) > tolerance:
                problems.append(
                    f"{model} / {dataset}: README says {expected:.3f}, run gives {mean:.3f} "
                    f"(difference {abs(mean - expected):.3f} > {tolerance})"
                )
    return problems


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
    parser.add_argument(
        "--check", action="store_true",
        help="compare the run with the README table and exit 1 if any cell differs "
             f"by more than {TOLERANCE}",
    )
    parser.add_argument(
        "--results", metavar="JSON",
        help="read results from this file if it exists, otherwise run and write them there",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.results and os.path.exists(args.results):
        with open(args.results, encoding="utf-8") as f:
            results = {m: {d: tuple(v) for d, v in s.items()} for m, s in json.load(f).items()}
    else:
        results = run(range(args.seeds))
        if args.results:
            with open(args.results, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)
    print_markdown_table(results)
    if args.check:
        problems = check_against_readme(results)
        if problems:
            print("\nREADME benchmark table does not match this run:", file=sys.stderr)
            for line in problems:
                print("  " + line, file=sys.stderr)
            sys.exit(1)
        print(f"\nREADME table matches (tolerance {TOLERANCE}).")

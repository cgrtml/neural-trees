# neural-trees

Soft decision trees, mixture of experts, and statistical model comparison tests for Python. A scikit-learn compatible library implementing classic machine learning algorithms from research papers, with a PyTorch backend.

<p align="center">
  <img src="assets/demo.gif" width="600">
</p>

<p align="center">
Decision boundary learning with Soft Decision Trees on a toy dataset.
</p>

[![PyPI](https://img.shields.io/pypi/v/neural-trees)](https://pypi.org/project/neural-trees/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/cgrtml/neural-trees/actions/workflows/tests.yml/badge.svg)](https://github.com/cgrtml/neural-trees/actions)
[![GitHub Stars](https://img.shields.io/github/stars/cgrtml/neural-trees?style=social)](https://github.com/cgrtml/neural-trees/stargazers)

## Features

- scikit-learn compatible API (`fit`, `predict`, `score`, works in `Pipeline`)
- PyTorch backend with GPU support
- Soft Decision Trees, Hierarchical Mixture of Experts, Omnivariate Trees, GAL
- Combined 5x2cv F test, McNemar's test, paired t-test for classifier comparison
- Tested on standard benchmarks (Iris, Wine, Breast Cancer)

## Installation

```bash
pip install neural-trees
```

### Install from source

```bash
git clone https://github.com/cgrtml/neural-trees.git
cd neural-trees
pip install -e .
```

## Quick Start

Train a Soft Decision Tree on the Iris dataset:

```python
from neural_trees import SoftDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = SoftDecisionTree(depth=4, max_epochs=40)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # ~0.97
```

Use it inside a scikit-learn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SoftDecisionTree(depth=4, max_epochs=40)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

## Benchmark

5-fold cross-validation accuracy with `StandardScaler` preprocessing:

| Model | Iris | Wine | Breast Cancer |
|-------|:----:|:----:|:-------------:|
| **Soft Decision Tree** (depth=4) | 0.96 | 0.95 | 0.95 |
| CART (sklearn) | 0.953 | 0.865 | 0.917 |
| Random Forest | 0.967 | 0.978 | 0.956 |
| SVM (RBF) | 0.967 | 0.983 | 0.974 |

Soft Decision Trees close most of the gap between CART and ensemble or kernel methods, while staying differentiable and interpretable.

## Algorithms

Implementations based on published research, including work by Ethem Alpaydın.

| Algorithm | Reference |
|-----------|-----------|
| **Soft Decision Trees** | İrsoy, Yıldız, Alpaydın (ICPR 2012) |
| **Omnivariate Decision Trees** | Yıldız & Alpaydın (IEEE TNN 2001) |
| **Hierarchical Mixture of Experts with Dropout** | İrsoy & Alpaydın (Neurocomputing 2021) |
| **GAL: Grow and Learn Networks** | Alpaydın (IJPRAI 1994) |
| **Combined 5x2cv F Test** | Alpaydın (Neural Computation 1999) |
| **McNemar's Test, Paired t-test** | Standard references |
| **Naive Bayes, Weighted KNN** | Textbook chapters 3 to 8 |

## Use Cases

**Research.** Reproduce or extend results from the original papers with a clean, tested codebase.

**Statistical model comparison.** Compare classifiers with proper hypothesis tests instead of ad hoc accuracy diffs:

```python
from neural_trees import combined_5x2cv_f_test
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

result = combined_5x2cv_f_test(
    DecisionTreeClassifier(),
    SVC(kernel="rbf"),
    X, y,
)

print(result)
```

**Education.** A working reference for soft splits and mixtures of experts beyond textbook diagrams.

## Why Soft Decision Trees

Standard decision trees use hard splits, which makes them non-differentiable and unstable to small input changes. Soft Decision Trees replace each split with a sigmoid gate, which means:

- The tree is fully differentiable and trains with gradient descent
- Predictions are smooth, not piecewise constant
- Performance often lands between CART and ensemble methods
- The tree stays interpretable, you can still read off split decisions

## Notebooks

- [`01_soft_decision_trees.ipynb`](notebooks/01_soft_decision_trees.ipynb): training, decision boundary visualization, comparison with CART
- [`02_classifier_comparison_tests.ipynb`](notebooks/02_classifier_comparison_tests.ipynb): when to use which statistical test

## Citation

If you use this library in academic work, please cite the original papers:

```bibtex
@inproceedings{irsoy2012soft,
  title     = {Soft Decision Trees},
  author    = {\.{I}rsoy, O{\u{g}}uzhan and Y{\i}ld{\i}z, Olcay Taner and Alpayd{\i}n, Ethem},
  booktitle = {ICPR},
  year      = {2012}
}

@article{alpaydin1999combined,
  title   = {Combined 5x2cv {F} Test for Comparing Supervised Classification Learning Algorithms},
  author  = {Alpayd{\i}n, Ethem},
  journal = {Neural Computation},
  volume  = {11},
  number  = {8},
  pages   = {1885--1892},
  year    = {1999}
}
```

## Contributing

Contributions are welcome. Good starting points:

- Add an algorithm from Alpaydın's papers
- Improve test coverage
- Add a notebook or example

### How to contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-username>/neural-trees.git
   cd neural-trees
   git checkout -b my-feature
   pip install -e ".[dev]"
   ```
3. Make your changes and run the tests:
   ```bash
   pytest
   ```
4. Commit, push, and open a **Pull Request** against `main`.

For larger changes, open an issue first to discuss the approach.

If this project is useful to you, a star helps others find it.

## License

MIT. See [LICENSE](LICENSE).

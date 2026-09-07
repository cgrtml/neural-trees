# neural-trees

Soft decision trees, mixture of experts, and statistical model comparison tests for Python. A scikit-learn compatible library implementing classic machine learning algorithms from research papers, with a PyTorch backend.

<p align="center">
  <img src="assets/demo.gif" width="600">
</p>

<p align="center">
Decision boundary learning with Soft Decision Trees on a toy dataset.
</p>

[![PyPI](https://img.shields.io/pypi/v/neural-trees)](https://pypi.org/project/neural-trees/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/neural-trees?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/neural-trees)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/cgrtml/neural-trees/actions/workflows/tests.yml/badge.svg)](https://github.com/cgrtml/neural-trees/actions)
[![GitHub Stars](https://img.shields.io/github/stars/cgrtml/neural-trees?style=social)](https://github.com/cgrtml/neural-trees/stargazers)

## Features

- scikit-learn compatible API (`fit`, `predict`, `score`, works in `Pipeline`)
- PyTorch backend with GPU support
- Soft Decision Trees, Hierarchical Mixture of Experts, Multivariate and Omnivariate Trees, GAL
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

5-fold stratified cross-validation accuracy with `StandardScaler` preprocessing,
averaged over 5 seeds. Every number comes from
[`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py), so the table can
be re-run and checked:

```bash
python benchmarks/run_benchmarks.py --seeds 5
```

| Model | Iris | Wine | Breast Cancer |
|-------|:----:|:----:|:-------------:|
| **Soft Decision Tree** (depth=4) | 0.900 | 0.979 | 0.976 |
| **Multivariate Tree** (depth=3) | 0.973 | 0.989 | 0.952 |
| CART (sklearn) | 0.943 | 0.917 | 0.920 |
| Random Forest | 0.945 | 0.980 | 0.960 |
| SVM (RBF) | 0.959 | 0.984 | 0.978 |

On Wine and Breast Cancer the soft tree closes most of the gap between CART and
kernel or ensemble methods while staying differentiable. On Iris it does not:
150 samples over 3 classes is too little data for a depth-4 tree with 15 gates
trained for 40 epochs, and a single oblique split does better. That is the
honest shape of the trade-off, and it is why the comparison scripts in
[`examples/`](examples) use a hypothesis test rather than a single accuracy
number.

## Algorithms

Implementations based on published research, including work by Ethem Alpaydın.

| Algorithm | Reference |
|-----------|-----------|
| **Soft Decision Trees** | İrsoy, Yıldız, Alpaydın (ICPR 2012) |
| **Multivariate Decision Trees** | Alpaydın & Çetin (1995), Yıldız & Alpaydın (IEEE TNN 2001) |
| **Omnivariate Decision Trees** | Yıldız & Alpaydın (IEEE TNN 2001) |
| **Hierarchical Mixture of Experts with subtree dropout** | İrsoy & Alpaydın (Neurocomputing 2021) |
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

## Examples

Runnable scripts in [`examples/`](examples):

| Script | What it shows |
|--------|---------------|
| [`01_iris_classification.py`](examples/01_iris_classification.py) | Minimal train/test loop on Iris |
| [`02_pipeline_with_scaler.py`](examples/02_pipeline_with_scaler.py) | `StandardScaler` + `SoftDecisionTree` in a `Pipeline`, 5-fold CV |
| [`03_classifier_comparison.py`](examples/03_classifier_comparison.py) | Combined 5x2cv F test against CART |
| [`04_decision_boundary.py`](examples/04_decision_boundary.py) | Decision boundary plot on `make_moons` |

```bash
python examples/01_iris_classification.py
```

## Notebooks

[![Open 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgrtml/neural-trees/blob/main/notebooks/01_soft_decision_trees.ipynb)
[![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgrtml/neural-trees/blob/main/notebooks/02_classifier_comparison_tests.ipynb)
[![Open 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cgrtml/neural-trees/blob/main/notebooks/03_multivariate_decision_trees.ipynb)

- [`01_soft_decision_trees.ipynb`](notebooks/01_soft_decision_trees.ipynb): training, decision boundary visualization, comparison with CART
- [`02_classifier_comparison_tests.ipynb`](notebooks/02_classifier_comparison_tests.ipynb): when to use which statistical test
- [`03_multivariate_decision_trees.ipynb`](notebooks/03_multivariate_decision_trees.ipynb): oblique splits against CART and soft trees

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

To cite this implementation:

```bibtex
@software{temel_neural_trees,
  author = {Temel, Cagri},
  title  = {neural-trees: scikit-learn compatible Soft Decision Trees and Mixture of Experts},
  year   = {2026},
  url    = {https://github.com/cgrtml/neural-trees}
}
```

## Limitations

neural-trees is not the right tool for every problem:

- **Very high-dimensional data.** Every internal node holds a dense weight
  vector, so parameter count grows as `2^depth x n_features`. Beyond a few
  thousand features, reduce dimensionality first or use a linear model.
- **Streaming or online learning.** Training is batch only; there is no
  `partial_fit`. Refit from scratch when new data arrives.
- **Sub-millisecond inference.** The PyTorch backend adds per-call overhead.
  For extreme latency budgets, export the learned gates and evaluate them in
  plain numpy.
- **Very large sample counts.** Training is full-batch gradient descent over
  epochs, not an optimized tree-growing routine like CART. Millions of rows
  will be slow on CPU.
- **Categorical features.** There is no built-in encoding; sigmoid gates
  expect continuous, scaled inputs. Encode and scale in a `Pipeline`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md). Version 0.2.0 fixes two models that did not
work in 0.1.x, so upgrade if you are on an earlier release.

## Contributing

Contributions are welcome. New to open source? See
[CONTRIBUTING.md](CONTRIBUTING.md) for a beginner-friendly walkthrough.

Good starting points:

- Browse issues tagged [`good first issue`](https://github.com/cgrtml/neural-trees/labels/good%20first%20issue)
- Add an algorithm from Alpaydın's papers
- Improve test coverage
- Add a notebook or example

For larger changes, open an issue first to discuss the approach. If this
project is useful to you, a star helps others find it.

## Contributors

Thanks to everyone who has improved this library.

- [@snoopuppy582](https://github.com/snoopuppy582) — symmetric McNemar disagreement test, development requirements
- [@aribaskagan](https://github.com/aribaskagan) — fixed the coverage target in CI, which had been measuring a module that no longer exists

Full list: [contributors graph](https://github.com/cgrtml/neural-trees/graphs/contributors).

### WSU Data and Analytics Breakout (May 15, 2026)

Students from Washington State University contributed via the live
GitHub Sprint segment of the workshop. Their merged pull requests
appear below as the event proceeds:

<!-- WSU-CONTRIBUTORS-START -->
*To be populated during and after the workshop.*
<!-- WSU-CONTRIBUTORS-END -->

## License

MIT. See [LICENSE](LICENSE).

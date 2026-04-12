# neural-trees

> PyTorch + sklearn implementations of the tree and mixture of experts algorithms
> from Alpaydın's research papers the ones that never got a proper open-source home.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/cgrtml/neural-trees/actions/workflows/tests.yml/badge.svg)](https://github.com/cgrtml/neural-trees/actions)

---

## Why?

I was reading through Alpaydın's *Introduction to Machine Learning* and his papers
and kept hitting the same wall: interesting algorithms, no usable Python code anywhere.
The Soft Decision Tree paper (ICPR 2012) alone has hundreds of citations but the implementations
floating around are incomplete, undocumented, or years out of date.

So I wrote them myself clean, tested, and fully compatible with the sklearn API.

Covered so far:

| Algorithm | Paper |
|-----------|-------|
| **Soft Decision Trees** | İrsoy, Yıldız, Alpaydın (ICPR 2012) |
| **Omnivariate Decision Trees** | Yıldız & Alpaydın (IEEE TNN 2001) |
| **Hierarchical Mixture of Experts + Dropout** | İrsoy & Alpaydın (Neurocomputing 2021) |
| **GAL: Grow and Learn Networks** | Alpaydın (IJPRAI 1994) |
| **Combined 5×2cv F Test** | Alpaydın (Neural Computation 1999) |
| **McNemar's Test + Paired t-test** | — |
| **Naive Bayes, Weighted KNN** | Textbook Ch. 3–8 |

---

## Installation

```bash
pip install neural-trees
```

---

## Usage

### Soft Decision Trees

Unlike hard decision trees, every sample reaches every leaf with some probability —
making the tree fully differentiable and trainable end-to-end with backpropagation.

```python
from neural_trees import SoftDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

sdt = SoftDecisionTree(depth=4, max_epochs=40)
sdt.fit(X_train, y_train)
print(sdt.score(X_test, y_test))

# what each leaf learned
leaf_distributions = sdt.get_leaf_distributions()  # (n_leaves, n_classes)
```

### Comparing two classifiers properly

The standard paired t-test is statistically unreliable for classifier comparison —
the training folds overlap, making the differences correlated. Alpaydın's F test fixes this.

```python
from neural_trees.statistical_tests import combined_5x2cv_f_test
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
result = combined_5x2cv_f_test(DecisionTreeClassifier(), SVC(kernel="rbf"), X, y)
print(result)
```

```
StatisticalTestResult(
  test       = Alpaydın's Combined 5×2cv F Test
  statistic  = 12.4731
  p-value    = 0.0083
  decision   = ✓ REJECT H0
)
```

### Hierarchical Mixture of Experts

```python
from neural_trees import HierarchicalMixtureOfExperts
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
moe = HierarchicalMixtureOfExperts(depth=2, branching_factor=4, dropout_rate=0.3)
moe.fit(X, y)
print(moe.score(X, y))
```

### sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("sdt", SoftDecisionTree(depth=4, max_epochs=30)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

---

## Notebooks

- [`01_soft_decision_trees.ipynb`](notebooks/01_soft_decision_trees.ipynb) — training, boundary visualization, comparison with CART
- [`02_classifier_comparison_tests.ipynb`](notebooks/02_classifier_comparison_tests.ipynb) — when to use which test

---

## Citation

If you use this in academic work, please cite the original papers:

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

---

## License

MIT — see [LICENSE](LICENSE).

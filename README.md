# neural-trees

> PyTorch + sklearn implementations of the tree and mixture-of-experts algorithms
> from Alpaydın's research papers — the ones that never got a proper open-source home.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/cgrtml/neural-trees/actions/workflows/tests.yml/badge.svg)](https://github.com/cgrtml/neural-trees/actions)
[![sklearn compatible](https://img.shields.io/badge/sklearn-compatible-orange)](https://scikit-learn.org)

---

## Why?

I was reading through Alpaydın's *Introduction to Machine Learning* and his papers
and kept hitting the same wall: interesting algorithms, no usable Python code anywhere.
The Soft Decision Tree paper (ICPR 2012) alone has hundreds of citations but the implementations
floating around are incomplete, undocumented, or years out of date.

So I wrote them myself — clean, tested, and fully compatible with the sklearn API.

Covered so far:

| Algorithm | Paper | Status |
|-----------|-------|--------|
| **Soft Decision Trees** | İrsoy, Yıldız, Alpaydın (ICPR 2012) | ✅ PyTorch + sklearn API |
| **Omnivariate Decision Trees** | Yıldız & Alpaydın (IEEE TNN 2001) | ✅ |
| **Hierarchical Mixture of Experts + Dropout** | İrsoy & Alpaydın (Neurocomputing 2021) | ✅ PyTorch |
| **GAL: Grow and Learn Networks** | Alpaydın (IJPRAI 1994) | ✅ |
| **Combined 5×2cv F Test** | Alpaydın (Neural Computation 1999) | ✅ Gold-standard classifier comparison |
| **McNemar's Test** | — | ✅ |
| **Naive Bayes (Gaussian/Bernoulli/Multinomial)** | Textbook Ch. 3 | ✅ |
| **Distance-Weighted KNN + CNN** | Alpaydın (AIR 1997) | ✅ |

---

## Installation

```bash
pip install neural-trees
```

Or install from source:

```bash
git clone https://github.com/cgrtml/neural-trees.git
cd neural-trees
pip install -e ".[dev]"
```

---

## Quick Start

### Soft Decision Trees

The flagship algorithm. Unlike hard decision trees, every sample reaches every leaf with some probability — making the tree **fully differentiable** and trainable end-to-end with backpropagation.

```python
from neural_trees import SoftDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

sdt = SoftDecisionTree(depth=4, max_epochs=40, penalty_coef=1e-3)
sdt.fit(X_train, y_train)

print(f"Accuracy: {sdt.score(X_test, y_test):.4f}")

# Inspect what each leaf learned
leaf_distributions = sdt.get_leaf_distributions()  # shape: (n_leaves, n_classes)

# Inspect the split direction at each internal node
split_weights = sdt.get_split_weights()             # list of weight vectors
```

**Key idea** (Irsoy, Yıldız, Alpaydın, 2012):

At each internal node *i*:

$$p_i(\mathbf{x}) = \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$$

The probability of reaching leaf $\ell$ is the product of gate values along the path. Final prediction:

$$P(y \mid \mathbf{x}) = \sum_\ell \mu_\ell(\mathbf{x}) \cdot Q_\ell(y)$$

---

### Comparing Two Classifiers — The Gold Standard Test

Alpaydın's **Combined 5×2cv F Test** (Neural Computation, 1999) is the statistically correct way to compare two classifiers. It overcomes the inflated Type I error of the paired t-test.

```python
from neural_trees.statistical_tests import combined_5x2cv_f_test
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

result = combined_5x2cv_f_test(
    clf_A=DecisionTreeClassifier(),
    clf_B=SVC(kernel="rbf"),
    X=X, y=y,
    alpha=0.05
)
print(result)
```

```
StatisticalTestResult(
  test       = Alpaydın's Combined 5×2cv F Test
  statistic  = 12.4731
  p-value    = 0.0083
  alpha      = 0.05
  decision   = ✓ REJECT H0
  note       = Classifiers significantly differ
)
```

**Why not just use a t-test?**
The paired t-test reuses training data across folds — the differences are correlated, inflating the false positive rate. Alpaydın's F test accounts for this by estimating variance within each 2-fold split, giving a much better calibrated test.

---

### Hierarchical Mixture of Experts with Dropout

```python
from neural_trees import HierarchicalMixtureOfExperts
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

moe = HierarchicalMixtureOfExperts(
    depth=2,
    branching_factor=4,   # 4^2 = 16 expert leaves
    dropout_rate=0.3,     # Dropout on gating networks (Irsoy & Alpaydın, 2021)
    max_epochs=50,
    verbose=True,
)
moe.fit(X, y)
print(f"Accuracy: {moe.score(X, y):.4f}")
```

---

### GAL — Grow and Learn Networks

No need to specify architecture. The network grows when it can't learn and prunes itself when neurons become redundant.

```python
from neural_trees.classical import GALNetwork
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

gal = GALNetwork(
    initial_hidden=2,
    max_hidden=40,
    grow_threshold=0.15,
    prune_threshold=1e-4,
    max_epochs=100,
    verbose=True,
)
gal.fit(X, y)
print(f"Final hidden units: {gal.n_hidden_final_}")
print(f"Accuracy: {gal.score(X, y):.4f}")
```

---

### Omnivariate Decision Trees

At each node, automatically selects the best split type (univariate, linear LDA, or nonlinear MLP) using cross-validation.

```python
from neural_trees import OmnivariateDecisionTree
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

odt = OmnivariateDecisionTree(max_depth=4, cv_folds=3)
odt.fit(X, y)

# See how many nodes used each split type
print(odt.get_split_type_distribution())
# {'univariate': 3, 'linear': 4, 'nonlinear': 1}
```

---

## All sklearn-compatible

Every model follows the `fit` / `predict` / `predict_proba` / `score` interface:

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

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"sdt__depth": [3, 4, 5], "sdt__penalty_coef": [1e-4, 1e-3, 1e-2]}
gs = GridSearchCV(pipe, param_grid, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_soft_decision_trees.ipynb`](notebooks/01_soft_decision_trees.ipynb) | Training, visualization, comparison with CART |
| [`02_classifier_comparison_tests.ipynb`](notebooks/02_classifier_comparison_tests.ipynb) | When to use which statistical test |
| [`03_hierarchical_moe.ipynb`](notebooks/03_hierarchical_moe.ipynb) | HMoE training and expert specialization |
| [`04_gal_network.ipynb`](notebooks/04_gal_network.ipynb) | Dynamic architecture growth/pruning |
| [`05_omnivariate_trees.ipynb`](notebooks/05_omnivariate_trees.ipynb) | Node-level split type analysis |

---

## About the Author

**Prof. Dr. Ethem Alpaydın** is one of the world's leading machine learning researchers.

- Professor Emeritus at Boğaziçi University (Istanbul), now at Özyeğin University
- Author of *Introduction to Machine Learning* (MIT Press, 4 editions, 2004–2020) — used in hundreds of universities globally
- Author of *Machine Learning: The New AI* (MIT Press, 2016)
- PhD from EPFL (1990); research stays at UC Berkeley, MIT, and IDIAP
- **34,000+ citations** on Google Scholar
- IEEE Senior Member; Pattern Recognition journal editorial board

His 1999 paper on the Combined 5×2cv F Test is the standard reference for classifier comparison. His Soft Decision Trees paper (2012) remains one of the most elegant proposals for differentiable tree models — predating the modern neural tree literature.

---

## Citation

If you use this library in academic work, please cite the original papers:

```bibtex
@book{alpaydin2020introduction,
  title     = {Introduction to Machine Learning},
  author    = {Alpayd{\i}n, Ethem},
  year      = {2020},
  edition   = {4th},
  publisher = {MIT Press}
}

@article{irsoy2021dropout,
  title   = {Dropout Regularization in Hierarchical Mixture of Experts},
  author  = {\.{I}rsoy, O{\u{g}}uzhan and Alpayd{\i}n, Ethem},
  journal = {Neurocomputing},
  volume  = {419},
  pages   = {148--156},
  year    = {2021}
}

@inproceedings{irsoy2012soft,
  title     = {Soft Decision Trees},
  author    = {\.{I}rsoy, O{\u{g}}uzhan and Y{\i}ld{\i}z, Olcay Taner and Alpayd{\i}n, Ethem},
  booktitle = {Proceedings of the 21st International Conference on Pattern Recognition (ICPR)},
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

## Roadmap

Things I'm planning to add:

- [ ] Multiple Kernel Learning (Gönen & Alpaydın, JMLR 2011)
- [ ] Localized Multiple Kernel Learning (ICML 2008)
- [ ] Convolutional Soft Decision Trees (ICANN 2018)
- [ ] Decision boundary visualization utilities
- [ ] Benchmark comparison on UCI datasets

If you find a bug or want to implement one of these, open an issue.

---

## License

MIT — see [LICENSE](LICENSE).

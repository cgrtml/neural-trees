# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-07

Every model in the library now works and is tested. Two of them did not learn
at all before this release.

### Fixed

- **`GALNetwork` was not learning.** `fit` took exactly one full-batch SGD step
  per epoch, so `max_epochs=100` meant 100 gradient steps in total, and the
  growth and pruning criteria were evaluated on a network that had barely moved
  from its initialization. Training now iterates mini-batches (new `batch_size`
  and `momentum` parameters), and the error driving growth is measured on the
  full training set at the end of each epoch.

  | | before | after |
  |---|:---:|:---:|
  | three separable blobs, train | 0.333 | 1.000 |
  | Iris, 5-fold | 0.333 | 0.893 |
  | Wine, 5-fold | 0.518 | 0.983 |
  | Breast Cancer, 5-fold | 0.627 | 0.974 |

- **Condensed nearest neighbour was inconsistent.** `WeightedKNN._condense`
  made a single sweep of the training set, so samples seen early were judged
  against a store that later grew, and the result did not classify the training
  set correctly. It now sweeps until a full pass adds nothing, which is Hart's
  algorithm, and honours the `metric` parameter instead of always using the
  Euclidean norm. 5-fold accuracy with `condense=True`: Iris 0.747 to 0.933,
  Wine 0.792 to 0.933, Breast Cancer 0.935 to 0.951.
- **Exact matches were diluted in `WeightedKNN`.** A zero nearest distance fell
  back to uniform weights, letting `k - 1` unrelated neighbours outvote a
  training sample identical to the query. Zero-distance neighbours now carry
  the whole vote.
- `metric`, `k`, `likelihood` and `dropout_type` are validated in `fit` rather
  than failing later inside a distance, likelihood or forward computation.

### Added

- **`OmnivariateDecisionTree.predict_proba`.** It was the only estimator without
  it, which ruled out ROC AUC, calibration and soft voting. Leaves keep the full
  class distribution now.
- Test files for `WeightedKNN` and `NaiveBayesClassifier`. Condensing, the
  manhattan metric, and the bernoulli and multinomial likelihoods previously had
  no coverage at all.
- `tests/test_app_contract.py`, which checks the Streamlit playground against
  the library API so it cannot drift out of step unnoticed.
- Notebooks 01 and 02 are executed with their outputs committed.

### Changed

- **`HierarchicalMixtureOfExperts` now implements the subtree dropout it cites.**
  The previous mechanism was a plain `nn.Dropout` on the gating networks' hidden
  activations, which perturbs a gate but never removes a branch. With
  probability `dropout_rate` a gating node now withholds all mass from one child
  and renormalizes the rest, switching off that subtree for the sample. Measured
  over 5 seeds on a noisy 20-feature problem, test accuracy goes from 0.588 to
  0.634 and the train/test gap from 0.412 to 0.364; on Breast Cancer every
  setting sits within noise of the others. `dropout_type="activation"` keeps the
  old behaviour.
- The README no longer claims the modules are implementations of the cited
  papers. They start from those algorithms and depart from them where this
  library makes its own choices; deliberate deviations are noted in the module
  docstrings.
- Test suite: 101 to 135, coverage 97%.

## [0.2.0] - 2026-09-06

Correctness release. Two of the four models shipped in 0.1.x did not work, and
neither had tests. If you are on an earlier version, upgrade.

### Fixed

- **`HierarchicalMixtureOfExperts` could not train at all.** `_compute_leaf_weights`
  wrote each child's weight in place into one preallocated tensor. Autograd had
  saved that storage for the multiplication backward, so every `loss.backward()`
  raised `RuntimeError: one of the variables needed for gradient computation has
  been modified by an inplace operation`. Node weights are now accumulated in a
  list.
- **`OmnivariateDecisionTree` ignored its own splits.** `predict_one` called the
  node classifier, discarded the result, and unconditionally descended into the
  right child, so every sample landed in the same leaf. Nodes also split on
  whether the node classifier was correct rather than on its decision. 5-fold
  accuracy at depth 3 goes from 0.000 to 0.947 on Iris, 0.072 to 0.887 on Wine,
  and 0.628 to 0.967 on Breast Cancer. Fold selection is now bounded by the size
  of the rarest group rather than the number of distinct classes, which used to
  raise on small or skewed nodes.
- **`GALNetwork` lost its learned output bias on every growth step.** The second
  layer's bias was reset to a fresh init whenever a hidden unit was added. Its
  SGD optimizer was also rebuilt on every epoch instead of only when the
  architecture changed.
- The `import torch` failure now raises an `ImportError` naming
  `pip install torch` instead of a bare `ModuleNotFoundError`.
- **scikit-learn estimator contract.** Every classifier now inherits
  `ClassifierMixin` before `BaseEstimator`, rejects a continuous target in
  `fit`, raises `NotFittedError` rather than `AttributeError` when `predict` is
  called before `fit`, and rejects an `X` at predict time whose feature count
  does not match `fit`. That last one used to return meaningless predictions
  instead of an error. `SoftDecisionTree`, `MultivariateDecisionTree`,
  `OmnivariateDecisionTree`, `WeightedKNN` and `NaiveBayesClassifier` now pass
  `sklearn.utils.estimator_checks.check_estimator` completely (55/55);
  `HierarchicalMixtureOfExperts` is at 54/55 and `GALNetwork` at 52/55, both
  tracked as open issues.
- **`NaiveBayesClassifier.predict_log_proba` returned unnormalized values.** It
  gave the joint log-likelihood, so exponentiated rows summed to arbitrary
  numbers rather than 1 and it did not match `predict_proba`. It is now
  normalized with `logsumexp`; the previous quantity is available as
  `_joint_log_likelihood`.
- **Predicting on a reversed or reordered view crashed.** A negatively strided
  array reached `torch.FloatTensor` and raised `ValueError: At least one stride
  in the given numpy array is negative`. Predict-time input is now made
  contiguous.

### Added

- **`MultivariateDecisionTree`**: nodes split on a linear discriminant,
  `w . x + b > 0`, instead of on a single feature. Classes are reduced to a
  two-group problem by centroid clustering when a node holds more than two, and
  every hyperplane is readable through `get_split_weights()`.
- `SoftDecisionTree.feature_importances_`, gate weight magnitudes weighted by the
  probability mass reaching each node.
- `SoftDecisionTree` early stopping: `early_stopping`, `validation_fraction`,
  `n_iter_no_change`, with best-epoch parameter restore and `n_iter_`.
- `SoftDecisionTree(learn_temperature=True)`, an opt-in per-node inverse
  temperature (Frosst & Hinton, 2017). Off by default: over 5 seeds of 5-fold CV
  at depth 4 it moved Iris from 0.900 to 0.928 and cost about 0.6 points on Wine
  and Breast Cancer.
- `random_state` on `HierarchicalMixtureOfExperts` and `GALNetwork`.
- `GALNetwork`, `WeightedKNN` and `NaiveBayesClassifier` are exported at package
  level.
- Four runnable scripts in `examples/`, `benchmarks/run_benchmarks.py`, and
  `notebooks/03_multivariate_decision_trees.ipynb`.

### Changed

- **`SoftDecisionTree` trains about 11x faster at depth 8.** All internal gates
  live in a single `Linear` layer, so a forward pass is one matmul instead of
  `2^depth - 1` small ones. Breast Cancer, 30 epochs: 19.2s to 1.7s at depth 8.
- Path probabilities are accumulated in log space. At depth 12 the smallest leaf
  arrival probability was flushing to exactly 0 in float32; it is now 1.6e-08.
- The entropy penalty follows Frosst & Hinton: a path-probability weighted
  average gate activation per node and a `2^-level` coefficient, computed in the
  same forward pass instead of a second one. The previous version used an
  unweighted batch mean and a flat coefficient.
- The README benchmark table is now generated by `benchmarks/run_benchmarks.py`.
  The previous table was not reproducible: it claimed 0.96 / 0.95 / 0.95 for the
  soft tree on Iris / Wine / Breast Cancer, while the measured values are
  0.900 / 0.979 / 0.976.
- Test suite: 21 tests to 101.

### Contributors

Thanks to @snoopuppy582 and @aribaskagan, whose pull requests are part of this
release.

## [0.1.1] - 2026-04

First public release: `SoftDecisionTree`, `OmnivariateDecisionTree`,
`HierarchicalMixtureOfExperts`, `GALNetwork`, and the classifier comparison
tests (combined 5x2cv F test, McNemar, paired t-test).

# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.6.0] - 2026-09-10

Ten open items closed. Weighting, device selection, warm starts and hard
exports are now consistent across the library, and two models implement the
mechanism they cite rather than an approximation of it.

### Added

- **`sample_weight` and `class_weight` for `GALNetwork` and
  `HierarchicalMixtureOfExperts`**, matching `SoftDecisionTree`. On a
  600-sample problem where one class holds 6%, `class_weight="balanced"` moves
  recall on that class from 0.476 to 0.976 for HMoE and from **0.000** to 0.714
  for GAL, which had been predicting the rare class not once while reporting
  0.930 accuracy. GAL's growth and pruning criteria see the weights too:
  weighting the loss while judging the architecture on unweighted accuracy let
  `sample_weight` change what the network fits and not what it builds.
- **`device="auto"`** on every torch-backed estimator, resolving to CUDA, then
  Apple silicon MPS, then CPU. Resolved once in `fit` and recorded as
  `device_`, so prediction runs where training did.
- **`warm_start`** on all three torch-backed estimators. Deliberately not
  `partial_fit`: that contract promises batch updates approach training on the
  union and requires handling unseen classes, and neither holds for a
  fixed-architecture mini-batch learner.
- **`SoftDecisionTree(growth="per_leaf")`**, the growth rule of İrsoy, Yıldız &
  Alpaydın (2012). Splits one leaf at a time, the one carrying the most
  expected error, so the tree can end up unbalanced. It builds by far the
  sparsest trees: on an 800x20 synthetic problem it beats a fixed depth-6 tree
  using **3.7 splits against 63**.
- **`HierarchicalMixtureOfExperts.to_hard_router()`**, the mixture's answer to
  `to_hard_tree()`: 2.4x to 2.8x faster prediction at 0.981 to 1.000 agreement.
  `route_counts()` shows how many samples reach each expert, which the mixture
  cannot tell you because it spreads every sample over all of them. On Wine one
  of four experts turns out to receive nothing.
- **`WeightedKNN(n_condensed_sets=...)`**, the plural in Alpaydın (1997),
  *Voting over Multiple Condensed Nearest Neighbors*. Voting beats a single
  condensed subset everywhere and reaches the uncondensed classifier on Wine
  and Breast Cancer while storing roughly a sixth of the data.
- **`OmnivariateDecisionTree(selection="test")`**, which picks the simplest
  split type that is not significantly worse using the 5x2cv F test this
  library ships, instead of the ad hoc accuracy comparison its README argues
  against. Opt-in: it buys simpler splits and costs one to two points, because
  significance at a node does not compose into performance of the tree.
- **`py.typed`** and annotated public signatures, so type checkers see the
  library's hints instead of ignoring them (PEP 561). `mypy` runs in CI.

### Changed

- `GALNetwork(growth_policy="validation")` reconsiders the architecture when
  validation *error* plateaus, not only when loss stops falling. A
  capacity-starved network keeps getting more confident about the same
  mistakes: on six separable blobs the loss fell from 1.81 to 1.17 while the
  error sat at 0.46, and the old rule never grew past three units. Iris 0.867
  to 0.947, six blobs 0.732 to 0.857.
- `GALNetwork` predicts in float64 on the CPU, as HMoE already did. The float32
  row-order sensitivity returned at larger epoch budgets, which is exactly when
  growth has built a network big enough for it to matter.
- The benchmark table covers every model in the library. GAL lands within noise
  of Random Forest and SVM on all three datasets.
- Python 3.9 to 3.13 in CI, and the estimators keep their `check_estimator`
  standing: 62/63 for the three that accept `sample_weight`, 55/55 for the rest.
- Test suite: 180 to 255.

### Fixed

- `early_stopping=True` and the growth modes fall back instead of raising on
  datasets too small to hold out a stratified validation split.
- The `HardDecisionTree` export understands unbalanced trees, so exported rules
  stop where the tree stops rather than printing nodes that were never grown.

## [0.5.0] - 2026-09-10

Closes the last of the open work items. The library has no known broken
behaviour and no open issues.

### Added

- **`SoftDecisionTree.to_hard_tree()`.** After training, each internal node has
  settled on a hyperplane, and the sign of that hyperplane is a decision.
  The export routes each sample down a single path in plain numpy, with
  readable rules through `export_text()`. Held-out 30%, depth 4:

  | | soft | hard | agreement | predict speedup |
  |---|:---:|:---:|:---:|:---:|
  | Iris | 0.911 | 0.867 | 0.911 | 5.9x |
  | Wine | 0.981 | 0.981 | 1.000 | 5.9x |
  | Breast Cancer | 0.947 | 0.942 | 0.994 | 5.5x |

  It is a different model, not a re-encoding: a mixture over leaves is not a
  single path. Measure the agreement before relying on it.

- **`SoftDecisionTree(growth="incremental")`.** Starts from a single split and
  deepens while the extra level improves validation loss, so depth is chosen by
  the data (İrsoy, Yıldız & Alpaydın, ICPR 2012). `depth` becomes an upper
  bound and `tree_depth_` reports what was kept. It wins where the right depth
  is not known in advance (a synthetic 800x20 problem: 0.872 at depth 4.5
  against 0.857 at a fixed depth 6) and costs a little on small clean sets,
  where dividing the epoch budget across rounds outweighs adaptive depth. The
  default stays `"none"`.

### Fixed

- **A deepened level could not learn anything.** Making the deeper tree compute
  exactly the same function left both new children with identical
  distributions, and with `Q_left = Q_right` the mixture does not depend on the
  new gate, so its gradient is exactly zero and the children stay identical
  forever. Growing that way reached 0.756 on Iris against 0.958 for a tree of
  the same depth trained from scratch. A small jitter breaks the symmetry.
- **`early_stopping=True` crashed on datasets too small to split.**
  `train_test_split` raised `The test_size = 1 should be greater or equal to
  the number of classes`. Both early stopping and growth now fall back to
  training on everything, and `growth_` records what ran.

### Changed

- **The HMoE tree is evaluated as stacked banks, in log space.** Gates and
  experts were `ModuleList`s evaluated one node at a time; a depth-3 tree with
  branching factor 4 meant 170 small matmuls per forward. Fit wall time on
  Breast Cancer, 20 epochs: depth 2 b=2 0.7s to 0.2s, depth 3 b=4 3.4s to 1.2s,
  between 2.4x and 3.2x across shapes. Accuracy over 5 seeds is unchanged or
  slightly better. Subtree dropout now applies in log space, so a dropped
  subtree carries exactly zero weight rather than a clamped small number.
- **CI enforces what it measures.** Coverage is gated at 95% (currently 96.9%),
  and a separate job executes every notebook and fails on the first cell error,
  so committed outputs cannot go stale unnoticed.
- Test suite: 156 to 180.

## [0.4.0] - 2026-09-09

Work on GALNetwork, plus weighting for the soft tree. Every classifier in the
library now passes scikit-learn's estimator checks.

### Added

- **`GALNetwork(growth_init="residual")`, now the default.** A new hidden unit
  used to arrive with random incoming *and* outgoing weights: it perturbed every
  logit the moment it appeared and then had to be trained up from noise. It is
  now fitted to the residual error of the frozen network, maximizing the
  correlation between its activation and `p - y` (Fahlman & Lebiere, 1990), and
  installed with **zero** outgoing weights, so the network computes the same
  function at the moment of growth. Growth can no longer make the model worse.

  Over 3 seeds of 5-fold CV, accuracy and mean final hidden units:

  | | random | residual |
  |---|:---:|:---:|
  | Iris | 0.931 / 22.9 u | **0.958 / 7.1 u** |
  | Wine | 0.983 / 6.8 u | 0.981 / **4.2 u** |
  | Breast Cancer | 0.975 / 2.0 u | 0.975 / 2.0 u |
  | 3 separable blobs | 1.000 / 6.9 u | 1.000 / **4.1 u** |
  | 6 separable blobs | 0.967 / 15.1 u | **0.989** / 17.9 u |

  Equal or better accuracy everywhere, with a third of the units on Iris, and no
  runtime cost. `growth_init="random"` restores the old behaviour.

- **`GALNetwork(growth_policy="validation")`**, opt-in. Holds out
  `validation_fraction`, leaves the architecture alone while validation loss is
  still improving by more than `tol`, and when it stalls tries removing the
  least useful unit before adding one. Stops after `patience` changes that fail
  to pay off, restoring the best epoch. It reaches far smaller networks and wins
  where capacity is not the constraint (Breast Cancer 0.977 with 3.9 units
  against 0.975 with 2.0), but under-grows where it is (6 blobs 0.733 against
  0.989), so it is not the default.

- **`SoftDecisionTree.fit(X, y, sample_weight=...)` and `class_weight`**
  (`None`, `"balanced"`, or a dict). They combine multiplicatively and are
  normalized to mean 1, so `learning_rate` and `penalty_coef` keep their
  meaning. On a 600-sample problem where one class holds 6%,
  `class_weight="balanced"` lifts recall on that class from 0.500 to 0.976 while
  accuracy moves 0.962 to 0.957.

### Fixed

- **HMoE predictions no longer depend on row order.** Rows are independent, but
  BLAS blocks differently for different memory layouts, so in float32 the same
  sample scored inside a reordered batch came out up to 1.2e-07 different, and a
  borderline argmax could flip with it. Prediction runs in float64, putting that
  at 2.2e-16. Training stays float32.
- **NaN in the GAL candidate objective.** It clamped after the square root,
  leaving `sqrt(0)` in the graph. Its gradient is infinite, so a candidate whose
  sigmoid saturated into a constant activation returned NaN and took every later
  candidate's score with it, leaving growth with no unit to install.
- GAL prunes on **contribution**, the spread of a unit's activation scaled by
  the size of its outgoing weights, rather than activation variance alone.
  Variance says nothing about whether a unit matters: a nearly constant unit
  with large outgoing weights still shifts every logit.

### Changed

- All seven classifiers pass `sklearn.utils.estimator_checks.check_estimator`.
  `SoftDecisionTree` is at 62 of 63: declaring `sample_weight` activates the
  weight checks, and
  `check_sample_weight_equivalence_on_dense_data` demands that weighting a row
  be bit-identical to repeating it, which no stochastic mini-batch learner can
  satisfy. The loss equivalence that does hold is tested directly.
- Python support is 3.9 to 3.13, all tested in CI. The package previously
  claimed 3.8, which was never exercised and reached end of life in 2024.
- Packaging moved to `pyproject.toml`, and `ruff` runs in CI, both contributed
  by @yunaremaia.
- Test suite: 135 to 156.

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

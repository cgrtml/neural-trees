# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.7.0] - 2026-09-23

### Added

- `SoftDecisionTree.explain(X, feature_names=...)`: per-prediction
  explanations. For each sample: the predicted class and probability, the leaf
  that received most of the sample's mass and its distribution, every gate on
  the path to it with the direction taken, its probability and the largest
  feature terms, a per-feature contribution score along the path, and the
  smallest single-feature change that flips the class, verified by
  re-predicting. Returned as `Explanation` objects with `to_text()` and
  `to_dict()`. The contribution score reads the linear gates directly and is
  not SHAP; the counterfactual is `None` when no single-feature change on the
  path flips the class. Documented in "Explaining a prediction", with a
  gallery example.
- `growth_init` in `{"random", "residual", "residual_gate"}`: how the
  symmetry between a new gate's two children is broken when a soft tree grows.
  Measured on 24 datasets, the direction does not change accuracy against
  random noise (mean +0.20 points, median 0.00); it lowers fold-to-fold
  variance by about 15% on multi-class problems. Also `growth_jitter`, the
  size of the perturbation (#97).
- `growth_budget` in `{"split", "full"}`: whether `max_epochs` is divided
  across growth rounds (the previous behaviour) or given to every round.

- `SoftDecisionTreeRegressor`: the soft tree for regression, with
  multi-output targets, `sample_weight`, early stopping, internal target
  scaling and leaves initialised at the target mean. Passes every
  scikit-learn regressor check except `check_sample_weight_equivalence`,
  for the same mini-batch reason as the classifier. Growth and the hard-tree
  export are not available for it yet (closes #103).
- `SoftDecisionTree.to_numpy()` and `NumpySoftTree`: a torch-free copy of
  the fitted tree with the mixture over leaves kept, so it is the same model
  rather than the hard-tree approximation. Predictions agree with the torch
  model to float32 precision; `to_json()`/`from_json()` round-trip exactly
  and the file carries a format tag. Meant for shipping a fitted tree to a
  service without PyTorch and for freezing one for audit.
- `sample_weight` in `NaiveBayesClassifier` and `WeightedKNN` (#94). Naive
  Bayes is exact: integer weights reproduce the fit on repeated rows, and it
  now passes all 63 scikit-learn checks. In the KNN a weight scales the
  neighbour's vote and a zero weight drops the row, which is not the same as
  repeating rows (a repeated row fills several of the `k` slots), so it
  passes 62 of 63 and the docstring says which one and why.
- `random_state` in `OmnivariateDecisionTree` (#96). It reaches the k-means
  class pairing, the stump and MLP candidates and the F test's folds; the
  hard-coded seed of 42 is gone. Measured on Breast Cancer with
  `selection="test"`, the split-type counts move between seeds, which the
  fixed seed had hidden.
- `early_stopping`, `validation_fraction` and `n_iter_no_change` in
  `HierarchicalMixtureOfExperts`, with the names, defaults and meaning they
  have on `SoftDecisionTree` (#99). The split is stratified and drawn with
  `random_state`, fresh on every call to `fit` so a `warm_start`
  continuation never scores against a stale one; `n_iter_` reports the
  epoch reached, and the best validation epoch's parameters are restored.
  Off by default and then bit-identical to the previous behaviour. With
  `max_epochs=400` it stopped at 62 epochs on Breast Cancer and 71 on
  Digits for the same accuracy (0.977 against 0.974 and 0.977 against
  0.977), 7 and 6 times faster.
- `n_jobs` in `OmnivariateDecisionTree` (#102): the three candidate split
  types at a node are cross-validated in parallel, never with more workers
  than candidates, and the tree is identical for every `n_jobs` on a fixed
  seed (tested for both selection rules). The gain is modest because nodes
  are small and the MLP dominates: on Breast Cancer at depth 3, 0.82 s to
  0.68 s with the accuracy rule and 2.72 s to 2.54 s with the test rule.
- `to_hard_tree(rule=...)` (#101): besides the default `"gate"` walk, a
  `"leaf"` rule (the leaf with the largest arrival probability) and a
  `"contribution"` rule (the leaf that contributes most to the soft
  mixture's winning class). Measured on Wine, Digits and satimage,
  `"contribution"` agrees with the soft model more on every dataset (0.994
  to 0.998, 0.992 to 0.999, 0.980 to 1.000) at 1.3 to 3 times the cost of
  `"gate"`, so the default stays `"gate"`; the table is in the user guide.
  Walking by the larger child probability is `"gate"` by identity and is
  not offered as a separate rule.
- `scipy.sparse` CSR input in `NaiveBayesClassifier` and `WeightedKNN`
  (#100), with the sparse tag declared so scikit-learn's sparse checks run.
  Predictions on `X` and `csr_matrix(X)` are identical; nothing is
  densified except the Manhattan distance, in bounded chunks. On 20
  newsgroups (130 107 features) the sparse naive Bayes fits 11 314
  documents in 0.04 s at 25 MB peak, where the dense path takes 2 s and
  60 s to predict for 2 000 documents; the sparse KNN predicts 20 times
  faster at 6 MB against 1.6 GB. The dense-only models (the torch models
  and the two multivariate trees) now raise a `TypeError` that names the
  model and the fix, `X.toarray()`.
- `sample_weight`, `class_weight` and `min_weight_fraction_leaf` in
  `MultivariateDecisionTree` and `OmnivariateDecisionTree` (#95). The weights
  reach inside the split search: the multivariate tree's node discriminant
  is solved with weighted means, pooled covariance and prior ratio, the Gini
  decrease and the leaves are weighted; the omnivariate tree weights the
  two-group construction, the model selection (weighted fold accuracy and
  the candidate's fit where it takes weights) and the leaves. Size limits
  count rows, as in scikit-learn's trees. Without weights the fits are
  unchanged. Measured with one class cut to a tenth, balancing changes
  which *kind* of split the omnivariate tree picks (Breast Cancer: from
  linear toward stumps and MLPs, Wine: the other way) and raises minority
  recall on two of three datasets; the table is in the user guide.
- A benchmark against the field, `benchmarks/rakipler.py`: nine models
  (the soft tree at depth 4 and with per-leaf growth, GAL, Random Forest,
  an MLP, XGBoost, LightGBM, GRANDE, NODE) on 24 datasets, three seeds of
  five-fold cross-validation, every model untuned and in its own process.
  Documented on the "Against the field" page with the tables generated by
  `benchmarks/rakipler_ozet.py`. Over all 24 the boosted ensembles lead by
  a point or two; on the eight datasets with at most 1 000 rows GAL beats
  XGBoost on every one and the per-leaf tree on seven, and the MLP does
  too, so the finding is about smooth models against untuned boosting on
  small tables. The split was chosen after seeing the data and is stated
  as exploratory.
- `benchmarks/run_benchmarks.py --check` compares a fresh run with the README
  table (tolerance 0.0005, i.e. the run must round to the README cell) and
  a CI job runs it when the
  models, the script or the README change, and weekly (#98).
  Its first run found one stale cell: the multivariate tree on Breast Cancer
  is 0.950 on the Linux runner against the 0.952 the README carried from a
  laptop; the README now shows the runner's value and says so.

### Changed

- Mini-batches in every torch model are sliced directly from tensors
  instead of going through `DataLoader(TensorDataset(...))`, which indexed
  one sample at a time. The
  new iterator draws exactly the permutation the loader drew from the same
  seed, so fits with a `random_state` are bit-identical to 0.6.2's; they are
  14-21% faster. A "Performance" page documents measured fit times from
  1 000 to 50 000 samples.

### Fixed

- `SoftDecisionTreeRegressor.predict` runs in float64 on the CPU, as the
  mixture of experts already did, so a row's prediction does not depend on
  the batch it is scored in; scikit-learn's subset-invariance check caught
  the float32 version on one CI runner.
- Both multivariate trees closed nodes that still held every class when one
  class at the node was rare (#104). The two-group problem at a node is built
  by clustering the class centroids; unweighted, a class with a sample or two
  far from the rest became a group of its own, the discriminant separated
  one sample from a thousand with a hyperplane of norm 1e15, `min_samples_leaf`
  refused that split and the node became a leaf. Two changes, in
  `_grouping.py`, shared by both trees: centroids are clustered with each
  class's mass as its weight, and when the discriminant fitted to that
  grouping still cannot make a legal split (a few far-away samples are
  separable, so it overfits them) a mass-balanced cut along the centroids'
  principal axis is tried before the node gives up. A third guard is in the
  multivariate tree alone: scikit-learn's SVD discriminant on ill-conditioned
  node data (pixel features, constant columns) can return a hyperplane of
  norm 1e18 that classifies its own training groups worse than the majority
  rule, and which side of a rank threshold that happens on depended on the
  BLAS thread count; such a hyperplane is now rejected and a shrinkage
  discriminant fitted instead. Digits at depth 6, standardised, 3 seeds of
  5-fold: `MultivariateDecisionTree` 0.354 to 0.932 (3 to 17 internal
  nodes), `OmnivariateDecisionTree` 0.490 to 0.948. Every published number
  for these two models was re-measured; the ones that moved are listed with
  them.
- Per-leaf growth started the two children of a split leaf as identical,
  untrained distributions, which discarded what the leaf had learned and, by
  Proposition 1 of the paper, left the new gate with no gradient. Children
  now inherit the parent and the symmetry is broken as in level-wise growth.
  On Digits this moves per-leaf growth from 0.703 to 0.927 at about ten
  splits, against 0.967 for a complete depth-six tree with 63. The previous
  behaviour is kept as `growth_init="uniform"` for reproducing the comparison.

### Measured

- `GALNetwork`'s `growth_policy="validation"` stays off by default, now with
  a reason measured on six datasets: it gives lower accuracy everywhere but
  Wine (2 to 7 points) and a smaller network only where the threshold rule
  had grown large, because a
  capacity-starved network keeps lowering validation loss slowly and the
  rule reads that as "no unit needed". Documented in "Design decisions"
  with the table; script `paper/arxiv/olcum/gal_politika.py`.
- The soft tree's probabilities are calibrated: expected calibration error
  0.023 / 0.052 / 0.030 on Breast Cancer / Wine / Digits, against 0.026 /
  0.048 / 0.022 for logistic regression and 0.038 / 0.098 / 0.201 for a
  random forest. `learn_temperature=True` does not help reliably and stays
  off by default. Documented under "Probabilities" in the user guide.
- The cost of exactly function-preserving deepening tracks the number of
  classes and not the number of features: over twenty OpenML CC-18 datasets,
  mean -0.2 points on the thirteen binary problems and 40.2 points on the
  seven multi-class ones (Spearman with K: 0.71; with p: 0.02).

## [0.6.2] - 2026-09-16

An archival release with no library behaviour changes. It exists because the
arXiv paper cites "the exact version used", and the version it used has to
contain the script that produced its numbers.

### Added

- `paper/arxiv/olcum/olc.py` measures every number in the paper under one
  protocol (features scaled on the training fold only, three seeds of
  stratified five-fold cross-validation, mean and standard deviation, the
  combined 5x2cv F test where a decision is needed) and writes `sonuc.json`.
  `doldur.py` substitutes those values into `main.tex.tmpl`; no number in the
  paper is typed by hand.
- Digits (1797 x 64, ten classes) and a 2000 x 50 synthetic problem, added
  because the original three datasets were small and low dimensional.

### Changed

- The zero-gradient regression test now asserts exact equality for gate
  weights, gate biases and sibling leaf gradients, matching what the paper
  claims and what single precision actually produces.
- Documentation, the JOSS draft and the `deepen` docstring were brought in line
  with the corrected measurements. The earlier numbers had been measured with
  features scaled on the whole dataset before splitting, which leaks test-fold
  statistics into training. Two claims did not survive the correction and the
  fourth dataset: residual-fitted units give a smaller GAL network but not a
  more accurate one, and per-leaf growth buys sparsity but loses accuracy on
  the harder synthetic problem. Both are now documented as trades.

## [0.6.1] - 2026-09-11

An archival release. No behaviour changes; the code is identical to 0.6.0 apart
from the version string. This release exists so that the papers can cite a
frozen, DOI-bearing version of exactly the code their numbers came from.

### Added

- `paper/paper.md` and `paper/paper.bib`: a Journal of Open Source Software
  submission draft.
- `paper/arxiv/`: a self-contained LaTeX source for an empirical paper on
  constructive tree-structured classifiers, with a submission checklist. Its
  central result is stated and proved: deepening a soft tree so that the
  function is exactly preserved makes the new gate's gradient identically zero
  and leaves both children with identical gradients, so the added level is a
  fixed point of the optimiser rather than a slow start.
- `.zenodo.json`: archive metadata, so the DOI record carries the ORCID,
  licence and keywords rather than repository defaults.

### Fixed

- Documentation numbers that had drifted as the library gained features. The
  benchmark table's two GAL cells moved after GAL gained `sample_weight`
  (Iris 0.951 to 0.952, Wine 0.980 to 0.982); the test count in the README
  still said 255; the `check_estimator` claim said "clean" without noting the
  one check no mini-batch learner can pass; the LICENSE carried a 2024
  copyright year for a repository first published in 2026. Every figure in the
  README and in both papers was re-measured against this code under one
  protocol before release.

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

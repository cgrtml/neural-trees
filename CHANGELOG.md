# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `DistilledTree(thresholds=k)` and `DistilledForest(thresholds=k)`: the student sees, next to the features, one 0/1 input per (feature, threshold) pair among the teacher's `k` highest-gain splits (`neural_trees.distillation.ThresholdBasis`; XGBoost, LightGBM and the scikit-learn tree models are read). Prediction, `explain` and `fidelity` take the original features, explanations name the threshold inputs ("bill_sep > 1500"), and `student_inputs(X)` gives what the exports take. With 512 thresholds and the tuned XGBoost of `benchmarks/kanit.py` as teacher (nine tables, three seeds; `benchmarks/esik.py`) one tree was 0.70 points under tuned XGBoost at 20 000 rows with 14 to 99 splits; with 500 to 5 000 rows, rivals re-tuned at each size, the three-tree group was level with tuned XGBoost and LightGBM and 0.40 points above XGBoost at 500 rows (18 of 27 cells, p=0.012). On many transfer rows the default student is pruned on a 40 000-row subsample.
- `DistilledForest(combine="mean")`: average the trees' probabilities instead of summing their log probabilities. The sum is overconfident (calibration slope 0.35 to 0.38, ECE 0.13 on the credit-default case at every size); the mean gives slope 1.05 to 1.15 and ECE about 0.02 at the same AUC (`cases/credit-default/esik.py`). The default stays "log".
- `neural_trees.DistilledForest(teacher=None, student=None, n_trees=3)`: the teacher and transfer rows of `DistilledTree`, then `n_trees` students, each on its own bootstrap sample of those rows, predicting with the sum of their log probabilities. Each tree in `estimators_` can be explained and exported on its own; `explain`, `to_hard_tree` and `to_onnx` on the group raise `NotImplementedError` and `to_numpy` returns one export per tree. With an XGBoost teacher and ten MUNGE passes on the 11 cells up to 20 000 rows (`MUNGE_PASSES=10 benchmarks/grup.py`) three depth-8 trees were 0.38 points above one (10 higher, none lower, p=0.001) and 0.9 below the teacher (p=0.12). `benchmarks/xgb_boyut.py` compares both students with XGBoost of 3 to 300 trees on the same cells: one tree (137 splits) is level with 30 trees (1 303 splits; -0.25, p=0.90), the group (about 450 splits) above 10 trees (+1.16, 10 of 11, p=0.003) and level with 100 (4 159 splits; -0.40, p=0.58). Five trees, depth-6 students, boosting on the teacher's logits, joint fine-tuning and linear leaves were tried and not kept (documentation page "Large tables").
- `benchmarks/kanit.py`: tuned XGBoost, LightGBM, CatBoost, HistGradientBoosting and CART (20 random configurations each, early stopping on a validation split) against `DistilledTree`, `DistilledForest` and the soft tree without a teacher, on nine datasets including three credit and marketing tables, with accuracy, ROC AUC and model size.
- `neural_trees.DistilledTree(teacher=None, student=None, alpha=0.0)`: a scikit-learn classifier that fits a teacher (default `HistGradientBoostingClassifier`, XGBoost and LightGBM work the same) on `(X, y)` and then one soft tree on the teacher's class probabilities. `predict`, `explain` and every export go through that single tree; `fidelity(X)` reports agreement with the teacher. `synthetic` adds MUNGE passes (default 10, see Changed) (`neural_trees.distillation.munge`; Bucila, Caruana and Niculescu-Mizil, KDD 2006): synthetic rows near the training rows that the teacher labels for the student. On six OpenML tables at 5 000 to 100 000 rows (XGBoost teacher, depth-8 student, 13 cells) this was 1.5 points above the same tree on labels and 0.9 above plain distillation, higher in 12 of 13 cells each (Wilcoxon p=0.0002 and 0.0005), and cut the mean gap to XGBoost from 3.3 to 1.8 points, at about three times the fit time (`benchmarks/munge.py`). Plain distillation alone was +0.6, not significant. With the default HistGradientBoosting teacher the synthetic rows gained less, +0.5 points over `synthetic=0` (8 of 12 fits higher, none lower, p=0.003; `benchmarks/munge_lib.py`). `fit(X, y, X_unlabeled=...)` lets the teacher label further rows for the student, for tables where labels are scarcer than rows: with twice as many unlabelled rows, +1.85 points over the tree on labels in 13 of 13 cells. It accepts NaN and passes scikit-learn's estimator checks except the sample-weight equivalence check, as `SoftDecisionTree` does. Measured on large tables in `benchmarks/damitma_olcek.py` and on the new documentation page "Large tables, ensembles and missing values".
- `benchmarks/gelistir.py`: a one-split harness for trying tree configurations on 20 000-row training sets, and `benchmarks/damitma_olcek.py`: XGBoost, the soft tree on labels and the distilled tree at 5k, 20k and 100k training rows on six OpenML tables.
- `SoftDecisionTreeRegressor` gains what the classifier gained in this release: `split_features`, `init="cart"` (a soft copy of a `DecisionTreeRegressor` on the standardised targets, multi-output included), `init_sharpness`, `init_prune` (cost-complexity strength chosen by 3-fold CV on weighted squared error) and missing values, with the numpy, hard-rule and ONNX exports carrying them. The CART copying and pruning are shared code with the classifier.
- Missing values in `SoftDecisionTree`: `fit`, `predict`, `explain` and the numpy, hard-rule and ONNX exports accept NaN (infinity is still rejected). A missing cell is set to 0 and every feature that had missing values in training gets an indicator input. For a linear gate this is exactly a learned per-gate default, the soft counterpart of the default direction a boosted tree learns per split: when the feature is missing, its term is replaced by a free weight of that gate. `missing_columns_` lists the features with indicators, `feature_importances_` folds an indicator into its feature, `explain` names it ("income is missing") and does not propose a counterfactual on a missing feature. A NaN in a feature that had none in training is set to 0.
- `init_prune=True` (default with `init="cart"`): the starting CART tree is pruned with a cost-complexity strength chosen by 3-fold cross-validation, since its soft copy cannot drop the splits it is handed. With `split_features=1` as well, each gate keeps CART's feature for the whole fit.
- `neural_trees.distill(teacher, X, X_eval=...)`: fit a soft tree to a fitted classifier's probabilities (each row repeated per class, weighted by the teacher's probability) and report agreement with the teacher for the tree and for its hard rules, the total-variation distance, the number of splits and the features per split; `neural_trees.fidelity(teacher, student, X)` for the agreement alone. On 24 datasets with XGBoost, LightGBM and random forest teachers the student agrees with the teacher on 0.934 of held-out rows against 0.901 for CART of depth 6 (43/16/13 dataset-teacher pairs, Wilcoxon p<0.0001) with about 11 splits against 38 leaves. The lead needs splits that use several features: a dense student pruned down to one or two features per split is the worse copy. Started from CART instead (`init="cart"`, `split_features=k`), it beats CART at every budget: +1.5 points at one feature per split (p=0.002), +2.5 at two (p=0.003), +3.2 at three (p=0.0008), as faithful as an unrestricted split (`benchmarks/distill.py seyrek_cart`).
- `SoftDecisionTree(split_features=k)`: at most `k` features per split. The tree is trained as usual, then pruned in rounds that halve the features each gate keeps, ranked by `|weight| * std(feature)`, with fine-tuning between rounds and the pruned weights held at exactly zero. Pruning in one step lost most of an oblique tree (Digits, depth 5, k=1: 0.94 to 0.53); halving keeps more (k=3: 0.72, k=5: 0.83, against 0.67 for CART of the same depth, one split, one seed). Rules and explanations print only the kept features.
- `SoftDecisionTree.from_hard_tree(tree, X, sharpness)` and `SoftDecisionTree(init="cart", init_sharpness=10.0)`: start from a soft copy of a CART tree (each split `x[f] <= t` becomes the gate `sigmoid(sharpness * (x[f] - t) / std(x[f]))`, CART's shape and class distributions kept) and fine-tune with gradients. Greedy search places sharp thresholds; gradients then move them jointly. In a first single-split look at depth 6 with `learn_temperature=True`: Digits 0.924 (CART 0.764, per-leaf soft tree from scratch 0.907), phoneme 0.870 (0.838, 0.846), segment 0.964 (0.919, 0.964), wall-robot 0.954 (0.991, 0.886). The sharpness trades the two ends: 100 keeps wall-robot at 0.986 but leaves Digits at 0.784. Cross-validated on the 24 datasets (3 seeds x 5 folds, sharpness chosen inside each training fold, `init_prune=True`; `benchmarks/cart_init.py --prune`): 5.0 points above CART of the same depth (21 wins, 2 losses, p=0.0001), level with the per-leaf soft tree overall, ahead of it above 1 000 rows (+1.1, not significant) and behind it below (-1.5).
- `benchmarks/soft_forest.py`: 25 bagged per-leaf soft trees on the sixteen datasets above 1 000 rows, same folds as the field benchmark. Averaging closes about half the gap to XGBoost (2.3 points to 1.0, Wilcoxon p = 0.32, 4/6/6) at 160 times its fit time and with no readable tree left; documented on the benchmarks page and the playground's field page as a measured answer, not a recommendation.

### Changed
- `DistilledTree(synthetic=10)` is the new default (was 2). Ten MUNGE passes put the student 0.5 points above two passes with an XGBoost teacher (11 cells up to 20 000 rows, 7 higher, none lower, p=0.001; `MUNGE_PASSES=10 benchmarks/munge.py`) and 0.3 points above with the default HistGradientBoosting teacher (8 of 12 fits higher, none lower, p=0.009; `benchmarks/munge_lib.py --synthetic 2 10`). Thirty passes were no better than ten. Fitting takes four to six times as long, about 15 minutes for 20 000 rows on one CPU thread; pass `synthetic=2` for the previous behaviour.
- `NumpySoftTree` (the torch-free export from `to_numpy()`) predicts from the gates that split only, with each acting leaf's path summed as one matrix product, instead of walking all `2**depth - 1` gates. Same model and same numbers (largest difference to the previous code 1e-14 over 300 random trees). On 10 000 rows with 28 features and a depth-8 tree it went from 303 ms to 12 ms at 44 splits, 220 ms to 22 ms at 94 and 162 ms to 58 ms with every gate splitting; one row from 0.11 ms to 0.05 ms. For scale: XGBoost with 300 depth-6 trees took 17 to 19 ms for 10 000 rows and 0.056 ms for one row on the same machine (Higgs, covertype, jannis; one thread each).

### Fixed
- `warm_start=True` with a different number of features now raises a `ValueError` naming both counts instead of a matrix-shape error from inside torch.
- `SoftDecisionTree` and `SoftDecisionTreeRegressor` dropped the sample weights of the validation split, so the validation loss behind `early_stopping`, `growth="incremental"` and `growth="per_leaf"` was unweighted while training was weighted, and per-leaf growth also chose the leaf to split by unweighted error. With soft targets (each row repeated per class, weighted by its probability) per-leaf growth stopped at the root: 0.20 accuracy on Digits, 0.97 after the fix. With `class_weight` the stopping and growth decisions followed the majority class. Unweighted fits are unchanged, since their weights are all one. Found while distilling tree ensembles into soft trees (`benchmarks/distill.py`).
- On Intel Macs `pip install neural-trees` pulled NumPy 2 next to torch 2.2.2, the last torch wheel for that platform, and the import failed with "Numpy is not available". NumPy is now pinned below 2 on that platform only. Found while checking the 0.8.0 wheel in a clean environment.

## [0.8.0] - 2026-10-01

A minor release rather than a patch: the ONNX export, the regressor's growth and exports, and the rebuilt playground are new surface, not fixes.

### Added

- `explain(..., feature_bounds=(lower, upper))` keeps a counterfactual inside the values a feature can take, so a one-hot column is never "set to 2.2". The clipped candidate is still verified by re-predicting.
- `cases/german-credit/`: the first auditable model package. One script measures a soft tree against logistic regression, CART, random forest, XGBoost and LightGBM on identical folds with the dataset's cost matrix, writes calibration, rules, three explained decisions with reachable counterfactuals, slices by age, sex and residency, refit stability, the JSON and ONNX model files with an agreement check, and a model document generated from the numbers.
- `SoftDecisionTreeRegressor` gains what the classifier had and it lacked:
  `growth` (`"incremental"`, `"per_leaf"`) with `growth_init`,
  `growth_jitter` and `growth_budget`, deciding on held-out squared error
  and splitting along the leaf's residual; `to_numpy()`, a `NumpySoftTree`
  of kind `"regressor"` that predicts in target units and round-trips
  through JSON; and `to_hard_tree()`, a new `HardRegressionTree` that
  prints one value per leaf and scores R^2. The module's docstring no
  longer lists these as missing; `explain()` still is.
- `to_onnx()` and `save_onnx()` on `SoftDecisionTree`,
  `SoftDecisionTreeRegressor` and `NumpySoftTree`: the fitted tree as an
  ONNX graph of standard operators, with feature names and class labels as
  metadata, predicting the same probabilities as the torch model to
  float32 precision. Optional extra `neural-trees[onnx]`; the file runs in
  any ONNX runtime without torch or this library. Measured single-row
  latency 11 microseconds in ONNX Runtime against 90 for the torch
  estimator, 33 for the numpy copy; the "Performance" page has the table
  and `benchmarks/latency.py` produces it.

### Playground

- The model page draws the structure each model ended up with: for the
  soft tree the gates with their heaviest features and the share of the
  data reaching each, the branch probabilities and the leaves, three
  levels deep with deeper subtrees folded into one box; the oblique cuts
  of the multivariate tree; the split type chosen at each node of the
  omnivariate tree; the routing tree of the mixture of experts with the
  share each expert receives; the layer the GAL network grew to. Start
  here gains an at-a-glance table of the ten models. `Explanation.to_text()`
  prints plain class values rather than numpy scalars.
- Free-text columns on the "Try it on your data" page: a string column
  whose values are sentences becomes its 500 most frequent words and word
  pairs (TF-IDF, fitted inside each fold), so the soft tree's rules and
  counterfactuals name words ("if message did not contain gun"). Measured
  on four newsgroups (1 500 posts, body text only): the soft tree 0.855,
  a random forest 0.855, naive Bayes 0.828, CART 0.605; a tuned text
  model or an LLM will do better on nuanced text, which the page does not
  claim otherwise. The hard rule tree prints plain class values rather
  than numpy scalars.
- A "Try it on your data" page in the playground: upload a CSV, pick the
  column to predict, and run the same comparison on it, with imputation,
  scaling and one-hot encoding fitted inside every fold. It reports the
  majority-class floor next to the models, prints the soft tree's rules,
  explains any row with its counterfactual, and hands back a `model.json`
  that predicts with numpy alone plus the Python that reproduces the run.
  Tables are used in the session only; the hosted app caps at 5 000 rows.
- The Streamlit playground is rebuilt as pages around the questions a
  visitor asks. *Start here* shows the boundary every model learns on the
  same 2D data; *Compare* trains the chosen models on the same folds and
  says who is highest, who is within fold noise and who is behind, with the
  5x2cv F test on any pair; *How a model works* explains one model, puts its
  knobs on a live boundary and shows what it learned (rules, an explained
  prediction, split types, growth and pruning, routing); *What was fixed and
  verified* and *Against the field* carry the library's own contribution and
  the 24-dataset benchmark. The model registry and every cached fit live in
  `playground/`, the pages in `views/`; the contract test parses all of
  them. The hosted app installs the released package.
- The Streamlit playground opens on a comparison table: every selected
  model's accuracy, its gap to the best and whether that gap is within fold
  noise (paired t-test over the folds), with a plain-language summary and a
  panel on what the library changed and how that was checked. Sidebar
  buttons select all models, the library's only, or the default four.
- Two more tabs in the playground. "Explain a prediction" runs
  `SoftDecisionTree.explain()` on any sample of the chosen dataset: the leaf
  it reached, every gate on the path with the deciding features, the
  contribution chart and the verified single-feature counterfactual.
  "Against the field" reads `benchmarks/rakipler-sonuc.json` and shows the
  nine-model, 24-dataset comparison with a subset switch (all, small, large,
  binary, multi-class) and the same caveats as the documentation page. The
  hosted app installs the released package.

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

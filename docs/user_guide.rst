User guide
==========

The library has three groups of estimators and one group of statistical tests.
All estimators follow the scikit-learn contract: ``fit``, ``predict``,
``predict_proba``, ``score``, ``get_params``/``set_params``, and attributes with
a trailing underscore that only exist after fitting.

Soft decision trees
-------------------

A hard decision tree sends each sample down exactly one path. A soft decision
tree replaces the threshold at each internal node with a sigmoid gate

.. math::

   g_n(x) = \sigma(w_n^\top x + b_n),

so a sample reaches *every* leaf with some probability, and the output is the
mixture of the leaf distributions weighted by those probabilities. The tree is
then differentiable end to end and is trained by gradient descent.

.. code-block:: python

   from neural_trees import SoftDecisionTree

   model = SoftDecisionTree(depth=4, max_epochs=100, random_state=0)
   model.fit(X_train, y_train)

Two consequences are worth knowing. Because :math:`w_n` is a vector, each split
is oblique rather than axis-aligned, which is often what makes a shallow soft
tree competitive with a much deeper CART. And because the model is a mixture,
``feature_importances_`` is computed from the gate weights rather than from
impurity decrease.

Probabilities
^^^^^^^^^^^^^

The mixture output is a genuine probability, and measured it is a calibrated
one. Expected calibration error over three seeds of five-fold
cross-validation (lower is better), against a logistic regression and a
300-tree random forest on the same folds:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - dataset
     - soft tree, depth 4
     - logistic regression
     - random forest
   * - Breast Cancer
     - 0.023
     - 0.026
     - 0.038
   * - Wine
     - 0.052
     - 0.048
     - 0.098
   * - Digits
     - 0.030
     - 0.022
     - 0.201

The soft tree sits with logistic regression and well clear of the forest,
whose probabilities are the usual over-confident vote fractions. Brier scores
tell the same story. ``learn_temperature=True`` does not improve calibration
reliably (it helped on Wine, hurt on the other two) and cost 3 points of
accuracy on Digits, so it stays off by default.

Growing the tree
^^^^^^^^^^^^^^^^

``depth`` fixes the size of a complete tree. The ``growth`` argument builds one
instead:

``growth="none"``
  Train a complete tree of the given depth. The default.

``growth="incremental"``
  Start shallow and add a level at a time, training between additions.

``growth="per_leaf"``
  Split one leaf at a time, choosing the leaf carrying the most expected error,
  which is the rule of İrsoy, Yıldız and Alpaydın. ``depth`` becomes an upper
  bound rather than a target, and the resulting tree is unbalanced.

Per-leaf growth is the one to reach for when the tree should stay small and
readable. It buys sparsity reliably and accuracy only sometimes. On a synthetic
problem with 800 samples and 20 features it reached 0.885 with 3.7 splits on
average, against 0.839 for a complete depth-6 tree using all 63; on a harder
problem with 2000 samples and 50 features it kept the sparsity (4.8 splits) and
lost 4.3 points to the complete tree. See :doc:`design_decisions`.

Reading the tree back out
^^^^^^^^^^^^^^^^^^^^^^^^^

``to_hard_tree()`` reads each gate as a hard decision and returns a plain NumPy
model that prints its rules and predicts faster.

.. code-block:: python

   hard = model.to_hard_tree()
   print(hard.export_text(feature_names=feature_names))

The export is an approximation, not a re-encoding: a mixture over leaves is not
a single path, so the hard tree does not always agree with the model it came
from. It reports its agreement rather than assuming it. Over fifteen folds on
Wine the mean agreement was 0.998, with 0.971 in the worst fold, and prediction
was about four times faster.

``rule`` chooses how a sample reaches a leaf. ``"gate"`` (the default) takes
the sign of each gate on the way down and is the routing the printed rules
describe. ``"leaf"`` sends the sample to the leaf with the largest arrival
probability; ``"contribution"`` to the leaf that contributes most to the soft
mixture's winning class. Both alternatives evaluate every gate rather than
``depth`` of them. Measured over three seeds of five-fold cross-validation
(depth 4, 40 epochs), agreement with the soft model and prediction cost:

.. list-table::
   :header-rows: 1
   :widths: 14 22 22 22 20

   * - Dataset
     - ``"gate"``
     - ``"leaf"``
     - ``"contribution"``
     - soft model
   * - Wine
     - 0.994 (worst 0.971), 2.6 ms
     - 0.996 (worst 0.972), 3.1 ms
     - 0.998 (worst 0.972), 3.4 ms
     - 12.3 ms
   * - Digits
     - 0.992 (worst 0.986), 0.65 ms
     - 0.995 (worst 0.986), 1.3 ms
     - 0.999 (worst 0.992), 1.4 ms
     - 2.4 ms
   * - satimage
     - 0.980 (worst 0.974), 0.37 ms
     - 0.987 (worst 0.979), 1.0 ms
     - 1.000 (worst 0.997), 1.1 ms
     - 1.3 ms

Times are per 1 000 predictions, best of five. ``"contribution"`` agrees
more on every dataset, and on satimage it closes the gap almost entirely,
but it costs 1.3 to 3 times the ``"gate"`` walk, so the default did not
change: the rule that is better on both axes does not exist here. Use
``"contribution"`` when the export must track the soft model and the extra
dot products are affordable; keep ``"gate"`` when the printed rules must be
the routing. Walking down by the larger child probability is not a fourth
rule: a sigmoid exceeds one half exactly when its argument is positive, so
it is ``"gate"``. The script is ``benchmarks/hard_rules.py``.

*Reference:* İrsoy, O., Yıldız, O. T. and Alpaydın, E. (2012). Soft Decision
Trees. *ICPR*, 1819-1822.

Shipping the tree without PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``to_numpy()`` is the other export. It is the *same* model, mixture over
leaves included: predictions agree with the torch model to float32
precision, and nothing in the prediction path imports torch. The object
serialises to JSON and back, so a fitted tree can be frozen for audit or
handed to a service that installs only numpy.

.. code-block:: python

   from neural_trees import NumpySoftTree

   npt = model.to_numpy(feature_names=feature_names)
   npt.to_json("tree.json")

   # elsewhere, without torch
   npt = NumpySoftTree.from_json("tree.json")
   npt.predict_proba(X_new)

The JSON carries a ``format`` field (``neural-trees/soft-tree/1``) so a
later layout change can be detected rather than misread.

Regression
^^^^^^^^^^

:class:`~neural_trees.SoftDecisionTreeRegressor` is the same tree with a
value in each leaf instead of a class distribution; the prediction is the
arrival-probability-weighted average of the leaf values, trained on the
weighted squared error. It accepts multi-output targets (``y`` of shape
``(n, k)``), ``sample_weight`` and early stopping, and passes the scikit-learn
regressor checks except the one no mini-batch learner can pass. Targets are
centred and scaled internally and leaves start at the target mean, so an
untrained tree predicts the mean.

Growth works as on the classifier: ``growth="incremental"`` adds a level at
a time and keeps it only if held-out squared error improves,
``growth="per_leaf"`` splits the leaf carrying the most weighted squared
error, and ``growth_init`` pushes a split's two children apart along the
leaf's residual (``y`` minus prediction) or at random, because identical
children would leave the new gate with a zero gradient here exactly as they
do for the classifier. Both exports exist too: ``to_numpy()`` is the same
model in numpy with a JSON round trip, ``to_hard_tree()`` a
:class:`~neural_trees.HardRegressionTree` that prints ``value = ...`` at
each leaf. The hard reading of a regressor can sit further from the soft
model than the classifier's does, because soft leaves compensate for gates
that share mass and the hard walk lands on those attenuated values; measure
``score`` on held-out data before relying on it. ``explain()`` is not
available for regression yet.

.. code-block:: python

   from neural_trees import SoftDecisionTreeRegressor

   reg = SoftDecisionTreeRegressor(depth=3, max_epochs=100, random_state=0).fit(X_train, y_train)
   reg.score(X_test, y_test)      # R^2
   reg.get_leaf_values()          # in target units

   grown = SoftDecisionTreeRegressor(depth=5, max_epochs=150, growth="per_leaf", random_state=0)
   grown.fit(X_train, y_train)
   grown.growth_, grown.tree_depth_
   print(grown.to_hard_tree().export_text(feature_names=feature_names))
   grown.to_numpy().to_json("reg.json")

Multivariate and omnivariate trees
----------------------------------

:class:`~neural_trees.MultivariateDecisionTree` splits on a linear combination
of features at every node.

:class:`~neural_trees.OmnivariateDecisionTree` chooses, at each node
independently, between a univariate threshold, a linear split and a nonlinear
split, so that simple regions of the space get simple splits.

.. code-block:: python

   from neural_trees import OmnivariateDecisionTree

   tree = OmnivariateDecisionTree(max_depth=3, selection="accuracy")
   tree.fit(X_train, y_train)
   print(tree.get_split_type_distribution())

``selection="accuracy"`` picks whichever split type scores best in
cross-validation at that node. ``selection="test"`` instead demands that the
more expressive split be *significantly* better by the combined 5x2cv F test,
and otherwise keeps the simpler one. The second option is opt-in for a reason;
see :doc:`design_decisions`.

*Reference:* Yıldız, O. T. and Alpaydın, E. (2001). Omnivariate Decision Trees.
*IEEE Transactions on Neural Networks*, 12(6), 1539-1546.

Hierarchical mixture of experts
-------------------------------

A hierarchical mixture of experts is a soft tree whose leaves are models rather
than class distributions, with gating networks routing input through the
hierarchy. :class:`~neural_trees.HierarchicalMixtureOfExperts` implements it
with subtree dropout, which drops whole subtrees during training so that no
branch is relied on exclusively.

.. code-block:: python

   from neural_trees import HierarchicalMixtureOfExperts

   moe = HierarchicalMixtureOfExperts(depth=2, dropout_rate=0.2, random_state=0)
   moe.fit(X_train, y_train)

``early_stopping=True`` holds out ``validation_fraction`` of the training
data, stratified and drawn with ``random_state``, stops after
``n_iter_no_change`` epochs without a validation improvement and restores the
best epoch; ``n_iter_`` says where it stopped. With ``max_epochs=400`` it
stopped at 62 epochs on Breast Cancer and 71 on Digits for the same accuracy,
six to seven times faster. On a separable set like Wine the validation loss
keeps falling and it correctly does not stop.

``to_hard_router()`` exports the trained router the way ``to_hard_tree()``
exports a soft tree.

*Reference:* İrsoy, O. and Alpaydın, E. (2021). Dropout Regularization in
Hierarchical Mixture of Experts. *Neurocomputing*, 419, 148-156.

Constructive networks
---------------------

:class:`~neural_trees.GALNetwork` grows a hidden layer while it trains, adding
units when the error stops improving and pruning units that stop contributing.

.. code-block:: python

   from neural_trees import GALNetwork

   net = GALNetwork(max_epochs=150, growth_init="residual", random_state=0)
   net.fit(X_train, y_train)
   print(net.n_hidden_final_, net.architecture_history_)

``growth_init`` decides how a new unit arrives. ``"random"`` is the original
scheme. ``"residual"`` fits the candidate unit to the residual error of the
frozen network first and installs it with zero outgoing weights, in the manner
of cascade-correlation. It reliably produces a smaller network (Iris: 6.9
hidden units against 17.2) but does not reliably produce a more accurate one;
on Digits it is significantly less accurate than ``"random"``. The measurements
are in :doc:`design_decisions`.

*Reference:* Alpaydın, E. (1994). GAL: Networks that grow when they learn and
shrink when they forget. *IJPRAI*, 8(1), 391-414.

Classical baselines
-------------------

:class:`~neural_trees.WeightedKNN` implements condensed nearest neighbour with
voting over several independently condensed prototype sets, which is the idea of
the paper it cites rather than a single condensed set.

:class:`~neural_trees.NaiveBayesClassifier` is a Gaussian naive Bayes with
normalised ``predict_log_proba``.

Sparse input
------------

:class:`~neural_trees.NaiveBayesClassifier` and
:class:`~neural_trees.WeightedKNN` accept ``scipy.sparse`` CSR matrices in
``fit`` and ``predict`` (other sparse formats are converted to CSR). Their
arithmetic has a sparse form: weighted sums and second moments for the naive
Bayes statistics, ``|a|^2 + |b|^2 - 2 a.b`` for Euclidean distances. Nothing
is densified except the Manhattan distance, which has no such identity and
densifies the store in bounded chunks. Predictions on ``X`` and on
``csr_matrix(X)`` are identical.

It is cheaper by a wide margin. On 20 newsgroups (token counts, 130 107
features), measured with ``tracemalloc``:

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14

   * - Setting
     - fit
     - predict
     - peak memory
     - accuracy
   * - Naive Bayes, sparse, 11 314 documents
     - 0.04 s
     - 0.03 s
     - 25 MB
     - 0.773
   * - Naive Bayes, sparse, 2 000 documents
     - 0.03 s
     - 0.03 s
     - 25 MB
     - 0.417
   * - Naive Bayes, dense, 2 000 documents
     - 2.0 s
     - 60 s
     - 250 MB
     - 0.417
   * - KNN, sparse, 1 000 stored rows, 1 000 features
     - 0.001 s
     - 0.03 s
     - 6 MB
     - 0.160
   * - KNN, dense, same
     - 0.001 s
     - 0.6 s
     - 1 602 MB
     - 0.160
   * - KNN, sparse, 1 000 rows, all 130 107 features
     - 0.001 s
     - 0.03 s
     - 6 MB
     - 0.175

The dense naive Bayes could not be run on the full training set at all (it
would be 11.8 GB), and the dense KNN broadcasts a ``(queries, store,
features)`` block, which is why its comparison is at 1 000 features. The
script is ``benchmarks/sparse_input.py``.

The torch-backed models and the two multivariate trees need dense input and
say so: passing a sparse matrix raises ``TypeError`` naming the model and the
fix (``X.toarray()``), rather than scikit-learn's generic message.

Weighting
---------

:class:`~neural_trees.SoftDecisionTree`,
:class:`~neural_trees.HierarchicalMixtureOfExperts`,
:class:`~neural_trees.GALNetwork`, :class:`~neural_trees.NaiveBayesClassifier`
and :class:`~neural_trees.WeightedKNN` accept ``sample_weight`` in ``fit``;
the first three also take a ``class_weight`` parameter. For
:class:`~neural_trees.GALNetwork` the weights are carried into the growth
criterion as well, so that weighting changes what the network *builds* and not
only what it learns.

What a weight means differs by model, and the docstrings say which. In naive
Bayes it is exact: priors and per-class statistics are weighted totals and
moments, so integer weights reproduce the fit on repeated rows. In the
nearest-neighbour rule a weight scales the neighbour's vote and a zero weight
drops the row; that is *not* the same as repeating rows, because a repeated row
fills several of the ``k`` neighbour slots while a weighted one fills one. The
three mini-batch learners batch a repeated dataset differently, so for them
the equivalence holds only in expectation.

:class:`~neural_trees.MultivariateDecisionTree` and
:class:`~neural_trees.OmnivariateDecisionTree` take ``sample_weight`` and
``class_weight`` as well, and the weights reach inside the split search, not
only the leaves. In the multivariate tree they enter the discriminant at every
node (weighted class means, pooled covariance and prior ratio), the Gini
decrease and the leaf distributions. In the omnivariate tree they enter the
two-group construction, the model selection (weighted fold accuracy, and the
candidate's own fit where it accepts weights: the stump does, scikit-learn's
LDA does not, its MLP only from 1.7) and the leaf distributions. Both count
rows, not weight, for ``min_samples_split`` and ``min_samples_leaf``, as
scikit-learn's trees do, and both take ``min_weight_fraction_leaf`` for the
case where a down-weighted class would otherwise keep leaves of its own.

Whether weighting changes the *kind* of split the omnivariate tree chooses
was the open question. Measured on three datasets with one class cut to a
tenth (three seeds of five-fold cross-validation; recall is of that class):

.. list-table::
   :header-rows: 1
   :widths: 26 16 14 14 30

   * - Dataset, model
     - ``class_weight``
     - accuracy
     - recall
     - splits per fit (uni / lin / nonlin)
   * - Breast Cancer, multivariate
     - none / balanced
     - 0.983 / 0.959
     - 0.82 / 0.52
     -
   * - Breast Cancer, omnivariate
     - none / balanced
     - 0.984 / 0.984
     - 0.77 / 0.85
     - 0.3, 1.3, 0.0 / 1.2, 1.0, 0.4
   * - Wine, multivariate
     - none / balanced
     - 0.978 / 0.990
     - 0.80 / 1.00
     -
   * - Wine, omnivariate
     - none / balanced
     - 0.970 / 0.985
     - 0.73 / 0.73
     - 0.2, 1.3, 0.7 / 0.1, 1.9, 0.1
   * - Digits (depth 6), multivariate
     - none / balanced
     - 0.938 / 0.947
     - 0.88 / 0.95
     -
   * - Digits (depth 6), omnivariate
     - none / balanced
     - 0.950 / 0.955
     - 0.78 / 0.90
     - 2.4, 3.4, 7.7 / 2.3, 2.8, 7.7

So yes, on Breast Cancer balancing moves the omnivariate tree from linear
splits toward stumps and MLPs while raising minority recall, and on Wine it
moves it the other way, toward linear splits; on Digits the mix hardly
changes. For the multivariate tree balancing helped on Wine and Digits and
hurt on Breast Cancer, where a 6% class weighted up sixteenfold pulled the
root hyperplane too far. The setting is a trade, not a default.

All six estimators with ``sample_weight`` therefore fail exactly one
scikit-learn check, ``check_sample_weight_equivalence_on_dense_data`` (and its
sparse twin for the KNN), which requires weighting a sample to be identical
to repeating it, except naive Bayes, which passes it. For the two trees the
reason is the check's data: 15 rows and 30 features, so every node
discriminant is under-determined and the weighted and unweighted solvers agree
only to floating point.

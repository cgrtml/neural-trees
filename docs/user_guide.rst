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

Per-leaf growth is usually the one to reach for when the tree should stay small
and readable. On a synthetic problem with 800 samples and 20 features it reached
0.885 with 3.7 splits on average, against 0.832 for a complete depth-6 tree
using all 63.

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

*Reference:* İrsoy, O., Yıldız, O. T. and Alpaydın, E. (2012). Soft Decision
Trees. *ICPR*, 1819–1822.

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
*IEEE Transactions on Neural Networks*, 12(6), 1539–1546.

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

``to_hard_router()`` exports the trained router the way ``to_hard_tree()``
exports a soft tree.

*Reference:* İrsoy, O. and Alpaydın, E. (2021). Dropout Regularization in
Hierarchical Mixture of Experts. *Neurocomputing*, 419, 148–156.

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
of cascade-correlation. On Iris over three seeds of five-fold cross-validation
the residual variant reached 0.956 with 6.6 hidden units against 0.938 with
17.4.

*Reference:* Alpaydın, E. (1994). GAL: Networks that grow when they learn and
shrink when they forget. *IJPRAI*, 8(1), 391–414.

Classical baselines
-------------------

:class:`~neural_trees.WeightedKNN` implements condensed nearest neighbour with
voting over several independently condensed prototype sets, which is the idea of
the paper it cites rather than a single condensed set.

:class:`~neural_trees.NaiveBayesClassifier` is a Gaussian naive Bayes with
normalised ``predict_log_proba``.

Weighting
---------

:class:`~neural_trees.SoftDecisionTree`,
:class:`~neural_trees.HierarchicalMixtureOfExperts` and
:class:`~neural_trees.GALNetwork` accept ``sample_weight`` in ``fit`` and a
``class_weight`` parameter. For :class:`~neural_trees.GALNetwork` the weights
are carried into the growth criterion as well, so that weighting changes what
the network *builds* and not only what it learns.

These three estimators fail exactly one scikit-learn check,
``check_sample_weight_equivalence_on_dense_data``, which requires weighting a
sample to be bit-identical to repeating it. No stochastic mini-batch learner can
satisfy it. The other four estimators pass all 55 checks.

Design decisions
================

The implementations begin from published algorithms and depart from them where
measurement justified it. This page records the departures and the numbers
behind them, so that a reader can disagree with a choice on the evidence rather
than on taste. Every figure below comes from running the code: three seeds of
stratified five-fold cross-validation, features standardised on the training
fold only, both arms of each comparison sharing seeds and folds.

Growth must not preserve the function exactly
---------------------------------------------

Deepening a soft tree looks like it should be done without disturbing what the
tree has learned. Turn every leaf into a gate with all-zero parameters, so it
sends half the arriving mass each way, and give both children the parent's class
distribution. The mixture is unchanged and the extra capacity is available.

This does not work, and the reason is exact rather than statistical. With
:math:`Q_L = Q_R` the subtree's output does not depend on the gate at all, so

.. math::

   \frac{\partial}{\partial g}\left( g\,Q_R + (1-g)\,Q_L \right) = Q_R - Q_L = 0,

and the gate's gradient is identically zero. With the gate at :math:`g = 1/2`
the two children receive identical gradients, so they stay identical, so the
gate's gradient stays zero. The state reproduces itself at every step: it is a
fixed point of the optimiser, not a slow start. The added level is dead weight
while costing full parameters and compute.

Perturbing the child distributions slightly at insertion breaks the symmetry.
The function is then preserved only approximately, which is the price of the
level being able to learn anything.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Growth
     - Iris
     - Wine
   * - Exactly function-preserving
     - 0.753
     - 0.754
   * - Perturbed by 0.05
     - 0.942
     - 0.979
   * - Perturbed by 0.2 (the default)
     - 0.942
     - 0.979
   * - Perturbed by 0.5
     - 0.938
     - 0.981
   * - Same depth, trained from scratch
     - 0.958
     - 0.977

The perturbation needs to be nonzero and otherwise needs no tuning: a tenfold
change in it moves accuracy by four tenths of a point on Iris and two tenths on
Wine, while setting it to zero costs about twenty. A regression test asserts
that the zero gradient exists when the symmetry is not broken.

New units should be fitted to the residual
------------------------------------------

The GAL network installs a new hidden unit with random weights. Such a unit
perturbs every output logit the moment it arrives, damaging a network that was
just judged to have stopped improving, and then has to be trained from noise.

``growth_init="residual"`` freezes the network, trains the candidate unit to
correlate with the residual error, and installs it with zero outgoing weights.
That is safe here — unlike the tree case above — because the outgoing weights
have a nonzero gradient as soon as the incoming weights are informative.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - New-unit initialisation
     - Accuracy
     - Hidden units
   * - ``"random"``
     - 0.938
     - 17.4
   * - ``"residual"``
     - 0.956
     - 6.6

The size difference is the more interesting half. Random units arrive unhelpful,
the error does not fall, and the growth criterion fires again, so the network
grows *because* growth is not working.

Node-level significance does not compose
----------------------------------------

The omnivariate tree picks each node's split type by cross-validated accuracy,
which prefers a nonlinear split for any improvement at all, however small.
Demanding statistical significance instead sounds strictly better: simpler nodes
at little cost.

Measured on Breast Cancer at maximum depth 3, it is worse on both axes.

.. list-table::
   :header-rows: 1
   :widths: 26 16 14 14 14 16

   * - ``selection``
     - Accuracy
     - Nodes
     - Univariate
     - Linear
     - Nonlinear
   * - ``"accuracy"``
     - 0.971
     - 4.1
     - 8
     - 0
     - 15
   * - ``"test"``
     - 0.960
     - 7.3
     - 20
     - 14
     - 13

Per node the rule does exactly what it was asked to do: nonlinear splits fall
from 15 to 13 and univariate splits rise from 8 to 20. But a simpler split
separates its node's data less cleanly, so its children inherit harder problems
and must themselves be split, and the tree grows from 4.1 nodes to 7.3 while
losing a point of accuracy. Parsimony enforced locally is paid for globally.

On Iris and Wine the rule never fires, because the test needs a minimum sample
count per node and almost every node below the root falls under it. A
significance criterion is unavailable exactly where a recursive method does most
of its work, since nodes get smaller with depth by construction.

This is why ``selection="test"`` is opt-in rather than the default.

The hard export reports its agreement
-------------------------------------

``to_hard_tree()`` could present itself as a faithful re-encoding of the soft
model. It is not one: a mixture over leaves is not a single path, and reading
each gate as a hard decision discards the mixing. Over fifteen folds on Wine the
export agreed with its source model on 0.998 of held-out samples on average,
with 0.971 in the worst fold, and predicted about four times faster. The method
reports that agreement rather than assuming it.

Four models that did not work
-----------------------------

An early release shipped four models that were broken, and none of them
announced it. They were found by writing tests, not by reading code.

:class:`~neural_trees.HierarchicalMixtureOfExperts`
  Could not take a single gradient step. Child weights were written in place
  into one preallocated tensor, which autograd had saved for the multiplication
  backward, so every call to ``backward()`` raised.

:class:`~neural_trees.OmnivariateDecisionTree`
  Computed its node classifier's decision, discarded it, and always descended
  right. Every sample reached the same leaf; five-fold accuracy at depth three
  was 0.000 on Iris.

:class:`~neural_trees.GALNetwork`
  Took one full-batch gradient step per epoch, so growth and pruning decisions
  were made on a network that had barely moved from its initialisation. It
  scored 0.333 on three well-separated blobs, which is chance for three classes.

:class:`~neural_trees.WeightedKNN`
  Condensed nearest neighbour made a single pass over the training set, leaving
  a prototype store that did not classify that set correctly.

The test suite now has 283 tests at 95% coverage, gated in CI, and every
estimator is checked against
:func:`sklearn.utils.estimator_checks.check_estimator`.

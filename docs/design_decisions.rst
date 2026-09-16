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
gate's gradient stays zero. The state reproduces itself at every step. It is a fixed point of the
optimiser. The added level is dead weight while costing full parameters and
compute. We checked this numerically rather than assuming it: in single
precision the new gates' gradients are exactly zero and sibling leaves receive
bitwise identical gradients, and the regression test asserts exact equality.

Perturbing the child distributions slightly at insertion breaks the symmetry.
The function is then preserved only approximately, which is the price of the
level being able to learn anything.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Growth
     - Iris
     - Wine
     - Digits
   * - Exactly function-preserving
     - 0.762 ± 0.103
     - 0.786 ± 0.145
     - 0.369 ± 0.035
   * - Perturbed by 0.05
     - 0.942 ± 0.041
     - 0.979 ± 0.020
     - 0.917 ± 0.034
   * - Perturbed by 0.2 (the default)
     - 0.942 ± 0.044
     - 0.979 ± 0.020
     - 0.921 ± 0.027
   * - Perturbed by 0.5
     - 0.938 ± 0.050
     - 0.983 ± 0.018
     - 0.917 ± 0.027
   * - Same depth, trained from scratch
     - 0.958 ± 0.032
     - 0.978 ± 0.016
     - 0.925 ± 0.029

Exact preservation costs about twenty points on Iris and Wine and fifty-six on
Digits, and it inflates the fold-to-fold standard deviation as well. The
perturbation needs to be nonzero and otherwise needs no tuning: a tenfold change
in it moves the mean by at most four tenths of a point on any of the three. The
combined 5x2cv F test against the default perturbation gives p = 0.0596 on Iris,
which we do not call significant, p = 0.0097 on Wine and p < 0.0001 on Digits.

New units should be fitted to the residual
------------------------------------------

The GAL network installs a new hidden unit with random weights. Such a unit
perturbs every output logit the moment it arrives, damaging a network that was
just judged to have stopped improving, and then has to be trained from noise.

``growth_init="residual"`` freezes the network, trains the candidate unit to
correlate with the residual error, and installs it with zero outgoing weights.
That is safe here, unlike the tree case above, because the outgoing weights
have a nonzero gradient as soon as the incoming weights are informative.

.. list-table::
   :header-rows: 1
   :widths: 16 20 22 22 20

   * - Dataset
     - New unit
     - Accuracy
     - Hidden units
     - 5x2cv F
   * - Iris
     - ``"random"``
     - 0.936 ± 0.046
     - 17.2 ± 2.4
     - p = 0.181
   * -
     - ``"residual"``
     - 0.956 ± 0.037
     - 6.9 ± 1.5
     -
   * - Wine
     - ``"random"``
     - 0.985 ± 0.015
     - 6.3 ± 2.2
     - p = 0.633
   * -
     - ``"residual"``
     - 0.985 ± 0.015
     - 4.3 ± 0.5
     -
   * - Digits
     - ``"random"``
     - 0.949 ± 0.013
     - 7.5 ± 0.6
     - p = 0.0012
   * -
     - ``"residual"``
     - 0.928 ± 0.015
     - 5.7 ± 0.5
     -

The size effect is consistent and the accuracy effect is not. Every dataset
gives a smaller network: random units arrive unhelpful, the error does not
fall, and the growth criterion fires again, so the network grows *because*
growth is not working. Accuracy rose on Iris and was identical on Wine, and the
F test rejects equality on neither; on Digits it fell by 2.1 points and the test
does reject equality, in favour of random initialisation. Read this as a
parsimony intervention, not an accuracy intervention. It is still the default
because a smaller network is what the growth mechanism is for, but if the
problem has enough classes and features to need the extra units, ``"random"``
is the setting to try.

Node-level significance does not compose
----------------------------------------

The omnivariate tree picks each node's split type by cross-validated accuracy,
which prefers a nonlinear split for any improvement at all, however small.
Demanding statistical significance instead sounds strictly better: simpler nodes
at little cost.

Measured at maximum depth 3, it is worse on both axes on both datasets.

.. list-table::
   :header-rows: 1
   :widths: 24 24 26 26

   * - Dataset
     - ``selection``
     - Accuracy
     - Nodes
   * - Breast Cancer
     - ``"accuracy"``
     - 0.971 ± 0.018
     - 3.8 ± 1.0
   * -
     - ``"test"``
     - 0.959 ± 0.020
     - 8.1 ± 2.5
   * - Digits
     - ``"accuracy"``
     - 0.510 ± 0.083
     - 10.1 ± 1.7
   * -
     - ``"test"``
     - 0.492 ± 0.076
     - 11.5 ± 2.3

The rule does what it was asked to do at each node and the tree is worse for
it. A simpler split separates its node's data less cleanly, so its children
inherit harder problems and must themselves be split, and the tree roughly
doubles on Breast Cancer while losing a point of accuracy. Parsimony enforced
locally is paid for globally. The Digits accuracies are low for a reason
unrelated to the rule: a depth-3 tree has at most eight leaves and Digits has
ten classes, so the row is a comparison between the rules, not a claim about
how well omnivariate trees classify digits.

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

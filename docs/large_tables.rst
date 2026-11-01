Large tables, ensembles and missing values
==========================================

A single soft tree is a strong model on small tables: on the eight datasets
under 1 000 rows of the :doc:`benchmarks` it matches or beats XGBoost. On
larger tables it is not. With 5 000 to 20 000 training rows a depth-6 to
depth-10 tree trained on the labels trails 300 boosted trees by 3 to 7
points, and neither a deeper tree nor a longer fit closes that: a sum of
hundreds of trees has far more capacity than one tree, and one tree is the
point of this library. This page is about what does help, measured, and
about what was tried and did not.

.. contents::
   :local:
   :depth: 1


Train one tree from a strong model
----------------------------------

The probabilities of a boosted model are a better training signal for a
tree than the labels it was fitted on. :class:`~neural_trees.DistilledTree`
fits a teacher on ``(X, y)`` and then one soft tree on the teacher's class
probabilities; the deployed model, its explanations and its exports are that
single tree.

.. code-block:: python

   from xgboost import XGBClassifier
   from neural_trees import DistilledTree

   model = DistilledTree(teacher=XGBClassifier(n_estimators=300, max_depth=6))
   model.fit(X_train, y_train)
   model.predict(X_test)          # one soft tree
   model.fidelity(X_test)         # agreement with the teacher
   model.explain(X_test[:1])      # rules and a counterfactual, from the tree

.. SCALE-RESULTS: filled in from benchmarks/damitma_olcek.py

Most of the gain is regularisation: mixing the true labels back in
(``alpha=0.5``) brought the overfitting back and lost most of it (jannis,
20 000 rows: 0.733 against 0.768 for the teacher's probabilities alone).

Unlabelled rows
~~~~~~~~~~~~~~~

Labels are often scarcer than rows. ``fit(X, y, X_unlabeled=X_more)`` lets
the teacher label further rows for the student; the teacher itself still
sees only ``(X, y)``. Because the teacher cannot use those rows, compare
this with the teacher only as an answer to a different question: what one
tree can reach when unlabelled data is available.


A readable copy of a model you already run
------------------------------------------

When the ensemble is already in production and the question is "why did it
decide this", :func:`~neural_trees.distill` fits a soft tree to the fitted
model's probabilities and reports how often the tree agrees with it.

.. code-block:: python

   from neural_trees import SoftDecisionTree, distill

   result = distill(xgb_model, X_train, X_eval=X_test,
                    student=SoftDecisionTree(depth=6, init="cart", split_features=3,
                                             learn_temperature=True, max_epochs=60))
   print(result)   # fidelity, fidelity of the hard rules, splits, features per split

Measured on 24 datasets with XGBoost as the teacher (``benchmarks/distill.py``,
agreement with the teacher on held-out rows, compared with a CART tree of
depth 6 trained on the same teacher labels):

==============================  ====================  ===========================
features per split              fidelity vs CART       Wilcoxon p (24 datasets)
==============================  ====================  ===========================
1, started from CART            +1.5 points            0.002
2, started from CART            +2.5 points            0.003
3, started from CART            +3.2 points            0.0008
unrestricted                    +3.2 points            0.02
==============================  ====================  ===========================

Three features per split are as faithful as an unrestricted split. Starting
from CART matters here: pruning a dense student down to the same budget
instead lost to CART at one and two features per split (-6.9 and -2.6
points).


Starting from CART
------------------

``SoftDecisionTree(init="cart")`` copies a CART tree of the same depth into
soft gates (``sigmoid(init_sharpness * (x[f] - t) / std(x[f]))``) and fine-tunes
it, so greedy search places sharp thresholds and gradients then move them
jointly. ``init_prune=True`` (the default) prunes that CART tree with a
cost-complexity strength chosen by 3-fold cross-validation, because the soft
copy cannot remove splits it was handed. Use it with ``learn_temperature=True``.
On the 24 datasets (``benchmarks/cart_init.py --prune``, 3 seeds x 5 folds,
sharpness chosen inside each training fold) it beats CART of the same depth by
5.0 points (21 wins, 2 losses, p=0.0001), ties the per-leaf soft tree overall,
is ahead of it above 1 000 rows (+1.1, not significant) and behind it below
(-1.5). It also repairs the sharp-threshold failure of soft trees: on
wall-robot-navigation 0.976 against 0.876 for the per-leaf tree.

Rule of thumb from these measurements: below about 1 000 rows grow a soft
tree from scratch (``growth="per_leaf"``); above, start from CART.


Missing values
--------------

``fit``, ``predict``, ``explain`` and every export accept ``NaN``. A missing
cell becomes 0 and every feature that had missing values in training gets an
indicator input. For a linear gate this is exactly a learned default per
split: when the feature is missing, its term is replaced by a free weight of
that gate, the soft counterpart of the default direction a boosted tree
learns. Explanations name the indicator ("income is missing") and never
propose a counterfactual on a missing value.


Tried and not kept
------------------

All on 20 000-row training sets of electricity, MiniBooNE and jannis, one
split (``benchmarks/gelistir.py``):

* Sharpening the gates after training (a scale chosen on held-out rows): the
  chooser kept the trained scale almost every time.
* Reading a depth-10 tree as hard rules: 1 to 2 points below the soft tree.
* Training three times longer: no gain.
* Depth 10 to 12 instead of 6 to 8: small gains on two datasets, a failed fit
  on a third, five to ten times the cost.
* Linear models in the leaves: overfits (jannis train 0.99, test 0.70).
* Growing the student leaf by leaf on the teacher's probabilities: below the
  CART-started student on four of five datasets.
* Bagging 25 soft trees (see :doc:`benchmarks`): halves the gap to boosting at
  160 times its cost and leaves no single tree to read.

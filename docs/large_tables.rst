Large tables, ensembles and missing values
==========================================

A single soft tree is a strong model on small tables: on the eight datasets
under 1 000 rows of the :doc:`benchmarks` it matches or beats XGBoost. On
larger tables it is not. With 5 000 to 20 000 training rows a depth-6 to
depth-10 tree trained on the labels trails 300 boosted trees by 2 to 7
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

   model = DistilledTree(teacher=XGBClassifier(n_estimators=300, max_depth=6),
                         synthetic=2)     # MUNGE rows for the teacher to label (default)
   model.fit(X_train, y_train)
   model.predict(X_test)          # one soft tree
   model.fidelity(X_test)         # agreement with the teacher
   model.explain(X_test[:1])      # rules and a counterfactual, from the tree

Measured on six OpenML tables from the Grinsztajn et al. (2022) benchmark
(``benchmarks/damitma_olcek.py``; fixed test set of up to 20 000 rows;
three training samples per size, one at 100 000 rows, marked ``*``). Every
tree is ``SoftDecisionTree(depth=8, init="cart", learn_temperature=True,
max_epochs=60)``; the teacher is 300 XGBoost trees of depth 6.

================  =========  =========  ===============  ===========  ===========  =================
dataset           rows       XGBoost    tree on labels   distilled    + MUNGE      + 2n unlabelled
================  =========  =========  ===============  ===========  ===========  =================
electricity       5 000      0.841      0.797            0.797        0.805        0.805
electricity       20 000     0.873      0.810            0.810        0.814        0.814
MagicTelescope    5 000      0.850      0.830            0.835        0.844        0.844
jannis            5 000      0.764      0.717            0.715        0.743        0.753
jannis            20 000     0.786      0.743            0.770        0.780        0.786
MiniBooNE         5 000      0.930      0.902            0.904        0.911        0.920
MiniBooNE         20 000     0.939      0.920            0.924        0.929        0.931
covertype         5 000      0.788      0.759            0.765        0.772        0.774
covertype         20 000     0.822      0.790            0.787        0.799        0.797
covertype         100 000 *  0.847      0.818            0.810        0.820        0.821
Higgs             5 000      0.695      0.659            0.674        0.690        0.700
Higgs             20 000     0.714      0.688            0.709        0.716        0.717
Higgs             100 000 *  0.731      0.718            0.727        0.726        0.730
================  =========  =========  ===============  ===========  ===========  =================

Accuracy on the test set. Over the 13 cells:

* Distilled against the tree on labels: +0.6 points on average, higher in 7
  cells, lower in 3 (Wilcoxon p=0.09). It helps where the tree overfits the
  labels (Higgs +0.9 to +2.1, jannis at 20 000 rows +2.7) and not where it
  is short of capacity (electricity, covertype): the teacher's probabilities
  are a better target, not a bigger tree.
* With synthetic rows (MUNGE, two passes, the ``DistilledTree`` default;
  ``benchmarks/munge.py``): +1.5 points over the tree on labels and +0.9
  over plain distillation, higher in 12 of 13 cells and level in the
  last (p=0.0002 and 0.0005; level at 100 000 rows, where the training
  rows are already many), and the mean gap to XGBoost falls from 3.3 to 1.8 points. No
  data beyond XGBoost's is used, so this column compares fairly with it.
  The cost is about three times the fit time of plain distillation.
* With twice as many real unlabelled rows: +1.85 points over the tree on
  labels, higher in all 13 cells (p=0.0002), a mean gap to XGBoost of 1.45
  points, and on Higgs at 5 000 and 20 000 rows and on jannis at 20 000 the
  single tree reaches its teacher. Read the section on unlabelled rows
  before comparing this column with XGBoost.

Readable rules cost little here. The same MUNGE student limited to three
features per split (``split_features=3``, started from CART; all 13 cells,
``benchmarks/munge_seyrek.py``) lost 0.25 points against the unrestricted
student (2 cells higher, 8 lower, p=0.09) and was 1.3 points above the
unrestricted tree trained on the labels (11 higher, 2 lower, p=0.008), 2.0
points behind XGBoost. The cost grows with the data: 0.13 points up to
20 000 rows, 1.3 on covertype and 0.5 on Higgs at 100 000, where many rows
can feed many features per split.

.. code-block:: python

   from neural_trees import DistilledTree, SoftDecisionTree

   model = DistilledTree(student=SoftDecisionTree(
       depth=8, init="cart", split_features=3, learn_temperature=True,
       max_epochs=60, batch_size=256))

It fits in about 1.5 times the time of the unrestricted student.

The same check through the library itself (``DistilledTree`` with its default
teacher, scikit-learn's ``HistGradientBoostingClassifier``, 20 000 rows of
jannis, Higgs, covertype and electricity, three seeds;
``benchmarks/munge_lib.py``) gave a smaller gain: +0.5 points for
``synthetic=2`` over ``synthetic=0``, higher in 8 of 12 fits and lower in
none (p=0.003). That teacher stops early on its own validation split and is
already a smoother target, so plain distillation from it is stronger
(jannis 0.780, Higgs 0.711) and the synthetic rows help mostly where the
tree is short of capacity (covertype +1.0, electricity +0.8).

More rows for the teacher to label is what moves the student: the
teacher's probabilities on the training rows alone are a better target than
the labels, but they are a target on the same few points. MUNGE
(Bucila, Caruana and Niculescu-Mizil, *Model compression*, KDD 2006) makes
each new row from a training row by redrawing about half of its features
near the values of the row's nearest neighbour, so the new rows stay where the
data is and the teacher's answers there are informative.

The gain of plain distillation is regularisation: mixing the true labels back in
(``alpha=0.5``) brought the overfitting back and lost most of it (jannis,
20 000 rows: 0.733 against 0.768 for the teacher's probabilities alone).

Below 5 000 rows
~~~~~~~~~~~~~~~~

On the 24 datasets of :doc:`benchmarks` (same folds; library defaults, the
HistGradientBoosting teacher; ``benchmarks/damitma_kucuk.py``) the distilled
tree is the wrong choice under 1 000 rows: 1.2 points below the per-leaf
soft tree grown from scratch, lower on 7 of 8 datasets (p=0.016). From 1 000
to 5 000 rows it is 1.05 points above the per-leaf tree (9 higher, 6 lower,
not significant) and 1.3 points below XGBoost, about where the CART-started
tree alone already is. The distillation earns its cost on the larger tables
above, not here.

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
tree from scratch (``growth="per_leaf"``); from 1 000 to about 5 000 start
from CART; above that, distil (``DistilledTree``).


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
* A depth-10 student on the MUNGE rows instead of depth 8 (5 000 and 20 000
  rows, three seeds; ``benchmarks/munge_derin.py``): +0.05 points on
  average, 3 cells higher and 3 lower. It helps where one tree is short of
  capacity and rows are many (electricity and covertype at 20 000 rows,
  +0.8 and +0.7) and loses on the small and noisy cells (Higgs at 5 000
  rows, -0.7), at about three times the fit time.
* Four MUNGE passes instead of two (the 11 cells up to 20 000 rows;
  ``MUNGE_PASSES=4 benchmarks/munge.py``): +0.15 points, 5 cells higher and
  2 lower (p=0.21), at about 1.8 times the fit time.
* Linear models in the leaves: overfits (jannis train 0.99, test 0.70).
* Growing the student leaf by leaf on the teacher's probabilities: below the
  CART-started student on four of five datasets.
* Bagging 25 soft trees (see :doc:`benchmarks`): halves the gap to boosting at
  160 times its cost and leaves no single tree to read.

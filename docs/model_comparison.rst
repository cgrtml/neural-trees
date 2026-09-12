Comparing classifiers
=====================

An accuracy difference on a single train/test split says very little. Split the
data differently and the ordering can reverse. The package ships three tests for
deciding whether a difference is real, in ``neural_trees``:

.. code-block:: python

   from neural_trees import combined_5x2cv_f_test, mcnemar_test, paired_t_test

Combined 5x2cv F test
---------------------

The default choice when comparing two learning *algorithms* on one dataset. It
runs five replications of two-fold cross-validation, giving ten accuracy
differences, and combines them into an F statistic. It is designed to have a
Type I error close to the nominal level, which the naive paired t-test over
k-fold splits does not, because those splits share training data and the
differences are not independent.

.. code-block:: python

   from sklearn.tree import DecisionTreeClassifier
   from neural_trees import SoftDecisionTree, combined_5x2cv_f_test

   statistic, p_value = combined_5x2cv_f_test(
       SoftDecisionTree(depth=4, random_state=0),
       DecisionTreeClassifier(random_state=0),
       X, y, random_state=0,
   )

*Reference:* Alpaydın, E. (1999). Combined 5x2cv F test for comparing supervised
classification learning algorithms. *Neural Computation*, 11(8), 1885–1892.

McNemar's test
--------------

For comparing two *fitted* classifiers on one held-out set. It looks only at the
samples the two models disagree on and asks whether the disagreement is
lopsided. Use it when refitting is expensive or when the models are given.

.. code-block:: python

   statistic, p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)

Paired t-test
-------------

Included for completeness and for comparison with the other two. It is the test
most often reached for and the one most likely to overstate significance on
cross-validation folds, for the reason given above. Prefer the 5x2cv F test when
you can afford the ten fits.

Choosing between them
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Situation
     - Test
     - Why
   * - Two algorithms, one dataset, refitting affordable
     - ``combined_5x2cv_f_test``
     - Calibrated Type I error
   * - Two already-fitted models, one test set
     - ``mcnemar_test``
     - No refitting required
   * - Repeated measurements you know to be independent
     - ``paired_t_test``
     - Assumptions actually hold

API reference
=============

Decision trees
--------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   neural_trees.SoftDecisionTree
   neural_trees.SoftDecisionTreeRegressor
   neural_trees.NumpySoftTree
   neural_trees.HardRegressionTree
   neural_trees.MultivariateDecisionTree
   neural_trees.OmnivariateDecisionTree
   neural_trees.HardDecisionTree

Mixtures of experts
-------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   neural_trees.HierarchicalMixtureOfExperts
   neural_trees.HardRoutedExperts

Constructive and classical models
---------------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   neural_trees.GALNetwork
   neural_trees.WeightedKNN
   neural_trees.NaiveBayesClassifier

Distillation
------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   neural_trees.DistilledTree
   neural_trees.DistilledForest

.. autosummary::
   :toctree: generated

   neural_trees.distill
   neural_trees.fidelity
   neural_trees.Distilled
   neural_trees.distillation.munge

Statistical tests
-----------------

.. autosummary::
   :toctree: generated

   neural_trees.combined_5x2cv_f_test
   neural_trees.mcnemar_test
   neural_trees.paired_t_test

Explanations
------------

.. autosummary::
   :toctree: generated

   neural_trees.Explanation
   neural_trees.GateStep
   neural_trees.Counterfactual

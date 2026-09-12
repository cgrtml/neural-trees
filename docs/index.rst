neural-trees
============

Differentiable and constructive classifiers with a scikit-learn API and a
PyTorch backend: soft decision trees, multivariate and omnivariate trees,
hierarchical mixtures of experts, a grow-and-prune network, and the statistical
tests used to decide whether one classifier is genuinely better than another.

Every model is an ordinary estimator, so it composes with
:class:`~sklearn.pipeline.Pipeline`, :class:`~sklearn.model_selection.GridSearchCV`
and the rest of the ecosystem.

.. code-block:: python

   from sklearn.datasets import load_wine
   from sklearn.model_selection import cross_val_score
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler

   from neural_trees import SoftDecisionTree

   X, y = load_wine(return_X_y=True)
   model = make_pipeline(StandardScaler(), SoftDecisionTree(depth=4, random_state=0))
   print(cross_val_score(model, X, y, cv=5).mean())

Install with ``pip install neural-trees``.

Why this library exists
-----------------------

Soft decision trees, omnivariate trees and hierarchical mixtures of experts are
standard material in machine learning courses and are cited regularly, but
working implementations with a familiar API are scarce. A practitioner who
wants to compare a soft tree against CART on their own data usually has to
reimplement the model from the paper, and the reimplementation is then
unverified.

The second reason is honest comparison. The package ships Alpaydın's combined
5x2cv F test alongside McNemar's test and the paired t-test, because an accuracy
difference on one split says little until you know how much that number moves
when only the fold assignment changes.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   model_comparison
   design_decisions
   api

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Citing
------

The software is archived on Zenodo. The concept DOI
`10.5281/zenodo.22718897 <https://doi.org/10.5281/zenodo.22718897>`_ always
resolves to the most recent release; cite the DOI of the specific version you
ran when the version matters.

When you use one of the underlying methods, cite the paper it comes from as
well. Each is listed in :doc:`user_guide`.

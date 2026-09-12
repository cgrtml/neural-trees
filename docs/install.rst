Installation
============

.. code-block:: bash

   pip install neural-trees

Requirements
------------

Python 3.9 to 3.13. The package depends on NumPy, scikit-learn and PyTorch.

PyTorch is imported lazily: ``import neural_trees`` does not pull it in, and the
classical estimators (:class:`~neural_trees.WeightedKNN`,
:class:`~neural_trees.NaiveBayesClassifier`) run without it. The differentiable
models raise a clear error if torch is missing rather than failing at fit time.

CPU-only installation
---------------------

The default PyTorch wheel is large because it carries CUDA. For a CPU-only
environment:

.. code-block:: bash

   pip install neural-trees --extra-index-url https://download.pytorch.org/whl/cpu

Device selection
----------------

Every differentiable estimator takes a ``device`` argument, defaulting to
``"cpu"``. Pass ``device="auto"`` to pick CUDA when it is available, then Apple
silicon's MPS, then CPU. Anything else is handed to torch as given, so
``"cuda:1"`` works. The choice is resolved once in ``fit`` and recorded as
``device_``, so prediction always runs where training did.

:class:`~neural_trees.HierarchicalMixtureOfExperts` and
:class:`~neural_trees.GALNetwork` additionally keep a float64 copy of the fitted
model on the CPU and predict from it, so that the same input gives the same
output across machines. This costs a little speed and buys reproducibility:
float32 matrix products are sensitive to row order and to the BLAS
implementation, which is enough to flip a prediction near a decision boundary.

From source
-----------

.. code-block:: bash

   git clone https://github.com/cgrtml/neural-trees
   cd neural-trees
   pip install -e ".[dev]"
   pytest

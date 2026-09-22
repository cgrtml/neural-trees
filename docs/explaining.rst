Explaining a prediction
=======================

A soft decision tree can say why it predicted what it predicted for one
sample. ``explain`` returns, for each row:

- the predicted class and its probability;
- the **dominant leaf**, the one that received most of the sample's
  probability mass, and that leaf's class distribution;
- every **gate on the path** to that leaf: which way the sample went, with
  what probability, and the largest feature terms behind the decision;
- a per-feature **contribution** score aggregated along the path;
- the smallest single-feature **counterfactual** that flips the class,
  verified by re-predicting the changed sample.

.. code-block:: python

   e = model.explain(x, feature_names=names)
   print(e.to_text())

.. code-block:: text

   predicted 1 with probability 0.913
   dominant leaf 9 received 0.842 of the sample's mass; its distribution is 0: 0.031, 1: 0.951, 2: 0.018
   path:
     gate 0: went left with p=0.902  (-1.204[proline] -0.877[color_intensity] +0.412[flavanoids])
     gate 1: went right with p=0.933  (+0.981[flavanoids] +0.522[hue] -0.310[alcalinity_of_ash])
     gate 4: went left with p=0.871  (...)
   largest contributions: proline 1.204, flavanoids 1.156, color_intensity 0.877, ...
   counterfactual: set proline from -0.612 to 0.401 and the prediction becomes 0 (p=0.702)

What the numbers mean, exactly
------------------------------

A soft tree's prediction is a mixture over leaves, so the sample did not
literally take one path. The explanation reports the path to the leaf that
received the most mass and says how much that was (``leaf_probability``). If
it is close to one, the tree behaved like a hard tree on this sample; if it
is 0.4, the prediction is genuinely a blend and the path is only the largest
part of the story.

The **contribution** of a feature is the size of its term, ``weight × value``,
in each gate on the path, weighted by how much of the sample reached that
gate. It reads the linear gates directly. It is not SHAP and makes no claim
beyond the gates on this path.

The **counterfactual** is found by moving one feature just past the zero
crossing of one gate on the path and re-predicting. It is reported only if
the class actually changes. If no single-feature change on the path flips the
class, the field is ``None`` rather than a guess. All values are in the
model's input units: if you fitted on standardised features, so are these.

Cost
----

Explaining a sample costs one forward pass plus, with ``counterfactual=True``,
one prediction per candidate feature per gate on the path. For many samples
pass ``counterfactual=False``.

See :ref:`sphx_glr_auto_examples_05_explain_prediction.py` for a runnable
example and :class:`~neural_trees.Explanation` for the returned object.

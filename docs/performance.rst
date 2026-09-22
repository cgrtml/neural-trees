Performance
===========

How long a soft decision tree takes to fit, measured rather than estimated.
One CPU thread (``OMP_NUM_THREADS=1``), Apple silicon, synthetic data with
50 features and 5 classes, depth 4, batch size 64, 30 epochs.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - samples
     - fit, 30 epochs
     - per epoch
     - predict (all samples)
   * - 1 000
     - 2.0 s
     - 66 ms
     - 3 ms
   * - 5 000
     - 6.3 s
     - 210 ms
     - 13 ms
   * - 20 000
     - 30 s
     - 1.0 s
     - 39 ms
   * - 50 000
     - 58 s
     - 1.9 s
     - 102 ms

Time is linear in the number of samples and, since every gate is evaluated
for every sample, linear in the number of gates: depth 6 (63 gates) costs
about 1.4 times depth 4 (15 gates) at 5 000 samples. At the default 150
epochs a depth-4 tree on 50 000 samples fits in about five minutes on one
thread.

Where the time goes
-------------------

About a third of a training step is the backward pass and a fifth was, until
0.7, the ``DataLoader`` indexing the dataset one sample at a time. Batches are
now sliced directly from the tensors (:mod:`neural_trees._batching`), which
draws the same permutation the loader would draw from the same seed, so fits
with a ``random_state`` are bit-identical to the previous version's and
14--21% faster.

What is not fast
----------------

- **Growth.** ``growth="incremental"`` trains up to ``depth`` rounds and
  ``growth="per_leaf"`` up to ``2 * depth``; with ``growth_budget="full"``
  each round gets the whole epoch budget, so the fit costs that many times a
  fixed tree.
- **Omnivariate trees** cross-validate an MLP at every node. They are for
  small problems.
- **Batch size.** Larger batches are faster per epoch and reach a worse
  model at the same number of epochs (on Digits, 20 epochs: batch 64 gives
  0.915 training accuracy in 1.6 s, batch 1024 gives 0.786 in 0.5 s). Raise
  ``max_epochs`` with the batch size.

Devices
-------

``device="auto"`` uses CUDA, then Apple MPS, then CPU. On problems of the
sizes above the CPU is not the bottleneck and a GPU rarely helps; it starts
to pay at hundreds of thousands of samples or deep trees.

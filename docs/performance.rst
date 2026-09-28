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

Prediction
----------

How fast a fitted tree predicts, by export, one thread. *One row* is the
latency a service answering single requests pays; *batch* is 10 000 rows
at once. The torch estimator, the numpy copy (``to_numpy()``), ONNX Runtime
on the ONNX export (``to_onnx()``) and the hard rule tree (``to_hard_tree()``)
are the same fitted tree; the first three predict the same probabilities to
float32 precision, the last reads each gate as a hard decision. Measured by
``benchmarks/latency.py``, best of seven, Apple silicon.

.. list-table::
   :header-rows: 1
   :widths: 30 18 16 18 16

   * - Export
     - Breast Cancer, depth 4, 30 features
     -
     - Digits, depth 6, 64 features
     -
   * -
     - one row
     - batch, rows/s
     - one row
     - batch, rows/s
   * - torch estimator
     - 93 µs
     - 2,485,509
     - 111 µs
     - 448,645
   * - numpy export
     - 33 µs
     - 1,176,864
     - 50 µs
     - 205,547
   * - ONNX Runtime
     - 11 µs
     - 3,022,593
     - 33 µs
     - 565,836
   * - hard rule tree
     - 44 µs
     - 7,259,967
     - 58 µs
     - 2,974,960

Single-row latency is dominated by call overhead, not arithmetic: a depth-4
tree is fifteen dot products. ONNX Runtime has the least overhead and is the
export to serve one request at a time; the hard rule tree is the fastest in
batch because it evaluates one path instead of every gate. Nothing here
needs a GPU, and the ONNX file needs neither torch nor this library where
it runs.

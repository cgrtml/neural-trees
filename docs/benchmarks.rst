Against the field
=================

How the models here compare with what people actually use on tabular data:
XGBoost, LightGBM, Random Forest, a two-layer MLP, and two differentiable
tree ensembles from the literature, GRANDE (Marton et al., ICLR 2024) and
NODE (Popov et al., ICLR 2020). Twenty-four datasets: Iris, Wine, Breast
Cancer, Digits and twenty from the OpenML CC-18 suite, capped at 5 000 rows
by stratified subsampling. Three seeds of stratified five-fold cross-validation,
features standardised on the training fold, accuracy averaged over the
fifteen folds. Every model in its own process, one BLAS thread.

**Nothing is tuned.** Each model runs one fixed configuration on every
dataset: XGBoost 300 trees of depth 6 at learning rate 0.1, LightGBM 300
trees of 31 leaves, Random Forest 300 unbounded trees, MLP 64-64 for 300
iterations, GRANDE 256 trees of depth 5 with its published defaults, NODE one
layer of 256 oblivious trees of depth 6 for 50 epochs with early stopping,
the soft tree at depth 4 for 150 epochs, the per-leaf tree at depth 6 for
180 epochs, GAL for 150 epochs with residual initialisation. A tuned
XGBoost would do better on the small datasets, where 300 deep trees overfit
150 rows; so would a tuned soft tree. The comparison is between defaults,
which is what a first run looks like, and it should be read that way.

The script is ``benchmarks/rakipler.py``; ``benchmarks/rakipler_ozet.py``
turns its output into the tables below. Bold marks the best in a row.

.. csv-table::
   :file: _generated/benchmark_full.csv
   :header-rows: 1

Paired against XGBoost
----------------------

Difference in accuracy points, model minus XGBoost, over the datasets; a tie
is a difference within half a point; the Wilcoxon signed-rank test is over
the per-dataset means.

.. csv-table::
   :file: _generated/benchmark_vs_xgboost.csv
   :header-rows: 1

Over all twenty-four datasets the gradient-boosted ensembles, Random Forest
and the MLP are the top group and the soft models sit a point or two below
them; NODE, with these defaults and this budget, is last everywhere. That is
the expected result and it is not the interesting one.

Small datasets
--------------

Split by size, the picture reverses. On the datasets with at most 1 000
rows:

.. csv-table::
   :file: _generated/benchmark_small_n.csv
   :header-rows: 1

and on the rest:

.. csv-table::
   :file: _generated/benchmark_large_n.csv
   :header-rows: 1

On the small datasets every gradient-trained smooth model beats XGBoost:
GAL on every one of them, the per-leaf soft tree on seven of eight, the
MLP on every one. Boosting with 300 deep trees has more capacity than a few
hundred rows can constrain, and a smooth model with a few dozen parameters
per node does not. Above 1 000 rows, sixteen datasets, the order flips and XGBoost, LightGBM
and Random Forest win almost every comparison against the soft models.

Two cautions. The split at 1 000 rows was chosen after seeing the data, and
eight datasets is a small sample, so the p-values in the small-data table are
exploratory, not confirmatory. And the MLP wins there too, so the finding is
about smooth gradient-trained models against untuned boosting on small
tables, not about trees in particular. What the soft tree adds over the
MLP in that regime is what it adds everywhere: a model that can be read
(:doc:`explaining`), exported as rules or as plain numpy, and whose
probabilities are calibrated (:doc:`user_guide`).

So the honest summary for someone choosing a model: with a few hundred rows
and a need to explain the predictions, a soft tree or GAL is a reasonable
first choice and will likely match or beat an untuned boosting model. With
thousands of rows and accuracy as the only criterion, use LightGBM.

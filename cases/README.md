# Case studies

Each directory is one complete, regenerable "auditable model package": a script that
measures a neural-trees model against the models a team would otherwise use on
identical folds, and a model document written from the numbers, with the rules,
explained decisions, calibration, slices, refit stability and the exported model
files a reviewer would ask for.

| case | data | what it shows |
|---|---|---|
| [german-credit](german-credit/report.md) | UCI German Credit, 1 000 applicants, cost matrix | soft tree ties logistic regression on accuracy, AUC and cost; untuned boosters lose on cost through calibration; rules, three explained decisions with reachable counterfactuals, slices by age, sex and residency |
| [credit-default](credit-default/report.md) | UCI credit card default, 30 000 card holders | one tree distilled from a tuned XGBoost (24 814 splits) keeps AUC 0.783 against 0.785 with 48 splits and takes the same decision for 98.5% of applicants; principal reasons for a decline read from the deployed tree, reachable counterfactuals, decisions by sex, marital status and age kept out of the inputs |

To regenerate a case: `python cases/<name>/run.py && python cases/<name>/report.py`
(the German Credit case needs `xgboost`, `lightgbm`, `onnxruntime` and, for the
Word version, `python-docx`; the credit default case needs `xgboost` and takes about an hour, most of it the distillation).

If you have a table of a few hundred to a few thousand rows where every decision has
to be explained, open an issue: the offer in the README of a free case study stands.

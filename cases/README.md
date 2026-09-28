# Case studies

Each directory is one complete, regenerable "auditable model package": a script that
measures a neural-trees model against the models a team would otherwise use on
identical folds, and a model document written from the numbers, with the rules,
explained decisions, calibration, slices, refit stability and the exported model
files a reviewer would ask for.

| case | data | what it shows |
|---|---|---|
| [german-credit](german-credit/report.md) | UCI German Credit, 1 000 applicants, cost matrix | soft tree ties logistic regression on accuracy, AUC and cost; untuned boosters lose on cost through calibration; rules, three explained decisions with reachable counterfactuals, slices by age, sex and residency |

To regenerate a case: `python cases/<name>/run.py && python cases/<name>/report.py`
(the German Credit case needs `xgboost`, `lightgbm`, `onnxruntime` and, for the
Word version, `python-docx`).

If you have a table of a few hundred to a few thousand rows where every decision has
to be explained, open an issue: the offer in the README of a free case study stands.

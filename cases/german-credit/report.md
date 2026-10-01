# German Credit: an auditable model package

neural-trees 0.7.0 · 2026-10-01 · every figure below is read from results.json, produced by run.py

## 1. Purpose and scope

A credit decision has to be defended one applicant at a time: the reviewer asks why this person was refused, what would have changed the answer, and whether the same rule was applied to everyone. This document shows what a soft decision tree from neural-trees gives a reviewer on the standard public credit dataset, measured against the models a credit team would otherwise use, on identical folds. It follows the headings a model validation under SR 11-7 or the EU AI Act's Annex IV documentation expects: purpose, data, method, performance, explanation, stability, limitations, and the shipped artefacts. It is a demonstration on public data, not a validated production model.

## 2. Data

OpenML 31 (UCI Statlog German Credit): 1000 loan applicants, 700 good and 300 bad, 7 numeric attributes (duration, credit_amount, installment_commitment, residence_since, age, existing_credits, num_dependents) and 13 categorical ones (checking_status, credit_history, purpose, savings_status, employment, personal_status, other_parties, property_magnitude, other_payment_plans, housing, job, own_telephone, foreign_worker). After one-hot encoding the model sees 61 inputs.

The dataset ships a cost matrix: a bad applicant accepted costs 5, a good applicant refused costs 1. The decision that minimises expected cost is therefore to refuse when P(bad) exceeds 1/6 ≈ 0.167, not 0.5. Both thresholds are reported; the cost column is the one that matters to the business.

Known caveat: the UCI coding of `personal_status` (used below for the sex slice) has been reported as unreliable (Grömping, 2019, "South German Credit Data: Correcting a Widely Used Data Set"). The sex slice is therefore illustrative of the procedure, not a finding about the population.

## 3. Method

3 seeds × 5-fold stratified cross-validation, 15 fits per model, the same folds for every model. Preprocessing: median impute + standardise numerics, one-hot categoricals, fitted inside each training fold. Nothing was tuned on this data: each model runs one fixed configuration, the same one used in the library's 24-dataset benchmark.

| model | configuration | seconds per fit |
|---|---|---|
| Soft tree, per-leaf | depth 6 grown per leaf, residual initialisation, 180 epochs | 1.1 |
| Soft tree, depth 4 | complete tree, 150 epochs | 1.3 |
| Logistic regression | scikit-learn defaults, the scorecard baseline | 0.0 |
| CART | scikit-learn defaults, unpruned | 0.0 |
| Random forest | 300 trees | 0.2 |
| XGBoost | 300 rounds, depth 6, learning rate 0.1 | 0.0 |
| LightGBM | 300 rounds, 31 leaves, learning rate 0.1 | 0.1 |

## 4. Performance

Mean ± standard deviation over the 15 fits. Cost is per applicant under the dataset's cost matrix; lower is better. Refusal rate is the share of applicants refused at the cost-optimal threshold.

| model | accuracy | balanced acc. | AUC | Brier | cost at 0.5 | cost at 1/6 | refused at 1/6 |
|---|---|---|---|---|---|---|---|
| Soft tree, per-leaf | 0.747 ± 0.021 | 0.667 ± 0.030 | 0.786 ± 0.032 | 0.168 ± 0.014 | 0.893 ± 0.099 | 0.548 ± 0.069 | 0.59 ± 0.07 |
| Soft tree, depth 4 | 0.710 ± 0.024 | 0.642 ± 0.033 | 0.731 ± 0.036 | 0.228 ± 0.022 | 0.921 ± 0.100 | 0.761 ± 0.096 | 0.39 ± 0.03 |
| Logistic regression | 0.747 ± 0.021 | 0.667 ± 0.031 | 0.787 ± 0.034 | 0.166 ± 0.014 | 0.892 ± 0.098 | 0.536 ± 0.084 | 0.58 ± 0.03 |
| CART | 0.677 ± 0.026 | 0.619 ± 0.027 | 0.619 ± 0.027 | 0.323 ± 0.026 | 0.955 ± 0.078 | 0.955 ± 0.078 | 0.31 ± 0.04 |
| Random forest | 0.763 ± 0.023 | 0.656 ± 0.034 | 0.794 ± 0.031 | 0.163 ± 0.008 | 0.970 ± 0.099 | 0.544 ± 0.047 | 0.71 ± 0.03 |
| XGBoost | 0.749 ± 0.032 | 0.673 ± 0.041 | 0.775 ± 0.034 | 0.186 ± 0.020 | 0.869 ± 0.111 | 0.625 ± 0.102 | 0.42 ± 0.04 |
| LightGBM | 0.746 ± 0.029 | 0.676 ± 0.039 | 0.775 ± 0.035 | 0.206 ± 0.025 | 0.855 ± 0.111 | 0.710 ± 0.114 | 0.35 ± 0.03 |

Best accuracy: Random forest. Best AUC: Random forest. Lowest cost at the cost-optimal threshold: Logistic regression. Best calibrated by Brier score: Random forest.

Reading. The per-leaf soft tree and logistic regression cannot be told apart on this data: accuracy 0.747 against 0.747, AUC 0.786 against 0.787, cost 0.548 against 0.536 (p = 0.34). Random forest is 1.6 points more accurate and no cheaper at the cost-optimal threshold (0.544), and it refuses 71% of applicants to get there against the soft tree's 59%. The boosted models match the soft tree on accuracy and lose on cost (XGBoost 0.625, LightGBM 0.710) because their untuned probabilities are poorly calibrated (section 5), and a cost-weighted decision at a 1/6 threshold is exactly where calibration is paid for. What the soft tree adds over logistic regression is sections 6 and 7: a small tree of rules and a per-applicant path with counterfactuals, at the same accuracy.

Paired tests of each model against the per-leaf soft tree over the 15 fits (difference is model minus soft tree; a negative cost difference favours the model):

| model | Δ accuracy | p | Δ AUC | p | Δ cost at 1/6 | p |
|---|---|---|---|---|---|---|
| Soft tree, depth 4 | -0.038 | 0.000 | -0.055 | 0.000 | +0.213 | 0.000 |
| Logistic regression | -0.001 | 0.809 | +0.001 | 0.698 | -0.012 | 0.340 |
| CART | -0.070 | 0.000 | -0.166 | 0.000 | +0.407 | 0.000 |
| Random forest | +0.016 | 0.011 | +0.008 | 0.078 | -0.004 | 0.774 |
| XGBoost | +0.001 | 0.870 | -0.010 | 0.063 | +0.077 | 0.003 |
| LightGBM | -0.001 | 0.885 | -0.011 | 0.053 | +0.162 | 0.000 |

p is a paired t-test over the fits; the fits share data, so the test is optimistic and a p just under 0.05 should be read as "probably", not "proven".

## 5. Calibration

Out-of-fold P(bad) from seed 0, cut into ten equal-width bins: what the model said against what happened. ECE is the expected calibration error, the bin-weighted gap between the two columns.

**Soft tree, per-leaf**, ECE 0.032

| P(bad) bin | applicants | mean predicted | observed bad rate |
|---|---|---|---|
| 0.0-0.1 | 279 | 0.060 | 0.079 |
| 0.1-0.2 | 180 | 0.147 | 0.200 |
| 0.2-0.3 | 126 | 0.245 | 0.230 |
| 0.3-0.4 | 97 | 0.349 | 0.381 |
| 0.4-0.5 | 90 | 0.448 | 0.378 |
| 0.5-0.6 | 92 | 0.548 | 0.554 |
| 0.6-0.7 | 84 | 0.650 | 0.595 |
| 0.7-0.8 | 28 | 0.747 | 0.750 |
| 0.8-0.9 | 22 | 0.839 | 0.818 |
| 0.9-1.0 | 2 | 0.924 | 1.000 |

**Logistic regression**, ECE 0.033

| P(bad) bin | applicants | mean predicted | observed bad rate |
|---|---|---|---|
| 0.0-0.1 | 290 | 0.051 | 0.069 |
| 0.1-0.2 | 181 | 0.148 | 0.177 |
| 0.2-0.3 | 111 | 0.252 | 0.270 |
| 0.3-0.4 | 108 | 0.353 | 0.370 |
| 0.4-0.5 | 90 | 0.450 | 0.478 |
| 0.5-0.6 | 60 | 0.554 | 0.433 |
| 0.6-0.7 | 70 | 0.650 | 0.571 |
| 0.7-0.8 | 49 | 0.742 | 0.735 |
| 0.8-0.9 | 29 | 0.842 | 0.828 |
| 0.9-1.0 | 12 | 0.932 | 0.750 |

**Random forest**, ECE 0.037

| P(bad) bin | applicants | mean predicted | observed bad rate |
|---|---|---|---|
| 0.0-0.1 | 149 | 0.057 | 0.054 |
| 0.1-0.2 | 186 | 0.146 | 0.091 |
| 0.2-0.3 | 181 | 0.250 | 0.227 |
| 0.3-0.4 | 168 | 0.346 | 0.321 |
| 0.4-0.5 | 144 | 0.450 | 0.431 |
| 0.5-0.6 | 104 | 0.547 | 0.606 |
| 0.6-0.7 | 46 | 0.642 | 0.761 |
| 0.7-0.8 | 22 | 0.741 | 0.909 |

**LightGBM**, ECE 0.163

| P(bad) bin | applicants | mean predicted | observed bad rate |
|---|---|---|---|
| 0.0-0.1 | 604 | 0.011 | 0.164 |
| 0.1-0.2 | 59 | 0.147 | 0.186 |
| 0.2-0.3 | 34 | 0.250 | 0.353 |
| 0.3-0.4 | 23 | 0.350 | 0.391 |
| 0.4-0.5 | 30 | 0.451 | 0.500 |
| 0.5-0.6 | 20 | 0.557 | 0.400 |
| 0.6-0.7 | 30 | 0.643 | 0.533 |
| 0.7-0.8 | 16 | 0.767 | 0.500 |
| 0.8-0.9 | 49 | 0.861 | 0.510 |
| 0.9-1.0 | 135 | 0.975 | 0.719 |

## 6. The rules

The final model is trained on all 1000 applicants (seed 0). It has 2 soft splits and 3 leaves; training accuracy 0.780. Reading each gate as a hard decision gives the rule list in rules.txt (5 lines); that hard reading agrees with the soft model on 93.0% of the training rows and, out of fold, on 92.0% of held-out rows on average. Each split is a weighted sum of the inputs, so a rule is not a single threshold; the print below shows the five largest raw weights per gate, and the table after it shows which inputs actually decide each gate, ranked by mean |weight × value| over the training rows, which puts a 0/1 one-hot column and a standardised numeric on the same footing.

```
if +1.290*foreign_worker=no -1.198*credit_history=no credits/all paid +1.156*checking_status=no checking +1.154*purpose=used car -1.042*checking_status=<0 +0.055 (+56 more) > 0:
    yes -> predict 'good' (p=0.973)
    no  -> if +0.346*savings_status=100<=X<500 +0.334*installment_commitment +0.317*employment=>=7 +0.299*residence_since -0.282*purpose=other +0.055 (+56 more) > 0:
        yes -> predict 'bad' (p=0.864)
        no  -> predict 'bad' (p=0.674)
```

| gate | inputs that decide it (share of the gate's mean absolute contribution) |
|---|---|
| 0 | checking_status=no checking (8%), duration (7%), installment_commitment (6%), credit_amount (6%), other_payment_plans=none (6%) |
| 1 | installment_commitment (14%), residence_since (13%), num_dependents (6%), other_payment_plans=none (6%), existing_credits (6%) |

## 7. Three decisions, explained

From the seed-0, fold-0 model, on applicants it never saw in training. Each explanation gives the predicted class and its probability, the leaf that received the applicant, the gates on the way with the inputs that decided them, and the smallest single change that flips the decision, verified by re-predicting the changed applicant.

### clearly refused (highest P(bad))

Applicant 973: actual outcome **bad**, P(bad) = 0.930, so refused at the cost-optimal threshold. Checking status '<0', duration 60 months, amount 7297, credit history 'existing paid', savings '<100', employment '>=7', age 36, purpose 'business'.

```
predicted 'bad' with probability 0.930
dominant leaf 6 received 0.936 of the sample's mass; its distribution is 'bad': 0.971, 'good': 0.029
path:
  gate 0: went right with p=0.986  (+1.831[duration] +1.025[checking_status=<0] +0.729[credit_amount])
  gate 2: went right with p=0.949  (+1.186[employment=>=7] +0.781[residence_since] +0.752[installment_commitment])
largest contributions: duration 1.842, employment=>=7 1.366, checking_status=<0 1.211, other_payment_plans=none 1.021, credit_amount 0.893
counterfactual: no single-feature change on this path flips the class
```

No single encoded input, kept inside its training range, flips this decision at the gates on the path.

Reachable single changes that flip the decision at the cost-optimal threshold of 1/6, tried on the raw table and re-encoded (81 candidates: every other value of each categorical column, each numeric column moved to a training decile), smallest first:

None: no single change of one column flips this decision.

### clearly accepted (lowest P(bad))

Applicant 851: actual outcome **good**, P(bad) = 0.024, so accepted at the cost-optimal threshold. Checking status 'no checking', duration 24 months, amount 4042, credit history 'critical/other existing credit', savings 'no known savings', employment '4<=X<7', age 43, purpose 'used car'.

```
predicted 'good' with probability 0.976
dominant leaf 4 received 0.799 of the sample's mass; its distribution is 'bad': 0.015, 'good': 0.985
path:
  gate 0: went left with p=0.998  (-1.188[credit_history=critical/other existing credit] -1.154[purpose=used car] -1.139[checking_status=no checking])
  gate 1: went right with p=0.801  (+0.430[purpose=used car] +0.375[checking_status=no checking] -0.210[existing_credits])
largest contributions: purpose=used car 1.583, checking_status=no checking 1.513, credit_history=critical/other existing credit 1.369, employment=4<=X<7 1.174, savings_status=no known savings 0.583
counterfactual: no single-feature change on this path flips the class
```

No single encoded input, kept inside its training range, flips this decision at the gates on the path.

Reachable single changes that flip the decision at the cost-optimal threshold of 1/6, tried on the raw table and re-encoded (85 candidates: every other value of each categorical column, each numeric column moved to a training decile), smallest first:

None: no single change of one column flips this decision.

### borderline (P(bad) nearest the 1/6 threshold)

Applicant 970: actual outcome **good**, P(bad) = 0.166, so accepted at the cost-optimal threshold. Checking status '0<=X<200', duration 15 months, amount 1514, credit history 'existing paid', savings '100<=X<500', employment '1<=X<4', age 22, purpose 'repairs'.

```
predicted 'good' with probability 0.834
dominant leaf 4 received 0.433 of the sample's mass; its distribution is 'bad': 0.015, 'good': 0.985
path:
  gate 0: went left with p=0.848  (-1.325[other_parties=guarantor] +0.689[purpose=repairs] -0.500[other_payment_plans=none])
  gate 1: went right with p=0.511  (-0.189[purpose=repairs] +0.164[savings_status=100<=X<500] +0.163[property_magnitude=real estate])
largest contributions: other_parties=guarantor 1.335, purpose=repairs 0.849, other_payment_plans=none 0.515, checking_status=0<=X<200 0.399, property_magnitude=real estate 0.391
counterfactual: no single-feature change on this path flips the class
```

No single encoded input, kept inside its training range, flips this decision at the gates on the path.

Reachable single changes that flip the decision at the cost-optimal threshold of 1/6, tried on the raw table and re-encoded (82 candidates: every other value of each categorical column, each numeric column moved to a training decile), smallest first:

| change | P(bad) after |
|---|---|
| job: 'skilled' → 'unskilled resident' | 0.167 |
| job: 'skilled' → 'high qualif/self emp/mgmt' | 0.171 |
| property_magnitude: 'real estate' → 'car' | 0.174 |
| employment: '1<=X<4' → '>=7' | 0.179 |
| property_magnitude: 'real estate' → 'no known property' | 0.190 |

## 8. Slices: is the same rule applied to everyone?

Out-of-fold decisions from seed 0 at the cost-optimal threshold. "Bad missed" is the share of actually bad applicants that were accepted; "good refused" the share of actually good applicants that were refused. A model can be equally accurate in two groups and still refuse one of them more often, so both are shown, for the soft tree and for logistic regression.

**Soft tree, per-leaf**

| group | applicants | actual bad rate | refusal rate | accuracy | bad missed | good refused |
|---|---|---|---|---|---|---|
| age < 25 | 149 | 0.409 | 0.772 | 0.644 | 0.082 | 0.670 |
| age >= 25 | 851 | 0.281 | 0.561 | 0.776 | 0.146 | 0.446 |
| female | 310 | 0.352 | 0.645 | 0.742 | 0.110 | 0.512 |
| male | 690 | 0.277 | 0.568 | 0.762 | 0.147 | 0.459 |
| foreign worker | 963 | 0.307 | 0.603 | 0.752 | 0.132 | 0.486 |
| not a foreign worker | 37 | 0.108 | 0.297 | 0.865 | 0.250 | 0.242 |

**Logistic regression**

| group | applicants | actual bad rate | refusal rate | accuracy | bad missed | good refused |
|---|---|---|---|---|---|---|
| age < 25 | 149 | 0.409 | 0.758 | 0.631 | 0.049 | 0.625 |
| age >= 25 | 851 | 0.281 | 0.550 | 0.771 | 0.159 | 0.436 |
| female | 310 | 0.352 | 0.645 | 0.723 | 0.101 | 0.507 |
| male | 690 | 0.277 | 0.552 | 0.762 | 0.157 | 0.441 |
| foreign worker | 963 | 0.307 | 0.593 | 0.744 | 0.135 | 0.472 |
| not a foreign worker | 37 | 0.108 | 0.270 | 0.919 | 0.250 | 0.212 |

Age and sex are inputs to the model here because they are in the public dataset; a lender in the EU or the US would remove them (and their proxies) before training, and would then use exactly these tables to check whether the remaining inputs still act as a proxy.

## 9. Stability across refits

Over the 15 refits the per-leaf soft tree grew between 1 and 4 splits (median 2). The input with the largest mean |weight × value| at the root gate was 'checking_status=no checking' in 9 of 15 fits, 'duration' in 4 of 15 fits, 'installment_commitment' in 2 of 15 fits. Hard-rule agreement with the soft model on held-out rows ranged from 0.805 to 0.985.

| seed | fold | splits | root gate, largest inputs | hard agreement |
|---|---|---|---|---|
| 0 | 0 | 3 | checking_status=no checking, duration, other_payment_plans=none | 0.985 |
| 0 | 1 | 1 | duration, checking_status=no checking, installment_commitment | 0.885 |
| 0 | 2 | 3 | checking_status=no checking, duration, other_payment_plans=none | 0.940 |
| 0 | 3 | 2 | duration, checking_status=no checking, other_payment_plans=none | 0.910 |
| 0 | 4 | 1 | installment_commitment, checking_status=no checking, duration | 0.895 |
| 1 | 0 | 4 | checking_status=no checking, duration, installment_commitment | 0.975 |
| 1 | 1 | 1 | duration, credit_amount, installment_commitment | 0.805 |
| 1 | 2 | 1 | checking_status=no checking, duration, installment_commitment | 0.920 |
| 1 | 3 | 2 | checking_status=no checking, duration, credit_amount | 0.950 |
| 1 | 4 | 3 | checking_status=no checking, installment_commitment, other_payment_plans=none | 0.965 |
| 2 | 0 | 1 | duration, checking_status=no checking, savings_status=<100 | 0.885 |
| 2 | 1 | 4 | checking_status=no checking, credit_amount, duration | 0.970 |
| 2 | 2 | 1 | checking_status=no checking, duration, other_payment_plans=none | 0.895 |
| 2 | 3 | 1 | checking_status=no checking, duration, installment_commitment | 0.875 |
| 2 | 4 | 2 | installment_commitment, checking_status=no checking, personal_status=male single | 0.945 |

## 10. Limitations

- 1 000 rows is small; the standard deviations above are the honest width of every claim, and several models are within one of them of each other.
- The rules are a hard reading of soft gates. Near a gate the two disagree; the agreement rates above measure how often.
- Two kinds of counterfactual are shown on purpose. The gate-level one reads the model's own gates and is bounded to the training range; the reachable one changes a column of the raw table. Both are statements about the model, not advice to the applicant.
- Age, sex and residency are inputs here only because they are in the public data. See section 8.
- Nothing was tuned, on purpose, so that the comparison is fair; every model would gain from tuning, and not by the same amount.

## 11. Shipped artefacts

| file | what it is |
|---|---|
| model.json | the fitted soft tree as plain arrays (85 KB); predicts with numpy alone via NumpySoftTree.from_json |
| model.onnx | the same tree as an ONNX graph (18 KB), standard operators only |
| preprocessing.json | the medians, means, scales and category lists a consumer must apply before calling the model, in input order |
| rules.txt | the hard-rule reading of the final model, with input names |
| results.json | every number in this document, fold by fold |
| summary.csv | the performance table |
| run.py, report.py | regenerate everything |

ONNX check on all 1000 training rows: the ONNX graph and the torch model agree on 100.0% of labels, largest probability difference 2.38e-07; one applicant scores in 6 µs in ONNX Runtime on one CPU thread.

## 12. Monitoring after deployment

What a reviewer would ask to see quarterly, all computable from the files here: the refusal rate and the observed bad rate per calibration bin against section 5; the slice table of section 8 on new applicants; the share of decisions where the hard rules and the soft model disagree, against section 6; and a refit on the new quarter with the root-gate inputs compared to section 9. A drift in any of these is a reason to look, not a reason to retrain automatically.

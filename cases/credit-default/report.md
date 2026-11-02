# Credit card default: a distilled, auditable model package

neural-trees 0.8.0 · 2026-11-01 · every figure below is read from results.json, produced by run.py

## 1. Purpose and scope

A bank already runs a gradient boosted model for credit decisions and has to defend each decision to a validator, an examiner and the applicant. This document shows what happens when that model is distilled into one soft decision tree with neural-trees: how much accuracy and ranking power the tree keeps, how often it takes the same decision as the bank's model, how a single decline is explained from the tree's own path, and whether decisions differ by group. It follows the headings a model validation under SR 11-7 expects: purpose, data, method, performance, explanation, fairness checks, stability, limitations and the shipped files. It is a demonstration on public data, not a validated production model.

## 2. Data

OpenML 42477 (UCI default of credit card clients, Yeh and Lien 2009), CC0: 30,000 card holders, 6,636 of whom defaulted the following month (22.1%). 24,000 rows train every model and the same 6,000 held-out rows score them.

Inputs (20): credit limit, education, the repayment status of the last six months (September back to April), the bill amount and the amount paid in each of those months. sex, marriage, age are kept out of the inputs, as a US lender would keep them out under ECOA and Regulation B, and are used only in section 8 to check the decisions by group.

Known caveats: the repayment status uses -1 for paid duly and 1 to 9 for months of delay; the codes -2 and 0 occur in the data but are not documented by the source (they are commonly read as "no use" and "revolving credit"). The data is from one bank in Taiwan in 2005 and says nothing about any US portfolio.

## 3. Method

The bank's model is XGBoost tuned on a 20% validation split of the training rows: 20 random configurations, each with early stopping, the best by validation AUC refitted on all training rows. The challengers:

| model | what it is | fit seconds |
|---|---|---|
| XGBoost (bank model) | tuned as above; the reference every other model is compared with | 1 |
| DistilledTree | the bank's XGBoost configuration as teacher, ten MUNGE passes, one depth-8 soft tree deployed | 980 |
| DistilledForest | the same teacher and rows, three soft trees on bootstrap samples, predicting as a group | 2881 |
| Logistic regression | standardised inputs, the scorecard baseline | 0.0 |
| CART | depth and leaf size chosen on the same validation split | 0.1 |

Three seeds (0, 1, 2) change the tuning draws and every model's own randomness; the rows stay the same. An applicant is declined when the model's P(default) exceeds 0.5. A lender sets that cut-off from its own costs; AUC and KS below do not depend on it.

## 4. Performance

Mean ± standard deviation over the three seeds, on the held-out rows. Splits count the decision nodes of every tree in the model.

| model | AUC | KS | accuracy | Brier | declined | splits | ms per applicant |
|---|---|---|---|---|---|---|---|
| XGBoost (bank model) | 0.785 ± 0.000 | 0.433 ± 0.002 | 0.823 ± 0.001 | 0.133 ± 0.000 | 0.116 ± 0.002 | 24,814 | 0.70 |
| DistilledTree | 0.783 ± 0.000 | 0.426 ± 0.003 | 0.823 ± 0.001 | 0.134 ± 0.000 | 0.114 ± 0.002 | 48 | 0.46 |
| DistilledForest | 0.784 ± 0.001 | 0.427 ± 0.001 | 0.823 ± 0.001 | 0.154 ± 0.001 | 0.113 ± 0.001 | 141 | 0.93 |
| Logistic regression | 0.711 ± 0.000 | 0.380 ± 0.000 | 0.812 ± 0.000 | 0.146 ± 0.000 | 0.070 ± 0.000 | linear | 0.20 |
| CART | 0.766 ± 0.003 | 0.415 ± 0.005 | 0.821 ± 0.000 | 0.137 ± 0.001 | 0.119 ± 0.005 | 72 | 0.14 |

Reading. The distilled tree's AUC is 0.783 against 0.785 for the bank's model (-0.22 points) with 517 times fewer splits, and it takes the same decision as the bank's model for 98.5% of applicants. Logistic regression reaches 0.711 and the tuned CART 0.766: accuracy alone hides the difference, because declining nobody is already right for most applicants, so AUC and KS, which measure how well the model ranks applicants by risk, are the columns to compare. The group of three trees ranks slightly better than one tree but its probabilities are less calibrated (Brier 0.154 against 0.134), because the group adds the trees' log probabilities; a lender who uses the probability itself, for pricing or loss provisions, would recalibrate it or use the single tree.

## 5. The rules

The tree a reviewer reads is the seed-0 distilled tree: 46 soft splits. Reading each gate as a hard decision gives the rule list below (93 lines, the four largest weights per gate, inputs standardised as in preprocessing.json); that hard reading takes the same decision as the soft tree for 92.2% of the held-out applicants.

Each gate is a weighted sum of all 20 inputs, not a test on one input as in CART, so a rule shows only its largest weights and is harder to read than a CART split. What the reviewer gets in exchange is a model of 46 decisions whose every gate can be printed and audited, with nearly the bank model's ranking power; the explanations in section 6 are computed from these gates.

```
if +16.724*repay_status_sep -1.112*paid_sep -0.854*credit_limit -0.733*bill_may -17.906 (+16 more) > 0:
    yes -> if -7.746*paid_aug -6.214*paid_jul +3.779*repay_status_sep -3.770*paid_sep -16.703 (+16 more) > 0:
        yes -> if +16.851*paid_aug +16.397*paid_may +15.144*paid_apr +12.810*repay_status_aug -28.562 (+16 more) > 0:
            yes -> predict 1 (p=0.538)
            no  -> if +15.479*paid_apr +12.328*paid_jul +8.749*paid_aug -7.911*paid_may +3.117 (+16 more) > 0:
                yes -> predict 1 (p=0.878)
                no  -> if +14.398*bill_apr +11.719*bill_may +11.071*paid_apr +7.323*paid_jul +6.676 (+16 more) > 0:
                    yes -> predict 1 (p=1.000)
                    no  -> if +17.601*bill_sep +6.023*bill_may +5.049*bill_aug +4.211*bill_jul +9.035 (+16 more) > 0:
                        yes -> predict 1 (p=0.545)
                        no  -> predict 1 (p=1.000)
        no  -> if +15.776*repay_status_aug +7.999*repay_status_sep -5.077*paid_may +2.480*paid_apr -6.039 (+16 more) > 0:
            yes -> if +15.408*education -6.022*paid_jun +5.209*paid_may +3.039*paid_aug -20.961 (+16 more) > 0:
                yes -> predict 0 (p=0.849)
                no  -> if +12.660*education -10.278*paid_jun -6.163*paid_may +6.077*repay_status_jul -7.531 (+16 more) > 0:
                    yes -> predict 1 (p=1.000)
                    no  -> if +13.627*bill_sep +6.712*bill_jun -6.175*paid_apr +5.557*bill_aug +7.548 (+16 more) > 0:
                        yes -> if +15.312*paid_may +8.264*paid_aug -8.164*education +3.206*paid_sep -7.073 (+16 more) > 0:
                            yes -> predict 0 (p=0.782)
                            no  -> predict 1 (p=1.000)
                        no  -> predict 0 (p=0.658)
            no  -> predict 0 (p=0.778)
    no  -> if +15.676*repay_status_aug -9.371*paid_apr -3.645*repay_status_sep +2.805*repay_status_jun -12.098 (+16 more) > 0:
        yes -> if +9.678*repay_status_sep +7.408*repay_status_may +7.353*paid_aug -5.733*credit_limit -12.122 (+16 more) > 0:
            yes -> if +11.912*repay_status_apr -9.961*paid_apr -7.682*paid_aug -3.400*credit_limit -15.262 (+16 more) > 0:
                yes -> if +26.448*paid_aug +8.917*paid_sep +7.597*paid_may +6.327*paid_apr +2.684 (+16 more) > 0:
                    yes -> predict 0 (p=0.944)
                    no  -> predict 1 (p=0.999)
                no  -> predict 0 (p=0.927)
            no  -> if +6.008*repay_status_jul +5.787*paid_apr +5.567*credit_limit +5.163*bill_apr +7.817 (+16 more) > 0:
                yes -> if -16.018*paid_apr -12.414*paid_aug -9.754*credit_limit -9.420*paid_may -17.314 (+16 more) > 0:
                    yes -> predict 0 (p=0.822)
                    no  -> predict 0 (p=1.000)
                no  -> if +11.778*paid_sep -9.164*bill_jul -9.138*bill_sep +8.823*credit_limit -12.898 (+16 more) > 0:
                    yes -> if +15.087*bill_sep +1.761*bill_aug -0.890*bill_apr -0.672*paid_sep +10.695 (+16 more) > 0:
                        yes -> predict 0 (p=1.000)
                        no  -> predict 1 (p=1.000)
                    no  -> if +19.897*paid_aug +6.390*paid_jul -4.965*paid_jun +4.880*paid_sep +7.615 (+16 more) > 0:
                        yes -> if +15.973*paid_sep +12.061*repay_status_apr -6.559*paid_jul -3.208*repay_status_aug +0.513 (+16 more) > 0:
                            yes -> predict 0 (p=1.000)
                            no  -> predict 0 (p=0.716)
                        no  -> predict 1 (p=0.956)
        no  -> if +21.832*paid_jul +5.243*repay_status_aug +2.438*paid_aug +1.705*credit_limit +6.784 (+16 more) > 0:
            yes -> if +13.471*repay_status_may +5.562*repay_status_apr +3.022*bill_jul -2.326*repay_status_jul -18.137 (+16 more) > 0:
                yes -> if -10.896*paid_aug +9.070*repay_status_aug +8.374*repay_status_jul -8.006*credit_limit -12.634 (+16 more) > 0:
                    yes -> predict 1 (p=0.591)
                    no  -> if +19.618*repay_status_jul +9.306*paid_may +8.781*paid_aug +6.785*credit_limit +4.738 (+16 more) > 0:
                        yes -> predict 0 (p=0.887)
                        no  -> predict 0 (p=0.817)
                no  -> if +11.147*credit_limit -8.324*bill_sep +4.393*repay_status_aug -3.032*paid_sep +9.354 (+16 more) > 0:
                    yes -> if +12.113*bill_sep +5.895*repay_status_sep +4.588*repay_status_jul +2.407*repay_status_aug +12.150 (+16 more) > 0:
                        yes -> if +14.247*repay_status_jul +7.965*repay_status_jun +5.221*repay_status_aug -4.094*repay_status_sep -14.016 (+16 more) > 0:
                            yes -> predict 0 (p=0.851)
                            no  -> if -16.784*paid_aug -11.474*repay_status_jun -10.798*repay_status_sep +10.425*credit_limit -30.039 (+16 more) > 0:
                                yes -> predict 0 (p=0.888)
                                no  -> predict 0 (p=1.000)
                        no  -> if +24.783*paid_sep +6.084*bill_sep +4.833*bill_aug -4.039*repay_status_sep +5.817 (+16 more) > 0:
                            yes -> predict 0 (p=0.967)
                            no  -> predict 0 (p=0.834)
                    no  -> if +17.681*paid_aug -7.069*education +5.033*paid_jun +4.706*credit_limit +3.225 (+16 more) > 0:
                        yes -> predict 0 (p=0.940)
                        no  -> if -18.010*paid_aug -8.072*credit_limit -6.947*education -6.741*paid_jun -8.134 (+16 more) > 0:
                            yes -> predict 0 (p=0.783)
                            no  -> if +11.996*paid_sep +11.322*credit_limit +6.018*paid_aug -4.455*bill_jul +12.554 (+16 more) > 0:
                                yes -> predict 0 (p=1.000)
                                no  -> predict 1 (p=0.858)
            no  -> if +12.168*repay_status_sep +7.998*paid_jul -6.843*repay_status_aug -6.702*repay_status_jun -4.172 (+16 more) > 0:
                yes -> if +13.026*paid_jun +7.373*paid_apr +6.801*paid_may +5.625*repay_status_sep +8.542 (+16 more) > 0:
                    yes -> predict 0 (p=0.905)
                    no  -> if +15.109*credit_limit -11.297*paid_apr -8.177*paid_sep -8.079*paid_may -14.042 (+16 more) > 0:
                        yes -> if +7.649*repay_status_jul +7.397*paid_apr +6.826*paid_sep -6.759*repay_status_sep +18.437 (+16 more) > 0:
                            yes -> predict 0 (p=0.991)
                            no  -> predict 1 (p=1.000)
                        no  -> if -7.765*repay_status_sep +6.986*paid_jun +5.903*credit_limit +5.767*paid_aug +8.503 (+16 more) > 0:
                            yes -> predict 0 (p=1.000)
                            no  -> predict 1 (p=0.981)
                no  -> if -12.069*paid_jul -7.339*bill_apr -6.274*bill_may -5.015*bill_aug -13.040 (+16 more) > 0:
                    yes -> if +9.318*paid_aug -8.033*paid_jul +6.791*paid_sep +6.371*bill_sep +2.490 (+16 more) > 0:
                        yes -> predict 0 (p=0.573)
                        no  -> if +14.927*paid_jun -7.938*paid_apr +7.439*paid_aug +5.832*paid_sep +7.471 (+16 more) > 0:
                            yes -> predict 0 (p=1.000)
                            no  -> predict 1 (p=1.000)
                    no  -> if +12.526*paid_jul +11.631*bill_sep +6.184*paid_jun +4.864*paid_apr +11.313 (+16 more) > 0:
                        yes -> if +13.115*repay_status_may -6.526*paid_sep -4.980*credit_limit +4.113*paid_aug -11.806 (+16 more) > 0:
                            yes -> predict 0 (p=0.610)
                            no  -> if +12.557*credit_limit -5.145*repay_status_jun +3.558*repay_status_jul -3.112*paid_apr +8.961 (+16 more) > 0:
                                yes -> predict 0 (p=1.000)
                                no  -> predict 0 (p=0.868)
                        no  -> if +19.307*paid_sep +9.039*paid_jun +7.341*paid_jul +6.101*paid_apr +5.907 (+16 more) > 0:
                            yes -> predict 0 (p=0.914)
                            no  -> if +9.377*credit_limit -6.188*paid_may -4.380*paid_apr +3.922*bill_may -15.015 (+16 more) > 0:
                                yes -> predict 1 (p=1.000)
                                no  -> predict 1 (p=0.523)
```

## 6. Three decisions, explained

From the seed-0 tree, on applicants it never saw in training. For each: the decision of the tree and of the bank's model, the principal reasons, the path through the tree, and the single changes that would flip the decision.

Principal reasons are read from the deployed tree itself, not from a separate explanation model: each input is set alone to its training median and the applicant re-scored, and the inputs whose neutral value lowers P(default) most are the reasons. This is the kind of ranking an adverse action notice draws on.

### declined, highest risk

Test applicant 25053: actually defaulted. Distilled tree: P(default) = 0.873, **declined**. Bank's model: P(default) = 0.845, declined. Key inputs: credit_limit 500,000, repay_status_sep 3 months late, repay_status_aug 2 months late, repay_status_jul 2 months late, bill_sep 4,957, paid_sep 0, paid_aug 0.

| principal reason | applicant | median applicant | P(default) if at median |
|---|---|---|---|
| repay_status_sep | 3 months late | 0 (undocumented code) | 0.355 |
| credit_limit | 500,000 | 140,000 | 0.826 |
| repay_status_apr | 2 months late | 0 (undocumented code) | 0.848 |
| repay_status_may | 2 months late | 0 (undocumented code) | 0.851 |

```
predicted 1 with probability 0.873
dominant leaf 56 received 0.405 of the sample's mass; its distribution is 0: 0.000, 1: 1.000
path:
  gate 0: went right with p=0.993  (+44.788[repay_status_sep] -2.201[credit_limit] -1.254[repay_status_aug])
  gate 2: went right with p=0.942  (+10.121[repay_status_sep] +5.959[repay_status_aug] -5.831[credit_limit])
  gate 6: went left with p=0.854  (+22.735[repay_status_aug] -15.617[credit_limit] -9.938[repay_status_apr])
  gate 13: went left with p=0.592  (+9.680[repay_status_sep] -7.254[repay_status_aug] -4.607[paid_apr])
  gate 27: went right with p=0.857  (+11.107[credit_limit] +9.006[repay_status_may] -8.243[bill_apr])
largest contributions: repay_status_sep 71.045, repay_status_aug 34.392, credit_limit 29.371, repay_status_apr 19.565, repay_status_may 13.715
counterfactual: set repay_status_sep from 2.678 to -1.741 and the prediction becomes 0 (p=0.682)
```

The path's own counterfactual, in the table's units: repay_status_sep from 3 months late to -1.98 would make the tree predict class 0 (probability 0.682). It moves one input just past a gate on the path and may not be a value a real applicant has.

Single changes to a value real applicants have (each input moved to a training decile, 129 candidates), smallest move first:

| change | P(default) after |
|---|---|
| repay_status_sep: 3 months late → 1 month late | 0.467 |
| repay_status_sep: 3 months late → 0 (undocumented code) | 0.355 |
| repay_status_sep: 3 months late → paid duly | 0.332 |

### declined, nearest the cut-off

Test applicant 7752: actually defaulted. Distilled tree: P(default) = 0.500, **declined**. Bank's model: P(default) = 0.414, approved. Key inputs: credit_limit 120,000, repay_status_sep 4 months late, repay_status_aug 3 months late, repay_status_jul 2 months late, bill_sep 149,229, paid_sep 0, paid_aug 0.

| principal reason | applicant | median applicant | P(default) if at median |
|---|---|---|---|
| repay_status_sep | 4 months late | 0 (undocumented code) | 0.210 |
| bill_jul | 127,516 | 20,202 | 0.483 |
| bill_jun | 124,530 | 19,147 | 0.483 |
| bill_sep | 149,229 | 22,468 | 0.493 |

```
predicted 1 with probability 0.500
dominant leaf 14 received 0.456 of the sample's mass; its distribution is 0: 0.462, 1: 0.538
path:
  gate 0: went right with p=1.000  (+59.633[repay_status_sep] -1.843[repay_status_aug] +1.034[education])
  gate 2: went right with p=0.460  (+13.476[repay_status_sep] -11.529[paid_apr] +8.755[repay_status_aug])
  gate 6: went right with p=0.992  (+99.880[paid_apr] +33.401[repay_status_aug] +11.905[repay_status_sep])
largest contributions: repay_status_sep 78.580, paid_apr 58.437, repay_status_aug 25.954, education 6.293, paid_aug 3.987
counterfactual: set paid_jun from 0.017 to -0.257 and the prediction becomes 0 (p=0.504)
```

The path's own counterfactual, in the table's units: paid_jun from 5,067 to 845.17 would make the tree predict class 0 (probability 0.504). It moves one input just past a gate on the path and may not be a value a real applicant has.

Single changes to a value real applicants have (each input moved to a training decile, 134 candidates), smallest move first:

| change | P(default) after |
|---|---|
| paid_aug: 0 → 240 | 0.500 |
| paid_sep: 0 → 316 | 0.500 |
| bill_jun: 124,530 → 122,342 | 0.500 |
| paid_may: 1,548 → 2,138 | 0.499 |
| paid_aug: 0 → 1,158 | 0.498 |

### approved, typical applicant

Test applicant 11943: actually did not default. Distilled tree: P(default) = 0.147, **approved**. Bank's model: P(default) = 0.162, approved. Key inputs: credit_limit 60,000, repay_status_sep 0 (undocumented code), repay_status_aug 0 (undocumented code), repay_status_jul 0 (undocumented code), bill_sep 57,171, paid_sep 2,464, paid_aug 1,932.

| principal reason | applicant | median applicant | P(default) if at median |
|---|---|---|---|
| credit_limit | 60,000 | 140,000 | 0.077 |
| bill_jul | 53,271 | 20,202 | 0.121 |
| bill_aug | 58,656 | 21,451 | 0.134 |
| bill_jun | 54,331 | 19,147 | 0.137 |

```
predicted 0 with probability 0.853
dominant leaf 299 received 0.150 of the sample's mass; its distribution is 0: 1.000, 1: 0.000
path:
  gate 0: went left with p=0.976  (+0.711[credit_limit] +0.252[repay_status_sep] +0.214[paid_sep])
  gate 1: went left with p=0.592  (+1.740[paid_apr] +1.717[repay_status_aug] +1.428[credit_limit])
  gate 3: went right with p=0.546  (-4.234[paid_jul] -1.419[credit_limit] +0.574[repay_status_aug])
  gate 8: went left with p=0.987  (+3.117[repay_status_may] +1.395[repay_status_apr] +0.995[credit_limit])
  gate 17: went right with p=0.531  (-9.275[credit_limit] -0.655[bill_sep] +0.585[paid_sep])
  gate 36: went right with p=0.947  (-1.577[credit_limit] +0.954[bill_sep] +0.631[repay_status_jul])
  gate 74: went left with p=1.000  (+1.959[repay_status_jul] +1.500[repay_status_jun] +0.572[repay_status_aug])
  gate 149: went left with p=0.962  (-8.675[credit_limit] +2.883[paid_aug] -2.161[repay_status_jun])
largest contributions: credit_limit 7.789, paid_jul 3.231, repay_status_aug 2.520, paid_apr 1.896, repay_status_may 1.577
counterfactual: set repay_status_sep from 0.015 to 1.965 and the prediction becomes 1 (p=0.723)
```

The path's own counterfactual, in the table's units: repay_status_sep from 0 (undocumented code) to 2.20 would make the tree predict class 1 (probability 0.723). It moves one input just past a gate on the path and may not be a value a real applicant has.

Single changes to a value real applicants have (each input moved to a training decile, 134 candidates), smallest move first:

| change | P(default) after |
|---|---|
| repay_status_sep: 0 (undocumented code) → 2 months late | 0.695 |

Most of these levers are past repayment behaviour, which the applicant cannot change today. That is a property of this data, where recent delays carry most of the signal, and it is the honest answer to "what would have changed the decision".

## 7. Agreement with the bank's model

| model | same decision as the bank's model | AUC difference (points) |
|---|---|---|
| DistilledTree | 0.985 ± 0.001 | -0.22 |
| DistilledForest | 0.985 ± 0.001 | -0.12 |
| Logistic regression | 0.950 ± 0.002 | -7.43 |
| CART | 0.972 ± 0.003 | -1.88 |

## 8. Decisions by group

Seed 0, held-out applicants, cut-off 0.5. Sex, marital status and age are not model inputs; the question is whether the remaining inputs still treat the groups differently. "Defaults approved" is the share of applicants who defaulted and were approved; "repayers declined" the share who repaid and were declined. The approval ratio divides each group's approval rate by the highest in its family; US practice reads a ratio below 0.8 as a sign of adverse impact to investigate.

**DistilledTree**

| group | applicants | default rate | declined | AUC | defaults approved | repayers declined | approval ratio |
|---|---|---|---|---|---|---|---|
| sex: male | 2436 | 0.242 | 0.115 | 0.772 | 0.666 | 0.045 | 0.993 |
| sex: female | 3564 | 0.207 | 0.109 | 0.791 | 0.630 | 0.041 | 1.000 |
| married | 2765 | 0.245 | 0.121 | 0.779 | 0.646 | 0.045 | 0.982 |
| single | 3149 | 0.201 | 0.105 | 0.789 | 0.642 | 0.042 | 1.000 |
| age under 30 | 1933 | 0.215 | 0.125 | 0.787 | 0.605 | 0.051 | 0.972 |
| age 30 to 44 | 2970 | 0.211 | 0.100 | 0.777 | 0.674 | 0.039 | 1.000 |
| age 45 and over | 1097 | 0.261 | 0.122 | 0.795 | 0.643 | 0.039 | 0.975 |

**XGBoost (bank model)**

| group | applicants | default rate | declined | AUC | defaults approved | repayers declined | approval ratio |
|---|---|---|---|---|---|---|---|
| sex: male | 2436 | 0.242 | 0.122 | 0.771 | 0.652 | 0.049 | 0.990 |
| sex: female | 3564 | 0.207 | 0.113 | 0.795 | 0.629 | 0.045 | 1.000 |
| married | 2765 | 0.245 | 0.126 | 0.776 | 0.631 | 0.047 | 0.981 |
| single | 3149 | 0.201 | 0.109 | 0.796 | 0.644 | 0.046 | 1.000 |
| age under 30 | 1933 | 0.215 | 0.129 | 0.793 | 0.600 | 0.055 | 0.971 |
| age 30 to 44 | 2970 | 0.211 | 0.103 | 0.782 | 0.668 | 0.042 | 1.000 |
| age 45 and over | 1097 | 0.261 | 0.129 | 0.781 | 0.633 | 0.046 | 0.971 |

## 9. Stability across seeds

| seed | XGBoost trees | XGBoost max depth | bank AUC | tree AUC | tree splits | same decision |
|---|---|---|---|---|---|---|
| 0 | 276 | 9 | 0.7849 | 0.7829 | 46 | 0.985 |
| 1 | 203 | 10 | 0.7857 | 0.7830 | 51 | 0.985 |
| 2 | 308 | 9 | 0.7852 | 0.7833 | 47 | 0.986 |

## 10. Limitations

- One public dataset from one bank in 2005. The numbers are a demonstration of the procedure, not evidence about any other portfolio.
- Three seeds on one split: the standard deviations are the spread of the method, not of the data.
- The rules in section 5 are a hard reading of soft gates; near a gate the two disagree, and the agreement rate measures how often. Each gate weighs every input, so the rules are less readable than a CART tree.
- Principal reasons and reachable changes are statements about the model, not advice to the applicant.
- Leaving protected attributes out of the inputs does not by itself remove their proxies; section 8 is the check, and it should be repeated on the lender's own data.
- Fitting the distilled tree takes minutes where XGBoost takes seconds; in production this happens at each refit, not at each decision.

## 11. Shipped files

| file | what it is |
|---|---|
| model.json | the seed-0 distilled tree as plain arrays (54 KB); predicts with numpy alone |
| preprocessing.json | the means and scales to apply before calling the model, in input order |
| rules.txt | the hard-rule reading of that tree |
| results.json | every number in this document, seed by seed |
| run.py, report.py | regenerate everything |

The numpy export and the fitted tree agree to 2.9e-07 in probability on every held-out applicant.

## 12. Monitoring after deployment

What a reviewer would ask to see each quarter, all computable from the files here: AUC, KS and the decline rate on new applicants against section 4; the share of decisions where the distilled tree and the bank's model disagree, against section 7; the group table of section 8; and a refit with the rules compared to section 5. A drift in any of these is a reason to look, not a reason to retrain automatically.

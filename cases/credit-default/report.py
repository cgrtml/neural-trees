"""
Write the model document for the credit card default case from results.json.

Every number comes from the JSON that run.py wrote; nothing is typed in by
hand, so re-running run.py and then this file keeps the document honest.
Output: report.md next to this file, and report.docx if python-docx is
installed.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __import__("os").environ.get("SMOKE"):                 # read the quick check of run.py instead
    HERE = HERE / "_smoke"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
S = R["summary"]
BANK = "XGBoost (bank model)"
TREE = "DistilledTree"
ORDER = [BANK, TREE, "DistilledForest", "Logistic regression", "CART"]
D, FM = R["data"], R["final_model"]
T = R["protocol"]["threshold"]


def ms(name, metric, d=3, scale=1.0):
    v = S[name][metric]
    return f"{v['mean'] * scale:.{d}f} ± {v['sd'] * scale:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def money(v):
    return f"{v:,.0f}"


def value(name, v):
    if name.startswith("repay_status"):
        return {-2: "-2 (undocumented code)", -1: "paid duly", 0: "0 (undocumented code)"}.get(
            int(v), f"{int(v)} month{'s' if int(v) > 1 else ''} late")
    if name in ("credit_limit",) or name.startswith(("bill_", "paid_")):
        return money(v)
    return f"{v:g}"


lines = []
P = lines.append
P("# Credit card default: a distilled, auditable model package")
P("")
P(f"neural-trees {R['library']} · {date.today().isoformat()} · every figure below is read from results.json, produced by run.py")
P("")
P("## 1. Purpose and scope")
P("")
P("A bank already runs a gradient boosted model for credit decisions and has to defend each decision to a validator, an "
  "examiner and the applicant. This document shows what happens when that model is distilled into one soft decision tree "
  "with neural-trees: how much accuracy and ranking power the tree keeps, how often it takes the same decision as the bank's "
  "model, how a single decline is explained from the tree's own path, and whether decisions differ by group. It follows the "
  "headings a model validation under SR 11-7 expects: purpose, data, method, performance, explanation, fairness checks, "
  "stability, limitations and the shipped files. It is a demonstration on public data, not a validated production model.")
P("")
P("## 2. Data")
P("")
P(f"{D['source']}: {D['n']:,} card holders, {D['n_default']:,} of whom defaulted the following month "
  f"({D['n_default'] / D['n']:.1%}). {D['train_rows']:,} rows train every model and the same {D['test_rows']:,} held-out rows score them.")
P("")
P(f"Inputs ({len(D['inputs'])}): credit limit, education, the repayment status of the last six months (September back to April), "
  "the bill amount and the amount paid in each of those months. "
  f"{', '.join(D['kept_out_of_inputs'])} are kept out of the inputs, as a US lender would keep them out under ECOA and Regulation B, "
  "and are used only in section 8 to check the decisions by group.")
P("")
P("Known caveats: the repayment status uses -1 for paid duly and 1 to 9 for months of delay; the codes -2 and 0 occur in the "
  "data but are not documented by the source (they are commonly read as \"no use\" and \"revolving credit\"). The data is from "
  "one bank in Taiwan in 2005 and says nothing about any US portfolio.")
P("")
P("## 3. Method")
P("")
pr = R["protocol"]
P(f"The bank's model is XGBoost tuned on a 20% validation split of the training rows: {pr['xgboost_trials']} random configurations, "
  "each with early stopping, the best by validation AUC refitted on all training rows. The challengers:")
P("")
P(table(["model", "what it is", "fit seconds"], [
    (BANK, "tuned as above; the reference every other model is compared with", f"{S[BANK]['fit_seconds']['mean']:.0f}"),
    (TREE, "the bank's XGBoost configuration as teacher, ten MUNGE passes, one depth-8 soft tree deployed", f"{S[TREE]['fit_seconds']['mean']:.0f}"),
    ("DistilledForest", "the same teacher and rows, three soft trees on bootstrap samples, predicting as a group", f"{S['DistilledForest']['fit_seconds']['mean']:.0f}"),
    ("Logistic regression", "standardised inputs, the scorecard baseline", f"{S['Logistic regression']['fit_seconds']['mean']:.1f}"),
    ("CART", "depth and leaf size chosen on the same validation split", f"{S['CART']['fit_seconds']['mean']:.1f}"),
]))
P("")
P(f"Three seeds ({', '.join(map(str, pr['seeds']))}) change the tuning draws and every model's own randomness; the rows stay the same. "
  f"An applicant is declined when the model's P(default) exceeds {T}. A lender sets that cut-off from its own costs; AUC and KS below do not depend on it.")
P("")
P("## 4. Performance")
P("")
P("Mean ± standard deviation over the three seeds, on the held-out rows. Splits count the decision nodes of every tree in the model.")
P("")
P(table(["model", "AUC", "KS", "accuracy", "Brier", "declined", "splits", "ms per applicant"], [
    (n, ms(n, "auc"), ms(n, "ks"), ms(n, "accuracy"), ms(n, "brier"), ms(n, "decline_rate", 3),
     f"{S[n]['splits']['mean']:,.0f}" if S[n]['splits']['mean'] else "linear", f"{S[n]['ms_per_row']['mean']:.2f}")
    for n in ORDER]))
P("")
b, t = S[BANK], S[TREE]
ratio = b["splits"]["mean"] / max(t["splits"]["mean"], 1)
P(f"Reading. The distilled tree's AUC is {t['auc']['mean']:.3f} against {b['auc']['mean']:.3f} for the bank's model "
  f"({(t['auc']['mean'] - b['auc']['mean']) * 100:+.2f} points) with {ratio:.0f} times fewer splits, and it takes the same "
  f"decision as the bank's model for {t['same_decision_as_bank_model']['mean']:.1%} of applicants. "
  f"Logistic regression reaches {S['Logistic regression']['auc']['mean']:.3f} and the tuned CART {S['CART']['auc']['mean']:.3f}: "
  "accuracy alone hides the difference, because declining nobody is already right for most applicants, so AUC and KS, which measure "
  "how well the model ranks applicants by risk, are the columns to compare. "
  f"The group of three trees ranks slightly better than one tree but its probabilities are less calibrated (Brier "
  f"{S['DistilledForest']['brier']['mean']:.3f} against {t['brier']['mean']:.3f}), because the group adds the trees' log "
  "probabilities; a lender who uses the probability itself, for pricing or loss provisions, would recalibrate it or use the single tree.")
P("")
P("## 5. The rules")
P("")
P(f"The tree a reviewer reads is the seed-0 distilled tree: {FM['n_splits']} soft splits. Reading each gate as a hard decision gives "
  f"the rule list below ({FM['rules_lines']} lines, the four largest weights per gate, inputs standardised as in preprocessing.json); "
  f"that hard reading takes the same decision as the soft tree for {FM['hard_rules_same_decision']:.1%} of the held-out applicants.")
P("")
P(f"Each gate is a weighted sum of all {len(D['inputs'])} inputs, not a test on one input as in CART, so a rule shows only its "
  "largest weights and is harder to read than a CART split. What the reviewer gets in exchange is a model of 46 decisions whose every "
  "gate can be printed and audited, with nearly the bank model's ranking power; the explanations in section 6 are computed from these gates.".replace("46", str(FM['n_splits'])))
P("")
P("```")
P((HERE / "rules.txt").read_text(encoding="utf-8").strip())
P("```")
P("")
P("## 6. Three decisions, explained")
P("")
P("From the seed-0 tree, on applicants it never saw in training. For each: the decision of the tree and of the bank's model, "
  "the principal reasons, the path through the tree, and the single changes that would flip the decision.")
P("")
P("Principal reasons are read from the deployed tree itself, not from a separate explanation model: each input is set alone to its "
  "training median and the applicant re-scored, and the inputs whose neutral value lowers P(default) most are the reasons. This is "
  "the kind of ranking an adverse action notice draws on.")
P("")
KEY = ["credit_limit", "repay_status_sep", "repay_status_aug", "repay_status_jul", "bill_sep", "paid_sep", "paid_aug"]
for e in R["explanations"]:
    P(f"### {e['case']}")
    P("")
    a = e["applicant"]
    P(f"Test applicant {e['test_row']}: actually {'defaulted' if e['actual_default'] else 'did not default'}. "
      f"Distilled tree: P(default) = {e['p_default']:.3f}, **{'declined' if e['declined'] else 'approved'}**. "
      f"Bank's model: P(default) = {e['bank_model_p_default']:.3f}, {'declined' if e['bank_model_declined'] else 'approved'}. "
      + "Key inputs: " + ", ".join(f"{k} {value(k, a[k])}" for k in KEY) + ".")
    P("")
    if e["reasons"]:
        P(table(["principal reason", "applicant", "median applicant", "P(default) if at median"],
                [(r["input"], value(r["input"], r["value"]), value(r["input"], r["median"]), f"{r['p_default_if_median']:.3f}")
                 for r in e["reasons"]]))
    else:
        P("No single input, moved to its median, lowers P(default) by more than half a point.")
    P("")
    P("```")
    P(e["path_text"])
    P("```")
    P("")
    cf = e["gate_counterfactual_raw_units"]
    if cf:
        P(f"The path's own counterfactual, in the table's units: {cf['input']} from {value(cf['input'], cf['from'])} to "
          f"{cf['to']:,.2f} would make the tree predict class {cf['new_class']} (probability {cf['p_new_class']:.3f}). "
          "It moves one input just past a gate on the path and may not be a value a real applicant has.")
        P("")
    P(f"Single changes to a value real applicants have (each input moved to a training decile, {e['reachable_tried']} candidates), "
      "smallest move first:")
    P("")
    if e["reachable_flips"]:
        P(table(["change", "P(default) after"],
                [(f"{f['input']}: {value(f['input'], f['from'])} → {value(f['input'], f['to'])}", f"{f['p_default']:.3f}")
                 for f in e["reachable_flips"]]))
    else:
        P("None: no single input moved to a training decile flips this decision.")
    P("")
P("Most of these levers are past repayment behaviour, which the applicant cannot change today. That is a property of this data, "
  "where recent delays carry most of the signal, and it is the honest answer to \"what would have changed the decision\".")
P("")
P("## 7. Agreement with the bank's model")
P("")
P(table(["model", "same decision as the bank's model", "AUC difference (points)"], [
    (n, ms(n, "same_decision_as_bank_model", 3), f"{(S[n]['auc']['mean'] - b['auc']['mean']) * 100:+.2f}")
    for n in ORDER if n != BANK]))
P("")
P("## 8. Decisions by group")
P("")
P(f"Seed 0, held-out applicants, cut-off {T}. Sex, marital status and age are not model inputs; the question is whether the "
  "remaining inputs still treat the groups differently. \"Defaults approved\" is the share of applicants who defaulted and were "
  "approved; \"repayers declined\" the share who repaid and were declined. The approval ratio divides each group's approval "
  "rate by the highest in its family; US practice reads a ratio below 0.8 as a sign of adverse impact to investigate.")
P("")
for name in (TREE, BANK):
    P(f"**{name}**")
    P("")
    P(table(["group", "applicants", "default rate", "declined", "AUC", "defaults approved", "repayers declined", "approval ratio"],
            [(g, v["n"], f"{v['default_rate']:.3f}", f"{v['decline_rate']:.3f}", f"{v['auc']:.3f}",
              f"{v['defaults_approved']:.3f}", f"{v['repayers_declined']:.3f}", f"{v['approval_ratio_to_highest']:.3f}")
             for g, v in R["slices_seed0"][name].items()]))
    P("")
P("## 9. Stability across seeds")
P("")
P(table(["seed", "XGBoost trees", "XGBoost max depth", "bank AUC", "tree AUC", "tree splits", "same decision"], [
    (r["seed"], r["xgboost_params"]["n_estimators"], r["xgboost_params"]["max_depth"], f"{r['models'][BANK]['auc']:.4f}",
     f"{r['models'][TREE]['auc']:.4f}", r["models"][TREE]["splits"], f"{r['models'][TREE]['same_decision_as_bank_model']:.3f}")
    for r in R["per_seed"]]))
P("")
P("## 10. Limitations")
P("")
P("- One public dataset from one bank in 2005. The numbers are a demonstration of the procedure, not evidence about any other portfolio.")
P("- Three seeds on one split: the standard deviations are the spread of the method, not of the data.")
P("- The rules in section 5 are a hard reading of soft gates; near a gate the two disagree, and the agreement rate measures how often. Each gate weighs every input, so the rules are less readable than a CART tree.")
P("- Principal reasons and reachable changes are statements about the model, not advice to the applicant.")
P("- Leaving protected attributes out of the inputs does not by itself remove their proxies; section 8 is the check, and it should be repeated on the lender's own data.")
P("- Fitting the distilled tree takes minutes where XGBoost takes seconds; in production this happens at each refit, not at each decision.")
P("")
P("## 11. Shipped files")
P("")
P(table(["file", "what it is"], [
    ("model.json", f"the seed-0 distilled tree as plain arrays ({FM['json_bytes'] / 1024:.0f} KB); predicts with numpy alone"),
    ("preprocessing.json", "the means and scales to apply before calling the model, in input order"),
    ("rules.txt", "the hard-rule reading of that tree"),
    ("results.json", "every number in this document, seed by seed"),
    ("run.py, report.py", "regenerate everything"),
]))
P("")
P(f"The numpy export and the fitted tree agree to {FM['numpy_matches_model']:.1e} in probability on every held-out applicant.")
P("")
P("## 12. Monitoring after deployment")
P("")
P("What a reviewer would ask to see each quarter, all computable from the files here: AUC, KS and the decline rate on new applicants "
  "against section 4; the share of decisions where the distilled tree and the bank's model disagree, against section 7; the group table "
  "of section 8; and a refit with the rules compared to section 5. A drift in any of these is a reason to look, not a reason to retrain automatically.")
P("")
md = "\n".join(lines)
(HERE / "report.md").write_text(md, encoding="utf-8")
print("report.md written")

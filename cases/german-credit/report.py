"""
Write the model document for the German Credit case from results.json.

Every number in the document comes from the JSON the run wrote; nothing is
typed in by hand, so re-running run.py and then this file keeps the document
honest. Output: report.md next to this file, and report.docx if python-docx
is installed.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
S = R["summary"]
ORDER = ["Soft tree, per-leaf", "Soft tree, depth 4", "Logistic regression", "CART", "Random forest", "XGBoost", "LightGBM"]
SOFT = "Soft tree, per-leaf"


def ms(name, metric, d=3):
    v = S[name][metric]
    return f"{v['mean']:.{d}f} ± {v['sd']:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def best(metric, low=False):
    return min(ORDER, key=lambda n: S[n][metric]["mean"] * (1 if low else -1))


lines = []
P = lines.append
D = R["data"]
FM = R["final_model"]
P("# German Credit: an auditable model package")
P("")
P(f"neural-trees {R['library']} · {date.today().isoformat()} · every figure below is read from results.json, produced by run.py")
P("")
P("## 1. Purpose and scope")
P("")
P("A credit decision has to be defended one applicant at a time: the reviewer asks why this person was refused, "
  "what would have changed the answer, and whether the same rule was applied to everyone. This document shows what a "
  "soft decision tree from neural-trees gives a reviewer on the standard public credit dataset, measured against the "
  "models a credit team would otherwise use, on identical folds. It follows the headings a model validation under "
  "SR 11-7 or the EU AI Act's Annex IV documentation expects: purpose, data, method, performance, explanation, "
  "stability, limitations, and the shipped artefacts. It is a demonstration on public data, not a validated production model.")
P("")
P("## 2. Data")
P("")
P(f"{D['source']}: {D['n']} loan applicants, {D['n_good']} good and {D['n_bad']} bad, "
  f"{len(D['numeric'])} numeric attributes ({', '.join(D['numeric'])}) and {len(D['categorical'])} categorical ones "
  f"({', '.join(D['categorical'])}). After one-hot encoding the model sees {FM['n_features_after_encoding']} inputs.")
P("")
P(f"The dataset ships a cost matrix: a bad applicant accepted costs {D['cost_matrix']['bad_called_good']:.0f}, a good applicant refused costs "
  f"{D['cost_matrix']['good_called_bad']:.0f}. The decision that minimises expected cost is therefore to refuse when P(bad) exceeds "
  f"1/6 ≈ {D['bayes_threshold_on_p_bad']:.3f}, not 0.5. Both thresholds are reported; the cost column is the one that matters to the business.")
P("")
P("Known caveat: the UCI coding of `personal_status` (used below for the sex slice) has been reported as unreliable (Grömping, 2019, "
  "\"South German Credit Data: Correcting a Widely Used Data Set\"). The sex slice is therefore illustrative of the procedure, not a finding about the population.")
P("")
P("## 3. Method")
P("")
pr = R["protocol"]
P(f"{len(pr['seeds'])} seeds × {pr['folds']}-fold stratified cross-validation, {pr['fits_per_model']} fits per model, "
  f"the same folds for every model. Preprocessing: {pr['preprocessing']}. Nothing was tuned on this data: each model runs one "
  "fixed configuration, the same one used in the library's 24-dataset benchmark.")
P("")
P(table(["model", "configuration", "seconds per fit"], [
    (SOFT, "depth 6 grown per leaf, residual initialisation, 180 epochs", f"{R['seconds_per_fit'][SOFT]:.1f}"),
    ("Soft tree, depth 4", "complete tree, 150 epochs", f"{R['seconds_per_fit']['Soft tree, depth 4']:.1f}"),
    ("Logistic regression", "scikit-learn defaults, the scorecard baseline", f"{R['seconds_per_fit']['Logistic regression']:.1f}"),
    ("CART", "scikit-learn defaults, unpruned", f"{R['seconds_per_fit']['CART']:.1f}"),
    ("Random forest", "300 trees", f"{R['seconds_per_fit']['Random forest']:.1f}"),
    ("XGBoost", "300 rounds, depth 6, learning rate 0.1", f"{R['seconds_per_fit']['XGBoost']:.1f}"),
    ("LightGBM", "300 rounds, 31 leaves, learning rate 0.1", f"{R['seconds_per_fit']['LightGBM']:.1f}"),
]))
P("")
P("## 4. Performance")
P("")
P("Mean ± standard deviation over the 15 fits. Cost is per applicant under the dataset's cost matrix; lower is better. "
  "Refusal rate is the share of applicants refused at the cost-optimal threshold.")
P("")
P(table(["model", "accuracy", "balanced acc.", "AUC", "Brier", "cost at 0.5", "cost at 1/6", "refused at 1/6"], [
    (n, ms(n, "accuracy"), ms(n, "balanced_accuracy"), ms(n, "auc"), ms(n, "brier"), ms(n, "cost_at_0.5"), ms(n, "cost_at_bayes"), ms(n, "refusal_rate_at_bayes", 2))
    for n in ORDER]))
P("")
P(f"Best accuracy: {best('accuracy')}. Best AUC: {best('auc')}. Lowest cost at the cost-optimal threshold: {best('cost_at_bayes', low=True)}. "
  f"Best calibrated by Brier score: {best('brier', low=True)}.")
P("")
lr, rf, lg, xg = S["Logistic regression"], S["Random forest"], S["LightGBM"], S["XGBoost"]
sf = S[SOFT]
P(f"Reading. The per-leaf soft tree and logistic regression cannot be told apart on this data: "
  f"accuracy {sf['accuracy']['mean']:.3f} against {lr['accuracy']['mean']:.3f}, AUC {sf['auc']['mean']:.3f} against {lr['auc']['mean']:.3f}, "
  f"cost {sf['cost_at_bayes']['mean']:.3f} against {lr['cost_at_bayes']['mean']:.3f} (p = {R['tests_vs_soft_per_leaf']['Logistic regression']['cost_at_bayes']['p_t']:.2f}). Random forest is "
  f"{(rf['accuracy']['mean'] - sf['accuracy']['mean']) * 100:.1f} points more accurate and no cheaper at the cost-optimal threshold "
  f"({rf['cost_at_bayes']['mean']:.3f}), and it refuses {rf['refusal_rate_at_bayes']['mean']:.0%} of applicants to get there against the soft tree's "
  f"{sf['refusal_rate_at_bayes']['mean']:.0%}. The boosted models match the soft tree on accuracy and lose on cost "
  f"(XGBoost {xg['cost_at_bayes']['mean']:.3f}, LightGBM {lg['cost_at_bayes']['mean']:.3f}) because their untuned probabilities are poorly calibrated "
  f"(section 5), and a cost-weighted decision at a 1/6 threshold is exactly where calibration is paid for. "
  "What the soft tree adds over logistic regression is sections 6 and 7: a small tree of rules and a per-applicant path with counterfactuals, at the same accuracy.")
P("")
P(f"Paired tests of each model against the per-leaf soft tree over the {pr['fits_per_model']} fits (difference is model minus soft tree; "
  "a negative cost difference favours the model):")
P("")
T = R["tests_vs_soft_per_leaf"]
P(table(["model", "Δ accuracy", "p", "Δ AUC", "p", "Δ cost at 1/6", "p"], [
    (n, f"{T[n]['accuracy']['diff_mean']:+.3f}", f"{T[n]['accuracy']['p_t']:.3f}",
        f"{T[n]['auc']['diff_mean']:+.3f}", f"{T[n]['auc']['p_t']:.3f}",
        f"{T[n]['cost_at_bayes']['diff_mean']:+.3f}", f"{T[n]['cost_at_bayes']['p_t']:.3f}")
    for n in ORDER if n != SOFT]))
P("")
P("p is a paired t-test over the fits; the fits share data, so the test is optimistic and a p just under 0.05 should be read as \"probably\", not \"proven\".")
P("")
P("## 5. Calibration")
P("")
P("Out-of-fold P(bad) from seed 0, cut into ten equal-width bins: what the model said against what happened. ECE is the expected calibration error, "
  "the bin-weighted gap between the two columns.")
P("")
for n in (SOFT, "Logistic regression", "Random forest", "LightGBM"):
    c = R["calibration_seed0_oof"][n]
    P(f"**{n}**, ECE {c['ece']:.3f}")
    P("")
    P(table(["P(bad) bin", "applicants", "mean predicted", "observed bad rate"],
            [(b["bin"], b["n"], f"{b['mean_predicted']:.3f}", f"{b['observed_bad_rate']:.3f}") for b in c["bins"]]))
    P("")
P("## 6. The rules")
P("")
P(f"The final model is trained on all {D['n']} applicants (seed 0). It has {FM['n_splits']} soft splits and {FM['n_leaves']} leaves; "
  f"training accuracy {FM['training_accuracy']:.3f}. Reading each gate as a hard decision gives the rule list in rules.txt "
  f"({FM['rules_lines']} lines); that hard reading agrees with the soft model on {FM['hard_agreement_in_sample']:.1%} of the training rows and, "
  f"out of fold, on {100 * sum(s['hard_agreement'] for s in R['structure_per_fit']) / len(R['structure_per_fit']):.1f}% of held-out rows on average. "
  "Each split is a weighted sum of the inputs, so a rule is not a single threshold; the print below shows the five largest raw weights per gate, "
  "and the table after it shows which inputs actually decide each gate, ranked by mean |weight × value| over the training rows, "
  "which puts a 0/1 one-hot column and a standardised numeric on the same footing.")
P("")
P("```")
P((HERE / "rules.txt").read_text(encoding="utf-8").strip())
P("```")
P("")
P(table(["gate", "inputs that decide it (share of the gate's mean absolute contribution)"],
        [(g["gate"], ", ".join(f"{n} ({v:.0%})" for n, v in g["inputs"])) for g in FM["gate_contributions"]]))
P("")
P("## 7. Three decisions, explained")
P("")
P("From the seed-0, fold-0 model, on applicants it never saw in training. Each explanation gives the predicted class and its probability, "
  "the leaf that received the applicant, the gates on the way with the inputs that decided them, and the smallest single change that flips the decision, "
  "verified by re-predicting the changed applicant.")
P("")
for e in R["explanations"]:
    P(f"### {e['case']}")
    P("")
    a = e["applicant"]
    P(f"Applicant {e['row_index']}: actual outcome **{e['actual']}**, P(bad) = {e['p_bad']:.3f}, "
      f"so {'refused' if e['p_bad'] > R['data']['bayes_threshold_on_p_bad'] else 'accepted'} at the cost-optimal threshold. "
      f"Checking status '{a['checking_status']}', duration {a['duration']} months, amount {a['credit_amount']}, "
      f"credit history '{a['credit_history']}', savings '{a['savings_status']}', employment '{a['employment']}', age {a['age']}, purpose '{a['purpose']}'.")
    P("")
    P("```")
    P(e["text"])
    P("```")
    P("")
    if e["counterfactual_in_words"]:
        P(f"The gate-level counterfactual above, in the table's own terms: {e['counterfactual_in_words']}. "
          "It moves one encoded input inside its training range and is a statement about the gate, not necessarily a reachable applicant.")
    else:
        P("No single encoded input, kept inside its training range, flips this decision at the gates on the path.")
    P("")
    P(f"Reachable single changes that flip the decision at the cost-optimal threshold of 1/6, tried on the raw table and re-encoded "
      f"({e['reachable_tried']} candidates: every other value of each categorical column, each numeric column moved to a training decile), smallest first:")
    P("")
    if e["reachable_flips"]:
        P(table(["change", "P(bad) after"], [(f"{f['column']}: '{f['from']}' → '{f['to']}'" if isinstance(f["to"], str) else f"{f['column']}: {f['from']} → {f['to']:g}", f"{f['p_bad']:.3f}") for f in e["reachable_flips"]]))
    else:
        P("None: no single change of one column flips this decision.")
    P("")
P("## 8. Slices: is the same rule applied to everyone?")
P("")
P("Out-of-fold decisions from seed 0 at the cost-optimal threshold. \"Bad missed\" is the share of actually bad applicants that were accepted; "
  "\"good refused\" the share of actually good applicants that were refused. A model can be equally accurate in two groups and still refuse one of them more often, "
  "so both are shown, for the soft tree and for logistic regression.")
P("")
for title, key in (("Soft tree, per-leaf", "slices_soft_per_leaf_seed0_oof"), ("Logistic regression", "slices_logistic_seed0_oof")):
    P(f"**{title}**")
    P("")
    P(table(["group", "applicants", "actual bad rate", "refusal rate", "accuracy", "bad missed", "good refused"],
            [(g, v["n"], f"{v['actual_bad_rate']:.3f}", f"{v['refusal_rate']:.3f}", f"{v['accuracy']:.3f}", f"{v['bad_missed_rate']:.3f}", f"{v['good_refused_rate']:.3f}")
             for g, v in R[key].items()]))
    P("")
P("Age and sex are inputs to the model here because they are in the public dataset; a lender in the EU or the US would remove them (and their proxies) before training, "
  "and would then use exactly these tables to check whether the remaining inputs still act as a proxy.")
P("")
P("## 9. Stability across refits")
P("")
st = R["structure_per_fit"]
roots = Counter(tuple(s["root_top_features"][:1]) for s in st)
P(f"Over the {len(st)} refits the per-leaf soft tree grew between {min(s['n_splits'] for s in st)} and {max(s['n_splits'] for s in st)} splits "
  f"(median {sorted(s['n_splits'] for s in st)[len(st) // 2]}). The input with the largest mean |weight × value| at the root gate was "
  + ", ".join(f"'{k[0]}' in {v} of {len(st)} fits" for k, v in roots.most_common(3)) + ". "
  f"Hard-rule agreement with the soft model on held-out rows ranged from {min(s['hard_agreement'] for s in st):.3f} to {max(s['hard_agreement'] for s in st):.3f}.")
P("")
P(table(["seed", "fold", "splits", "root gate, largest inputs", "hard agreement"],
        [(s["seed"], s["fold"], s["n_splits"], ", ".join(s["root_top_features"]), f"{s['hard_agreement']:.3f}") for s in st]))
P("")
P("## 10. Limitations")
P("")
P("- 1 000 rows is small; the standard deviations above are the honest width of every claim, and several models are within one of them of each other.")
P("- The rules are a hard reading of soft gates. Near a gate the two disagree; the agreement rates above measure how often.")
P("- Two kinds of counterfactual are shown on purpose. The gate-level one reads the model's own gates and is bounded to the training range; the reachable one changes a column of the raw table. Both are statements about the model, not advice to the applicant.")
P("- Age, sex and residency are inputs here only because they are in the public data. See section 8.")
P("- Nothing was tuned, on purpose, so that the comparison is fair; every model would gain from tuning, and not by the same amount.")
P("")
P("## 11. Shipped artefacts")
P("")
P(table(["file", "what it is"], [
    ("model.json", f"the fitted soft tree as plain arrays ({FM['json_bytes'] / 1024:.0f} KB); predicts with numpy alone via NumpySoftTree.from_json"),
    ("model.onnx", f"the same tree as an ONNX graph ({FM['onnx_bytes'] / 1024:.0f} KB), standard operators only"),
    ("preprocessing.json", "the medians, means, scales and category lists a consumer must apply before calling the model, in input order"),
    ("rules.txt", "the hard-rule reading of the final model, with input names"),
    ("results.json", "every number in this document, fold by fold"),
    ("summary.csv", "the performance table"),
    ("run.py, report.py", "regenerate everything"),
]))
P("")
P(f"ONNX check on all {D['n']} training rows: the ONNX graph and the torch model agree on {FM['onnx_label_agreement']:.1%} of labels, "
  f"largest probability difference {FM['onnx_max_abs_diff']:.2e}; one applicant scores in {FM['onnx_microseconds_per_row']:.0f} µs in ONNX Runtime on one CPU thread.")
P("")
P("## 12. Monitoring after deployment")
P("")
P("What a reviewer would ask to see quarterly, all computable from the files here: the refusal rate and the observed bad rate per calibration bin against section 5; "
  "the slice table of section 8 on new applicants; the share of decisions where the hard rules and the soft model disagree, against section 6; and a refit on the new quarter "
  "with the root-gate inputs compared to section 9. A drift in any of these is a reason to look, not a reason to retrain automatically.")
P("")
md = "\n".join(lines)
(HERE / "report.md").write_text(md, encoding="utf-8")
print("report.md written")

try:
    from docx import Document
    from docx.shared import Pt
except ImportError:
    raise SystemExit(0)

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)
in_code, code = False, []
for line in md.splitlines():
    if line.startswith("```"):
        if in_code:
            p = doc.add_paragraph()
            r = p.add_run("\n".join(code))
            r.font.name = "Menlo"
            r.font.size = Pt(8)
            code = []
        in_code = not in_code
        continue
    if in_code:
        code.append(line)
        continue
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-"):
            continue
        if not getattr(doc, "_tbl", None):
            doc._tbl = doc.add_table(rows=0, cols=len(cells))
            doc._tbl.style = "Light Grid Accent 1"
        row = doc._tbl.add_row().cells
        for c, t in zip(row, cells):
            c.text = t.replace("**", "")
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
        continue
    doc._tbl = None
    if line.startswith("# "):
        doc.add_heading(line[2:], 0)
    elif line.startswith("## "):
        doc.add_heading(line[3:], 1)
    elif line.startswith("### "):
        doc.add_heading(line[4:], 2)
    elif line.startswith("- "):
        doc.add_paragraph(line[2:].replace("**", ""), style="List Bullet")
    elif line.strip():
        p = doc.add_paragraph()
        for i, chunk in enumerate(line.split("**")):
            r = p.add_run(chunk)
            r.bold = i % 2 == 1
doc.save(HERE / "report.docx")
print("report.docx written")

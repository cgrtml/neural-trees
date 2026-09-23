"""
Summarise benchmarks/rakipler-sonuc.json: the table, mean ranks, paired
comparisons against XGBoost, and the small-sample subset. Writes
benchmarks/rakipler-ozet.md. Ties are differences within half a point.
"""
import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

KOK = pathlib.Path(__file__).resolve().parent
S = json.load(open(KOK / "rakipler-sonuc.json"))
MODELS = S["_modeller"]
DATA = {k: v for k, v in S.items() if not k.startswith("_")}
ACC = {m: np.array([DATA[d][m]["acc"] for d in DATA]) for m in MODELS}
N = np.array([DATA[d]["n"] for d in DATA])
K = np.array([DATA[d]["K"] for d in DATA])
names = list(DATA)


def ranks():
    M = np.array([ACC[m] for m in MODELS]).T  # (datasets, models)
    R = np.zeros_like(M)
    for i, row in enumerate(M):
        order = (-row).argsort()
        r = np.empty(len(row))
        r[order] = np.arange(1, len(row) + 1)
        # average ranks for ties within 0.0005
        for v in np.unique(np.round(row, 3)):
            same = np.abs(row - v) < 0.0006
            r[same] = r[same].mean()
        R[i] = r
    return R


def paired(a, b, mask=None):
    da = ACC[a] if mask is None else ACC[a][mask]
    db = ACC[b] if mask is None else ACC[b][mask]
    diff = (da - db) * 100
    w, t, lo = int((diff > 0.5).sum()), int((np.abs(diff) <= 0.5).sum()), int((diff < -0.5).sum())
    p = wilcoxon(diff).pvalue if len(diff) >= 6 and np.any(diff != 0) else float("nan")
    return diff.mean(), np.median(diff), w, t, lo, p


out = []
out.append("| Dataset | n | K | " + " | ".join(MODELS) + " |")
out.append("|---|---:|---:|" + "|".join(["---:"] * len(MODELS)) + "|")
for i, d in enumerate(names):
    best = max(ACC[m][i] for m in MODELS)
    cells = []
    for m in MODELS:
        a = ACC[m][i]
        cells.append(f"**{a:.3f}**" if a >= best - 0.0005 else f"{a:.3f}")
    out.append(f"| {d} | {N[i]} | {K[i]} | " + " | ".join(cells) + " |")
R = ranks()
out.append("| **mean accuracy** | | | " + " | ".join(f"{ACC[m].mean():.3f}" for m in MODELS) + " |")
out.append("| **mean rank** | | | " + " | ".join(f"{R[:, j].mean():.2f}" for j in range(len(MODELS))) + " |")
out.append("| **mean fit time (s)** | | | " + " | ".join(f"{np.mean([DATA[d][m]['fit_sn'] for d in DATA]):.1f}" for m in MODELS) + " |")
out.append("")
out.append("Paired against XGBoost over all datasets (difference in points, model minus XGBoost; win/tie/loss with ties within 0.5; Wilcoxon signed-rank p):")
out.append("")
out.append("| Model | mean | median | W/T/L | p |")
out.append("|---|---:|---:|---|---:|")
for m in MODELS:
    if m == "XGBoost":
        continue
    mean, med, w, t, lo, p = paired(m, "XGBoost")
    out.append(f"| {m} | {mean:+.2f} | {med:+.2f} | {w}/{t}/{lo} | {p:.3f} |")
for label, mask in (("n <= 1000", N <= 1000), ("n > 1000", N > 1000), ("binary", K == 2), ("multi-class", K > 2)):
    out.append("")
    out.append(f"Subset {label} ({int(mask.sum())} datasets):")
    out.append("")
    out.append("| Model | mean acc | mean rank | vs XGBoost mean | W/T/L | p |")
    out.append("|---|---:|---:|---:|---|---:|")
    for j, m in enumerate(MODELS):
        mean, med, w, t, lo, p = paired(m, "XGBoost", mask)
        out.append(f"| {m} | {ACC[m][mask].mean():.3f} | {R[mask][:, j].mean():.2f} | {mean:+.2f} | {w}/{t}/{lo} | {p:.3f} |")
text = "\n".join(out)
(KOK / "rakipler-ozet.md").write_text(text + "\n", encoding="utf-8")
print(text)

# CSV files for the documentation's csv-table directives.
DOCS = KOK.parent / "docs" / "_generated"
DOCS.mkdir(exist_ok=True)
with open(DOCS / "benchmark_full.csv", "w", encoding="utf-8") as f:
    f.write("Dataset,n,K," + ",".join(MODELS) + "\n")
    for i, d in enumerate(names):
        best = max(ACC[m][i] for m in MODELS)
        f.write(f"{d},{N[i]},{K[i]}," + ",".join(
            (f"**{ACC[m][i]:.3f}**" if ACC[m][i] >= best - 0.0005 else f"{ACC[m][i]:.3f}") for m in MODELS) + "\n")
    f.write("**mean accuracy**,,," + ",".join(f"{ACC[m].mean():.3f}" for m in MODELS) + "\n")
    f.write("**mean rank**,,," + ",".join(f"{R[:, j].mean():.2f}" for j in range(len(MODELS))) + "\n")
    f.write("**mean fit time (s)**,,," + ",".join(f"{np.mean([DATA[d][m]['fit_sn'] for d in DATA]):.1f}" for m in MODELS) + "\n")
for label, fname, mask in (
    ("all", "benchmark_vs_xgboost.csv", np.ones(len(names), bool)),
    ("small", "benchmark_small_n.csv", N <= 1000),
    ("large", "benchmark_large_n.csv", N > 1000),
):
    with open(DOCS / fname, "w", encoding="utf-8") as f:
        f.write("Model,mean accuracy,mean rank,vs XGBoost (points),win/tie/loss,Wilcoxon p\n")
        for j, m in enumerate(MODELS):
            mean, med, w, t, lo, pv = paired(m, "XGBoost", mask)
            pcell = "" if m == "XGBoost" else f"{pv:.3f}"
            f.write(f"{m},{ACC[m][mask].mean():.3f},{R[mask][:, j].mean():.2f},{mean:+.2f},{w}/{t}/{lo},{pcell}\n")
print("csv written to", DOCS)

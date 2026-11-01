#!/usr/bin/env python3
"""
DistilledTree on the 24 datasets of rakipler.py, same protocol (features
scaled on the training fold only, 3 seeds x 5 stratified folds, rows capped
at 5 000), so its accuracy compares with the per-leaf soft tree and XGBoost
there. Library defaults throughout: HistGradientBoosting teacher, depth-8
CART-started student, with (synthetic=2, the default) and without
(synthetic=0) MUNGE rows. The teacher's own accuracy is recorded too.

    python3 benchmarks/damitma_kucuk.py hazirla
    python3 benchmarks/damitma_kucuk.py kos [--n-jobs 8]
    python3 benchmarks/damitma_kucuk.py ozet
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

KOK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
ARA = KOK / "_kucuk_cache"
CIKTI = KOK / "damitma_kucuk-sonuc.json"
AYARLAR = (2, 0)


def hazirla():
    import rakipler
    ARA.mkdir(exist_ok=True)
    for ad, X, y in rakipler.kumeler(False):
        np.savez(ARA / f"{ad}.npz", X=X, y=y)
        print(ad, X.shape, flush=True)


def bir(yol, s, k, synthetic):
    import torch
    torch.set_num_threads(1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from neural_trees import DistilledTree
    out = ARA / f"{yol.stem}__{s}__{k}__syn{synthetic}.json"
    if out.exists():
        return
    d = np.load(yol)
    X, y = d["X"], d["y"]
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y))[k]
    sc = StandardScaler().fit(X[tr])
    t0 = time.time()
    m = DistilledTree(teacher=HistGradientBoostingClassifier(random_state=s), synthetic=synthetic,
                      random_state=s).fit(sc.transform(X[tr]), y[tr])
    Xte = sc.transform(X[te])
    out.write_text(json.dumps({"acc": float((m.predict(Xte) == y[te]).mean()),
                               "teacher": float((m.teacher_.predict(Xte) == y[te]).mean()),
                               "sn": time.time() - t0, "splits": int(m.student_.model_.is_split.sum())}))


def kos(n_jobs):
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*.npz"), key=lambda p: -np.load(p)["y"].size)
    Parallel(n_jobs=n_jobs)(delayed(bir)(p, s, k, syn) for p in yollar for syn in AYARLAR
                            for s in (0, 1, 2) for k in range(5))


def ozet():
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    R = json.loads((KOK / "rakipler-sonuc.json").read_text())
    S = {"_protokol": __doc__.split("    python3")[0].strip(), "_commit": commit}
    rows = []
    for yol in sorted(ARA.glob("*.npz")):
        ad = yol.stem
        r = {}
        for syn in AYARLAR:
            f = [json.loads(p.read_text()) for p in ARA.glob(f"{ad}__*__*__syn{syn}.json")]
            if len(f) != 15:
                break
            r[f"distilled syn{syn}"] = float(np.mean([x["acc"] for x in f]))
            r["teacher"] = float(np.mean([x["teacher"] for x in f]))
            r[f"splits syn{syn}"] = float(np.mean([x["splits"] for x in f]))
            r[f"sn syn{syn}"] = float(np.mean([x["sn"] for x in f]))
        else:
            r.update({"n": R[ad]["n"], "per-leaf": R[ad]["SoftTree per-leaf"]["acc"],
                      "XGBoost": R[ad]["XGBoost"]["acc"]})
            S[ad] = {k: round(v, 4) for k, v in r.items()}
            rows.append((ad, S[ad]))
    print(f"{'dataset':28s} {'n':>5s}  {'XGB':>6s} {'HGB':>6s} {'perleaf':>7s} {'dist0':>6s} {'dist2':>6s}  splits")
    for ad, r in sorted(rows, key=lambda t: t[1]["n"]):
        print(f"{ad:28s} {r['n']:5.0f}  {r['XGBoost']:.3f}  {r['teacher']:.3f}  {r['per-leaf']:.3f}  "
              f"{r['distilled syn0']:.3f}  {r['distilled syn2']:.3f}  {r['splits syn2']:.0f}")
    summ = {}
    for lab, sel in (("all", lambda r: True), ("n<=1000", lambda r: r["n"] <= 1000),
                     ("n>1000", lambda r: r["n"] > 1000)):
        sub = [r for _, r in rows if sel(r)]
        if len(sub) < 2:
            continue
        for a, b in (("distilled syn2", "distilled syn0"), ("distilled syn2", "per-leaf"),
                     ("distilled syn2", "XGBoost"), ("per-leaf", "XGBoost")):
            d = np.array([r[a] - r[b] for r in sub])
            p = float(wilcoxon(d).pvalue)
            print(f"{lab:8s} {a} - {b}: {d.mean()*100:+.2f} pt, {(d > 0.002).sum()}/{(d < -0.002).sum()} "
                  f"of {len(d)}, p={p:.4f}")
            summ[f"{lab}: {a} - {b}"] = {"mean_pt": round(float(d.mean() * 100), 2), "higher": int((d > 0.002).sum()),
                                          "lower": int((d < -0.002).sum()), "n": len(d), "p": round(p, 4)}
    S["_summary"] = summ
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["hazirla", "kos", "ozet"])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    {"hazirla": hazirla, "kos": lambda: kos(a.n_jobs), "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

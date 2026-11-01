#!/usr/bin/env python3
"""
Does training one soft tree on XGBoost's probabilities (DistilledTree) close
the gap to XGBoost on large tables, and how does that change with size?

Datasets and test sets as in olcek.py (fixed stratified test set of
min(20 000, 20 %) rows). Training sets of 5k, 20k and 100k rows drawn with
three seeds below 100k and one at 100k; the seed also seeds every model.
Models, all on the same training rows:

    XGBoost      300 trees x depth 6 (as rakipler.py), the teacher
    soft d8      SoftDecisionTree(depth=8, init="cart", learn_temperature=True,
                 max_epochs=60, batch_size=256) on the labels
    distilled d8 the same tree on XGBoost's probabilities (DistilledTree, alpha=0)

Phase 1 fits the teachers (xgboost only), phase 2 the trees (torch only):
the two libraries can deadlock on libomp in one process.

    python3 benchmarks/damitma_olcek.py ogretmen
    python3 benchmarks/damitma_olcek.py ogrenci [--n-jobs 8]
    python3 benchmarks/damitma_olcek.py ozet
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
ARA = KOK / "_damitma_cache"
CIKTI = KOK / "damitma_olcek-sonuc.json"
KUMELER = ("electricity", "MagicTelescope", "jannis", "MiniBooNE", "covertype", "Higgs")
BOYUTLAR = (5_000, 20_000, 100_000)


def bolum(ad, n, seed):
    import olcek
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = olcek.yukle(ad)
    n_te = min(20_000, len(y) // 5)
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=n_te, random_state=0, stratify=y)
    if n > len(yp):
        return None
    idx = np.random.default_rng(seed).permutation(len(yp))[:n]
    sc = StandardScaler().fit(Xp[idx])
    return sc.transform(Xp[idx]).astype(np.float32), yp[idx], sc.transform(Xte).astype(np.float32), yte


def tohumlar(n):
    return (0,) if n >= 100_000 else (0, 1, 2)


def ogretmen():
    import xgboost as xgb
    ARA.mkdir(exist_ok=True)
    for ad in KUMELER:
        for n in BOYUTLAR:
            for s in tohumlar(n):
                yol = ARA / f"{ad}__{n}__{s}.npz"
                if yol.exists():
                    continue
                try:
                    b = bolum(ad, n, s)
                except Exception as exc:          # a download that fails skips the dataset, loudly
                    print(f"{ad}: skipped, {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    break
                if b is None:
                    continue
                Xtr, ytr, Xte, yte = b
                t0 = time.time()
                m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=s,
                                      n_jobs=4, verbosity=0).fit(Xtr, ytr)
                np.savez_compressed(yol, Ptr=m.predict_proba(Xtr), acc=float((m.predict(Xte) == yte).mean()),
                                    sn=time.time() - t0)
                print(f"{ad:14s} n={n:6d} seed {s}  XGB {float((m.predict(Xte) == yte).mean()):.4f}", flush=True)


def _ogrenci(yol, isim):
    import torch
    torch.set_num_threads(1)
    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__{isim.replace(' ', '_')}.json")
    if out.exists():
        return
    ad, n, s = yol.stem.split("__")
    Xtr, ytr, Xte, yte = bolum(ad, int(n), int(s))
    t0 = time.time()
    m = SoftDecisionTree(depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                         random_state=int(s))
    if isim == "soft d8":
        m.fit(Xtr, ytr)
    else:
        P = np.load(yol)["Ptr"]
        classes = np.unique(ytr)
        i, c = np.nonzero(P >= 0.01)
        m.fit(Xtr[i], classes[c], sample_weight=P[i, c])
    out.write_text(json.dumps({"acc": float((m.predict(Xte) == yte).mean()), "sn": time.time() - t0,
                               "splits": int(m.model_.is_split.sum())}))
    print(f"{yol.stem:32s} {isim:12s} {float((m.predict(Xte) == yte).mean()):.4f}  {time.time()-t0:.0f} s",
          flush=True)


def ogrenci(n_jobs):
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*.npz"), key=lambda p: -int(p.stem.split("__")[1]))
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci)(p, isim) for p in yollar for isim in ("soft d8", "distilled d8"))


def ozet():
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    S = {"_protokol": __doc__.split("Phase 1")[0].strip(), "_commit": commit}
    farklar = {"soft": [], "distilled": []}
    for p in sorted(ARA.glob("*.npz")):
        ad, n, s = p.stem.split("__")
        a = p.with_name(f"{p.stem}__soft_d8.json")
        b = p.with_name(f"{p.stem}__distilled_d8.json")
        if not (a.exists() and b.exists()):
            continue
        r = S.setdefault(ad, {}).setdefault(n, {"XGBoost": [], "soft d8": [], "distilled d8": []})
        r["XGBoost"].append(float(np.load(p)["acc"]))
        r["soft d8"].append(json.loads(a.read_text())["acc"])
        r["distilled d8"].append(json.loads(b.read_text())["acc"])
    print(f"{'dataset':14s} {'n':>6s}  {'XGBoost':>8s} {'soft d8':>8s} {'distilled':>9s}  gap soft -> distilled")
    for ad in KUMELER:
        for n, r in sorted(S.get(ad, {}).items(), key=lambda kv: int(kv[0])):
            m = {k: float(np.mean(v)) for k, v in r.items()}
            r.update({f"{k} mean": round(v, 4) for k, v in m.items()})
            farklar["soft"].append(m["soft d8"] - m["XGBoost"])
            farklar["distilled"].append(m["distilled d8"] - m["XGBoost"])
            print(f"{ad:14s} {n:>6s}  {m['XGBoost']:8.4f} {m['soft d8']:8.4f} {m['distilled d8']:9.4f}  "
                  f"{(m['soft d8']-m['XGBoost'])*100:+.2f} -> {(m['distilled d8']-m['XGBoost'])*100:+.2f} "
                  f"({len(r['XGBoost'])} seeds)")
    d = np.array(farklar["distilled"]) - np.array(farklar["soft"])
    if len(d) > 1:
        print(f"distilled - soft over {len(d)} dataset x size cells: mean {d.mean()*100:+.2f} pt, "
              f"higher/lower {(d > 0.002).sum()}/{(d < -0.002).sum()}, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
        S["_summary"] = {"cells": int(len(d)), "mean_gain_pt": round(float(d.mean() * 100), 2),
                         "gap_soft_pt": round(float(np.mean(farklar['soft']) * 100), 2),
                         "gap_distilled_pt": round(float(np.mean(farklar['distilled']) * 100), 2)}
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["ogretmen", "ogrenci", "ozet"])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    {"ogretmen": ogretmen, "ogrenci": lambda: ogrenci(a.n_jobs), "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

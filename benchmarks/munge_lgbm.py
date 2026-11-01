#!/usr/bin/env python3
"""
Does the MUNGE gain hold with a LightGBM teacher? Same datasets, 5 000 and
20 000 training rows, seeds, test sets and student as munge.py; the teacher
is LGBMClassifier(n_estimators=300, num_leaves=31), as in rakipler.py. The
student learns the teacher's probabilities on the training rows only
(synthetic=0) and on those plus two MUNGE passes (synthetic=2).

Phase 1 fits the teachers (lightgbm only), phase 2 the trees (torch only).

    python3 benchmarks/munge_lgbm.py ogretmen
    python3 benchmarks/munge_lgbm.py ogrenci [--n-jobs 8]
    python3 benchmarks/munge_lgbm.py ozet
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
ARA = KOK / "_lgbm_cache"
CIKTI = KOK / "munge_lgbm-sonuc.json"
BOYUTLAR = (5_000, 20_000)


def ogretmen():
    import damitma_olcek as D
    import lightgbm as lgb
    import munge as M
    ARA.mkdir(exist_ok=True)
    for ad in D.KUMELER:
        for n in BOYUTLAR:
            for s in (0, 1, 2):
                yol = ARA / f"{ad}__{n}__{s}.npz"
                if yol.exists():
                    continue
                b = D.bolum(ad, n, s)
                if b is None:
                    continue
                Xtr, ytr, Xte, yte = b
                m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=s, n_jobs=4,
                                       verbose=-1).fit(Xtr, ytr)
                acc = float((m.predict(Xte) == yte).mean())
                np.savez_compressed(yol, Ptr=m.predict_proba(Xtr), Pm=m.predict_proba(M.munge(Xtr, s)), acc=acc)
                print(f"{ad:14s} n={n:6d} seed {s}  LightGBM {acc:.4f}", flush=True)


def _ogrenci(yol, syn):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D
    import munge as M

    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__syn{syn}.json")
    if out.exists():
        return
    ad, n, s = yol.stem.split("__")
    Xtr, ytr, Xte, yte = D.bolum(ad, int(n), int(s))
    t0 = time.time()
    d = np.load(yol)
    P, Xs = d["Ptr"], Xtr
    if syn:
        P, Xs = np.vstack([P, d["Pm"]]), np.vstack([Xtr, M.munge(Xtr, int(s))])
    classes = np.unique(ytr)
    i, c = np.nonzero(P >= 0.01)
    m = SoftDecisionTree(depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                         random_state=int(s)).fit(Xs[i], classes[c], sample_weight=P[i, c])
    acc = float((m.predict(Xte) == yte).mean())
    out.write_text(json.dumps({"acc": acc, "sn": time.time() - t0}))
    print(f"{yol.stem:32s} syn{syn} {acc:.4f}  {time.time()-t0:.0f} s", flush=True)


def ogrenci(n_jobs):
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*.npz"), key=lambda p: -int(p.stem.split("__")[1]))
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci)(p, syn) for syn in (2, 0) for p in yollar)


def ozet():
    import damitma_olcek as D
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    S = {"_protokol": __doc__.split("Phase 1")[0].strip(), "_commit": commit}
    rows = []
    for ad in D.KUMELER:
        for n in BOYUTLAR:
            r = {"LightGBM": [], "soft d8": [], "syn0": [], "syn2": []}
            for s in (0, 1, 2):
                stem = f"{ad}__{n}__{s}"
                f0, f2 = ARA / f"{stem}__syn0.json", ARA / f"{stem}__syn2.json"
                if not (f0.exists() and f2.exists()):
                    continue
                r["LightGBM"].append(float(np.load(ARA / f"{stem}.npz")["acc"]))
                r["soft d8"].append(json.loads((D.ARA / f"{stem}__soft_d8.json").read_text())["acc"])
                r["syn0"].append(json.loads(f0.read_text())["acc"])
                r["syn2"].append(json.loads(f2.read_text())["acc"])
            if not r["syn2"]:
                continue
            m = {k: float(np.mean(v)) for k, v in r.items()}
            S.setdefault(ad, {})[str(n)] = {**r, **{f"{k} mean": round(v, 4) for k, v in m.items()}}
            rows.append(m)
            print(f"{ad:14s} {n:6d}  LightGBM {m['LightGBM']:.4f}  labels {m['soft d8']:.4f}  "
                  f"distilled {m['syn0']:.4f}  +MUNGE {m['syn2']:.4f}", flush=True)
    summ = {"cells": len(rows)}
    for a, b in (("syn2", "syn0"), ("syn2", "soft d8"), ("syn0", "soft d8"), ("syn2", "LightGBM"),
                 ("soft d8", "LightGBM")):
        d = np.array([r[a] - r[b] for r in rows])
        p = float(wilcoxon(d).pvalue)
        print(f"{a} - {b}: {d.mean()*100:+.2f} pt, {(d > 0.002).sum()}/{(d < -0.002).sum()}, p={p:.4f}")
        summ[f"{a} - {b}"] = {"mean_pt": round(float(d.mean() * 100), 2), "higher": int((d > 0.002).sum()),
                              "lower": int((d < -0.002).sum()), "p": round(p, 4)}
    S["_summary"] = summ
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["ogretmen", "ogrenci", "ozet"])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    {"ogretmen": ogretmen, "ogrenci": lambda: ogrenci(a.n_jobs), "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

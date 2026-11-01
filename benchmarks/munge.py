#!/usr/bin/env python3
"""
Can synthetic rows replace unlabelled ones? damitma_olcek.py found that a
distilled tree gains most when the teacher labels further rows for it
(2n unlabelled pool rows: +1.85 points over the tree on labels, 13 of 13
cells). Here the further rows are generated from the training rows alone
with MUNGE (Bucila, Caruana and Niculescu-Mizil, "Model compression",
KDD 2006): each new row starts from a training row and, feature by
feature with probability p, takes a value drawn around its nearest
neighbour's (normal, sd = |difference| / s; features with at most ten
distinct values copy the neighbour's value). Two passes give 2n rows, the
budget of the unlabelled column. MUNGE_PASSES=4 in the environment runs
the same with four passes (cells up to 20 000 rows), as an ablation. p = 0.5, s = 5 are our choice, not tuned.

Same datasets, sizes, seeds, test sets and teacher as damitma_olcek.py.

    python3 benchmarks/munge.py ogretmen
    python3 benchmarks/munge.py ogrenci [--n-jobs 8]
    python3 benchmarks/munge.py ozet
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
PASSES = int(os.environ.get("MUNGE_PASSES", "2"))   # other values: ablation, 20 000 rows at most
EK = "" if PASSES == 2 else f"_p{PASSES}"
ARA = KOK / f"_munge_cache{EK}"
DAMITMA = KOK / "_damitma_cache"
CIKTI = KOK / f"munge{EK}-sonuc.json"
P_SWAP, S_SPREAD = 0.5, 5.0


def munge(X, seed, p=P_SWAP, s=S_SPREAD, passes=PASSES):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)[:, 1]
    discrete = np.array([len(np.unique(X[:, j])) <= 10 for j in range(X.shape[1])])
    out = []
    for _ in range(passes):
        E, N = X.copy(), X[nn]
        swap = rng.random(X.shape) < p
        sd = np.abs(E - N) / s
        drawn = np.where(discrete, N, rng.normal(N, sd))
        out.append(np.where(swap, drawn, E))
    return np.vstack(out).astype(np.float32)


def ogretmen():
    import damitma_olcek as D
    import xgboost as xgb
    ARA.mkdir(exist_ok=True)
    for ad in D.KUMELER:
        for n in D.BOYUTLAR:
            for s in D.tohumlar(n):
                yol = ARA / f"{ad}__{n}__{s}.npz"
                if yol.exists() or not (DAMITMA / yol.name).exists() or (EK and n > 20_000):
                    continue
                Xtr, ytr, Xte, yte = D.bolum(ad, n, s)
                m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=s,
                                      n_jobs=4, verbosity=0).fit(Xtr, ytr)
                Xm = munge(Xtr, s)
                np.savez_compressed(yol, Pm=m.predict_proba(Xm), acc=float((m.predict(Xte) == yte).mean()))
                print(f"{ad:14s} n={n:6d} seed {s}  XGB {float((m.predict(Xte) == yte).mean()):.4f}", flush=True)


def _ogrenci(yol):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D

    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__munge_d8.json")
    if out.exists():
        return
    ad, n, s = yol.stem.split("__")
    Xtr, ytr, Xte, yte = D.bolum(ad, int(n), int(s))
    t0 = time.time()
    P = np.vstack([np.load(DAMITMA / yol.name)["Ptr"], np.load(yol)["Pm"]])
    Xs = np.vstack([Xtr, munge(Xtr, int(s))])
    classes = np.unique(ytr)
    i, c = np.nonzero(P >= 0.01)
    m = SoftDecisionTree(depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                         random_state=int(s)).fit(Xs[i], classes[c], sample_weight=P[i, c])
    acc = float((m.predict(Xte) == yte).mean())
    out.write_text(json.dumps({"acc": acc, "sn": time.time() - t0, "splits": int(m.model_.is_split.sum())}))
    print(f"{yol.stem:32s} munge d8     {acc:.4f}  {time.time()-t0:.0f} s", flush=True)


def ogrenci(n_jobs):
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*.npz"), key=lambda p: -int(p.stem.split("__")[1]))
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci)(p) for p in yollar)


def ozet():
    import damitma_olcek as D
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    S = {"_protokol": __doc__.split("Same datasets")[0].strip(), "_commit": commit}
    cols = ("XGBoost", "soft d8", "distilled d8", "unlabeled d8", "munge d8")
    M = {c: [] for c in cols}
    print(f"{'dataset':14s} {'n':>6s}  " + " ".join(f"{c:>12s}" for c in cols))
    for ad in D.KUMELER:
        for n in D.BOYUTLAR:
            r = {c: [] for c in cols}
            for s in D.tohumlar(n):
                stem = f"{ad}__{n}__{s}"
                mj = ARA / f"{stem}__munge_d8.json"
                if not mj.exists():
                    continue
                r["XGBoost"].append(float(np.load(DAMITMA / f"{stem}.npz")["acc"]))
                for c in cols[1:4]:
                    r[c].append(json.loads((DAMITMA / f"{stem}__{c.replace(' ', '_')}.json").read_text())["acc"])
                r["munge d8"].append(json.loads(mj.read_text())["acc"])
            if not r["munge d8"]:
                continue
            m = {c: float(np.mean(v)) for c, v in r.items()}
            S.setdefault(ad, {})[str(n)] = {**r, **{f"{c} mean": round(v, 4) for c, v in m.items()}}
            for c in cols:
                M[c].append(m[c])
            print(f"{ad:14s} {n:6d}  " + " ".join(f"{m[c]:12.4f}" for c in cols), flush=True)
    A = {c: np.array(v) for c, v in M.items()}
    summ = {"cells": int(len(A["soft d8"]))}
    for a, b in (("munge d8", "soft d8"), ("munge d8", "distilled d8"), ("munge d8", "unlabeled d8"),
                 ("munge d8", "XGBoost")):
        d = A[a] - A[b]
        p = float(wilcoxon(d).pvalue) if len(d) > 1 else float("nan")
        print(f"{a} - {b}: mean {d.mean()*100:+.2f} pt, higher/lower {(d > 0.002).sum()}/{(d < -0.002).sum()}, "
              f"Wilcoxon p={p:.4f}")
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

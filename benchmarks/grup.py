#!/usr/bin/env python3
"""
A few soft trees instead of one. munge.py left the single distilled tree
1.8 points below XGBoost on average (13 cells), with the largest gaps where
one depth-8 tree lacks capacity (electricity, covertype). Here the student
is a short sum of soft regression trees fitted to the teacher's logits, one
after another on the residual (gradient boosting with squared loss on
logits; logit matching as in Ba and Caruana, NIPS 2014):

    logit d8   one depth-8 regression tree on the teacher's logits
    group3 d6  three depth-6 trees, each on the residual of the previous
    group5 d6  five depth-6 trees

Logits: log p clipped at 1e-4, centred per row (binary: one column,
log p1 - log p0). Prediction: argmax of the summed logits. Rows: the
training rows plus two MUNGE passes, labelled by the same XGBoost teacher
as in munge.py (its cached probabilities), so the comparison with the
"munge d8" column there is like for like.

Same datasets, sizes, seeds, test sets and teacher as damitma_olcek.py.

    python3 benchmarks/grup.py ogrenci [--n-jobs 8] [--en-buyuk 20000]
    python3 benchmarks/grup.py ozet
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
DAMITMA = KOK / "_damitma_cache"
MUNGE = KOK / "_munge_cache"
ARA = KOK / "_grup_cache"
CIKTI = KOK / "grup-sonuc.json"
OGRENCILER = {"logit d8": (1, 8), "group3 d6": (3, 6), "group5 d6": (5, 6)}


def logits(P):
    L = np.log(np.clip(P, 1e-4, 1.0))
    if P.shape[1] == 2:
        return L[:, 1] - L[:, 0]
    return L - L.mean(axis=1, keepdims=True)


def _ogrenci(stem):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D
    import munge as Mu

    from neural_trees import SoftDecisionTreeRegressor
    out = ARA / f"{stem}.json"
    if out.exists():
        return
    ad, n, s = stem.split("__")
    Xtr, ytr, Xte, yte = D.bolum(ad, int(n), int(s))
    classes = np.unique(ytr)
    P = np.vstack([np.load(DAMITMA / f"{stem}.npz")["Ptr"], np.load(MUNGE / f"{stem}.npz")["Pm"]])
    Xs = np.vstack([Xtr, Mu.munge(Xtr, int(s))])
    T = logits(P)
    res = {}
    for name, (k, depth) in OGRENCILER.items():
        t0 = time.time()
        F, Fte, splits = np.zeros_like(T), 0.0, 0
        for j in range(k):
            m = SoftDecisionTreeRegressor(depth=depth, init="cart", max_epochs=60, batch_size=256,
                                          learning_rate=0.01, random_state=int(s) + j).fit(Xs, T - F)
            F = F + m.predict(Xs)
            Fte = Fte + m.predict(Xte)
            splits += int(m.model_.is_split.sum())
        if np.ndim(Fte) == 1:
            pred = classes[(Fte > 0).astype(int)]
        else:
            pred = classes[np.argmax(Fte, axis=1)]
        res[name] = {"acc": float((pred == yte).mean()), "sn": time.time() - t0, "splits": splits}
        print(f"{stem:32s} {name:10s} {res[name]['acc']:.4f}  {splits:4d} splits  {res[name]['sn']:.0f} s", flush=True)
    out.write_text(json.dumps(res))


def hucreler(en_buyuk):
    import damitma_olcek as D
    for ad in D.KUMELER:
        for n in D.BOYUTLAR:
            if n > en_buyuk:
                continue
            for s in D.tohumlar(n):
                stem = f"{ad}__{n}__{s}"
                if (MUNGE / f"{stem}__munge_d8.json").exists():
                    yield stem


def ogrenci(n_jobs, en_buyuk):
    from joblib import Parallel, delayed
    ARA.mkdir(exist_ok=True)
    stems = sorted(hucreler(en_buyuk), key=lambda t: -int(t.split("__")[1]))
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci)(t) for t in stems)


def ozet():
    import damitma_olcek as D
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    S = {"_protokol": __doc__.split("Same datasets")[0].strip(), "_commit": commit}
    cols = ("XGBoost", "munge d8", *OGRENCILER)
    M = {c: [] for c in cols}
    print(f"{'dataset':14s} {'n':>6s}  " + " ".join(f"{c:>10s}" for c in cols))
    for ad in D.KUMELER:
        for n in D.BOYUTLAR:
            r = {c: [] for c in cols}
            for s in D.tohumlar(n):
                stem = f"{ad}__{n}__{s}"
                g = ARA / f"{stem}.json"
                if not g.exists():
                    continue
                r["XGBoost"].append(float(np.load(DAMITMA / f"{stem}.npz")["acc"]))
                r["munge d8"].append(json.loads((MUNGE / f"{stem}__munge_d8.json").read_text())["acc"])
                G = json.loads(g.read_text())
                for c in OGRENCILER:
                    r[c].append(G[c]["acc"])
            if not r["XGBoost"]:
                continue
            m = {c: float(np.mean(v)) for c, v in r.items()}
            S.setdefault(ad, {})[str(n)] = {**r, **{f"{c} mean": round(v, 4) for c, v in m.items()}}
            for c in cols:
                M[c].append(m[c])
            print(f"{ad:14s} {n:6d}  " + " ".join(f"{m[c]:10.4f}" for c in cols), flush=True)
    A = {c: np.array(v) for c, v in M.items()}
    summ = {"cells": int(len(A["XGBoost"]))}
    for a in OGRENCILER:
        for b in ("munge d8", "XGBoost"):
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
    ap.add_argument("asama", choices=["ogrenci", "ozet"])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--en-buyuk", type=int, default=20_000)
    ap.add_argument("--pilot", nargs="*")
    a = ap.parse_args()
    if a.pilot:
        ARA.mkdir(exist_ok=True)
        from joblib import Parallel, delayed
        Parallel(n_jobs=len(a.pilot))(delayed(_ogrenci)(t) for t in a.pilot)
        return
    {"ogrenci": lambda: ogrenci(a.n_jobs, a.en_buyuk), "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

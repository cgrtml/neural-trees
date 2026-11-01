#!/usr/bin/env python3
"""
A few soft trees instead of one. munge.py left the single distilled tree
1.8 points below XGBoost on average (13 cells), with the largest gaps where
one tree lacks capacity (electricity, covertype). Here the student is a
small group of distilled soft trees, each fitted like the munge.py student
(teacher probabilities on the training rows plus two MUNGE passes, CART
start, learnt temperature, 60 epochs) on its own bootstrap sample of those
rows, and the group predicts with the sum of the trees' log probabilities:

    group3 d6  three depth-6 trees
    group3 d8  three depth-8 trees

A pilot on three 5k cells (seed 0) found no gain from five trees over three,
nor from fine-tuning the summed group jointly afterwards, nor from boosting
soft regression trees on the teacher's logits; those are not in this run.

Same datasets, sizes, seeds, test sets and teacher as damitma_olcek.py; the
"munge d8" column is the single tree from munge.py. MUNGE_PASSES=10 in the
environment runs the same with munge.py's 10-pass rows and teacher labels.

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
PASSES = int(os.environ.get("MUNGE_PASSES", "2"))   # as in munge.py: 10 reads its 10-pass cache
EK = "" if PASSES == 2 else f"_p{PASSES}"
MUNGE = KOK / f"_munge_cache{EK}"
ARA = KOK / f"_grup_cache{EK}"
CIKTI = KOK / f"grup{EK}-sonuc.json"
OGRENCILER = {"group3 d6": (3, 6), "group3 d8": (3, 8)}


def _ogrenci(stem):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D
    import munge as Mu

    from neural_trees import SoftDecisionTree
    out = ARA / f"{stem}.json"
    if out.exists():
        return
    ad, n, s = stem.split("__")
    Xtr, ytr, Xte, yte = D.bolum(ad, int(n), int(s))
    classes = np.unique(ytr)
    P = np.vstack([np.load(DAMITMA / f"{stem}.npz")["Ptr"], np.load(MUNGE / f"{stem}.npz")["Pm"]])
    Xs = np.vstack([Xtr, Mu.munge(Xtr, int(s))])
    Xt = torch.tensor(Xte)
    res = {}
    for name, (k, depth) in OGRENCILER.items():
        t0 = time.time()
        rng = np.random.default_rng(int(s))
        L, splits = 0.0, 0
        for j in range(k):
            b = rng.integers(0, len(Xs), len(Xs))
            i, c = np.nonzero(P[b] >= 0.01)
            m = SoftDecisionTree(depth=depth, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                                 random_state=int(s) + j).fit(Xs[b][i], classes[c], sample_weight=P[b][i, c])
            with torch.no_grad():
                L = L + m.model_.log_forward(Xt).numpy()
            splits += int(m.model_.is_split.sum())
        pred = classes[np.argmax(L, axis=1)]
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

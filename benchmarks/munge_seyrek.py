#!/usr/bin/env python3
"""
What does a readable student cost? The MUNGE student of munge.py uses
oblique splits over many features. Here the same depth-8 student on the
same rows (munge.py's cached teacher probabilities) with at most three
features per split (split_features=3, started from CART), and the same
restriction on the tree trained on the labels, at 5 000 and 20 000 rows,
three seeds (SEYREK_ENBUYUK=100000 adds the 100 000-row cells, one seed);
compared with the unrestricted columns.

    python3 benchmarks/munge_seyrek.py ogrenci [--n-jobs 8]
    python3 benchmarks/munge_seyrek.py ozet
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
CIKTI = KOK / "munge_seyrek-sonuc.json"
K = 3
ENBUYUK = int(os.environ.get("SEYREK_ENBUYUK", "20000"))   # 100000 adds the two 100k cells, one seed


def _ogrenci(yol, isim):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D
    import munge as M

    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__{isim.replace(' ', '_')}.json")
    if out.exists():
        return
    ad, n, s = yol.stem.split("__")
    Xtr, ytr, Xte, yte = D.bolum(ad, int(n), int(s))
    t0 = time.time()
    m = SoftDecisionTree(depth=8, init="cart", split_features=K, learn_temperature=True, max_epochs=60,
                         batch_size=256, random_state=int(s))
    if isim.startswith("soft"):
        m.fit(Xtr, ytr)
    else:
        P = np.vstack([np.load(M.DAMITMA / yol.name)["Ptr"], np.load(yol)["Pm"]])
        Xs = np.vstack([Xtr, M.munge(Xtr, int(s))])
        classes = np.unique(ytr)
        i, c = np.nonzero(P >= 0.01)
        m.fit(Xs[i], classes[c], sample_weight=P[i, c])
    acc = float((m.predict(Xte) == yte).mean())
    out.write_text(json.dumps({"acc": acc, "sn": time.time() - t0, "splits": int(m.model_.is_split.sum())}))
    print(f"{yol.stem:32s} {isim:12s} {acc:.4f}  {time.time()-t0:.0f} s", flush=True)


def ogrenci(n_jobs):
    import munge as M
    from joblib import Parallel, delayed
    yollar = sorted((p for p in M.ARA.glob("*.npz") if int(p.stem.split("__")[1]) <= ENBUYUK),
                    key=lambda p: -int(p.stem.split("__")[1]))
    isler = [(p, i) for i in (f"munge d8 k{K}", f"soft d8 k{K}") for p in yollar]
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci)(p, i) for p, i in isler)


def ozet():
    import damitma_olcek as D
    import munge as M
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    cols = ("XGBoost", "soft d8", "munge d8", f"soft d8 k{K}", f"munge d8 k{K}")
    S = {"_protokol": __doc__.split("    python3")[0].strip(), "_commit": commit}
    A = {c: [] for c in cols}
    print(f"{'dataset':14s} {'n':>6s}  " + " ".join(f"{c:>10s}" for c in cols))
    for ad in D.KUMELER:
        for n in (5_000, 20_000, 100_000):
            r = {c: [] for c in cols}
            for s in D.tohumlar(n):
                stem = f"{ad}__{n}__{s}"
                yer = {c: M.ARA / f"{stem}__{c.replace(' ', '_')}.json" for c in cols[2:]}
                yer["soft d8"] = M.DAMITMA / f"{stem}__soft_d8.json"
                if not all(v.exists() for v in yer.values()):
                    continue
                r["XGBoost"].append(float(np.load(M.DAMITMA / f"{stem}.npz")["acc"]))
                for c, v in yer.items():
                    r[c].append(json.loads(v.read_text())["acc"])
            if not r[cols[-1]]:
                continue
            m = {c: float(np.mean(v)) for c, v in r.items()}
            S.setdefault(ad, {})[str(n)] = {**r, **{f"{c} mean": round(v, 4) for c, v in m.items()}}
            for c in cols:
                A[c].append(m[c])
            print(f"{ad:14s} {n:6d}  " + " ".join(f"{m[c]:10.4f}" for c in cols), flush=True)
    A = {c: np.array(v) for c, v in A.items()}
    summ = {"cells": int(len(A["XGBoost"]))}
    for a, b in ((f"munge d8 k{K}", "munge d8"), (f"soft d8 k{K}", "soft d8"), (f"munge d8 k{K}", "soft d8"),
                 (f"munge d8 k{K}", f"soft d8 k{K}"),
                 (f"munge d8 k{K}", "XGBoost")):
        d = A[a] - A[b]
        p = float(wilcoxon(d).pvalue) if len(d) > 1 else float("nan")
        print(f"{a} - {b}: mean {d.mean()*100:+.2f} pt, higher/lower {(d > 0.002).sum()}/{(d < -0.002).sum()}, "
              f"p={p:.4f}")
        summ[f"{a} - {b}"] = {"mean_pt": round(float(d.mean() * 100), 2), "higher": int((d > 0.002).sum()),
                              "lower": int((d < -0.002).sum()), "p": round(p, 4)}
    S["_summary"] = summ
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["ogrenci", "ozet"])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    {"ogrenci": lambda: ogrenci(a.n_jobs), "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

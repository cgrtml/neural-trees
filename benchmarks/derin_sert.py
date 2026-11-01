#!/usr/bin/env python3
"""
A deep hard tree as the student. On electricity the depth-8 soft student of
munge.py stays 4 to 6 points below XGBoost, and neither a deeper soft tree
(munge_derin.py) nor a group of three (grup.py) closes that. A pilot found
that one deep axis-aligned CART fitted to the teacher's labels on many
MUNGE rows comes much closer there (electricity 20k: 0.855 against XGBoost
0.870 and the soft student 0.814), while on Higgs and covertype it stays
below the soft student. This measures that on every cell.

Student: sklearn DecisionTreeClassifier (min_samples_leaf=5) on the
training rows plus 20 MUNGE passes, every row labelled with the XGBoost
teacher's predicted class. Depth is chosen among 8, 12, 16 and unlimited by
agreement with the teacher on 2 further MUNGE passes drawn with another
seed (no labels used). "hard pick" is that choice; the depth columns are
reported as well.

Same datasets, sizes, seeds, test sets and teacher as damitma_olcek.py; the
"munge d8" column is the soft student from munge.py.

    python3 benchmarks/derin_sert.py hesap
    python3 benchmarks/derin_sert.py ozet
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

KOK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
DAMITMA = KOK / "_damitma_cache"
MUNGE = KOK / "_munge_cache"
ARA = KOK / "_derin_sert_cache"
CIKTI = KOK / "derin_sert-sonuc.json"
DERINLIKLER = (8, 12, 16, None)


def hesap():
    import damitma_olcek as D
    import munge as Mu
    import xgboost as xgb
    from sklearn.tree import DecisionTreeClassifier
    ARA.mkdir(exist_ok=True)
    for ad in D.KUMELER:
        for n in D.BOYUTLAR:
            for s in D.tohumlar(n):
                stem = f"{ad}__{n}__{s}"
                out = ARA / f"{stem}.json"
                if out.exists() or not (MUNGE / f"{stem}__munge_d8.json").exists():
                    continue
                Xtr, ytr, Xte, yte = D.bolum(ad, n, s)
                m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=s,
                                      n_jobs=4, verbosity=0).fit(Xtr, ytr)
                Xa = np.vstack([Xtr, Mu.munge(Xtr, s, passes=20)])
                Xv = Mu.munge(Xtr, s + 1000, passes=2)
                ya, yv = m.predict(Xa), m.predict(Xv)
                res = {"XGBoost": float((m.predict(Xte) == yte).mean())}
                best = None
                for d in DERINLIKLER:
                    t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, random_state=0).fit(Xa, ya)
                    fid = float((t.predict(Xv) == yv).mean())
                    res[f"hard d{d}"] = {"acc": float((t.predict(Xte) == yte).mean()), "fid": fid,
                                         "leaves": int(t.get_n_leaves())}
                    if best is None or fid > res[best]["fid"]:
                        best = f"hard d{d}"
                res["pick"] = best
                out.write_text(json.dumps(res))
                print(f"{stem:28s} XGB {res['XGBoost']:.4f}  pick {best} {res[best]['acc']:.4f} "
                      f"({res[best]['leaves']} leaves)", flush=True)


def ozet():
    import damitma_olcek as D
    from scipy.stats import wilcoxon
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    S = {"_protokol": __doc__.split("Same datasets")[0].strip(), "_commit": commit}
    cols = ("XGBoost", "munge d8", "hard pick", "leaves")
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
                G = json.loads(g.read_text())
                r["XGBoost"].append(G["XGBoost"])
                r["munge d8"].append(json.loads((MUNGE / f"{stem}__munge_d8.json").read_text())["acc"])
                r["hard pick"].append(G[G["pick"]]["acc"])
                r["leaves"].append(G[G["pick"]]["leaves"])
            if not r["XGBoost"]:
                continue
            m = {c: float(np.mean(v)) for c, v in r.items()}
            S.setdefault(ad, {})[str(n)] = {**r, **{f"{c} mean": round(v, 4) for c, v in m.items()}}
            for c in cols:
                M[c].append(m[c])
            print(f"{ad:14s} {n:6d}  " + " ".join(f"{m[c]:10.4f}" for c in cols[:3]) + f" {m['leaves']:10.0f}",
                  flush=True)
    A = {c: np.array(v) for c, v in M.items()}
    A["best of two"] = np.maximum(A["munge d8"], A["hard pick"])
    summ = {"cells": int(len(A["XGBoost"]))}
    for a, b in (("hard pick", "munge d8"), ("hard pick", "XGBoost"), ("best of two", "XGBoost")):
        d = A[a] - A[b]
        p = float(wilcoxon(d).pvalue) if len(d) > 1 else float("nan")
        print(f"{a} - {b}: mean {d.mean()*100:+.2f} pt, higher/lower {(d > 0.002).sum()}/{(d < -0.002).sum()}, "
              f"Wilcoxon p={p:.4f}")
        summ[f"{a} - {b}"] = {"mean_pt": round(float(d.mean() * 100), 2), "higher": int((d > 0.002).sum()),
                              "lower": int((d < -0.002).sum()), "p": round(p, 4)}
    summ["note"] = "best of two picks per cell by test accuracy: an upper bound for a chooser, not a result"
    S["_summary"] = summ
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["hesap", "ozet"])
    a = ap.parse_args()
    {"hesap": hesap, "ozet": ozet}[a.asama]()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Soft trees started from CART (`init="cart"`) on the 24 datasets of
rakipler.py, under its protocol (3 seeds x 5 stratified folds,
StandardScaler on the training fold), next to CART of the same depth and the
reference columns already in rakipler-sonuc.json.

Question: does a soft copy of CART, fine-tuned with gradients, keep CART's
sharp thresholds where they matter (wall-robot) and still learn oblique
structure where they do not (Digits)?

The sharpness is chosen inside each training fold: 20 % of it is held out,
one tree per sharpness in GRID is fitted on the rest, the best held-out
accuracy wins, and that sharpness is refitted on the whole training fold. The
test fold is never seen during the choice. Every fixed sharpness is also
reported, so the cost of choosing is visible.

    python3 benchmarks/cart_init.py            # 24 datasets
    python3 benchmarks/cart_init.py --hizli    # Digits and wall-robot only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rakipler import TOHUM, kumeler  # noqa: E402

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "cart_init-sonuc.json"
DEPTH, EPOCHS, GRID = 6, 60, (4.0, 10.0, 30.0, 100.0)


def _soft(sharpness, seed):
    from neural_trees import SoftDecisionTree
    return SoftDecisionTree(depth=DEPTH, max_epochs=EPOCHS, init="cart", init_sharpness=sharpness,
                            learn_temperature=True, random_state=seed)


def _bir_kat(X, y, tr, te, seed):
    import torch
    torch.set_num_threads(1)
    from sklearn.tree import DecisionTreeClassifier
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
    R, t0 = {}, time.time()
    R["CART d6"] = float(DecisionTreeClassifier(max_depth=DEPTH, random_state=seed).fit(Xtr, ytr).score(Xte, yte))
    for s in GRID:
        R[f"init s{s:g}"] = float((_soft(s, seed).fit(Xtr, ytr).predict(Xte) == yte).mean())
    # choose the sharpness on a held-out part of the training fold only
    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    val = {s: float((_soft(s, seed).fit(a, ya).predict(b) == yb).mean()) for s in GRID}
    best = max(GRID, key=lambda s: (val[s], -s))
    R["init chosen"] = R[f"init s{best:g}"]
    R["_chosen"] = best
    R["_sn"] = round(time.time() - t0, 1)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=10)
    a = ap.parse_args()
    ref = json.loads((KOK / "rakipler-sonuc.json").read_text(encoding="utf-8"))
    S = json.loads(CIKTI.read_text(encoding="utf-8")) if CIKTI.exists() and not a.hizli else {}
    S["_protokol"] = (f"depth {DEPTH}, {EPOCHS} epochs, learn_temperature=True, sharpness grid {list(GRID)}; "
                      "'init chosen' picks the sharpness on 20 % of each training fold; "
                      f"seeds {list(TOHUM)} x 5-fold, StandardScaler on the training fold; "
                      "reference columns from rakipler-sonuc.json")
    for ad, X, y in kumeler(hizli=False):
        if ad in S and not a.hizli:
            continue
        if a.hizli and ad not in ("Digits", "wall-robot-navigation"):
            continue
        t0 = time.time()
        isler = [(tr, te, s) for s in TOHUM
                 for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)]
        katlar = Parallel(n_jobs=a.n_jobs)(delayed(_bir_kat)(X, y, tr, te, s) for tr, te, s in isler)
        r = {k: round(float(np.mean([f[k] for f in katlar])), 4) for k in katlar[0] if not k.startswith("_")}
        r["sd_chosen"] = round(float(np.std([f["init chosen"] for f in katlar], ddof=1)), 4)
        r["chosen_counts"] = {f"{s:g}": sum(f["_chosen"] == s for f in katlar) for s in GRID}
        for k in ("SoftTree per-leaf", "XGBoost", "RandomForest"):
            r[k] = ref[ad][k]["acc"]
        r["n"], r["K"] = int(len(y)), int(len(np.unique(y)))
        S[ad] = r
        print(f"{ad[:26]:26s} n={r['n']:5d}  CART {r['CART d6']:.3f}  chosen {r['init chosen']:.3f}  "
              f"per-leaf {r['SoftTree per-leaf']:.3f}  XGB {r['XGBoost']:.3f}  "
              f"grid {[r[f'init s{s:g}'] for s in GRID]}  picks {r['chosen_counts']}  {time.time()-t0:.0f} s",
              flush=True)
        if not a.hizli:
            CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

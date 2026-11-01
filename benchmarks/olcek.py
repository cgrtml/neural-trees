#!/usr/bin/env python3
"""
Scale curve: how the gap between soft trees and boosting moves as the
training set grows from 1 000 to 500 000 rows.

Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022) report that the gap between
neural networks and trees tends to shrink with more data and leave the
measurement open. This measures it for soft trees, on the numerical versions
of their datasets (OpenML), with the same test set at every size.

Per dataset: a fixed stratified test set of min(20 000, 20 %) rows; training
sets of 1k, 5k, 20k, 100k, 500k rows drawn from the rest (nested, so a larger
set contains the smaller ones), as far as the data allows. StandardScaler on
the training set. Models:

    CART d6        sklearn, depth 6
    XGBoost        300 trees x depth 6, learning rate 0.1 (as rakipler.py)
    soft per-leaf  depth <= 6, per-leaf growth, residual init
    soft cart      init="cart", depth 6, learned temperature, sharpness
                   chosen on 20 % of the training set from (4, 10, 30, 100)

The soft trees use batch size 64 below 20 000 rows and 1 024 from there, and
max(10, 60 * 5 000 / n) epochs capped at 60, so the number of optimiser steps
stays bounded; the budget is recorded with every result. One seed per size
(the test set is large), three below 20 000 rows.

    python3 benchmarks/olcek.py                    # all datasets
    python3 benchmarks/olcek.py --kume electricity --boyut 1000 5000
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "olcek-sonuc.json"
KUMELER = {                       # OpenML ids of the Grinsztajn et al. numerical versions
    "electricity": 44120,
    "MagicTelescope": 44125,
    "jannis": 44131,
    "MiniBooNE": 44128,
    "covertype": 44121,
    "Higgs": 44129,
}
BOYUTLAR = (1_000, 5_000, 20_000, 100_000, 500_000)
GRID = (4.0, 10.0, 30.0, 100.0)


def yukle(ad):
    d = fetch_openml(data_id=KUMELER[ad], as_frame=True, parser="auto")
    frame = d.data.copy()
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype("category").cat.codes
    X = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0)
    y = LabelEncoder().fit_transform(np.asarray(d.target))
    return X, y


def butce(n):
    return {"batch_size": 64 if n < 20_000 else 1024,
            "max_epochs": int(min(60, max(10, 60 * 5_000 / n)))}


def _model(isim, n, seed, sharpness=None):
    if isim == "CART d6":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=6, random_state=seed)
    if isim == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed,
                                 n_jobs=1, verbosity=0)
    from neural_trees import SoftDecisionTree
    b = butce(n)
    if isim == "soft per-leaf":
        return SoftDecisionTree(depth=6, growth="per_leaf", growth_init="residual", random_state=seed, **b)
    return SoftDecisionTree(depth=6, init="cart", init_sharpness=sharpness, learn_temperature=True,
                            random_state=seed, **b)


def _bir(isim, Xtr, ytr, Xte, yte, seed):
    """One model at one size; xgboost and torch never share a process (libomp)."""
    if isim != "XGBoost":
        import torch
        torch.set_num_threads(1)
    t0 = time.time()
    ek = {}
    if isim == "soft cart":
        a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
        val = {s: float((_model(isim, len(ya), seed, s).fit(a, ya).predict(b) == yb).mean()) for s in GRID}
        best = max(GRID, key=lambda s: (val[s], -s))
        m, ek = _model(isim, len(ytr), seed, best).fit(Xtr, ytr), {"sharpness": best}
    else:
        m = _model(isim, len(ytr), seed).fit(Xtr, ytr)
    acc = float((np.asarray(m.predict(Xte)) == yte).mean())
    return {"acc": acc, "sn": round(time.time() - t0, 1), **ek}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", nargs="*", default=list(KUMELER))
    ap.add_argument("--boyut", nargs="*", type=int, default=list(BOYUTLAR))
    ap.add_argument("--n-jobs", type=int, default=10)
    a = ap.parse_args()
    S = json.loads(CIKTI.read_text(encoding="utf-8")) if CIKTI.exists() else {}
    S["_protokol"] = __doc__.split("    python3")[0].strip()
    import subprocess
    S["_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                  cwd=KOK).stdout.strip()
    for ad in a.kume:
        X, y = yukle(ad)
        n_te = min(20_000, len(y) // 5)
        Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=n_te, random_state=0, stratify=y)
        sira = np.random.default_rng(0).permutation(len(ypool))       # nested training sets
        isler = []
        for n in a.boyut:
            if n > len(ypool) or str(n) in S.get(ad, {}):
                continue
            idx = sira[:n]
            sc = StandardScaler().fit(Xpool[idx])
            Xtr, Xs = sc.transform(Xpool[idx]), sc.transform(Xte)
            for seed in ((0, 1, 2) if n < 20_000 else (0,)):
                for isim in ("CART d6", "XGBoost", "soft per-leaf", "soft cart"):
                    isler.append((n, seed, isim, Xtr, ypool[idx], Xs))
        if not isler:
            continue
        # xgboost in its own batch of processes, torch models in another
        sonuc = []
        for grup in (("CART d6", "XGBoost"), ("soft per-leaf", "soft cart")):
            alt = [i for i in isler if i[2] in grup]
            alt.sort(key=lambda i: -i[0])
            out = Parallel(n_jobs=a.n_jobs)(delayed(_bir)(isim, Xtr, ytr, Xs, yte, seed)
                                            for n, seed, isim, Xtr, ytr, Xs in alt)
            sonuc += [(i[0], i[1], i[2], o) for i, o in zip(alt, out)]
        r = S.setdefault(ad, {"n_total": int(len(y)), "n_test": int(n_te), "K": int(len(np.unique(y)))})
        for n in sorted({s[0] for s in sonuc}):
            satir = {}
            for isim in ("CART d6", "XGBoost", "soft per-leaf", "soft cart"):
                kat = [o for nn, _, ii, o in sonuc if nn == n and ii == isim]
                satir[isim] = round(float(np.mean([o["acc"] for o in kat])), 4)
                satir[f"{isim} sn"] = round(float(np.mean([o["sn"] for o in kat])), 1)
            satir["sharpness"] = [o["sharpness"] for nn, _, ii, o in sonuc if nn == n and ii == "soft cart"]
            satir["budget"] = butce(n)
            satir["seeds"] = 3 if n < 20_000 else 1
            r[str(n)] = satir
            print(f"{ad:14s} n={n:7d}  CART {satir['CART d6']:.3f}  XGB {satir['XGBoost']:.3f}  "
                  f"per-leaf {satir['soft per-leaf']:.3f}  cart-init {satir['soft cart']:.3f}  "
                  f"s={satir['sharpness']}  times {satir['XGBoost sn']:.0f}/{satir['soft per-leaf sn']:.0f}/"
                  f"{satir['soft cart sn']:.0f} s", flush=True)
        CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evidence package: tuned boosting models against the library's distilled
students, on 9 datasets (the 6 Grinsztajn et al. numerical sets used so far,
plus 3 credit and marketing sets a bank would recognise), 20 000 training
rows, three seeds, accuracy and ROC AUC on the same held-out test rows.

Two steps, in separate processes (xgboost and torch never share one):

  rakip    Tuned XGBoost, LightGBM, CatBoost, HistGradientBoosting and a CART
           on the real training rows. Each gets 20 random configurations,
           scored on a validation split (20% of the training rows), with
           early stopping on that split for the boosters; the best one is
           refitted on all training rows. Model size is the number of
           internal (split) nodes over all trees.
  ogrenci  The library as a user would call it: DistilledTree and
           DistilledForest(n_trees=3) with synthetic=10 and the tuned
           HistGradientBoosting configuration as teacher, plus the same soft
           tree trained on the real rows with no teacher.

    python3 benchmarks/kanit.py rakip   [--n-jobs 8]
    python3 benchmarks/kanit.py ogrenci [--n-jobs 8]
    python3 benchmarks/kanit.py ozet
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
ARA = KOK / os.environ.get("KANIT_ARA", "_kanit_cache")
CIKTI = KOK / "kanit-sonuc.json"
KUMELER = {                      # OpenML ids
    "electricity": 44120, "MagicTelescope": 44125, "jannis": 44131, "MiniBooNE": 44128,
    "covertype": 44121, "Higgs": 44129,
    "credit-default": 42477, "bank-marketing": 1461, "adult": 1590,
}
N = int(os.environ.get("KANIT_N", 20_000))
TOHUMLAR = (0, 1, 2)
DENEME = int(os.environ.get("KANIT_DENEME", 20))
RAKIPLER = ("XGBoost", "LightGBM", "CatBoost", "HistGB", "CART")
OGRENCILER = ("DistilledTree", "DistilledForest", "SoftTree")


def bolum(ad, seed):
    """Same recipe as damitma_olcek.bolum: fixed test rows, seeded training sample, scaled."""
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    d = fetch_openml(data_id=KUMELER[ad], as_frame=True, parser="auto")
    frame = d.data.copy()
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype("category").cat.codes
    X = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0)
    y = LabelEncoder().fit_transform(np.asarray(d.target))
    n_te = min(20_000, len(y) // 5)
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=n_te, random_state=0, stratify=y)
    idx = np.random.default_rng(seed).permutation(len(yp))[:N]
    sc = StandardScaler().fit(Xp[idx])
    return sc.transform(Xp[idx]).astype(np.float32), yp[idx], sc.transform(Xte).astype(np.float32), yte


def skor(model, X, y):
    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(X)
    auc = roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr")
    return float((model.classes_[p.argmax(1)] == y).mean()), float(auc)


def _uzay(isim, rng):
    lr = float(10 ** rng.uniform(-2, -0.5))
    if isim == "XGBoost":
        return {"learning_rate": lr, "max_depth": int(rng.integers(3, 11)),
                "min_child_weight": float(10 ** rng.uniform(-1, 1.5)), "subsample": float(rng.uniform(0.5, 1)),
                "colsample_bytree": float(rng.uniform(0.5, 1)), "reg_lambda": float(10 ** rng.uniform(-2, 1.5))}
    if isim == "LightGBM":
        return {"learning_rate": lr, "num_leaves": int(rng.integers(8, 256)),
                "min_child_samples": int(rng.integers(5, 100)), "subsample": float(rng.uniform(0.5, 1)),
                "subsample_freq": 1, "colsample_bytree": float(rng.uniform(0.5, 1)),
                "reg_lambda": float(10 ** rng.uniform(-2, 1.5))}
    if isim == "CatBoost":
        return {"learning_rate": lr, "depth": int(rng.integers(4, 11)),
                "l2_leaf_reg": float(10 ** rng.uniform(0, 1.5))}
    if isim == "HistGB":
        return {"learning_rate": lr, "max_leaf_nodes": int(rng.integers(8, 256)),
                "min_samples_leaf": int(rng.integers(5, 100)), "l2_regularization": float(10 ** rng.uniform(-3, 1))}
    return {"max_depth": int(rng.integers(3, 16)), "min_samples_leaf": int(rng.integers(1, 100))}


def _kur(isim, p, seed, Xv=None, yv=None):
    """Fit-ready model; with a validation split the boosters stop early on it."""
    if isim == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50 if Xv is not None else None,
                                 random_state=seed, n_jobs=1, verbosity=0, **p)
    if isim == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=2000, random_state=seed, n_jobs=1, verbose=-1, **p)
    if isim == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, random_seed=seed, thread_count=1, verbose=0,
                                  early_stopping_rounds=50 if Xv is not None else None, **p)
    if isim == "HistGB":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=2000, early_stopping=True, n_iter_no_change=50,
                                              random_state=seed, **p)
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=seed, **p)


def _fit(isim, m, X, y, Xv=None, yv=None):
    if isim == "XGBoost" and Xv is not None:
        return m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    if isim == "LightGBM" and Xv is not None:
        import lightgbm as lgb
        return m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    if isim == "CatBoost" and Xv is not None:
        return m.fit(X, y, eval_set=(Xv, yv))
    return m.fit(X, y)


def _agac_sayisi(isim, m):
    if isim == "XGBoost":
        try:
            return int(m.best_iteration) + 1
        except AttributeError:                               # refitted without a validation split
            return int(m.n_estimators)
    if isim == "LightGBM":
        return int(m.best_iteration_) or int(m.n_estimators)
    if isim == "CatBoost":
        b = m.get_best_iteration()
        return int(m.tree_count_) if b is None else int(b) + 1
    if isim == "HistGB":
        return int(m.n_iter_)
    return 1


def boyut(isim, m):
    """Internal (split) nodes over all trees."""
    if isim == "XGBoost":
        df = m.get_booster().trees_to_dataframe()
        return int((df["Feature"] != "Leaf").sum())
    if isim == "LightGBM":
        return int(sum(t["num_leaves"] - 1 for t in m.booster_.dump_model()["tree_info"]))
    if isim == "CatBoost":                                # symmetric trees
        return int(m.tree_count_ * (2 ** m.get_all_params()["depth"] - 1))
    if isim == "HistGB":
        return int(sum((p.nodes["is_leaf"] == 0).sum() for it in m._predictors for p in it))
    return int(m.tree_.node_count - m.tree_.n_leaves)


def rakip_bir(ad, seed):
    yol = ARA / f"{ad}__{seed}__rakip.json"
    if yol.exists():
        return json.loads(yol.read_text())
    from sklearn.model_selection import train_test_split
    Xtr, ytr, Xte, yte = bolum(ad, seed)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    sonuc = {}
    for isim in RAKIPLER:
        rng = np.random.default_rng(1000 * seed + RAKIPLER.index(isim))
        t0, en = time.time(), None
        for _ in range(DENEME):
            p = _uzay(isim, rng)
            m = _fit(isim, _kur(isim, p, seed, Xv, yv), Xa, ya, Xv, yv)
            acc = float((m.predict(Xv).ravel() == yv).mean())
            if en is None or acc > en[0]:
                en = (acc, p, _agac_sayisi(isim, m))
        p = dict(en[1])
        if isim in ("XGBoost", "LightGBM", "CatBoost"):        # refit on all rows, trees fixed by early stopping
            key = "iterations" if isim == "CatBoost" else "n_estimators"
            m = _kur(isim, p, seed)
            m.set_params(**{key: en[2]})
        elif isim == "HistGB":                                   # early stopping on its own split
            m = _kur(isim, p, seed)
        else:
            m = _kur(isim, p, seed)
        m = _fit(isim, m, Xtr, ytr)
        acc, auc = skor(m, Xte, yte)
        sonuc[isim] = {"acc": acc, "auc": auc, "splits": boyut(isim, m), "trees": _agac_sayisi(isim, m),
                       "params": p, "val_acc": en[0], "sn": time.time() - t0}
        print(ad, seed, isim, f"{acc:.4f}", sonuc[isim]["splits"], flush=True)
    yol.write_text(json.dumps(sonuc))
    return sonuc


def ogrenci_bir(ad, seed, isim):
    yol = ARA / f"{ad}__{seed}__{isim}.json"
    if yol.exists():
        return json.loads(yol.read_text())
    import torch
    torch.set_num_threads(1)
    from sklearn.ensemble import HistGradientBoostingClassifier

    from neural_trees import DistilledForest, DistilledTree, SoftDecisionTree
    Xtr, ytr, Xte, yte = bolum(ad, seed)
    p = json.loads((ARA / f"{ad}__{seed}__rakip.json").read_text())["HistGB"]["params"]
    teacher = HistGradientBoostingClassifier(max_iter=2000, early_stopping=True, n_iter_no_change=50,
                                             random_state=seed, **p)
    t0 = time.time()
    if isim == "DistilledTree":
        m = DistilledTree(teacher=teacher, synthetic=10, random_state=seed).fit(Xtr, ytr)
        trees = [m.student_]
    elif isim == "DistilledForest":
        m = DistilledForest(teacher=teacher, n_trees=3, synthetic=10, random_state=seed).fit(Xtr, ytr)
        trees = list(m.estimators_)
    else:
        m = SoftDecisionTree(depth=8, init="cart", learn_temperature=True, max_epochs=60, batch_size=256,
                             random_state=seed).fit(Xtr, ytr)
        trees = [m]
    sn = time.time() - t0
    acc, auc = skor(m, Xte, yte)
    r = {"acc": acc, "auc": auc, "sn": sn,
         "splits": int(sum(int(np.asarray(t.to_numpy().is_split_).sum()) for t in trees))}
    if isim != "SoftTree":
        r["fidelity"] = float(m.fidelity(Xte))
        r["teacher_acc"], r["teacher_auc"] = skor(m.teacher_, Xte, yte)
    print(ad, seed, isim, f"{acc:.4f}", r["splits"], f"{sn:.0f}s", flush=True)
    yol.write_text(json.dumps(r))
    return r


def ozet():
    from scipy.stats import wilcoxon
    tum = {}
    for ad in KUMELER:
        for s in TOHUMLAR:
            yol = ARA / f"{ad}__{s}__rakip.json"
            if not yol.exists():
                continue
            satir = json.loads(yol.read_text())
            for isim in OGRENCILER:
                o = ARA / f"{ad}__{s}__{isim}.json"
                if o.exists():
                    satir[isim] = json.loads(o.read_text())
            tum[f"{ad}__{s}"] = satir
    modeller = RAKIPLER + OGRENCILER
    print(f"{'dataset':16s}" + "".join(f"{m:>16s}" for m in modeller))
    hucre = {}
    for ad in KUMELER:
        rs = [tum[k] for k in tum if k.startswith(ad + "__")]
        if not rs:
            continue
        hucre[ad] = {m: {f: float(np.mean([r[m][f] for r in rs if m in r]))
                         for f in ("acc", "auc", "splits")} for m in modeller if any(m in r for r in rs)}
        print(f"{ad:16s}" + "".join(f"{hucre[ad][m]['acc']:>9.4f}{hucre[ad][m]['splits']:>7.0f}"
                                     if m in hucre[ad] else f"{'':>16s}" for m in modeller))
    karsilastirma = {}
    for a in OGRENCILER:
        for b in RAKIPLER + OGRENCILER:
            if a == b:
                continue
            for f in ("acc", "auc"):
                d = [100 * (hucre[c][a][f] - hucre[c][b][f]) for c in hucre if a in hucre[c] and b in hucre[c]]
                if len(d) < 3:
                    continue
                p = float(wilcoxon(d).pvalue) if any(d) else 1.0
                karsilastirma[f"{a} - {b} ({f})"] = {"mean_pt": float(np.mean(d)), "higher": sum(x > 0 for x in d),
                                                     "lower": sum(x < 0 for x in d), "p": p, "cells": len(d)}
    for k, v in karsilastirma.items():
        if "(acc)" in k or k.startswith("DistilledForest"):
            print(f"{k}: mean {v['mean_pt']:+.2f} pt, higher/lower {v['higher']}/{v['lower']}, "
                  f"Wilcoxon p={v['p']:.4f} ({v['cells']} datasets)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    CIKTI.write_text(json.dumps({"_protokol": __doc__, "_commit": commit, "datasets": hucre,
                                 "comparisons": karsilastirma, "runs": tum}, indent=1))


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("adim", choices=("rakip", "ogrenci", "ozet"))
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--kume", nargs="*", default=list(KUMELER))
    a = ap.parse_args()
    ARA.mkdir(exist_ok=True)
    if a.adim == "rakip":
        Parallel(n_jobs=a.n_jobs)(delayed(rakip_bir)(ad, s) for ad in a.kume for s in TOHUMLAR)
    elif a.adim == "ogrenci":
        isler = [(ad, s, i) for i in ("DistilledForest", "DistilledTree", "SoftTree") for ad in a.kume
                 for s in TOHUMLAR]
        Parallel(n_jobs=a.n_jobs)(delayed(ogrenci_bir)(*x) for x in isler)
    ozet()


if __name__ == "__main__":
    main()

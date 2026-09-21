#!/usr/bin/env python3
"""
Yönlü derinleştirme (growth_init) karşılaştırması: 4 UCI + 20 OpenML kümesi.

    OMP_NUM_THREADS=1 python3 paper/arxiv/olcum/buyume.py

Kollar, hepsi depth=4 / 150 epoch: random (mevcut jitter), residual,
residual_gate, sıfırdan. Protokol olc.py ile aynı; karar için residual_gate
ile random arasında 5x2cv F testi. Sonuç her veri kümesinden sonra
buyume-sonuc.json'a yazılır, koşu kaldığı yerden devam eder.
"""
import json
import pathlib
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from neural_trees import SoftDecisionTree, combined_5x2cv_f_test

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "buyume-sonuc.json"
TOHUM = (0, 1, 2)
N_MAKS = 5000
UCI = {"Iris": load_iris, "Wine": load_wine, "Cancer": load_breast_cancer, "Digits": load_digits}
OPENML = [1462, 1464, 1489, 37, 54, 40984, 182, 28, 32, 44,
          1504, 1480, 40994, 40499, 1497, 1494, 1487, 1067, 1068, 1050]
KOLLAR = ("random", "residual", "residual_gate", "sıfırdan")


def yap(kol, s):
    if kol == "sıfırdan":
        return SoftDecisionTree(depth=4, max_epochs=150, random_state=s)
    return SoftDecisionTree(depth=4, max_epochs=150, growth="incremental",
                            growth_init=kol, random_state=s)


def onisleyici(X):
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    sayisal = X.select_dtypes(include="number").columns.tolist()
    kategorik = [c for c in X.columns if c not in sayisal]
    return X, ColumnTransformer([
        ("s", StandardScaler(), sayisal),
        ("k", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorik),
    ])


def cv(kol, X, y):
    acc = []
    for s in TOHUM:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            X, on = onisleyici(X)
            on.fit(X.iloc[tr])
            e = yap(kol, s).fit(on.transform(X.iloc[tr]), y[tr])
            acc.append(float((e.predict(on.transform(X.iloc[te])) == y[te]).mean()))
    return [round(np.mean(acc), 4), round(np.std(acc, ddof=1), 4)]


def kumeler():
    for ad, yuk in UCI.items():
        X, y = yuk(return_X_y=True)
        yield ad, X, y
    for did in OPENML:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X, y = d.data, LabelEncoder().fit_transform(d.target)
        if len(y) > N_MAKS:
            X, _, y, _ = train_test_split(X, y, train_size=N_MAKS, stratify=y, random_state=0)
        yield d.details["name"], X.reset_index(drop=True), y


def main():
    S = json.loads(CIKTI.read_text()) if CIKTI.exists() else {}
    for ad, X, y in kumeler():
        if ad in S:
            continue
        t0 = time.time()
        R = {"n": int(len(y)), "p": int(X.shape[1]), "K": int(len(set(y)))}
        print(f"\n### {ad}  n={R['n']} p={R['p']} K={R['K']}", flush=True)
        for kol in KOLLAR:
            R[kol] = cv(kol, X, y)
            print(f"  {kol:14s} {R[kol][0]:.3f} ± {R[kol][1]:.3f}", flush=True)
        Xd, on = onisleyici(X)
        r = combined_5x2cv_f_test(yap("random", 0), yap("residual_gate", 0),
                                  on.fit_transform(Xd), y, random_state=0)
        R["test_gate_vs_random"] = [round(float(r.statistic), 3), round(float(r.p_value), 6)]
        print(f"  F test          F={R['test_gate_vs_random'][0]}  p={R['test_gate_vs_random'][1]}", flush=True)
        R["sure_sn"] = round(time.time() - t0)
        S[ad] = R
        CIKTI.write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("\nBİTTİ:", len(S), "veri kümesi")


if __name__ == "__main__":
    main()

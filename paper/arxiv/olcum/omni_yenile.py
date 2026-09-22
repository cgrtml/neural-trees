#!/usr/bin/env python3
"""
Omnivariate seçim-kuralı ölçümünü (sonuc.json["omni"]) yeniden üretir.

olc.py'nin 5. bölümüyle aynı protokol: 3 tohum x katmanlı 5-kat, ölçek
eğitim katında, max_depth=3, düğüm sayısı ağaç üzerinden. olc.py ana koruma
olmadan hepsini yeniden koşturduğu için bu dosya kendi başına durur. Ayrıca
docstring'deki min_samples_test=20/50 karşılaştırmasını (Wine) yeniler.
"""
import json
import pathlib
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from neural_trees import OmnivariateDecisionTree

warnings.filterwarnings("ignore")
KOK = pathlib.Path(__file__).resolve().parent
TOHUM = (0, 1, 2)
VERI = {"Cancer": load_breast_cancer, "Digits": load_digits, "Wine": load_wine}


def dugum(e):
    n = [0]

    def w(nd):
        n[0] += 1
        if nd.left is not None:
            w(nd.left)
            w(nd.right)

    w(e.root_)
    return n[0]


def cv(yap, ad):
    X, y = VERI[ad](return_X_y=True)
    acc, ek = [], []
    for s in TOHUM:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            e = yap(s).fit(sc.transform(X[tr]), y[tr])
            acc.append((e.predict(sc.transform(X[te])) == y[te]).mean())
            ek.append(dugum(e))
    return np.mean(acc), np.std(acc, ddof=1), np.mean(ek), np.std(ek, ddof=1)


S = json.loads((KOK / "sonuc.json").read_text())
S["omni"] = {}
for ad in ("Cancer", "Digits"):
    for sec in ("accuracy", "test"):
        m, sd, bm, bsd = cv(lambda s, ss=sec: OmnivariateDecisionTree(max_depth=3, selection=ss, random_state=s), ad)
        S["omni"].setdefault(ad, {})[sec] = [round(m, 4), round(sd, 4), round(bm, 2), round(bsd, 2)]
        print(f"{ad:7s} {sec:9s} acc {m:.3f} ± {sd:.3f}   düğüm {bm:.1f} ± {bsd:.1f}", flush=True)
(KOK / "sonuc.json").write_text(json.dumps(S, indent=2, ensure_ascii=False))

print("--- min_samples_test, Wine, selection='test' ---")
for mst in (20, 50):
    m, sd, bm, bsd = cv(lambda s, mm=mst: OmnivariateDecisionTree(max_depth=3, selection="test", min_samples_test=mm, random_state=s), "Wine")
    print(f"min_samples_test={mst:3d}  acc {m:.3f} ± {sd:.3f}   düğüm {bm:.1f} ± {bsd:.1f}", flush=True)
print("BİTTİ")

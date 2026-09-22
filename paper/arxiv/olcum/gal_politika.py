#!/usr/bin/env python3
"""GAL büyüme politikası: error_threshold (varsayılan) vs validation, random vs residual."""
import json
import pathlib

import numpy as np
from sklearn.datasets import fetch_openml, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_trees import GALNetwork

KOK = pathlib.Path(__file__).resolve().parent


def kumeler():
    for ad, yuk in (("Iris", load_iris), ("Wine", load_wine), ("Digits", load_digits)):
        X, y = yuk(return_X_y=True)
        yield ad, X.astype(float), y
    for did in (54, 40984, 182):
        d = fetch_openml(data_id=did, as_frame=False, parser="auto")
        X, y = d.data.astype(float), LabelEncoder().fit_transform(d.target)
        if len(y) > 5000:
            X, _, y, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=0)
        yield d.details["name"], X, y


S = {}
for ad, X, y in kumeler():
    S[ad] = {}
    for pol in ("error_threshold", "validation"):
        for init in ("random", "residual"):
            acc, birim = [], []
            for s in (0, 1, 2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
                    sc = StandardScaler().fit(X[tr])
                    m = GALNetwork(max_epochs=150, growth_policy=pol, growth_init=init, random_state=s)
                    m.fit(sc.transform(X[tr]), y[tr])
                    acc.append(float((m.predict(sc.transform(X[te])) == y[te]).mean()))
                    birim.append(float(m.n_hidden_final_))
            S[ad][f"{pol}/{init}"] = [round(np.mean(acc), 4), round(np.std(acc, ddof=1), 4),
                                      round(np.mean(birim), 1), round(np.std(birim, ddof=1), 1)]
            r = S[ad][f"{pol}/{init}"]
            print(f"{ad:14s} {pol:16s} {init:9s} acc {r[0]:.3f} ± {r[1]:.3f}  birim {r[2]:5.1f} ± {r[3]:.1f}", flush=True)
    (KOK / "gal_politika-sonuc.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
print("BİTTİ")

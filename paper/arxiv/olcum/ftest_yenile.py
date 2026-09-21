#!/usr/bin/env python3
"""olc.py'deki 5x2cv F testlerini kat-içi ölçeklemeyle yeniden koş, sonuc.json'u güncelle.

olc.py içe aktarılmıyor: ana korumasız olduğu için import etmek tüm ölçümü
yeniden koşturuyor. Gereken iki parça (veri yükleme, jitter sarmalayıcı) burada.
"""
import json
import pathlib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import neural_trees.decision_trees.soft_decision_tree as sdt_mod
from neural_trees import GALNetwork, SoftDecisionTree, combined_5x2cv_f_test

orij = sdt_mod._SoftTreeModule.deepen


def jitterli(j):
    def deepen(self, nc, jitter=0.0, _j=j):
        return orij(self, nc, _j)
    return deepen


class Buyuyen(ClassifierMixin, BaseEstimator):
    def __init__(self, jitter=0.2, depth=4, max_epochs=150, random_state=0):
        self.jitter = jitter
        self.depth = depth
        self.max_epochs = max_epochs
        self.random_state = random_state

    def fit(self, X, y):
        sdt_mod._SoftTreeModule.deepen = jitterli(self.jitter)
        try:
            self.ic_ = SoftDecisionTree(depth=self.depth, max_epochs=self.max_epochs,
                                        growth="incremental", random_state=self.random_state).fit(X, y)
        finally:
            sdt_mod._SoftTreeModule.deepen = orij
        self.classes_ = self.ic_.classes_
        return self

    def predict(self, X):
        return self.ic_.predict(X)


VERI = {"Iris": load_iris, "Wine": load_wine, "Digits": load_digits}
P = pathlib.Path(__file__).resolve().parent / "sonuc.json"
S = json.loads(P.read_text(encoding="utf-8"))
for ad, yuk in VERI.items():
    X, y = yuk(return_X_y=True)
    r = combined_5x2cv_f_test(make_pipeline(StandardScaler(), Buyuyen(jitter=0.0)),
                              make_pipeline(StandardScaler(), Buyuyen(jitter=0.2)), X, y, random_state=0)
    g = combined_5x2cv_f_test(
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="random", random_state=0)),
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="residual", random_state=0)),
        X, y, random_state=0)
    print(f"{ad:7s} jitter eski {S['jitter_test'][ad]} → yeni [{r.statistic:.3f}, {r.p_value:.6f}]"
          f" | GAL eski {S['gal_test'][ad]} → yeni [{g.statistic:.3f}, {g.p_value:.6f}]", flush=True)
    S["jitter_test"][ad] = [round(float(r.statistic), 3), round(float(r.p_value), 6)]
    S["gal_test"][ad] = [round(float(g.statistic), 3), round(float(g.p_value), 6)]
    P.write_text(json.dumps(S, indent=2, ensure_ascii=False))
print("BİTTİ")

#!/usr/bin/env python3
"""
Aynı üç karşılaştırmayı OpenML CC-18'den seçilmiş 20 veri kümesinde tekrarla.

    OMP_NUM_THREADS=1 python3 paper/arxiv/olcum/openml.py

olc.py'deki protokolün aynısı: özellikler yalnız eğitim katından ölçeklenir,
3 tohum x 5-kat, ortalama ± sapma; ana karşılaştırma için 5x2cv F testi.
Fark: veri kümeleri OpenML'den, n > 5000 olanlar tabakalı olarak 5000'e
indirilir (süre için; sonuçta belirtilir). Omnivariate ağaç dahil değil, her
düğümde MLP çapraz doğrulaması 20 veri kümesinde günler alır.

Sonuç her veri kümesinden sonra openml-sonuc.json'a yazılır; yarım kalan koşu
kaldığı yerden devam eder.
"""
import json, pathlib, sys, time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import neural_trees.decision_trees.soft_decision_tree as sdt_mod
from neural_trees import GALNetwork, SoftDecisionTree, combined_5x2cv_f_test

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "openml-sonuc.json"
TOHUM = (0, 1, 2)
N_MAKS = 5000
VERI = [1462, 1464, 1489, 37, 54, 40984, 182, 28, 32, 44,
        1504, 1480, 40994, 40499, 1497, 1494, 1487, 1067, 1068, 1050]

orij = sdt_mod._SoftTreeModule.deepen


def jitterli(j):
    def deepen(self, nc, jitter=0.0, _j=j):
        return orij(self, nc, _j)
    return deepen


class Buyuyen(ClassifierMixin, BaseEstimator):
    """5x2cv testi klonlayarak çalıştığı için jitter'ı taşıyan sarmalayıcı."""

    def __init__(self, jitter=0.2, depth=4, max_epochs=150, random_state=0):
        self.jitter = jitter
        self.depth = depth
        self.max_epochs = max_epochs
        self.random_state = random_state

    def fit(self, X, y):
        sdt_mod._SoftTreeModule.deepen = jitterli(self.jitter)
        try:
            self.ic_ = SoftDecisionTree(depth=self.depth, max_epochs=self.max_epochs,
                                        growth="incremental",
                                        random_state=self.random_state).fit(X, y)
        finally:
            sdt_mod._SoftTreeModule.deepen = orij
        self.classes_ = self.ic_.classes_
        return self

    def predict(self, X):
        return self.ic_.predict(X)


def yukle(did):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = d.data, LabelEncoder().fit_transform(d.target)
    if len(y) > N_MAKS:
        X, _, y, _ = train_test_split(X, y, train_size=N_MAKS, stratify=y, random_state=0)
    return d.details["name"], X.reset_index(drop=True), y


def onisleyici(X):
    sayisal = X.select_dtypes(include="number").columns.tolist()
    kategorik = [c for c in X.columns if c not in sayisal]
    return ColumnTransformer([
        ("s", StandardScaler(), sayisal),
        ("k", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorik),
    ])


def cv(yap, X, y, olcu=None):
    acc, ek = [], []
    for s in TOHUM:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            on = onisleyici(X).fit(X.iloc[tr])
            e = yap(s).fit(on.transform(X.iloc[tr]), y[tr])
            acc.append(float((e.predict(on.transform(X.iloc[te])) == y[te]).mean()))
            if olcu:
                ek.append(float(olcu(e)))
    out = [round(np.mean(acc), 4), round(np.std(acc, ddof=1), 4)]
    if ek:
        out += [round(np.mean(ek), 2), round(np.std(ek, ddof=1), 2)]
    return out


def main():
    S = json.loads(CIKTI.read_text()) if CIKTI.exists() else {}
    for did in VERI:
        if str(did) in S:
            continue
        t0 = time.time()
        ad, X, y = yukle(did)
        R = {"ad": ad, "n": int(len(y)), "p": int(X.shape[1]), "K": int(len(set(y)))}
        print(f"\n### {did} {ad}  n={R['n']} p={R['p']} K={R['K']}", flush=True)

        J = {}
        for j in (0.0, 0.2, None):
            if j is None:
                def yap(s):
                    return SoftDecisionTree(depth=4, max_epochs=150, random_state=s)
            else:
                sdt_mod._SoftTreeModule.deepen = jitterli(j)

                def yap(s):
                    return SoftDecisionTree(depth=4, max_epochs=150,
                                            growth="incremental", random_state=s)
            J["sıfırdan" if j is None else f"jitter={j}"] = cv(yap, X, y)
            sdt_mod._SoftTreeModule.deepen = orij
        R["jitter"] = J
        print("  jitter :", J, flush=True)

        on = onisleyici(X).fit(X)
        r = combined_5x2cv_f_test(Buyuyen(0.0), Buyuyen(0.2), on.transform(X), y, random_state=0)
        R["jitter_test"] = [round(float(r.statistic), 3), round(float(r.p_value), 6)]
        print("  F test :", R["jitter_test"], flush=True)

        G = {}
        for init in ("random", "residual"):
            G[init] = cv(lambda s, i=init: GALNetwork(max_epochs=150, growth_init=i, random_state=s),
                         X, y, olcu=lambda e: e.n_hidden_final_)
        R["gal"] = G
        print("  gal    :", G, flush=True)

        P = {}
        for g in ("none", "per_leaf"):
            P[g] = cv(lambda s, gg=g: SoftDecisionTree(depth=6, max_epochs=180, growth=gg, random_state=s),
                      X, y, olcu=lambda e: len(e.get_split_weights()))
        R["per_leaf"] = P
        print("  perleaf:", P, flush=True)

        R["sure_sn"] = round(time.time() - t0)
        S[str(did)] = R
        CIKTI.write_text(json.dumps(S, indent=1, ensure_ascii=False))
        print(f"  ({R['sure_sn']}s) kaydedildi", flush=True)
    print("\nBİTTİ:", len(S), "veri kümesi")


if __name__ == "__main__":
    main()

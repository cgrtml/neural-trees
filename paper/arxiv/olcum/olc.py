#!/usr/bin/env python3
"""
Makaledeki her sayıyı üreten betik. Çıktı: sonuc.json + ekrana tablo.

    OMP_NUM_THREADS=1 python3 paper/arxiv/olcum/olc.py

Önceki sürümde yalnız ortalamalar vardı. Hakem ilk soracağı şeyi soramasın
diye artık her ortalamanın yanında standart sapması ve kritik karşılaştırmalar
için kütüphanenin kendi 5x2cv F testi de var.
"""

import json
import pathlib
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import neural_trees.decision_trees.soft_decision_tree as sdt_mod
from neural_trees import (
    GALNetwork,
    OmnivariateDecisionTree,
    SoftDecisionTree,
    combined_5x2cv_f_test,
)

TOHUM = (0, 1, 2)
SONUC = {}


def veri(ad):
    if ad == "Iris":
        X, y = load_iris(return_X_y=True)
    elif ad == "Wine":
        X, y = load_wine(return_X_y=True)
    elif ad == "Cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif ad == "Digits":
        X, y = load_digits(return_X_y=True)
    elif ad == "Sentetik-800x20":
        X, y = make_classification(
            n_samples=800,
            n_features=20,
            n_informative=6,
            n_redundant=0,
            flip_y=0.05,
            class_sep=0.9,
            random_state=0,
        )
    elif ad == "Sentetik-2000x50":
        X, y = make_classification(
            n_samples=2000,
            n_features=50,
            n_informative=12,
            n_redundant=8,
            flip_y=0.05,
            class_sep=0.8,
            n_classes=4,
            n_clusters_per_class=2,
            random_state=0,
        )
    return X, y


def cv(yap, ad, olcu=None):
    """yap(seed) -> estimator. Katman başına doğruluk ve isteğe bağlı bir ölçü."""
    X, y = veri(ad)
    acc, ek = [], []
    for s in TOHUM:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            e = yap(s).fit(sc.transform(X[tr]), y[tr])
            acc.append((e.predict(sc.transform(X[te])) == y[te]).mean())
            if olcu:
                ek.append(olcu(e))
    return (
        np.mean(acc),
        np.std(acc, ddof=1),
        (np.mean(ek), np.std(ek, ddof=1)) if ek else None,
    )


def yaz(baslik, satirlar):
    print(f"\n### {baslik}")
    for s in satirlar:
        print("  " + s)
    sys.stdout.flush()


# ── 1 · JITTER · ana bulgu ────────────────────────────────────────────
orij = sdt_mod._SoftTreeModule.deepen


def jitterli(j):
    def deepen(self, nc, jitter=0.0, _j=j):
        return orij(self, nc, _j)

    return deepen


SONUC["jitter"] = {}
for ad in ("Iris", "Wine", "Digits"):
    satir = []
    for j in (0.0, 0.05, 0.2, 0.5, None):
        t0 = time.time()
        if j is None:

            def yap(s):
                return SoftDecisionTree(depth=4, max_epochs=150, random_state=s)
        else:
            sdt_mod._SoftTreeModule.deepen = jitterli(j)

            def yap(s):
                return SoftDecisionTree(
                    depth=4, max_epochs=150, growth="incremental", random_state=s
                )

        m, sd, _ = cv(yap, ad)
        sdt_mod._SoftTreeModule.deepen = orij
        etiket = "sıfırdan" if j is None else f"jitter={j}"
        SONUC["jitter"].setdefault(ad, {})[etiket] = [round(m, 4), round(sd, 4)]
        satir.append(f"{etiket:12s} {m:.3f} ± {sd:.3f}   ({time.time() - t0:.0f}s)")
    yaz(f"JITTER · {ad}", satir)


# ── 2 · jitter=0 ile jitter=0.2 arasında 5x2cv F testi ────────────────
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
            self.ic_ = SoftDecisionTree(
                depth=self.depth,
                max_epochs=self.max_epochs,
                growth="incremental",
                random_state=self.random_state,
            ).fit(X, y)
        finally:
            sdt_mod._SoftTreeModule.deepen = orij
        self.classes_ = self.ic_.classes_
        return self

    def predict(self, X):
        return self.ic_.predict(X)


SONUC["jitter_test"] = {}
for ad in ("Iris", "Wine"):
    X, y = veri(ad)
    # Ölçekleme testin kendi katlarında (Pipeline); tüm veriyi önceden
    # ölçeklemek test katının istatistiklerini eğitime sızdırır.
    r = combined_5x2cv_f_test(
        make_pipeline(StandardScaler(), Buyuyen(jitter=0.0)),
        make_pipeline(StandardScaler(), Buyuyen(jitter=0.2)),
        X, y, random_state=0,
    )
    SONUC["jitter_test"][ad] = [
        round(float(r.statistic), 3),
        round(float(r.p_value), 5),
    ]
    yaz(
        f"5x2cv F · jitter 0 vs 0.2 · {ad}", [f"F={r.statistic:.3f}  p={r.p_value:.5f}"]
    )

# ── 3 · GAL · rastgele vs residual ────────────────────────────────────
SONUC["gal"] = {}
for ad in ("Iris", "Wine", "Digits"):
    satir = []
    for init in ("random", "residual"):
        m, sd, (bm, bsd) = cv(
            lambda s, i=init: GALNetwork(max_epochs=150, growth_init=i, random_state=s),
            ad,
            olcu=lambda e: e.n_hidden_final_,
        )
        SONUC["gal"].setdefault(ad, {})[init] = [
            round(m, 4),
            round(sd, 4),
            round(bm, 2),
            round(bsd, 2),
        ]
        satir.append(f"{init:9s} acc {m:.3f} ± {sd:.3f}   birim {bm:.1f} ± {bsd:.1f}")
    yaz(f"GAL · {ad}", satir)

for ad in ("Iris", "Wine"):
    X, y = veri(ad)
    r = combined_5x2cv_f_test(
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="random", random_state=0)),
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="residual", random_state=0)),
        X,
        y,
        random_state=0,
    )
    SONUC.setdefault("gal_test", {})[ad] = [
        round(float(r.statistic), 3),
        round(float(r.p_value), 5),
    ]
    yaz(
        f"5x2cv F · GAL random vs residual · {ad}",
        [f"F={r.statistic:.3f}  p={r.p_value:.5f}"],
    )

# ── 4 · yaprak başına büyütme ─────────────────────────────────────────
SONUC["per_leaf"] = {}
for ad in ("Sentetik-800x20", "Sentetik-2000x50"):
    satir = []
    for g in ("none", "per_leaf"):
        m, sd, (bm, bsd) = cv(
            lambda s, gg=g: SoftDecisionTree(
                depth=6, max_epochs=180, growth=gg, random_state=s
            ),
            ad,
            olcu=lambda e: len(e.get_split_weights()),
        )
        SONUC["per_leaf"].setdefault(ad, {})[g] = [
            round(m, 4),
            round(sd, 4),
            round(bm, 2),
            round(bsd, 2),
        ]
        satir.append(f"{g:9s} acc {m:.3f} ± {sd:.3f}   bölme {bm:.1f} ± {bsd:.1f}")
    yaz(f"PER-LEAF · {ad}", satir)


# ── 5 · omnivariate seçim kuralı ──────────────────────────────────────
def dugum(e):
    n = [0]

    def w(nd):
        n[0] += 1
        if nd.left is not None:
            w(nd.left)
            w(nd.right)

    w(e.root_)
    return n[0]


SONUC["omni"] = {}
for ad in ("Cancer", "Digits"):
    satir = []
    for sec in ("accuracy", "test"):
        m, sd, (bm, bsd) = cv(
            lambda s, ss=sec: OmnivariateDecisionTree(max_depth=3, selection=ss),
            ad,
            olcu=dugum,
        )
        SONUC["omni"].setdefault(ad, {})[sec] = [
            round(m, 4),
            round(sd, 4),
            round(bm, 2),
            round(bsd, 2),
        ]
        satir.append(f"{sec:9s} acc {m:.3f} ± {sd:.3f}   düğüm {bm:.1f} ± {bsd:.1f}")
    yaz(f"OMNIVARIATE · {ad}", satir)

pathlib.Path("paper/arxiv/olcum/sonuc.json").write_text(
    json.dumps(SONUC, indent=2, ensure_ascii=False)
)
print("\n→ paper/arxiv/olcum/sonuc.json yazıldı")

#!/usr/bin/env python3
"""
Per-leaf bölme başlatması ve büyütme bütçesi: farkın görünebileceği kümelerde.

    OMP_NUM_THREADS=1 python3 paper/arxiv/olcum/perleaf.py

Kümeler: Digits + OpenML'in yedi çok sınıflı kümesi (ölü seviyenin bedeli ve
per-leaf'in kaybı yalnız bunlarda görüldü; ikili kümelerde her kol zaten eşit).

Kollar
  per-leaf, depth 6 / 180 epoch:  uniform (0.7 öncesi) · random · residual ·
                                  residual_gate   (+ tam ağaç referansı)
  artımlı,  depth 4 / 150 epoch:  budget=split · budget=full · sıfırdan
Kararlar: 5x2cv F, per-leaf residual_gate vs uniform; artımlı full vs split.
Sonuç her kümeden sonra perleaf-sonuc.json'a yazılır.
"""
import json
import pathlib
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_trees import SoftDecisionTree, combined_5x2cv_f_test

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "perleaf-sonuc.json"
TOHUM = (0, 1, 2)
N_MAKS = 5000
OPENML = [54, 40984, 182, 28, 32, 40499, 1497]   # vehicle segment satimage optdigits pendigits texture wall-robot


def perleaf(init, s):
    return SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf",
                            growth_init=init, random_state=s)


def artimli(butce, s):
    return SoftDecisionTree(depth=4, max_epochs=150, growth="incremental",
                            growth_budget=butce, random_state=s)


def cv(yap, X, y, olcu=None):
    acc, ek = [], []
    for s in TOHUM:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            e = yap(s).fit(sc.transform(X[tr]), y[tr])
            acc.append(float((e.predict(sc.transform(X[te])) == y[te]).mean()))
            if olcu:
                ek.append(float(olcu(e)))
    out = [round(np.mean(acc), 4), round(np.std(acc, ddof=1), 4)]
    if ek:
        out += [round(np.mean(ek), 2), round(np.std(ek, ddof=1), 2)]
    return out


def ftest(a, b, X, y):
    # ölçekleme testin kendi katlarında
    r = combined_5x2cv_f_test(make_pipeline(StandardScaler(), a),
                              make_pipeline(StandardScaler(), b), X, y, random_state=0)
    return [round(float(r.statistic), 3), round(float(r.p_value), 6)]


def kumeler():
    X, y = load_digits(return_X_y=True)
    yield "Digits", X, y
    for did in OPENML:
        d = fetch_openml(data_id=did, as_frame=False, parser="auto")
        X, y = d.data.astype(float), LabelEncoder().fit_transform(d.target)
        if len(y) > N_MAKS:
            X, _, y, _ = train_test_split(X, y, train_size=N_MAKS, stratify=y, random_state=0)
        yield d.details["name"], X, y


def main():
    S = json.loads(CIKTI.read_text()) if CIKTI.exists() else {}
    for ad, X, y in kumeler():
        if ad in S:
            continue
        t0 = time.time()
        R = {"n": int(len(y)), "p": int(X.shape[1]), "K": int(len(set(y)))}
        print(f"\n### {ad}  n={R['n']} p={R['p']} K={R['K']}", flush=True)
        R["tam"] = cv(lambda s: SoftDecisionTree(depth=6, max_epochs=180, random_state=s),
                      X, y, olcu=lambda e: len(e.get_split_weights()))
        print(f"  tam depth-6      {R['tam'][0]:.3f} ± {R['tam'][1]:.3f}  bölme {R['tam'][2]:.1f}", flush=True)
        # Büyüyen ağaçlar eğitim katının %10'unu doğrulamaya ayırır; tam ağaç
        # ayırmaz. Adil kontrol: aynı ayrımla, erken durmadan (patience = bütçe),
        # doğrulamada en iyi durumu tutan tam ağaç.
        R["tam_val"] = cv(lambda s: SoftDecisionTree(depth=6, max_epochs=180, early_stopping=True,
                                                     n_iter_no_change=180, random_state=s),
                          X, y, olcu=lambda e: len(e.get_split_weights()))
        print(f"  tam depth-6 %90  {R['tam_val'][0]:.3f} ± {R['tam_val'][1]:.3f}", flush=True)
        for init in ("uniform", "random", "residual", "residual_gate"):
            R[f"perleaf_{init}"] = cv(lambda s, i=init: perleaf(i, s), X, y,
                                      olcu=lambda e: len(e.get_split_weights()))
            r = R[f"perleaf_{init}"]
            print(f"  perleaf {init:14s} {r[0]:.3f} ± {r[1]:.3f}  bölme {r[2]:.1f} ± {r[3]:.1f}", flush=True)
        R["test_perleaf_gate_vs_uniform"] = ftest(perleaf("uniform", 0), perleaf("residual_gate", 0), X, y)
        print(f"  F perleaf        {R['test_perleaf_gate_vs_uniform']}", flush=True)
        for butce in ("split", "full"):
            R[f"artimli_{butce}"] = cv(lambda s, b=butce: artimli(b, s), X, y,
                                       olcu=lambda e: e.tree_depth_)
            r = R[f"artimli_{butce}"]
            print(f"  artimli {butce:6s}   {r[0]:.3f} ± {r[1]:.3f}  derinlik {r[2]:.1f}", flush=True)
        R["sifirdan4"] = cv(lambda s: SoftDecisionTree(depth=4, max_epochs=150, random_state=s), X, y)
        print(f"  sıfırdan depth-4 {R['sifirdan4'][0]:.3f} ± {R['sifirdan4'][1]:.3f}", flush=True)
        R["sifirdan4_val"] = cv(lambda s: SoftDecisionTree(depth=4, max_epochs=150, early_stopping=True,
                                                           n_iter_no_change=150, random_state=s), X, y)
        print(f"  sıfırdan d-4 %90 {R['sifirdan4_val'][0]:.3f} ± {R['sifirdan4_val'][1]:.3f}", flush=True)
        R["test_artimli_full_vs_split"] = ftest(artimli("split", 0), artimli("full", 0), X, y)
        print(f"  F artimli        {R['test_artimli_full_vs_split']}", flush=True)
        R["sure_sn"] = round(time.time() - t0)
        S[ad] = R
        CIKTI.write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("\nBİTTİ:", len(S), "veri kümesi")


if __name__ == "__main__":
    main()

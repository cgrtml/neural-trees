#!/usr/bin/env python3
"""
Distil a tree ensemble (XGBoost, LightGBM, RandomForest) into one tree, and
measure how faithfully the single tree reproduces the ensemble.

Question: as a readable surrogate of an ensemble a company already runs, does
a soft tree copy the teacher more faithfully than a hard tree trained on
exactly the same data, and where does the difference come from?

Protocol, per dataset (the 24 of rakipler.py), seed 0, 5 stratified folds:
  * StandardScaler on the training fold.
  * Teacher fitted on the true training labels.
  * Transfer set = training rows + 2 perturbed copies, labelled by the
    teacher. Continuous columns get Gaussian noise (sd 0.15, standardised
    units); columns with at most 10 distinct training values are categorical
    or ordinal codes, so instead of noise each cell is swapped, with
    probability 0.3, for the value of a random training row. Noise on a code
    would create values that never occur.
  * Students, all on the transfer set:
      cart6      CART depth 6 on teacher labels
      cart_eq    CART with as many leaves as the soft tree has (equal budget)
      multi6     hard oblique tree depth 6 (MultivariateDecisionTree), which
                 separates "oblique splits" from "soft routing"
      soft       soft tree per-leaf depth <= 6 on teacher labels
      soft_hard  the same soft tree read as hard rules (to_hard_tree), which
                 is what a person would actually read
      soft_prob  soft tree on the teacher's probabilities (each row repeated
                 per class with weight = teacher probability, which is exactly
                 the soft-target cross-entropy)
      soft_noaug soft tree on the training rows only (XGBoost teacher only),
                 to see what the perturbed copies are worth
  * On the test fold: fidelity (label agreement with the teacher), accuracy,
    total-variation distance to the teacher's probabilities, top-label ECE,
    and for soft trees the mean number of features that carry 90 % of a
    split's weight (the readability baseline for the sparsity experiment).

Teachers run in one process and students in another, because xgboost and
torch in one process can deadlock on libomp (see rakipler.py).

    python3 benchmarks/distill.py ogretmen [--hizli]
    python3 benchmarks/distill.py ogrenci  [--hizli] [--n-jobs 10]
    python3 benchmarks/distill.py ozet
    python3 benchmarks/distill.py istatistik      # paired tests over datasets
    python3 benchmarks/distill.py seyrek          # fidelity vs split_features
    python3 benchmarks/distill.py seyrek_cart     # the same, students started from CART
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KOK = pathlib.Path(__file__).resolve().parent
ARA = KOK / "_distill_cache"
CIKTI = KOK / "distill-sonuc.json"
SEED, KAT, KOPYA, GURULTU, TAKAS, AYRIK = 0, 5, 2, 0.15, 0.3, 10
HIZLI_KUMELER = {"Cancer", "Digits", "phoneme"}
OGRENCILER = ["cart6", "cart_eq", "multi6", "soft", "soft_hard", "soft_prob", "soft_noaug"]


def tasima_kumesi(Xtr, rng):
    ayrik = np.array([len(np.unique(Xtr[:, j])) <= AYRIK for j in range(Xtr.shape[1])])
    kopyalar = [Xtr]
    for _ in range(KOPYA):
        Z = Xtr + rng.normal(0, GURULTU, Xtr.shape) * ~ayrik
        if ayrik.any():
            kaynak = Xtr[rng.integers(0, len(Xtr), len(Xtr))]
            takas = (rng.random(Xtr.shape) < TAKAS) & ayrik
            Z = np.where(takas, kaynak, Z)
        kopyalar.append(Z)
    return np.vstack(kopyalar)


def ogretmen_asamasi(hizli):
    from rakipler import Lgbm, Rf, Xgb, kumeler
    ARA.mkdir(exist_ok=True)
    for ad, X, y in kumeler(hizli=False):
        if hizli and ad not in HIZLI_KUMELER:
            continue
        K = len(np.unique(y))
        for k, (tr, te) in enumerate(StratifiedKFold(KAT, shuffle=True, random_state=SEED).split(X, y)):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            Xa = tasima_kumesi(Xtr, np.random.default_rng(SEED * 100 + k))
            for T in (Xgb, Lgbm, Rf):
                yol = ARA / f"{ad}__{T.ad}__{k}.npz"
                if yol.exists():
                    continue
                t0 = time.time()
                t = T(SEED).fit(Xtr, y[tr])
                Pa, Pte = t.m.predict_proba(Xa), t.m.predict_proba(Xte)
                np.savez_compressed(yol, Xa=Xa, Pa=Pa, Xte=Xte, yte=y[te], Pte=Pte, n_tr=len(tr), K=K)
                print(f"{ad:28s} {T.ad:12s} fold {k}  teacher acc {np.mean(Pte.argmax(1) == y[te]):.3f}  "
                      f"{time.time()-t0:.0f} s", flush=True)


def _tam(P, sinif, K):
    """Probabilities over all K classes from a model that saw only `sinif`."""
    out = np.zeros((len(P), K))
    out[:, np.asarray(sinif, dtype=int)] = P
    return out


def _ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    kenar = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(kenar[:-1], kenar[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return float(e)


def _okunurluk(sdt):
    """Mean number of features that carry 90 % of each split's |weight|."""
    W = np.abs(sdt.model_.gates.weight.detach().cpu().numpy())
    sayi = []
    for w in W:
        if w.sum() == 0:
            continue
        s = np.sort(w)[::-1].cumsum() / w.sum()
        sayi.append(int(np.searchsorted(s, 0.9) + 1))
    return float(np.mean(sayi)) if sayi else 0.0


def _soft(seed):
    from neural_trees import SoftDecisionTree
    return SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf", growth_init="residual",
                            random_state=seed)


def _ogrenci_bir(yol):
    import torch
    torch.set_num_threads(1)
    from sklearn.tree import DecisionTreeClassifier

    from neural_trees import MultivariateDecisionTree
    out = yol.with_suffix(".json")
    if out.exists():
        return
    d = np.load(yol)
    Xa, Pa, Xte, yte, Pte, n_tr, K = d["Xa"], d["Pa"], d["Xte"], d["yte"], d["Pte"], int(d["n_tr"]), int(d["K"])
    ya, tte = Pa.argmax(1), Pte.argmax(1)
    ogretmen = yol.stem.split("__")[1]
    R = {"teacher": {"acc": float(np.mean(tte == yte)), "ece": _ece(Pte, yte)}}

    def kaydet(isim, P, sn, **ek):
        R[isim] = {"fid": float(np.mean(P.argmax(1) == tte)), "acc": float(np.mean(P.argmax(1) == yte)),
                   "tv": float(0.5 * np.abs(P - Pte).sum(1).mean()), "ece": _ece(P, yte),
                   "sn": round(sn, 1), **ek}

    t0 = time.time()
    s = _soft(SEED).fit(Xa, ya)
    kaydet("soft", _tam(s.predict_proba(Xte), s.classes_, K), time.time() - t0,
           splits=len(s.get_split_weights()), feat90=_okunurluk(s))
    h = s.to_hard_tree()
    kaydet("soft_hard", _tam(np.eye(len(s.classes_))[np.searchsorted(s.classes_, h.predict(Xte))], s.classes_, K), 0.0)
    yaprak = len(s.get_split_weights()) + 1

    for isim, m in (("cart6", DecisionTreeClassifier(max_depth=6, random_state=SEED)),
                    ("cart_eq", DecisionTreeClassifier(max_leaf_nodes=max(yaprak, 2), random_state=SEED)),
                    ("multi6", MultivariateDecisionTree(max_depth=6, random_state=SEED))):
        t0 = time.time()
        m.fit(Xa, ya)
        kaydet(isim, _tam(m.predict_proba(Xte), m.classes_, K), time.time() - t0,
               leaves=int(m.get_n_leaves()) if hasattr(m, "get_n_leaves") else None)

    # soft targets: repeat each row once per class the teacher gives >= 1 %
    i, c = np.nonzero(Pa >= 0.01)
    w = Pa[i, c]
    t0 = time.time()
    sp = _soft(SEED).fit(Xa[i], c, sample_weight=w)
    kaydet("soft_prob", _tam(sp.predict_proba(Xte), sp.classes_, K), time.time() - t0,
           splits=len(sp.get_split_weights()), feat90=_okunurluk(sp))

    if ogretmen == "XGBoost":
        t0 = time.time()
        sn = _soft(SEED).fit(Xa[:n_tr], ya[:n_tr])
        kaydet("soft_noaug", _tam(sn.predict_proba(Xte), sn.classes_, K), time.time() - t0)

    out.write_text(json.dumps(R))
    print(f"{yol.stem:44s} fid cart6 {R['cart6']['fid']:.3f} multi6 {R['multi6']['fid']:.3f} "
          f"soft {R['soft']['fid']:.3f} soft_prob {R['soft_prob']['fid']:.3f} "
          f"hard {R['soft_hard']['fid']:.3f}", flush=True)


def ogrenci_asamasi(hizli, n_jobs):
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*.npz"))
    if hizli:
        yollar = [p for p in yollar if p.stem.split("__")[0] in HIZLI_KUMELER]
    yollar.sort(key=lambda p: -p.stat().st_size)          # long fits first
    Parallel(n_jobs=n_jobs)(delayed(_ogrenci_bir)(p) for p in yollar)


def ozet():
    S = {"_protokol": __doc__.split("Teachers run")[0].strip()}
    for js in sorted(ARA.glob("*.json")):
        ad, ogretmen, _ = js.stem.split("__")
        n = int(np.load(js.with_suffix(".npz"))["n_tr"])
        r = S.setdefault(ad, {}).setdefault(ogretmen, {"n_train": n, "folds": []})
        r["folds"].append(json.loads(js.read_text()))
    for ad, tt in S.items():
        if ad.startswith("_"):
            continue
        for ogretmen, r in tt.items():
            ort = {}
            for isim in ["teacher"] + OGRENCILER:
                kat = [f[isim] for f in r["folds"] if isim in f]
                if kat:
                    ort[isim] = {k: round(float(np.mean([f[k] for f in kat])), 4)
                                 for k in kat[0] if kat[0][k] is not None}
            r["mean"], r["n_folds"] = ort, len(r["folds"])
            del r["folds"]
    CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(f"{'dataset':26s} {'teacher':12s} {'n':>5s} {'T acc':>6s} | fidelity: {'cart6':>6s} {'cartEq':>6s} "
          f"{'multi6':>6s} {'soft':>6s} {'hard':>6s} {'prob':>6s} {'noaug':>6s} | feat90")
    satir = [(ad, o, r) for ad, tt in S.items() if not ad.startswith("_") for o, r in tt.items()]
    for ad, o, r in sorted(satir, key=lambda x: (x[1], x[2]["n_train"])):
        m = r["mean"]
        f = lambda k: f"{m[k]['fid']:6.3f}" if k in m else "     -"  # noqa: E731
        print(f"{ad[:26]:26s} {o:12s} {r['n_train']:5d} {m['teacher']['acc']:6.3f} | fidelity: {f('cart6')} "
              f"{f('cart_eq')} {f('multi6')} {f('soft')} {f('soft_hard')} {f('soft_prob')} {f('soft_noaug')} | "
              f"{m.get('soft', {}).get('feat90', 0):.1f}  ({r['n_folds']} folds)")


def istatistik():
    """Paired Wilcoxon tests over the dataset x teacher pairs of distill-sonuc.json."""
    from scipy.stats import wilcoxon
    S = json.loads(CIKTI.read_text(encoding="utf-8"))
    rows = [(ad, t, r["mean"]) for ad, tt in S.items() if not ad.startswith("_") for t, r in tt.items()]

    def fark(a, b, metric="fid", secim=lambda ad, t, m: True, etiket=""):
        d = np.array([m[a][metric] - m[b][metric] for ad, t, m in rows if secim(ad, t, m) and a in m and b in m])
        w, lo = int((d > 0.005).sum()), int((d < -0.005).sum())
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"{etiket:24s} {metric:3s} {a:>10s} - {b:<10s} mean {d.mean()*100:+.2f} pt  median "
              f"{np.median(d)*100:+.2f}  higher/tie/lower {w}/{len(d)-w-lo}/{lo}  p={p:.4f}  n={len(d)}")

    for a, b in [("soft", "cart6"), ("soft", "cart_eq"), ("soft_prob", "cart6"), ("soft_hard", "cart6"),
                 ("multi6", "cart6"), ("soft", "multi6"), ("soft_prob", "soft"), ("soft_hard", "soft")]:
        fark(a, b)
    fark("soft", "soft_noaug", secim=lambda ad, t, m: t == "XGBoost", etiket="XGBoost teacher")
    for T in ("XGBoost", "LightGBM", "RandomForest"):
        fark("soft_prob", "cart6", secim=lambda ad, t, m, T=T: t == T, etiket=T)
    fark("soft_prob", "cart6", secim=lambda ad, t, m: m["soft"]["feat90"] >= 20, etiket="soft feat90 >= 20")
    fark("soft_prob", "cart6", secim=lambda ad, t, m: m["soft"]["feat90"] < 10, etiket="soft feat90 < 10")
    fark("soft_prob", "cart6", metric="tv", etiket="TV, lower is closer")
    for a in ("cart6", "multi6", "soft", "soft_prob", "soft_hard"):
        print(f"{a:10s} fidelity {np.mean([m[a]['fid'] for _, _, m in rows]):.4f}  accuracy - teacher "
              f"{np.mean([m[a]['acc'] - m['teacher']['acc'] for _, _, m in rows])*100:+.2f} pt  "
              f"TV {np.mean([m[a]['tv'] for _, _, m in rows]):.3f}  ECE {np.mean([m[a]['ece'] for _, _, m in rows]):.3f}")
    print(f"CART depth 6 leaves {np.mean([m['cart6']['leaves'] for _, _, m in rows]):.1f}, "
          f"soft tree splits {np.mean([m['soft']['splits'] for _, _, m in rows]):.1f}")


SEYREK_K = (1, 2, 3, 5)
SEYREK_CIKTI = KOK / "distill-seyrek.json"


def _seyrek_bir(yol, k):
    import torch
    torch.set_num_threads(1)
    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__k{k}.seyrek.json")
    if out.exists():
        return
    d = np.load(yol)
    Xa, Pa, Xte, Pte, K = d["Xa"], d["Pa"], d["Xte"], d["Pte"], int(d["K"])
    k_eff = min(k, Xa.shape[1])
    i, c = np.nonzero(Pa >= 0.01)
    m = SoftDecisionTree(depth=6, max_epochs=180, growth="per_leaf", growth_init="residual",
                         split_features=k_eff, random_state=SEED).fit(Xa[i], c, sample_weight=Pa[i, c])
    P = _tam(m.predict_proba(Xte), m.classes_, K)
    hard = m.to_hard_tree().predict(Xte)
    out.write_text(json.dumps({"fid": float(np.mean(P.argmax(1) == Pte.argmax(1))),
                               "fid_hard": float(np.mean(hard == Pte.argmax(1))),
                               "splits": len(m.get_split_weights())}))


def _seyrek_cart_bir(yol, k):
    """Student started from CART (init="cart"), k features per split, soft targets."""
    import torch
    torch.set_num_threads(1)
    from neural_trees import SoftDecisionTree
    out = yol.with_name(f"{yol.stem}__k{k}.cart.json")
    if out.exists():
        return
    d = np.load(yol)
    Xa, Pa, Xte, Pte, K = d["Xa"], d["Pa"], d["Xte"], d["Pte"], int(d["K"])
    i, c = np.nonzero(Pa >= 0.01)
    m = SoftDecisionTree(depth=6, max_epochs=60, init="cart", learn_temperature=True,
                         split_features=min(k, Xa.shape[1]), random_state=SEED)
    m.fit(Xa[i], c, sample_weight=Pa[i, c])
    P = _tam(m.predict_proba(Xte), m.classes_, K)
    hard = m.to_hard_tree().predict(Xte)
    out.write_text(json.dumps({"fid": float(np.mean(P.argmax(1) == Pte.argmax(1))),
                               "fid_hard": float(np.mean(hard == Pte.argmax(1))),
                               "splits": int(m.model_.is_split.sum())}))


def seyrek_cart(n_jobs):
    """Fidelity of a CART-started student at k = 1, 2, 3 features per split, against CART."""
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    yollar = sorted(ARA.glob("*__XGBoost__*.npz"), key=lambda p: -p.stat().st_size)
    Parallel(n_jobs=n_jobs)(delayed(_seyrek_cart_bir)(p, k) for p in yollar for k in (1, 2, 3))
    S = {}
    for p in yollar:
        ad = p.stem.split("__")[0]
        tam = json.loads(p.with_suffix(".json").read_text())
        r = S.setdefault(ad, {"cart6": []})
        r["cart6"].append(tam["cart6"]["fid"])
        for k in (1, 2, 3):
            js = json.loads(p.with_name(f"{p.stem}__k{k}.cart.json").read_text())
            r.setdefault(f"cart-k{k}", []).append(js["fid"])
            r.setdefault(f"cart-k{k}_hard", []).append(js["fid_hard"])
    S = {ad: {k: round(float(np.mean(v)), 4) for k, v in r.items()} for ad, r in S.items()}
    (KOK / "distill-seyrek-cart.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    for k in ("cart-k1", "cart-k1_hard", "cart-k2", "cart-k3"):
        d = np.array([r[k] - r["cart6"] for r in S.values()])
        w, lo = int((d > 0.005).sum()), int((d < -0.005).sum())
        print(f"{k:13s} - cart6  mean {d.mean()*100:+.2f} pt  median {np.median(d)*100:+.2f}  "
              f"higher/tie/lower {w}/{len(d)-w-lo}/{lo}  p={wilcoxon(d).pvalue:.4f}")


def seyrek(n_jobs):
    """Fidelity of the soft-target student under split_features=k, XGBoost teacher."""
    from joblib import Parallel, delayed
    yollar = sorted(ARA.glob("*__XGBoost__*.npz"), key=lambda p: -p.stat().st_size)
    Parallel(n_jobs=n_jobs)(delayed(_seyrek_bir)(p, k) for p in yollar for k in SEYREK_K)
    S = {}
    for p in yollar:
        ad, _, kat = p.stem.split("__")
        tam = json.loads(p.with_suffix(".json").read_text())
        r = S.setdefault(ad, {"cart6": [], "dense": []})
        r["cart6"].append(tam["cart6"]["fid"])
        r["dense"].append(tam["soft_prob"]["fid"])
        for k in SEYREK_K:
            js = json.loads(p.with_name(f"{p.stem}__k{k}.seyrek.json").read_text())
            r.setdefault(f"k{k}", []).append(js["fid"])
            r.setdefault(f"k{k}_hard", []).append(js["fid_hard"])
    S = {ad: {k: round(float(np.mean(v)), 4) for k, v in r.items()} for ad, r in S.items()}
    SEYREK_CIKTI.write_text(json.dumps(S, indent=1), encoding="utf-8")
    cols = ["cart6"] + [f"k{k}" for k in SEYREK_K] + ["dense"]
    print(f"{'dataset':26s} " + " ".join(f"{c:>7s}" for c in cols))
    for ad, r in S.items():
        print(f"{ad[:26]:26s} " + " ".join(f"{r[c]:7.3f}" for c in cols))
    print(f"{'mean':26s} " + " ".join(f"{np.mean([r[c] for r in S.values()]):7.3f}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asama", choices=["ogretmen", "ogrenci", "ozet", "istatistik", "seyrek", "seyrek_cart"])
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=10)
    a = ap.parse_args()
    {"ogretmen": lambda: ogretmen_asamasi(a.hizli),
     "ogrenci": lambda: ogrenci_asamasi(a.hizli, a.n_jobs),
     "ozet": ozet, "istatistik": istatistik, "seyrek": lambda: seyrek(a.n_jobs),
     "seyrek_cart": lambda: seyrek_cart(a.n_jobs)}[a.asama]()


if __name__ == "__main__":
    main()

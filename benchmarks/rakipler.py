#!/usr/bin/env python3
"""
neural-trees'i modern rakiplerle aynı protokolde karşılaştır.

    python3 benchmarks/rakipler.py            # tümü (4 UCI + 20 OpenML)
    python3 benchmarks/rakipler.py --hizli    # yalnız Iris, uçtan uca deneme

Rakipler: XGBoost, LightGBM, Random Forest, scikit-learn MLP, GRANDE
(Marton ve ark., ICLR 2024), NODE (Popov ve ark., ICLR 2020; pytorch-tabular).
Bizden: SoftDecisionTree sıfırdan (depth 4) ve per-leaf büyütme (depth 6,
residual), GALNetwork.

Protokol olcum/olc.py ile aynı: özellikler yalnız eğitim katından ölçeklenir,
3 tohum x 5-kat, ortalama ± sapma, kat başına uydurma süresi. HİÇBİR MODEL
AYARLANMADI: herkes makul varsayılanlarla koşuyor, bu yüzden tablo "kim en
iyi" değil "ayarsız hâlde kim nerede duruyor" sorusuna cevap veriyor. Bunu
yazmadan tabloyu alıntılama.

Yorumlanabilirlik sütunu: modelin karar için gösterebildiği yapı — ağaç
bölme sayısı, ağaç x derinlik, ya da yok. Rakiplerin sayıları
karşılaştırılabilir olsun diye bizim ağaçlar için bölme sayısı, topluluklar
için ağaç sayısı x derinlik yazılır.

Sonuç her kümeden sonra benchmarks/rakipler-sonuc.json'a yazılır; yarım kalan
koşu kaldığı yerden devam eder. GRANDE ve NODE kurulu değilse atlanır ve
bu JSON'da belirtilir.

HER MODEL KENDİ ALT SÜRECİNDE koşar; bu süreç yalnız orkestratördür ve
numpy/pandas/sklearn dışında hiçbir şey yüklemez. Sebep ölçüldü: torch,
TensorFlow ve xgboost/lightgbm aynı süreçte (hepsi kendi libomp'unu getirir)
macOS'te sessizce kilitleniyor, 0% CPU. Her alt süreç için zaman aşımı var;
takılan model tabloya "zaman aşımı" olarak girer, koşu devam eder.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "rakipler-sonuc.json"
TOHUM = (0, 1, 2)
N_MAKS = 5000
UCI = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Cancer": load_breast_cancer,
    "Digits": load_digits,
}
OPENML = [
    1462,
    1464,
    1489,
    37,
    54,
    40984,
    182,
    28,
    32,
    44,
    1504,
    1480,
    40994,
    40499,
    1497,
    1494,
    1487,
    1067,
    1068,
    1050,
]


# ── rakip sarmalayıcıları · hepsi fit(X, y) / predict(X) ve `yapi` verir ──
class Xgb:
    ad = "XGBoost"

    def __init__(self, s):
        import xgboost as xgb

        self.m = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=s,
            n_jobs=1,
            verbosity=0,
        )

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return f"{self.m.n_estimators} trees x depth 6"


class Lgbm:
    ad = "LightGBM"

    def __init__(self, s):
        import lightgbm as lgb

        self.m = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.1,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return "300 trees x 31 leaves"


class Rf:
    ad = "RandomForest"

    def __init__(self, s):
        from sklearn.ensemble import RandomForestClassifier

        self.m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=1)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return "300 trees, unbounded"


class Mlp:
    ad = "MLP"

    def __init__(self, s):
        from sklearn.neural_network import MLPClassifier

        self.m = MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=300, random_state=s
        )

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return "none (64-64 MLP)"


class SoftScratch:
    ad = "SoftTree d4"

    def __init__(self, s):
        from neural_trees import SoftDecisionTree

        self.m = SoftDecisionTree(depth=4, max_epochs=150, random_state=s)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return f"{len(self.m.get_split_weights())} soft splits"


class SoftPerLeaf:
    ad = "SoftTree per-leaf"

    def __init__(self, s):
        from neural_trees import SoftDecisionTree

        self.m = SoftDecisionTree(
            depth=6,
            max_epochs=180,
            growth="per_leaf",
            growth_init="residual",
            random_state=s,
        )

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return f"{len(self.m.get_split_weights())} soft splits"


class Gal:
    ad = "GAL"

    def __init__(self, s):
        from neural_trees import GALNetwork

        self.m = GALNetwork(max_epochs=150, growth_init="residual", random_state=s)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def yapi(self):
        return f"{self.m.n_hidden_final_} hidden units"


class Grande:
    ad = "GRANDE"

    def __init__(self, s):
        from GRANDE import GRANDE

        self.s = s
        params = {
            "depth": 5,
            "n_estimators": 256,
            "learning_rate_weights": 0.005,
            "learning_rate_index": 0.01,
            "learning_rate_values": 0.01,
            "learning_rate_leaf": 0.01,
            "optimizer": "adam",
            "cosine_decay_steps": 0,
            "loss": "crossentropy",
            "focal_loss": False,
            "temperature": 0.0,
            "from_logits": True,
            "use_class_weights": True,
            "dropout": 0.0,
            "selected_variables": 0.8,
            "data_subset_fraction": 1.0,
        }
        args = {
            "epochs": 100,
            "early_stopping_epochs": 20,
            "batch_size": 64,
            "cat_idx": [],
            "objective": "classification",
            "random_seed": s,
            "verbose": 0,
        }
        self.m = GRANDE(params=params, args=args)

    def fit(self, X, y):
        cols = [f"f{i}" for i in range(X.shape[1])]
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=self.s
        )
        self.cols = cols
        self.m.fit(
            X_train=pd.DataFrame(Xtr, columns=cols),
            y_train=pd.Series(ytr),
            X_val=pd.DataFrame(Xva, columns=cols),
            y_val=pd.Series(yva),
        )
        return self

    def predict(self, X):
        p = np.asarray(self.m.predict(pd.DataFrame(X, columns=self.cols)))
        return p.argmax(1) if p.ndim == 2 else p

    def yapi(self):
        return "256 trees x depth 5"


class Node:
    ad = "NODE"

    def __init__(self, s):
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import NodeConfig

        self.s = s
        self.TM = TabularModel
        self.DC = DataConfig
        self.OC = OptimizerConfig
        self.TC = TrainerConfig
        self.NC = NodeConfig

    def fit(self, X, y):
        cols = [f"f{i}" for i in range(X.shape[1])]
        self.cols = cols
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=self.s
        )
        tr = pd.DataFrame(Xtr, columns=cols)
        tr["target"] = ytr
        va = pd.DataFrame(Xva, columns=cols)
        va["target"] = yva
        self.m = self.TM(
            data_config=self.DC(
                target=["target"], continuous_cols=cols, categorical_cols=[]
            ),
            model_config=self.NC(
                task="classification",
                num_layers=1,
                num_trees=256,
                depth=6,
                learning_rate=1e-3,
            ),
            optimizer_config=self.OC(),
            trainer_config=self.TC(
                max_epochs=50,
                batch_size=128,
                accelerator="cpu",
                progress_bar="none",
                early_stopping="valid_loss",
                early_stopping_patience=10,
                checkpoints=None,
                seed=self.s,
                trainer_kwargs={"enable_model_summary": False},
            ),
            verbose=False,
            suppress_lightning_logger=True,
        )
        self.m.fit(train=tr, validation=va)
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.cols)
        p = self.m.predict(df)
        return p["target_prediction"].values

    def yapi(self):
        return "256 oblivious trees x depth 6"


IMPORT = {
    "SoftTree d4": "neural_trees",
    "SoftTree per-leaf": "neural_trees",
    "GAL": "neural_trees",
    "RandomForest": "sklearn",
    "MLP": "sklearn",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "GRANDE": "GRANDE",
    "NODE": "pytorch_tabular",
}
ZAMAN_ASIMI_SN = 3 * 3600


TUM_MODELLER = (SoftScratch, SoftPerLeaf, Gal, Rf, Mlp, Xgb, Lgbm, Grande, Node)


def uygun(cls):
    """Yalnız kurulu mu diye bak; ayrı süreçte çalışacaklar için ayrı süreçte bak."""
    r = subprocess.run(
        [sys.executable, "-c", f"import {IMPORT[cls.ad]}"],
        capture_output=True,
        timeout=300,
    )
    ok = r.returncode == 0
    if not ok:
        print(f"  {cls.ad}: kurulu değil / açılamadı, atlanıyor", flush=True)
    return ok


def cv_ayri_surec(cls, X, y, tohumlar):
    """cv() ile aynı işi yeni bir yorumlayıcıda yap, JSON ile geri al."""
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "veri.npz")
        np.savez(npz, X=X, y=y)
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--tek",
            cls.ad,
            "--npz",
            npz,
            "--tohumlar",
            ",".join(map(str, tohumlar)),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=ZAMAN_ASIMI_SN
            )
        except subprocess.TimeoutExpired:
            return {"hata": f"zaman aşımı ({ZAMAN_ASIMI_SN // 3600} saat)"}
        satir = [ln for ln in r.stdout.splitlines() if ln.startswith("SONUC ")]
        if r.returncode != 0 or not satir:
            return {"hata": (r.stderr.strip().splitlines() or ["çıktı yok"])[-1][:160]}
        return json.loads(satir[-1][6:])


def kumeler(hizli):
    for ad, yuk in UCI.items():
        X, y = yuk(return_X_y=True)
        yield ad, X.astype(float), y
        if hizli:
            return
    for did in OPENML:
        d = fetch_openml(data_id=did, as_frame=False, parser="auto")
        X, y = d.data.astype(float), LabelEncoder().fit_transform(d.target)
        if len(y) > N_MAKS:
            X, _, y, _ = train_test_split(
                X, y, train_size=N_MAKS, stratify=y, random_state=0
            )
        yield d.details["name"], X, y


def cv(cls, X, y, tohumlar):
    acc, sure, yapi = [], [], None
    for s in tohumlar:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            sc = StandardScaler().fit(X[tr])
            t0 = time.time()
            m = cls(s).fit(sc.transform(X[tr]), y[tr])
            sure.append(time.time() - t0)
            acc.append(
                float((np.asarray(m.predict(sc.transform(X[te]))) == y[te]).mean())
            )
            yapi = m.yapi()
    return {
        "acc": round(float(np.mean(acc)), 4),
        "sd": round(float(np.std(acc, ddof=1)), 4),
        "fit_sn": round(float(np.mean(sure)), 2),
        "yapi": yapi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--tek", help="(iç kullanım) tek modeli ayrı süreçte koş")
    ap.add_argument("--npz")
    ap.add_argument("--tohumlar")
    a = ap.parse_args()
    if a.tek:
        cls = {c.ad: c for c in TUM_MODELLER}[a.tek]
        d = np.load(a.npz)
        tohum = tuple(int(t) for t in a.tohumlar.split(","))
        print("SONUC " + json.dumps(cv(cls, d["X"], d["y"], tohum)), flush=True)
        return 0
    tohumlar = (0,) if a.hizli else TOHUM
    cikti = KOK / ("rakipler-hizli.json" if a.hizli else "rakipler-sonuc.json")
    S = json.loads(cikti.read_text()) if cikti.exists() else {}
    modeller = [c for c in TUM_MODELLER if uygun(c)]
    S["_modeller"] = [c.ad for c in modeller]
    for ad, X, y in kumeler(a.hizli):
        if ad in S:
            continue
        R = {"n": int(len(y)), "p": int(X.shape[1]), "K": int(len(set(y)))}
        print(f"\n### {ad}  n={R['n']} p={R['p']} K={R['K']}", flush=True)
        for c in modeller:
            try:
                R[c.ad] = cv_ayri_surec(c, X, y, tohumlar)
                r = R[c.ad]
                if "hata" in r:
                    print(f"  {c.ad:18s} HATA {r['hata'][:80]}", flush=True)
                else:
                    print(
                        f"  {c.ad:18s} {r['acc']:.3f} ± {r['sd']:.3f}  {r['fit_sn']:6.1f}s  {r['yapi']}",
                        flush=True,
                    )
            except Exception as e:
                R[c.ad] = {"hata": f"{type(e).__name__}: {str(e)[:120]}"}
                print(
                    f"  {c.ad:18s} HATA {type(e).__name__}: {str(e)[:80]}", flush=True
                )
        S[ad] = R
        cikti.write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("\nBİTTİ:", len([k for k in S if not k.startswith("_")]), "veri kümesi")


if __name__ == "__main__":
    sys.exit(main())

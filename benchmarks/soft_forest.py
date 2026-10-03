#!/usr/bin/env python3
"""
A bagged ensemble of per-leaf soft trees on the datasets with more than
1 000 rows, under the protocol of rakipler.py, against the XGBoost and
single-tree numbers already in rakipler-sonuc.json.

The question is narrow: how much of the gap to boosting on larger tables
does averaging soft trees close, and what it costs in fit time. Nothing is
tuned here either; the trees use the per-leaf configuration of the main
benchmark with half the epochs, because each tree sees a bootstrap sample.

    python3 benchmarks/soft_forest.py            # 16 datasets, 3 seeds x 5 folds
    python3 benchmarks/soft_forest.py --hizli    # one dataset, one seed, 3 trees
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
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rakipler import TOHUM, cv, kumeler  # noqa: E402

KOK = pathlib.Path(__file__).resolve().parent
CIKTI = KOK / "soft_forest-sonuc.json"
N_TREES, EPOCHS, N_JOBS = 25, 90, 10


def _fit_one(X, y, seed, i, n_classes):
    import torch
    torch.set_num_threads(1)
    from neural_trees import SoftDecisionTree
    rng = np.random.default_rng(seed * 1000 + i)
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < n_classes:          # keep every class in the bag
        idx = np.concatenate([idx, [np.flatnonzero(y == c)[0] for c in range(n_classes) if c not in y[idx]]])
    m = SoftDecisionTree(depth=6, max_epochs=EPOCHS, growth="per_leaf", growth_init="residual", random_state=seed * 1000 + i)
    m.fit(X[idx], y[idx])
    return m.to_numpy()                             # picklable, torch-free


class SoftForest:
    ad = "SoftForest"

    def __init__(self, s, n_trees=N_TREES):
        self.s, self.n_trees = s, n_trees

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees_ = Parallel(n_jobs=N_JOBS)(delayed(_fit_one)(X, y, self.s, i, len(self.classes_)) for i in range(self.n_trees))
        return self

    def predict(self, X):
        p = np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        return self.classes_[np.argmax(p, axis=1)]

    def yapi(self):
        return f"{self.n_trees} per-leaf trees, {np.mean([t.n_internal_ for t in self.trees_]):.0f} internal nodes each"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true")
    a = ap.parse_args()
    ref = json.loads((KOK / "rakipler-sonuc.json").read_text(encoding="utf-8"))
    S = json.loads(CIKTI.read_text(encoding="utf-8")) if CIKTI.exists() and not a.hizli else {}
    S["_protokol"] = f"{N_TREES} bagged per-leaf soft trees (depth<=6, {EPOCHS} epochs, residual init), 3 seeds x 5-fold, StandardScaler on the training fold; reference columns from rakipler-sonuc.json"
    tohumlar = (0,) if a.hizli else TOHUM
    for ad, X, y in kumeler(hizli=False):
        if len(y) <= 1000 or (ad in S and not a.hizli):
            continue
        if a.hizli and ad != "Digits":
            continue
        t0 = time.time()
        cls = (lambda s: SoftForest(s, 3)) if a.hizli else SoftForest
        r = cv(cls, X, y, tohumlar)
        r["XGBoost"] = ref[ad]["XGBoost"]["acc"]
        r["SoftTree per-leaf"] = ref[ad]["SoftTree per-leaf"]["acc"]
        r["n"], r["K"] = int(len(y)), int(len(np.unique(y)))
        S[ad] = r
        print(f"{ad:28s} forest {r['acc']:.3f}±{r['sd']:.3f}  tree {r['SoftTree per-leaf']:.3f}  xgb {r['XGBoost']:.3f}  {r['fit_sn']:.0f} s/fit  total {time.time()-t0:.0f} s", flush=True)
        if not a.hizli:
            CIKTI.write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()

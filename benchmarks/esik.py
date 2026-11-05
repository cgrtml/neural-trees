"""
Threshold inputs (`DistilledTree(thresholds=512)`, `DistilledForest(thresholds=512)`) against
the tuned rivals of benchmarks/kanit.py: same nine tables, same test rows, three seeds,
accuracy and ROC AUC. The teacher is kanit.py's tuned XGBoost (same parameters and number of
trees); rival results are read from (or written to) kanit.py's cache, so KANIT_N and KANIT_ARA
choose the training size as there.

    KANIT_N=2000 KANIT_ARA=_kanit_cache_n2000 python benchmarks/esik.py tek|grup|ozet
    env JOBS (default 3), ADS (comma-separated subset), COMBINE (log|mean, grup only)

One cell at 20 000 rows takes 10 to 40 minutes on one CPU thread (the group three times
that) and up to 10 GB of memory; at 2 000 rows about two minutes.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kanit as KN  # noqa: E402
import numpy as np  # noqa: E402

K = 512
COMBINE = os.environ.get("COMBINE", "log")


def _out(ad, seed, mode):
    tag = "" if mode == "tek" or COMBINE == "log" else "_mean"
    return KN.ARA / f"{ad}__{seed}__esik_{mode}{tag}.json"


def bir(ad, seed, mode):
    o = _out(ad, seed, mode)
    if o.exists():
        return json.loads(o.read_text())
    import torch
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    from neural_trees import DistilledForest, DistilledTree
    torch.set_num_threads(1)
    KN.ARA.mkdir(exist_ok=True)
    rp = KN.ARA / f"{ad}__{seed}__rakip.json"
    riv = json.loads(rp.read_text()) if rp.exists() else KN.rakip_bir(ad, seed)
    Xtr, ytr, Xte, yte = KN.bolum(ad, seed)
    T = xgb.XGBClassifier(n_estimators=riv["XGBoost"]["trees"], random_state=seed, n_jobs=1, verbosity=0,
                          **riv["XGBoost"]["params"])
    t0 = time.time()
    if mode == "tek":
        m = DistilledTree(teacher=T, thresholds=K, synthetic=10, random_state=seed).fit(Xtr, ytr)
        splits = int(m.student_.model_.is_split.sum())
    else:
        m = DistilledForest(teacher=T, thresholds=K, synthetic=10, n_trees=3, combine=COMBINE,
                            random_state=seed).fit(Xtr, ytr)
        splits = int(sum(int(t.model_.is_split.sum()) for t in m.estimators_))
    P = m.predict_proba(Xte)
    pred = m.classes_[P.argmax(1)]
    r = {"ad": ad, "seed": seed, "mode": mode, "combine": COMBINE if mode == "grup" else None, "n": int(len(ytr)),
         "acc": float((pred == yte).mean()), "auc": float(roc_auc_score(yte, P[:, 1])), "splits": splits,
         "K": len(m.basis_.pairs), "fidelity": float(m.fidelity(Xte)), "sec": round(time.time() - t0)}
    o.write_text(json.dumps(r))
    print(json.dumps(r), flush=True)
    return r


def ozet():
    from scipy.stats import wilcoxon
    out = {"n": KN.N, "rows": []}
    for mode in ("tek", "grup"):
        rows = []
        for ad in KN.KUMELER:
            for s in KN.TOHUMLAR:
                p, rp = _out(ad, s, mode), KN.ARA / f"{ad}__{s}__rakip.json"
                if p.exists() and rp.exists():
                    rows.append((json.loads(p.read_text()), json.loads(rp.read_text())))
        if not rows:
            continue
        for riv in ("XGBoost", "LightGBM", "CatBoost", "HistGB", "CART"):
            d = np.array([(a["acc"] - b[riv]["acc"]) * 100 for a, b in rows])
            line = {"model": mode, "vs": riv, "cells": len(d), "mean": float(d.mean()), "higher": int((d > 0).sum()),
                    "lower": int((d < 0).sum()), "p": float(wilcoxon(d).pvalue) if d.any() else 1.0,
                    "splits": float(np.mean([a["splits"] for a, _ in rows]))}
            out["rows"].append(line)
            print(f"n={KN.N} {mode:4s} vs {riv:8s} {line['mean']:+.2f} ({line['higher']}/{line['lower']}, "
                  f"p={line['p']:.3f}, {line['cells']} cells, {line['splits']:.0f} splits)")
    Path(__file__).with_name(f"esik_n{KN.N}-sonuc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ozet":
        ozet()
    else:
        from joblib import Parallel, delayed
        ads = os.environ.get("ADS", ",".join(KN.KUMELER)).split(",")
        Parallel(n_jobs=int(os.environ.get("JOBS", "3")))(delayed(bir)(a, s, mode) for a in ads for s in KN.TOHUMLAR)

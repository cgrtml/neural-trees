#!/usr/bin/env python3
"""
Development harness for the large-data accuracy work: one training set of
`--n` rows per dataset and a fixed test set of up to 20 000 rows, both drawn
as in olcek.py, and a list of named configurations, each fitted in its own
process. Prints test accuracy (and train accuracy) per configuration and
appends every result to gelistir-sonuc.jsonl with the commit it ran on.

Not a benchmark: one seed, one split, meant for deciding what to measure
properly in cart_init.py / olcek.py.

    python3 benchmarks/gelistir.py --kume electricity MiniBooNE --n 20000 --config cart10 hard10
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KOK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
CIKTI = KOK / "gelistir-sonuc.jsonl"


def veri(ad, n):
    import olcek
    X, y = olcek.yukle(ad)
    n_te = min(20_000, len(y) // 5)
    Xp, Xte, yp, yte = train_test_split(X, y, test_size=n_te, random_state=0, stratify=y)
    idx = np.random.default_rng(0).permutation(len(yp))[:n]
    sc = StandardScaler().fit(Xp[idx])
    return sc.transform(Xp[idx]), yp[idx], sc.transform(Xte), yte


def _sharpen(m, Xv, yv):
    """Scale every gate's logit by the factor in GRID that is best on (Xv, yv)."""
    import torch
    base = m.model_.log_beta.detach().clone()
    best, best_acc = 0.0, -1.0
    for c in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        with torch.no_grad():
            m.model_.log_beta.copy_(base + float(np.log(c)))
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc + 1e-9:
            best, best_acc = c, acc
    with torch.no_grad():
        m.model_.log_beta.copy_(base + float(np.log(best)))
    return best


def calistir(config, Xtr, ytr, Xte, yte, seed=0):
    """Fit one named configuration; returns (model-like with predict, extra dict)."""
    if config.startswith("xgb"):
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed,
                              n_jobs=1, verbosity=0).fit(Xtr, ytr)
        return m, {}
    import torch
    torch.set_num_threads(1)
    from sklearn.tree import DecisionTreeClassifier

    from neural_trees import SoftDecisionTree
    d = int("".join(ch for ch in config if ch.isdigit()) or 10)
    if config.startswith("cart"):
        return DecisionTreeClassifier(max_depth=d, random_state=seed).fit(Xtr, ytr), {}
    base = dict(depth=d, init="cart", init_sharpness=10.0, learn_temperature=True, batch_size=256,
                max_epochs=60, random_state=seed)
    if config.startswith("soft"):                       # baseline fine-tuned copy
        return SoftDecisionTree(**base).fit(Xtr, ytr), {}
    if config.startswith("hard"):                       # same, read as hard rules
        m = SoftDecisionTree(**base).fit(Xtr, ytr)
        return m.to_hard_tree(), {}
    if config.startswith("sharp"):                      # same, gates sharpened on a held-out part
        a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
        m = SoftDecisionTree(**base).fit(a, ya)
        c = _sharpen(m, b, yb)
        return m, {"sharpen": c}
    if config.startswith("long"):                       # 200 epochs
        return SoftDecisionTree(**{**base, "max_epochs": 200}).fit(Xtr, ytr), {}
    if config.startswith("mix"):                         # half true label, half teacher probabilities
        P = np.load(os.environ["GELISTIR_TEACHER"])
        classes, yi = np.unique(ytr, return_inverse=True)
        i, c = np.nonzero(P >= 0.01)
        Xm = np.vstack([Xtr, Xtr[i]])
        ym = np.concatenate([classes[yi], classes[c]])
        wm = np.concatenate([np.full(len(yi), 0.5), 0.5 * P[i, c]])
        return SoftDecisionTree(**base).fit(Xm, ym, sample_weight=wm), {}
    if config.startswith("pdistil"):                     # per-leaf growth on the teacher's probabilities
        P = np.load(os.environ["GELISTIR_TEACHER"])
        i, c = np.nonzero(P >= 0.01)
        m = SoftDecisionTree(depth=d, growth="per_leaf", growth_init="residual", max_epochs=180,
                             batch_size=256, random_state=seed)
        return m.fit(Xtr[i], np.unique(ytr)[c], sample_weight=P[i, c]), {}
    if config.startswith("distil") or config.startswith("dlin"):   # learn XGBoost's probabilities
        P = np.load(os.environ["GELISTIR_TEACHER"])
        i, c = np.nonzero(P >= 0.01)
        if config.startswith("dlin"):
            return _linear_leaves(Xtr[i], c, d, seed, epochs=60, weights=P[i, c], classes=np.unique(ytr)), {}
        m = SoftDecisionTree(**base).fit(Xtr[i], np.unique(ytr)[c], sample_weight=P[i, c])
        return m, {}
    if config.startswith("lin"):                        # linear leaves (prototype)
        return _linear_leaves(Xtr, ytr, d, seed, epochs=60), {}
    if config.startswith("llong"):
        return _linear_leaves(Xtr, ytr, d, seed, epochs=150), {}
    raise ValueError(config)


class _LinearLeafModel:
    """Prototype: a CART-started soft tree whose nodes hold a logistic model, not a constant."""

    def __init__(self, module, W, classes):
        self.module, self.W, self.classes_ = module, W, classes

    def log_proba(self, x):
        import torch
        import torch.nn.functional as F
        _, _, terminal = self.module._walk(x)
        parts = []
        for log_mu, idx in terminal:
            logits = self.module.node_logits[idx].unsqueeze(0) + torch.einsum("bp,mpk->bmk", x, self.W[idx])
            parts.append(log_mu.unsqueeze(2) + F.log_softmax(logits, dim=2))
        return torch.logsumexp(torch.cat(parts, dim=1), dim=1)

    def predict(self, X):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(self.log_proba(torch.as_tensor(X[i:i + 4096], dtype=torch.float32)).argmax(1).numpy())
        return self.classes_[np.concatenate(out)]


def _linear_leaves(Xtr, ytr, depth, seed, epochs=60, l2=1e-4, lr=0.01, batch=256, weights=None, classes=None):
    import torch
    import torch.nn.functional as F
    from sklearn.tree import DecisionTreeClassifier

    from neural_trees import SoftDecisionTree
    torch.manual_seed(seed)
    if classes is None:
        classes, y = np.unique(ytr, return_inverse=True)
    else:
        y = np.asarray(ytr)                     # already class indices
    w_all = torch.ones(len(y)) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    cart = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(
        Xtr, y, sample_weight=None if weights is None else weights)
    est = SoftDecisionTree.from_hard_tree(cart, Xtr, sharpness=10.0, learn_temperature=True)
    mod = est.model_
    n_nodes, K = mod.node_logits.shape
    W = torch.nn.Parameter(torch.zeros(n_nodes, Xtr.shape[1], K))
    model = _LinearLeafModel(mod, W, classes)
    opt = torch.optim.Adam(list(mod.parameters()) + [W], lr=lr)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]
            nll = F.nll_loss(model.log_proba(Xt[b]), yt[b], reduction="none")
            loss = (nll * w_all[b]).sum() / w_all[b].sum() + l2 * (W ** 2).sum() / len(yt) * len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _bir(config, ad, n, Xtr, ytr, Xte, yte):
    os.environ["GELISTIR_TEACHER"] = str(KOK / "_gelistir_teacher" / f"{ad}_{n}.npy")
    t0 = time.time()
    m, ek = calistir(config, Xtr, ytr, Xte, yte)
    return {"kume": ad, "n": n, "config": config,
            "test": round(float((np.asarray(m.predict(Xte)) == yte).mean()), 4),
            "train": round(float((np.asarray(m.predict(Xtr)) == ytr).mean()), 4),
            "sn": round(time.time() - t0, 1), **ek}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--ogretmen", action="store_true", help="only write XGBoost probabilities on the training rows")
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    ogr = KOK / "_gelistir_teacher"
    if a.ogretmen:
        ogr.mkdir(exist_ok=True)
        for ad in a.kume:
            Xtr, ytr, Xte, yte = veri(ad, a.n)
            m, _ = calistir("xgb", Xtr, ytr, Xte, yte)
            np.save(ogr / f"{ad}_{a.n}.npy", m.predict_proba(Xtr))
            print(ad, "teacher test", round(float((m.predict(Xte) == yte).mean()), 4), flush=True)
        return
    isler = []
    for ad in a.kume:
        Xtr, ytr, Xte, yte = veri(ad, a.n)
        isler += [(c, ad, a.n, Xtr, ytr, Xte, yte) for c in a.config]
    out = Parallel(n_jobs=a.n_jobs)(delayed(_bir)(*i) for i in isler)
    with CIKTI.open("a", encoding="utf-8") as f:
        for r in out:
            r["commit"] = commit
            f.write(json.dumps(r) + "\n")
    for r in sorted(out, key=lambda r: (r["kume"], r["config"])):
        ek = {k: v for k, v in r.items() if k not in ("kume", "n", "config", "test", "train", "sn", "commit")}
        print(f"{r['kume']:14s} n={r['n']:6d} {r['config']:10s} test {r['test']:.4f}  train {r['train']:.4f}  "
              f"{r['sn']:6.0f} s  {ek if ek else ''}", flush=True)


if __name__ == "__main__":
    main()

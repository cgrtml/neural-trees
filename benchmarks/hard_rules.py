"""
Which export rule of to_hard_tree() agrees most with its soft model? (#101)

Three seeds of stratified five-fold cross-validation. For each fold the soft
tree is fitted on the training part and every rule's hard export is scored
on the held-out part: agreement with the soft model's own predictions, and
wall-clock per 1000 predictions. Writes benchmarks/hard_rules-sonuc.json.
"""
import json
import pathlib
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_openml, load_digits, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_trees import SoftDecisionTree

warnings.filterwarnings("ignore")
RULES = ("gate", "leaf", "contribution")


def datasets():
    yield "Wine", *load_wine(return_X_y=True)
    yield "Digits", *load_digits(return_X_y=True)
    d = fetch_openml(data_id=182, as_frame=False, parser="auto")  # satimage
    yield "satimage", d.data.astype(float), LabelEncoder().fit_transform(d.target)


def timed(fn, X, repeats=5):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t0)
    return 1e6 * best / len(X)  # seconds per sample -> ms per 1000 samples


out = {}
for name, X, y in datasets():
    agree = {r: [] for r in RULES}
    ms = {r: [] for r in RULES}
    ms_soft = []
    for seed in (0, 1, 2):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            m = SoftDecisionTree(depth=4, max_epochs=40, random_state=seed).fit(Xtr, y[tr])
            soft = m.predict(Xte)
            ms_soft.append(timed(m.predict, Xte))
            for r in RULES:
                h = m.to_hard_tree(rule=r)
                agree[r].append(float((h.predict(Xte) == soft).mean()))
                ms[r].append(timed(h.predict, Xte))
    out[name] = {
        "n": int(len(y)), "p": int(X.shape[1]), "K": int(len(np.unique(y))),
        "soft_ms_per_1000": float(np.median(ms_soft)),
        "rules": {r: {"agree_mean": float(np.mean(agree[r])), "agree_worst": float(np.min(agree[r])),
                      "agree_sd": float(np.std(agree[r])), "ms_per_1000": float(np.median(ms[r]))}
                  for r in RULES},
    }
    print(f"### {name}  soft predict {np.median(ms_soft):.2f} ms/1000", flush=True)
    for r in RULES:
        d = out[name]["rules"][r]
        print(f"  {r:13s} agree {d['agree_mean']:.4f} (worst {d['agree_worst']:.3f})  {d['ms_per_1000']:.2f} ms/1000", flush=True)
pathlib.Path(__file__).with_name("hard_rules-sonuc.json").write_text(json.dumps(out, indent=1))
print("BİTTİ")

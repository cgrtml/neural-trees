"""
How fast a fitted soft tree predicts, by export.

Four ways to run the same model: the torch estimator, the numpy copy
(`to_numpy`), ONNX Runtime on the ONNX export (`to_onnx`), and the hard
rule tree (`to_hard_tree`). Two shapes: one row at a time, which is what a
service answering single requests pays, and a batch of 10 000 rows. One
thread each (OMP_NUM_THREADS=1, torch and ONNX Runtime set to one), so the
numbers are per core. Writes benchmarks/latency-sonuc.json and prints the
table the documentation carries.
"""
import json
import os
import pathlib
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort
import torch
from sklearn.datasets import load_breast_cancer, load_digits

from neural_trees import SoftDecisionTree

torch.set_num_threads(1)
KOK = pathlib.Path(__file__).resolve().parent


def best_of(fn, repeats=7, calls=1):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, (time.perf_counter() - t0) / calls)
    return best


out = {}
for name, loader, depth in (("Breast Cancer", load_breast_cancer, 4), ("Digits", load_digits, 4), ("Digits", load_digits, 6)):
    X, y = loader(return_X_y=True)
    sd = X.std(0)
    sd[sd == 0] = 1
    X = ((X - X.mean(0)) / sd).astype(np.float64)
    tree = SoftDecisionTree(depth=depth, max_epochs=20, random_state=0).fit(X, y)
    npt = tree.to_numpy()
    hard = tree.to_hard_tree()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(tree.to_onnx().SerializeToString(), so, providers=["CPUExecutionProvider"])
    one = X[:1]
    one32 = one.astype(np.float32)
    rng = np.random.RandomState(0)
    big = X[rng.randint(0, len(X), 10000)]
    big32 = big.astype(np.float32)
    runners = {
        "torch estimator": (lambda: tree.predict_proba(one), lambda: tree.predict_proba(big)),
        "numpy export": (lambda: npt.predict_proba(one), lambda: npt.predict_proba(big)),
        "ONNX Runtime": (lambda: sess.run(None, {"X": one32}), lambda: sess.run(None, {"X": big32})),
        "hard rule tree": (lambda: hard.predict_proba(one), lambda: hard.predict_proba(big)),
    }
    key = f"{name} depth {depth} p={X.shape[1]}"
    out[key] = {}
    print(f"\n### {key}")
    for r, (f1, fb) in runners.items():
        f1()
        fb()
        us = best_of(f1, calls=20) * 1e6
        rows_per_s = 10000 / best_of(fb, calls=3)
        out[key][r] = {"one_row_us": round(us, 1), "rows_per_s_batch": int(rows_per_s)}
        print(f"  {r:16s} one row {us:8.1f} us   batch {rows_per_s:12,.0f} rows/s")
(KOK / "latency-sonuc.json").write_text(json.dumps(out, indent=1))
print("\nBİTTİ")

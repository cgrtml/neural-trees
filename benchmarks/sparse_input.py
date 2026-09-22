"""
Is sparse input actually cheaper for NaiveBayesClassifier and WeightedKNN? (#100)

20 newsgroups (bydate split), token counts over the training vocabulary.
Pass the directory the archive was extracted into. Peak memory is measured with
tracemalloc, which sees numpy's and scipy's allocations. The dense KNN path
allocates an (n_query, n_store, n_features) block, so its comparison uses
feature and row counts that fit in memory; the sizes are printed alongside.
Writes benchmarks/sparse_input-sonuc.json.
"""
import json
import os
import pathlib
import sys
import time
import tracemalloc
import warnings

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

from neural_trees import NaiveBayesClassifier, WeightedKNN

warnings.filterwarnings("ignore")


def load_20news(root):
    """
    Token counts from the 20news-bydate archive extracted under `root`.

    Counts rather than scikit-learn's normalised tf-idf vectors, because the
    multinomial model is the one for counts; the vocabulary is fitted on the
    training split only.
    """
    tr = load_files(os.path.join(root, "20news-bydate-train"), encoding="latin1")
    te = load_files(os.path.join(root, "20news-bydate-test"), encoding="latin1")
    vec = CountVectorizer(dtype=np.float64)
    return vec.fit_transform(tr.data).tocsr(), tr.target, vec.transform(te.data).tocsr(), te.target


def measure(make, X_fit, y_fit, X_pred):
    tracemalloc.start()
    t0 = time.perf_counter()
    est = make().fit(X_fit, y_fit)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = est.predict(X_pred)
    pred_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"fit_s": round(fit_s, 3), "predict_s": round(pred_s, 3), "peak_mb": round(peak / 1e6, 1)}, pred


root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/scikit_learn_data/20news_home")
Xtr, ytr, Xte, yte = load_20news(root)
print("train", Xtr.shape, "nnz", Xtr.nnz, "test", Xte.shape, flush=True)
out = {"shapes": {"train": list(Xtr.shape), "test": list(Xte.shape), "train_nnz": int(Xtr.nnz)}}

# Naive Bayes, multinomial, the natural model for counts. Dense at full width
# is 11 314 documents x the full vocabulary in float64, more than this machine
# holds, so the dense arm uses a 2 000-document subset and the sparse arm is
# run at both sizes.
sub = np.arange(2000)
for name, Xf, yf, Xp, yp in (
    ("nb_sparse_full", Xtr, ytr, Xte, yte),
    ("nb_sparse_2000", Xtr[sub], ytr[sub], Xte, yte),
    ("nb_dense_2000", Xtr[sub].toarray(), ytr[sub], Xte.toarray(), yte),
):
    r, pred = measure(lambda: NaiveBayesClassifier(likelihood="multinomial"), Xf, yf, Xp)
    r["accuracy"] = round(float((pred == yp).mean()), 4)
    r["shape"] = [int(Xf.shape[0]), int(Xf.shape[1])]
    out[name] = r
    print(name, r, flush=True)
assert np.array_equal(
    NaiveBayesClassifier(likelihood="multinomial").fit(Xtr[sub], ytr[sub]).predict(Xte[:500]),
    NaiveBayesClassifier(likelihood="multinomial").fit(Xtr[sub].toarray(), ytr[sub]).predict(Xte[:500].toarray()),
), "sparse and dense predictions differ"

# KNN, euclidean. The dense path broadcasts (n_query, n_store, p), so the
# comparison keeps the 1 000 most frequent features and 1 000 store rows with
# 200 queries: dense broadcast 200 x 1000 x 1000 x 8 B = 1.6 GB.
df = np.asarray((Xtr > 0).sum(axis=0)).ravel()
top = np.argsort(-df)[:1000]
Xs, ys = Xtr[:1000][:, top].tocsr(), ytr[:1000]
Xq, yq = Xte[:200][:, top].tocsr(), yte[:200]
for name, Xf, Xp in (("knn_sparse", Xs, Xq), ("knn_dense", Xs.toarray(), Xq.toarray())):
    r, pred = measure(lambda: WeightedKNN(k=5), Xf, ys, Xp)
    r["accuracy"] = round(float((pred == yq).mean()), 4)
    r["shape"] = [int(Xf.shape[0]), int(Xf.shape[1])]
    out[name] = r
    print(name, r, flush=True)
# The sparse KNN at full width, which the dense path cannot do at all.
r, pred = measure(lambda: WeightedKNN(k=5), Xtr[:1000], ytr[:1000], Xte[:200])
r["accuracy"] = round(float((pred == yte[:200]).mean()), 4)
r["shape"] = [1000, int(Xtr.shape[1])]
out["knn_sparse_full_width"] = r
print("knn_sparse_full_width", r, flush=True)
pathlib.Path(__file__).with_name("sparse_input-sonuc.json").write_text(json.dumps(out, indent=1))
print("BİTTİ")

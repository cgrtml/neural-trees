#!/usr/bin/env python3
"""
Checks the library's DistilledTree (default HistGradientBoosting teacher)
with and without its synthetic rows, on the 20 000-row cells of
damitma_olcek.py that showed the largest and the smallest gains (jannis,
Higgs, covertype, electricity), three seeds each. The teacher here is
scikit-learn's, so xgboost and torch never share the process.
`--synthetic 2 10` compares two other settings (written to
munge_lib_2_10-sonuc.json).

    python3 benchmarks/munge_lib.py [--n-jobs 8] [--synthetic 0 2]
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

KOK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
CIKTI = KOK / "munge_lib-sonuc.json"
KUMELER = ("jannis", "Higgs", "covertype", "electricity")


def bir(ad, s, synthetic):
    import torch
    torch.set_num_threads(1)
    import damitma_olcek as D
    from sklearn.ensemble import HistGradientBoostingClassifier

    from neural_trees import DistilledTree
    Xtr, ytr, Xte, yte = D.bolum(ad, 20_000, s)
    t0 = time.time()
    m = DistilledTree(teacher=HistGradientBoostingClassifier(random_state=s), synthetic=synthetic,
                      random_state=s).fit(Xtr, ytr)
    r = {"ad": ad, "seed": s, "synthetic": synthetic, "acc": float((m.predict(Xte) == yte).mean()),
         "teacher": float((m.teacher_.predict(Xte) == yte).mean()), "sn": time.time() - t0}
    print(json.dumps(r), flush=True)
    return r


def main():
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--synthetic", type=int, nargs=2, default=(0, 2))
    a = ap.parse_args()
    k0, k2 = a.synthetic
    cikti = CIKTI if (k0, k2) == (0, 2) else KOK / f"munge_lib_{k0}_{k2}-sonuc.json"
    R = Parallel(n_jobs=a.n_jobs)(delayed(bir)(ad, s, k) for ad in KUMELER for s in (0, 1, 2) for k in (k0, k2))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=KOK).stdout.strip()
    d = []
    for ad in KUMELER:
        a0 = np.mean([r["acc"] for r in R if r["ad"] == ad and r["synthetic"] == k0])
        a2 = np.mean([r["acc"] for r in R if r["ad"] == ad and r["synthetic"] == k2])
        t = np.mean([r["teacher"] for r in R if r["ad"] == ad])
        print(f"{ad:12s} teacher {t:.4f}  synthetic={k0} {a0:.4f}  synthetic={k2} {a2:.4f}  {100*(a2-a0):+.2f} pt")
    for r0 in R:
        if r0["synthetic"] == k0:
            r2 = next(r for r in R if r["ad"] == r0["ad"] and r["seed"] == r0["seed"] and r["synthetic"] == k2)
            d.append(r2["acc"] - r0["acc"])
    d = np.array(d)
    print(f"synthetic={k2} - synthetic={k0} over {len(d)} fits: mean {100*d.mean():+.2f} pt, "
          f"higher/lower {(d > 0.002).sum()}/{(d < -0.002).sum()}, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
    cikti.write_text(json.dumps({"_protokol": __doc__.strip(), "_commit": commit, "fits": R}, indent=1))


if __name__ == "__main__":
    main()

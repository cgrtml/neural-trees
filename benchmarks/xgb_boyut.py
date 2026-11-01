#!/usr/bin/env python3
"""
XGBoost at 3, 10, 30, 100 and 300 trees of depth 6, on the same cells, seeds
and test rows as munge.py and grup.py (six OpenML tables, 5 000 and 20 000
training rows, three seeds), to compare the distilled students with an
XGBoost model of about their size. Learning rate 0.3 up to 30 trees, 0.1
from 100 (the library default of 300 trees at 0.1 is the reference model
elsewhere). Size is the number of split nodes over all trees.

    python3 benchmarks/xgb_boyut.py           # fits, then the comparison
    python3 benchmarks/xgb_boyut.py ozet      # comparison only

The comparison reads the single distilled tree (ten MUNGE passes) from
_munge_cache_p10 and the group of three from _grup_cache_p10, both written
by `MUNGE_PASSES=10 munge.py` and `MUNGE_PASSES=10 grup.py`.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

KOK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KOK))
CIKTI = KOK / "xgb_boyut-sonuc.json"
ARA = KOK / "_xgb_boyut_cache.json"
AGACLAR = (3, 10, 30, 100, 300)


def olc():
    import xgboost as xgb

    import damitma_olcek as D
    out = json.loads(ARA.read_text()) if ARA.exists() else {}
    for ad in D.KUMELER:
        for n in (5_000, 20_000):
            for s in D.tohumlar(n):
                if all(f"{ad}__{n}__{s}__{t}" in out for t in AGACLAR):
                    continue
                Xtr, ytr, Xte, yte = D.bolum(ad, n, s)
                for t in AGACLAR:
                    m = xgb.XGBClassifier(n_estimators=t, max_depth=6, learning_rate=0.3 if t <= 30 else 0.1,
                                          random_state=s, n_jobs=1, verbosity=0).fit(Xtr, ytr)
                    df = m.get_booster().trees_to_dataframe()
                    out[f"{ad}__{n}__{s}__{t}"] = {"acc": float((m.predict(Xte) == yte).mean()),
                                                   "splits": int((df["Feature"] != "Leaf").sum())}
                print(ad, n, s, {t: round(out[f"{ad}__{n}__{s}__{t}"]["acc"], 4) for t in AGACLAR}, flush=True)
                ARA.write_text(json.dumps(out))
    return out


def ozet(out):
    from scipy.stats import wilcoxon
    ogrenciler = {"one tree": ("_munge_cache_p10", "__munge_d8.json", None),
                  "group of three": ("_grup_cache_p10", ".json", "group3 d8")}
    sonuc = {"xgboost_runs": out, "comparisons": {}}
    for isim, (klasor, ek, anahtar) in ogrenciler.items():
        hucre = {}
        for k in out:
            ad, n, s, t = k.split("__")
            yol = KOK / klasor / f"{ad}__{n}__{s}{ek}"
            if not yol.exists():
                continue
            r = json.loads(yol.read_text())
            r = r[anahtar] if anahtar else r
            h = hucre.setdefault(f"{ad}__{n}", {"acc": {}, "splits": {}, t: [], f"s{t}": []})
            h.setdefault(t, []).append(out[k]["acc"])
            h.setdefault(f"s{t}", []).append(out[k]["splits"])
            h["acc"][s], h["splits"][s] = r["acc"], r["splits"]
        if not hucre:
            continue
        boyut = float(np.mean([np.mean(list(h["splits"].values())) for h in hucre.values()]))
        print(f"{isim}: {len(hucre)} cells, {boyut:.0f} splits on average")
        sonuc["comparisons"][isim] = {"cells": len(hucre), "splits": boyut}
        for t in AGACLAR:
            d = [100 * (np.mean(list(h["acc"].values())) - np.mean(h[t])) for h in hucre.values()]
            xs = float(np.mean([np.mean(h[f"s{t}"]) for h in hucre.values()]))
            v = {"xgb_splits": xs, "mean_pt": float(np.mean(d)), "higher": int(sum(x > 0 for x in d)),
                 "lower": int(sum(x < 0 for x in d)), "p": float(wilcoxon(d).pvalue)}
            sonuc["comparisons"][isim][f"XGBoost {t} trees"] = v
            print(f"  vs XGBoost {t:3d} trees ({xs:6.0f} splits): {v['mean_pt']:+.2f} pt, "
                  f"higher/lower {v['higher']}/{v['lower']}, Wilcoxon p={v['p']:.3f}")
    sonuc["_protokol"] = __doc__
    sonuc["_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                      cwd=KOK).stdout.strip()
    CIKTI.write_text(json.dumps(sonuc, indent=1))


if __name__ == "__main__":
    ozet(json.loads(ARA.read_text()) if sys.argv[1:] == ["ozet"] else olc())

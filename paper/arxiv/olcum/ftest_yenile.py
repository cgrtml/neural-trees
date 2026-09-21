#!/usr/bin/env python3
"""olc.py'deki 5x2cv F testlerini kat-içi ölçeklemeyle yeniden koş, sonuc.json'u güncelle."""
import json
import pathlib

import olc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_trees import GALNetwork, combined_5x2cv_f_test

P = pathlib.Path(__file__).resolve().parent / "sonuc.json"
S = json.loads(P.read_text(encoding="utf-8"))
for ad in ("Iris", "Wine", "Digits"):
    X, y = olc.veri(ad)
    r = combined_5x2cv_f_test(make_pipeline(StandardScaler(), olc.Buyuyen(jitter=0.0)),
                              make_pipeline(StandardScaler(), olc.Buyuyen(jitter=0.2)), X, y, random_state=0)
    g = combined_5x2cv_f_test(
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="random", random_state=0)),
        make_pipeline(StandardScaler(), GALNetwork(max_epochs=150, growth_init="residual", random_state=0)),
        X, y, random_state=0)
    print(f"{ad:7s} jitter eski {S['jitter_test'][ad]} → yeni [{r.statistic:.3f}, {r.p_value:.6f}]"
          f" | GAL eski {S['gal_test'][ad]} → yeni [{g.statistic:.3f}, {g.p_value:.6f}]", flush=True)
    S["jitter_test"][ad] = [round(float(r.statistic), 3), round(float(r.p_value), 6)]
    S["gal_test"][ad] = [round(float(g.statistic), 3), round(float(g.p_value), 6)]
    P.write_text(json.dumps(S, indent=2, ensure_ascii=False))
print("BİTTİ")

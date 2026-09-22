#!/usr/bin/env python3
"""
Şablondaki @@...@@ yer tutucularını ölçüm çıktısıyla doldurur.

    python3 paper/arxiv/olcum/doldur.py

Amaç, makaledeki hiçbir sayının elle yazılmaması. Tablo ile kod arasındaki
sessiz kayma bu depoda bir kez yaşandı; bir daha yaşanmasın diye sayılar
ölçüm dosyasından geliyor. Eksik yer tutucu kalırsa betik hata veriyor.
"""

import json
import pathlib
import re
import sys

KOK = pathlib.Path(__file__).resolve().parents[3]
S = json.loads((KOK / "paper/arxiv/olcum/sonuc.json").read_text(encoding="utf-8"))


def ms(x):  # "0.942 ± 0.044"
    return f"{x[0]:.3f} $\\pm$ {x[1]:.3f}"


def msn(x):  # birim/bölme/düğüm sayısı: "6.6 ± 1.9"
    return f"{x[2]:.1f} $\\pm$ {x[3]:.1f}"


def puan(a, b):  # yüzde puan farkı
    return f"{abs(a - b) * 100:.1f}"


J, G, P, OM = S["jitter"], S["gal"], S["per_leaf"], S["omni"]
JT, GT = S["jitter_test"], S["gal_test"]

D = {}
for ad, kisa in (("Iris", "IRIS"), ("Wine", "WINE"), ("Digits", "DIGITS")):
    for et, k in (
        ("jitter=0.0", "0"),
        ("jitter=0.05", "005"),
        ("jitter=0.2", "02"),
        ("jitter=0.5", "05"),
        ("sıfırdan", "S"),
    ):
        D[f"T_J_{kisa}_{k}"] = ms(J[ad][et])
    D[f"JITTER_{kisa}_DROP"] = puan(J[ad]["sıfırdan"][0], J[ad]["jitter=0.0"][0])
D["T_J_IRIS_0_SD"] = f"{J['Iris']['jitter=0.0'][1]:.3f}"
D["T_J_IRIS_02_SD"] = f"{J['Iris']['jitter=0.2'][1]:.3f}"

# Bozulmanın dağılıma etkisi: sapma kaç katına çıkıyor
oran = [
    J[a]["jitter=0.0"][1] / J[a]["jitter=0.2"][1] for a in ("Iris", "Wine", "Digits")
]
D["JITTER_SD_RATIO"] = f"{min(oran):.1f} to {max(oran):.1f}"

# jitter 0.05 ile 0.5 arasındaki en büyük oynama
D["JITTER_SPREAD"] = (
    f"{max(100 * abs(J[a]['jitter=0.05'][0] - J[a]['jitter=0.5'][0]) for a in ('Iris', 'Wine', 'Digits')):.1f}"
)


def ftest(d):
    # p çok küçükse 4 hane hepsini sıfır gösteriyor
    pp = "p < 0.0001" if d[1] < 1e-4 else f"p = {d[1]:.4f}"
    return f"$F = {d[0]:.2f}$, ${pp}$"


for ad, kisa in (("Iris", "IRIS"), ("Wine", "WINE"), ("Digits", "DIGITS")):
    D[f"FTEST_JITTER_{kisa}"] = ftest(JT[ad])
    D[f"FTEST_GAL_{kisa}"] = ftest(GT[ad])

for ad, kisa in (("Iris", "IRIS"), ("Wine", "WINE"), ("Digits", "DIGITS")):
    for init, k in (("random", "RND"), ("residual", "RES")):
        D[f"T_G_{kisa}_{k}"] = ms(G[ad][init])
        D[f"T_G_{kisa}_{k}_H"] = msn(G[ad][init])
D["GAL_IRIS_RES_ACC"] = f"{G['Iris']['residual'][0]:.3f}"
D["GAL_IRIS_RND_ACC"] = f"{G['Iris']['random'][0]:.3f}"
D["GAL_IRIS_RES_H"] = f"{G['Iris']['residual'][2]:.1f}"
D["GAL_IRIS_RND_H"] = f"{G['Iris']['random'][2]:.1f}"

for ad, kisa in (("Sentetik-800x20", "S1"), ("Sentetik-2000x50", "S2")):
    for g, k in (("none", "NONE"), ("per_leaf", "PL")):
        D[f"T_P_{kisa}_{k}"] = ms(P[ad][g])
        D[f"T_P_{kisa}_{k}_N"] = msn(P[ad][g])
D["PL_S1_GROW_ACC"] = f"{P['Sentetik-800x20']['per_leaf'][0]:.3f}"
D["PL_S1_GROW_N"] = f"{P['Sentetik-800x20']['per_leaf'][2]:.1f}"
D["PL_S1_FIXED_ACC"] = f"{P['Sentetik-800x20']['none'][0]:.3f}"
D["PL_S2_DROP"] = puan(
    P["Sentetik-2000x50"]["none"][0], P["Sentetik-2000x50"]["per_leaf"][0]
)
D["GAL_DIGITS_DROP"] = puan(G["Digits"]["random"][0], G["Digits"]["residual"][0])

for ad, kisa in (("Cancer", "CANCER"), ("Digits", "DIGITS")):
    for sec, k in (("accuracy", "ACC"), ("test", "TEST")):
        D[f"T_O_{kisa}_{k}"] = ms(OM[ad][sec])
        D[f"T_O_{kisa}_{k}_N"] = msn(OM[ad][sec])

# ══════════════ v2 · OpenML, yönlü büyütme, per-leaf tabloları ══════════════
import numpy as _np  # noqa: E402

def _oku(ad):
    q = KOK / "paper/arxiv/olcum" / ad
    return json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}

def _pm(x):
    return f"{x[0]:.3f} $\\pm$ {x[1]:.3f}"

def _p(pv):
    return "$<0.0001$" if pv < 1e-4 else f"{pv:.3f}"

def _spearman(a, b):
    ra = _np.argsort(_np.argsort(a)); rb = _np.argsort(_np.argsort(b))
    return float(_np.corrcoef(ra, rb)[0, 1])

OM = _oku("openml-sonuc.json")
rows, bedel, Ks, Ps, sig = [], [], [], [], 0
for did, R in sorted(OM.items(), key=lambda kv: (kv[1]["K"], kv[1]["ad"])):
    J = R["jitter"]; b = (J["sıfırdan"][0] - J["jitter=0.0"][0]) * 100
    bedel.append(b); Ks.append(R["K"]); Ps.append(R["p"]); pv = R["jitter_test"][1]; sig += pv < 0.05
    ad = R["ad"].replace("-", "\\mbox{-}")
    rows.append(f"{ad} & {R['n']} & {R['p']} & {R['K']} & {_pm(J['jitter=0.0'])} & {_pm(J['jitter=0.2'])} & {_pm(J['sıfırdan'])} & {b:+.1f} & {_p(pv)} \\\\")
D["TABLE_OPENML"] = "\n".join(rows)
bedel = _np.array(bedel); Ks = _np.array(Ks)
D["OM_N"] = str(len(OM))
D["OM_BIN_N"] = str(int((Ks == 2).sum())); D["OM_MC_N"] = str(int((Ks > 2).sum()))
D["OM_BIN_COST"] = f"{bedel[Ks == 2].mean():.1f}"; D["OM_MC_COST"] = f"{bedel[Ks > 2].mean():.1f}"
D["OM_MC_MIN"] = f"{bedel[Ks > 2].min():.0f}"; D["OM_MC_MAX"] = f"{bedel[Ks > 2].max():.0f}"
D["OM_SPEAR_K"] = f"{_spearman(Ks, bedel):.2f}"; D["OM_SPEAR_P"] = f"{_spearman(_np.array(Ps), bedel):.2f}"
D["OM_SIG"] = str(sig)
G_w = [0, 0, 0]; G_ratio = []; P_w = [0, 0, 0]; P_gap = []
for R in OM.values():
    dg = R["gal"]["residual"][0] - R["gal"]["random"][0]
    G_w[0 if dg > 0.005 else 2 if dg < -0.005 else 1] += 1
    G_ratio.append(R["gal"]["residual"][2] / R["gal"]["random"][2])
    dp = R["per_leaf"]["per_leaf"][0] - R["per_leaf"]["none"][0]
    P_w[0 if dp > 0.005 else 2 if dp < -0.005 else 1] += 1; P_gap.append(dp * 100)
D["OM_GAL_WIN"], D["OM_GAL_TIE"], D["OM_GAL_LOSS"] = map(str, G_w)
D["OM_GAL_RATIO"] = f"{100 * (1 - _np.mean(G_ratio)):.0f}"
D["OM_PL_WIN"], D["OM_PL_TIE"], D["OM_PL_LOSS"] = map(str, P_w)
D["OM_PL_GAP"] = f"{_np.mean(P_gap):+.1f}"

BU = _oku("buyume-sonuc.json")
rows, gaps, sd_r, sd_g, sig = [], [], [], [], 0
for ad, R in BU.items():
    g = (R["residual_gate"][0] - R["random"][0]) * 100; gaps.append(g)
    sd_r.append(R["random"][1]); sd_g.append(R["residual_gate"][1]); pv = R["test_gate_vs_random"][1]; sig += pv < 0.05
    rows.append(f"{ad.replace('-', chr(92) + 'mbox{-}')} & {R['K']} & {_pm(R['random'])} & {_pm(R['residual'])} & {_pm(R['residual_gate'])} & {_pm(R['sıfırdan'])} & {_p(pv)} \\\\")
D["TABLE_BUYUME"] = "\n".join(rows)
gaps = _np.array(gaps)
D["BU_N"] = str(len(BU)); D["BU_MEAN"] = f"{gaps.mean():+.2f}"; D["BU_MEDIAN"] = f"{_np.median(gaps):+.2f}"
D["BU_MIN"] = f"{gaps.min():+.1f}"; D["BU_MAX"] = f"{gaps.max():+.1f}"; D["BU_SIG"] = str(sig)
D["BU_WIN"] = str(int((gaps > 0.5).sum())); D["BU_LOSS"] = str(int((gaps < -0.5).sum()))
D["BU_SD_R"] = f"{_np.mean(sd_r):.4f}"; D["BU_SD_G"] = f"{_np.mean(sd_g):.4f}"
mc = [k for k, R in BU.items() if R["K"] > 2]
D["BU_SD_R_MC"] = f"{_np.mean([BU[k]['random'][1] for k in mc]):.4f}"; D["BU_SD_G_MC"] = f"{_np.mean([BU[k]['residual_gate'][1] for k in mc]):.4f}"

PL = _oku("perleaf-sonuc.json")
rows, rows2 = [], []
for ad, R in PL.items():
    n = ad.replace("-", "\\mbox{-}")
    rows.append(f"{n} & {R['K']} & {_pm(R['tam'])} & {_pm(R['tam_val'])} & {_pm(R['perleaf_uniform'])} & {_pm(R['perleaf_random'])} & {_pm(R['perleaf_residual'])} & {R['perleaf_residual'][2]:.1f} & {_p(R['test_perleaf_gate_vs_uniform'][1])} \\\\")
    rows2.append(f"{n} & {R['K']} & {_pm(R['sifirdan4'])} & {_pm(R['sifirdan4_val'])} & {_pm(R['artimli_split'])} & {_pm(R['artimli_full'])} & {_p(R['test_artimli_full_vs_split'][1])} \\\\")
D["TABLE_PERLEAF"] = "\n".join(rows) if rows else "\\multicolumn{9}{c}{(pending)} \\\\"
D["TABLE_BUDGET"] = "\n".join(rows2) if rows2 else "\\multicolumn{7}{c}{(pending)} \\\\"
D["PL_N"] = str(len(PL))
if PL:
    old_gap = _np.mean([(R["perleaf_uniform"][0] - R["tam"][0]) * 100 for R in PL.values()])
    new_gap = _np.mean([(R["perleaf_residual"][0] - R["tam"][0]) * 100 for R in PL.values()])
    frac = _np.mean([R["perleaf_residual"][2] / 63 for R in PL.values()])
    D["PL_OLD_GAP"] = f"{old_gap:+.1f}"; D["PL_NEW_GAP"] = f"{new_gap:+.1f}"; D["PL_FRAC"] = f"{frac * 100:.0f}"
    D["PL_VAL_GAP"] = f"{_np.mean([(R['sifirdan4'][0] - R['sifirdan4_val'][0]) * 100 for R in PL.values()]):+.1f}"
    D["PL_INC_VS_VAL"] = f"{_np.mean([(R['artimli_full'][0] - R['sifirdan4_val'][0]) * 100 for R in PL.values()]):+.1f}"
    D["PL_FULL_VS_SPLIT"] = f"{_np.mean([(R['artimli_full'][0] - R['artimli_split'][0]) * 100 for R in PL.values()]):+.1f}"
else:
    for k in ("PL_OLD_GAP", "PL_NEW_GAP", "PL_FRAC", "PL_VAL_GAP", "PL_INC_VS_VAL", "PL_FULL_VS_SPLIT"):
        D[k] = "?"

sab = (KOK / "paper/arxiv/main.tex.tmpl").read_text(encoding="utf-8")
eksik = sorted(set(re.findall(r"@@([A-Z0-9_]+)@@", sab)) - set(D))
if eksik:
    sys.exit("karşılığı olmayan yer tutucu: " + ", ".join(eksik))
for k, v in D.items():
    sab = sab.replace(f"@@{k}@@", v)
kalan = re.findall(r"@@[^@]*@@", sab)
if kalan:
    sys.exit("doldurulamayan: " + ", ".join(kalan))
sab = sab.replace(
    "% THIS IS A TEMPLATE.",
    "% GENERATED FILE, DO NOT EDIT. Source: paper/arxiv/main.tex.tmpl\n%",
)
(KOK / "paper/arxiv/main.tex").write_text(sab, encoding="utf-8")
print(f"{len(D)} değer dolduruldu → paper/arxiv/main.tex")

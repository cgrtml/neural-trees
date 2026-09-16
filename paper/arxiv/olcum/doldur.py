#!/usr/bin/env python3
"""
Şablondaki @@...@@ yer tutucularını ölçüm çıktısıyla doldurur.

    python3 paper/arxiv/olcum/doldur.py

Amaç, makaledeki hiçbir sayının elle yazılmaması. Tablo ile kod arasındaki
sessiz kayma bu depoda bir kez yaşandı; bir daha yaşanmasın diye sayılar
ölçüm dosyasından geliyor. Eksik yer tutucu kalırsa betik hata veriyor.
"""
import json, pathlib, re, sys

KOK = pathlib.Path(__file__).resolve().parents[3]
S = json.loads((KOK / "paper/arxiv/olcum/sonuc.json").read_text(encoding="utf-8"))

def ms(x):           # "0.942 ± 0.044"
    return f"{x[0]:.3f} $\\pm$ {x[1]:.3f}"
def msn(x):          # birim/bölme/düğüm sayısı: "6.6 ± 1.9"
    return f"{x[2]:.1f} $\\pm$ {x[3]:.1f}"
def puan(a, b):      # yüzde puan farkı
    return f"{abs(a - b) * 100:.1f}"

J, G, P, O = S["jitter"], S["gal"], S["per_leaf"], S["omni"]
JT, GT = S["jitter_test"], S["gal_test"]

D = {}
for ad, kisa in (("Iris","IRIS"), ("Wine","WINE"), ("Digits","DIGITS")):
    for et, k in (("jitter=0.0","0"), ("jitter=0.05","005"),
                  ("jitter=0.2","02"), ("jitter=0.5","05"), ("sıfırdan","S")):
        D[f"T_J_{kisa}_{k}"] = ms(J[ad][et])
    D[f"JITTER_{kisa}_DROP"] = puan(J[ad]["sıfırdan"][0], J[ad]["jitter=0.0"][0])
D["T_J_IRIS_0_SD"]  = f"{J['Iris']['jitter=0.0'][1]:.3f}"
D["T_J_IRIS_02_SD"] = f"{J['Iris']['jitter=0.2'][1]:.3f}"

# Bozulmanın dağılıma etkisi: sapma kaç katına çıkıyor
oran = [J[a]["jitter=0.0"][1] / J[a]["jitter=0.2"][1] for a in ("Iris","Wine","Digits")]
D["JITTER_SD_RATIO"] = f"{min(oran):.1f} to {max(oran):.1f}"

# jitter 0.05 ile 0.5 arasındaki en büyük oynama
D["JITTER_SPREAD"] = f"{max(100*abs(J[a]['jitter=0.05'][0]-J[a]['jitter=0.5'][0]) for a in ('Iris','Wine','Digits')):.1f}"

def ftest(d):
    # p çok küçükse 4 hane hepsini sıfır gösteriyor
    pp = f"p < 0.0001" if d[1] < 1e-4 else f"p = {d[1]:.4f}"
    return f"$F = {d[0]:.2f}$, ${pp}$"
for ad, kisa in (("Iris","IRIS"), ("Wine","WINE"), ("Digits","DIGITS")):
    D[f"FTEST_JITTER_{kisa}"] = ftest(JT[ad])
    D[f"FTEST_GAL_{kisa}"]    = ftest(GT[ad])

for ad, kisa in (("Iris","IRIS"), ("Wine","WINE"), ("Digits","DIGITS")):
    for init, k in (("random","RND"), ("residual","RES")):
        D[f"T_G_{kisa}_{k}"]   = ms(G[ad][init])
        D[f"T_G_{kisa}_{k}_H"] = msn(G[ad][init])
D["GAL_IRIS_RES_ACC"] = f"{G['Iris']['residual'][0]:.3f}"
D["GAL_IRIS_RND_ACC"] = f"{G['Iris']['random'][0]:.3f}"
D["GAL_IRIS_RES_H"]   = f"{G['Iris']['residual'][2]:.1f}"
D["GAL_IRIS_RND_H"]   = f"{G['Iris']['random'][2]:.1f}"

for ad, kisa in (("Sentetik-800x20","S1"), ("Sentetik-2000x50","S2")):
    for g, k in (("none","NONE"), ("per_leaf","PL")):
        D[f"T_P_{kisa}_{k}"]   = ms(P[ad][g])
        D[f"T_P_{kisa}_{k}_N"] = msn(P[ad][g])
D["PL_S1_GROW_ACC"]  = f"{P['Sentetik-800x20']['per_leaf'][0]:.3f}"
D["PL_S1_GROW_N"]    = f"{P['Sentetik-800x20']['per_leaf'][2]:.1f}"
D["PL_S1_FIXED_ACC"] = f"{P['Sentetik-800x20']['none'][0]:.3f}"
D["PL_S2_DROP"] = puan(P["Sentetik-2000x50"]["none"][0], P["Sentetik-2000x50"]["per_leaf"][0])
D["GAL_DIGITS_DROP"] = puan(G["Digits"]["random"][0], G["Digits"]["residual"][0])

for ad, kisa in (("Cancer","CANCER"), ("Digits","DIGITS")):
    for sec, k in (("accuracy","ACC"), ("test","TEST")):
        D[f"T_O_{kisa}_{k}"]   = ms(O[ad][sec])
        D[f"T_O_{kisa}_{k}_N"] = msn(O[ad][sec])

sab = (KOK / "paper/arxiv/main.tex.tmpl").read_text(encoding="utf-8")
eksik = sorted(set(re.findall(r"@@([A-Z0-9_]+)@@", sab)) - set(D))
if eksik:
    sys.exit("karşılığı olmayan yer tutucu: " + ", ".join(eksik))
for k, v in D.items():
    sab = sab.replace(f"@@{k}@@", v)
kalan = re.findall(r"@@[^@]*@@", sab)
if kalan:
    sys.exit("doldurulamayan: " + ", ".join(kalan))
sab = sab.replace("% THIS IS A TEMPLATE.", "% GENERATED FILE, DO NOT EDIT. Source: paper/arxiv/main.tex.tmpl\n%")
(KOK / "paper/arxiv/main.tex").write_text(sab, encoding="utf-8")
print(f"{len(D)} değer dolduruldu → paper/arxiv/main.tex")

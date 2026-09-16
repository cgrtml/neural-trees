# arXiv gönderimi — kontrol listesi

Bu klasör arXiv'e gönderilecek kaynak dosyayı içerir. `main.tex` tek başına
yeterlidir: harici `.bib`, figür veya stil dosyası yoktur.

## Overleaf'e yükleme

1. Overleaf → **New Project → Upload Project** → `neural-trees-arxiv.zip`.
2. Derleyiciyi **pdfLaTeX**'e ayarla (Menu → Compiler). XeLaTeX/LuaLaTeX
   gerekmiyor.
3. Derle. Uyarı çıkmamalı, tanımsız referans yok.

Not: bu dosya yerelde `tectonic` (XeTeX) ile `hyperref` yüzünden hata veriyor;
bu tectonic'in paket sürümüyle ilgili, belgede sorun yok. Overleaf ve arXiv
pdfLaTeX kullanır, ikisinde de temiz derlenir.

## arXiv'in kurallarına göre neye dikkat edildi

| Kural | Bu dosyada |
|---|---|
| TeX ile yazılmış makalelerde **PDF değil kaynak** gönderilir | `main.tex` gönderilecek |
| arXiv **BibTeX çalıştırmaz**, `.bbl` dosyasını sen eklemelisin | `.bib` hiç kullanılmadı; kaynakça `thebibliography` içinde gömülü, yani `.bbl` derdi yok |
| pdfLaTeX kullanılacaksa ilk satırlarda `\pdfoutput=1` bulunmalı | 3. satırda var |
| Standart dışı stil dosyaları pakete dahil edilmeli | Sadece standart paketler: `amsmath`, `amssymb`, `amsthm`, `booktabs`, `graphicx`, `geometry`, `hyperref`, `inputenc`, `fontenc` |
| Kullanılmayan dosya gönderme | Klasörde sadece `main.tex` ve bu README (README'yi zip'e koyma) |
| Toplam boyut 50 MB altında | ~30 KB |
| Dosya adlarında boşluk/özel karakter olmasın | `main.tex` |

## Web formunda doldurulacaklar

- **Primary category:** `cs.AI` (Artificial Intelligence) — alınan endorsement bu
  kategori için; `cs.LG` seçilemiyor
- **Cross-list:** `cs.LG` ve `stat.ML` eklemeyi dene; endorsement isterse boş bırak.
  Duyurulduktan sonra kategori ekletme talebi gönderilebilir
- **Title / Authors:** `main.tex` ile birebir aynı olmalı
- **Abstract:** düz metin olarak yapıştırılır, **1920 karakter sınırı** var.
  Makaledeki özet bu sınırın altında ama LaTeX komutlarını (`$0.958
  \rightarrow 0.753$` gibi) düz metne çevirmen gerekir: "0.958 -> 0.753".
- **License:** CC BY 4.0 öneririm — kodun MIT olduğu için tutarlı olur.
- **ACM/MSC class:** boş bırakılabilir.
- **Comments:** "10 pages, 4 tables. Code: https://doi.org/10.5281/zenodo.22718897"

Hafta içi 14:00 ET'den önce gönderilen makaleler aynı gün 20:00 ET'de duyurulur.
Moderasyon birkaç gün sürebilir.

## Göndermeden ÖNCE yapılması gerekenler

1. **Zenodo DOI'sini yerleştir.** `main.tex` içinde
   `\textsc{[Zenodo DOI to be inserted]}` yazan yeri gerçek DOI ile değiştir.
   Makale "bu sayıların hangi kod sürümünden çıktığı" sorusuna cevap
   veremezse hakem/okuyucu tekrar üretemez.
2. **Özeti kendi cümlelerinle gözden geçir.** Metin taslak; sayılar ölçülmüş
   ve doğru ama üslup senin olmalı.
3. **AI beyanı bölümünü oku.** Kalsın mı kalmasın mı senin kararın; arXiv
   zorunlu tutmuyor (JOSS tutuyor). Dürüstlük açısından kalmasını öneriyorum.

## Makaledeki her sayının nereden geldiği

Hiçbiri elle yazılmadı. `main.tex` bir şablondan üretiliyor:

```bash
OMP_NUM_THREADS=1 python3 paper/arxiv/olcum/olc.py    # ölçer, sonuc.json yazar
python3 paper/arxiv/olcum/doldur.py                   # main.tex.tmpl -> main.tex
```

`main.tex` ÜRETİLMİŞ dosyadır, elle düzenleme; değişiklikler `main.tex.tmpl`
içine yapılır. Bir yer tutucunun karşılığı yoksa betik hata verip durur, yani
tablo ile kod arasındaki sessiz kayma mümkün değil.

Protokol: özellikler yalnız eğitim katından ölçekleniyor, her sayı 3 tohum x
5-kat çapraz doğrulamanın (15 uydurma) ortalaması ve standart sapması. Karar
gerektiren yerlerde kütüphanenin kendi 5x2cv F testi kullanılıyor.

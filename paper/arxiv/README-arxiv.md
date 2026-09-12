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

- **Primary category:** `cs.LG` (Machine Learning)
- **Cross-list:** `stat.ML`, istersen `cs.NE` (Neural and Evolutionary Computing)
- **Title / Authors:** `main.tex` ile birebir aynı olmalı
- **Abstract:** düz metin olarak yapıştırılır, **1920 karakter sınırı** var.
  Makaledeki özet bu sınırın altında ama LaTeX komutlarını (`$0.958
  \rightarrow 0.753$` gibi) düz metne çevirmen gerekir: "0.958 -> 0.753".
- **License:** CC BY 4.0 öneririm — kodun MIT olduğu için tutarlı olur.
- **ACM/MSC class:** boş bırakılabilir.
- **Comments:** "7 pages, 3 tables. Code: https://github.com/cgrtml/neural-trees"

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

Hepsi mevcut kodla, 3 tohum × 5-kat çapraz doğrulama ile yeniden ölçüldü
(11 Eyl 2026):

| Tablo | İddia | Ölçüm |
|---|---|---|
| 1 | jitter=0 çöküyor | Iris 0.753, Wine 0.754 |
| 1 | jitter 0.05 / 0.2 / 0.5 | Iris 0.942 / 0.942 / 0.938; Wine 0.979 / 0.979 / 0.981 |
| 1 | sıfırdan eğitim | Iris 0.958, Wine 0.977 |
| 2 | GAL rastgele birim | 0.938 doğruluk, 17.4 birim |
| 2 | GAL residual'a uydurulmuş | 0.956 doğruluk, 6.6 birim |
| 3 (metin) | yaprak başına büyütme | 0.885 / 3.7 bölme, sabit derinlik 0.832 / 63 |
| 3 | omnivariate accuracy seçimi | 0.971 doğruluk, 4.1 düğüm, 8/0/15 bölme tipi |
| 3 | omnivariate test seçimi | 0.960 doğruluk, 7.3 düğüm, 20/14/13 bölme tipi |

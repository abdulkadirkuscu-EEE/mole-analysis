# Cilt Lezyonu (Ben) Analiz Sistemi

Kalibrasyon karti uzerine konulan cilt lezyonu (ben) fotograflarini analiz eden
bir goruntu isleme sistemidir. Iki farkli zamanda cekilmis ayni benin
goruntulerini karsilastirarak **renk ve boyut degisimini** otomatik olarak
tespit eder.

## Ozet

- **8 adimlik kod akisi**: Goruntu isleme + renk kalibrasyonu + olcek ve boyut analizi
- **Kalibrasyon karti**: 16 renk yamasi + ortada lezyon bolgesi
- **CIEDE2000** renk farki metrigi (insan gozunun algiladigi farka en yakin)
- **En kucuk kareler** ile RGB kanal kalibrasyonu
- **Yamalar arasi mesafeden olcek** tahmini (kamera-karta uzaklik degisse bile)
- Hem **HTTP API** (FastAPI) hem **mobil uyumlu web arayuzu** (app.html)
- **Iki dil destegi**: Turkce ve Ingilizce kod dosyalari

## Kod Adimlari

| Adim | Aciklama | Cikti |
|------|----------|-------|
| 1 | BW (siyah-beyaz) goruntu uretimi | `bw.png` |
| 2 | Etiket sinir kutusu tespiti | `output.jpg`, `crop.jpg` |
| 3 | Perspektif duzeltme | `perspective.jpg`, `color_square.jpg` |
| 4 | Yama renk kaymasi (drift) tespiti | (sayisal) |
| 5 | RGB kanal kalibrasyonu (en kucuk kareler) | (a, b, R² katsayilari) |
| 6 | Ben rengi degisimi (ΔE2000) | (renk farki ve esik) |
| 7 | Olcek faktoru hesabi | `step7_scale.jpg` |
| 8 | Ben boyut degisimi | `step8_size.jpg`, `comparison.jpg` |

## Klasor Yapisi

```
Cilt_Lezyonu_Analizi/
├── label_pipeline.py       Adim 1-3 fonksiyonlari (Turkce)
├── label_pipeline_en.py    Adim 1-3 fonksiyonlari (Ingilizce)
├── api_server.py           Adim 4-8 + REST API (Turkce)
├── api_server_en.py        Adim 4-8 + REST API (Ingilizce)
├── app.html                Mobil uyumlu web arayuzu
├── requirements.txt        Python bagimliliklari
├── README.md               Bu dosya
├── README_en.md            Ingilizce dokumantasyon
└── ornek_goruntuler/       Ornek girdi ve cikti goruntuleri
```

## Kurulum

```bash
# 1) Python 3.9+ gerekli (3.10 veya 3.11 onerilir)
python --version

# 2) Bagimliliklari yukle
pip install -r requirements.txt
```

## Kullanim

### Yontem 1: API Sunucusu (Onerilen)

```bash
python api_server.py
```

Sonra tarayicidan:
- **Web uygulamasi**: <http://localhost:8000/app>
- **API dokumantasyonu** (otomatik Swagger UI): <http://localhost:8000/docs>

Web uygulamasi acildiginda:
1. Telefonda bir ben adi gir (orn: `sol_kol_1`)
2. Bene "FOTOGRAF" butonuyla cekim yap
3. Yeni cekimleri ekledikce eski cekimlerle karsilastirma yapabilirsin

### Yontem 2: Klasor Izleme Modu (sadece Adim 1-3)

```bash
python label_pipeline.py
```

Bu mod `input_images/` klasorunu izler. Oraya birakilan her resim icin
otomatik olarak Adim 1-3'u calistirir ve ciktilari ilgili klasorlere yazar.

### Ingilizce Versiyonu Calistirmak

```bash
python api_server_en.py    # Ingilizce yorumlu API sunucusu
python label_pipeline_en.py # Ingilizce yorumlu klasor izleme
```

> **NOT**: Ingilizce ve Turkce sunuculari ayni anda calistirma. Her ikisi de
> 8000 portunu kullanir. Birini calistirip diger dosyayi sadece okuma amaciyla
> incele.

## API Endpoint'leri

| Method | Endpoint | Aciklama |
|--------|----------|----------|
| GET    | `/` | Saglik kontrolu |
| GET    | `/app` | Web uygulamasini sun |
| GET    | `/moles` | Tum benleri listele |
| GET    | `/moles/{name}/captures` | Bir benin cekimlerini listele |
| POST   | `/analyze` | Yeni fotograf isle (Adim 1-3) |
| POST   | `/compare` | Iki cekimi karsilastir (Adim 4-8) |
| PUT    | `/moles/{name}` | Ben adini degistir |
| DELETE | `/moles/{name}` | Beni sil |
| DELETE | `/moles/{name}/{timestamp}` | Cekimi sil |
| GET    | `/image/{name}/{timestamp}/{file}` | Adim goruntusunu getir |

## Teknik Detaylar

### Kalibrasyon Karti

Kart 16 doygun renk yamasini U-sekli bir yerlesimle bulundurur:
- Ust sirada 4 yatay yama
- Alt sirada 4 yatay yama
- Sol sutunda 4 dikey yama
- Sag sutunda 4 dikey yama
- Ortada lezyon bolgesi

Her yama 0.075 x 0.075 normalize boyutta. Yama merkezleri kart sablonundan
elle olculmustur (`label_pipeline.py` icindeki `PATCHES` sabiti).

### Yama Tespiti: Bant + Projeksiyon Yontemi

1. Doygunluk (S) kanalindan 4 renk yamasi bandi tespit edilir
2. Her bantta parlaklik (V) projeksiyonu alinir
3. Vadi-tepe analiziyle her bant 4 yamaya bolunur
4. Toplam 16 yama elde edilir

Bu yontem **sabit koordinatlardan daha saglamdir** cunku perspektif
duzeltme sonrasinda yama konumlari hafifce kayabilir.

### Renk Karsilastirmasi: CIEDE2000

CIE'nin tavsiye ettigi en gelismis renk farki formulu. Insan gozunun
algi egrisine en yakin sonucu verir.

| ΔE2000 | Anlam |
|--------|-------|
| < 1.0 | Fark insan gozune fark edilmez |
| 1-2 | Yakindan bakildiginda fark edilir |
| 2-10 | Bir bakista fark edilir |
| > 10 | Belirgin fark |

Renk degisim esigi: **max(ortalama_artik + 3σ, 6.0)** — yani en az 6.0
ΔE2000 farki olmadan "degismis" karari verilmez.

### Olcek Faktoru ve Boyut Esikleri

Iki cekimde kamera-karta uzakligi farkli olabilir. Bunu cozmek icin:

```
scale = median(test_yama_mesafeleri / referans_yama_mesafeleri)
```

Boyut degisim esikleri (kucuk benlerde Otsu segmentasyonu birkac piksel
oynar — bu yuzden esikler muhafazakar):
- **Cap esigi**: minimum ±%18
- **Alan esigi**: minimum ±%35

Karar: hem cap hem alan esigi asarsa "boyut degismis" sayilir.

## Veri Saklama

```
BENLER/                       (Turkce sunucu) veya MOLES/ (Ingilizce sunucu)
├── sol_kol_1/                Ben adi
│   ├── 2025-11-01_10-30-00/  Cekim zaman damgasi
│   │   ├── input.jpg         Orijinal fotograf
│   │   ├── bw.png            Adim 1 ciktisi
│   │   ├── output.jpg        Adim 2 ciktisi
│   │   ├── crop.jpg          Kirpilmis
│   │   ├── perspective.jpg   Adim 3 ciktisi
│   │   ├── color_square.jpg  Yama + ben isaretli gorsel
│   │   ├── data.json         Yama RGB + mesafe verileri
│   │   ├── step7_scale.jpg   Karsilastirma sonrasi (Adim 7)
│   │   ├── step8_size.jpg    Karsilastirma sonrasi (Adim 8)
│   │   ├── comparison.jpg    3-panelli karsilastirma gorseli
│   │   └── comparison_data.json  Karsilastirma sonucu
│   └── 2025-11-15_14-22-00/  Sonraki cekim
└── sag_yanak/                Ikinci ben
```

## Ornek Goruntuler

`ornek_goruntuler/` klasorune **ornek girdi fotograflari ve isleme
ciktilarini ekleyiniz**. Bu klasor su anda bostur — hocaya teslimden once
sablon kart uzerinde cekilmis fotograflarinizi ve isleme ciktilarini bu
klasore koyun.

## Gelistirme Notlari

- **Cikti tutarliligi**: Her cekim ayni klasor yapisinda saklandigi icin iki
  cekim arasi karsilastirma sirasinda dosyalara dogrudan erisilebilir.
- **data.json onbellegi**: Yama RGB ve mesafeleri her cekimden sonra
  data.json'a yazilir. Karsilastirma sirasinda once buradan okunur,
  bulamazsa dinamik tespit calistirilir.
- **Fallback'lar**: Dinamik yama tespiti basarisiz olursa sabit PATCHES
  koordinatlari kullanilir. BW dortgen bulunamazsa dilate edilmis maske
  denenir.
- **Ben tespiti**: Goruntu merkezinde ±%13 ROI icinde, dairesel ve buyuk
  blob skorlanarak secilir.

## Yazar

Cilt Lezyonu Analiz Sistemi - Dijital Goruntu Isleme Projesi

# Basketball Game Tracker

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/CUDA-önerilir-76B900" />
  <img src="https://img.shields.io/badge/Durum-Aktif-brightgreen" />
</p>

Basketbol maç videoları için bilgisayarlı görü pipeline'ı. Oyuncu ve hakemleri kare kare tespit edip takip eder, forma numaralarını okur ve her oyuncunun konumunu gerçek zamanlı 2D taktik saha görünümüne yansıtır.

İngilizce dokümantasyon: [README.md](README.md)

---

## Demo

**Oyuncu takibi**

<video src="https://github.com/user-attachments/assets/58786f0c-562e-4b5d-89ac-d41856fe7738" controls width="75%"></video>

**Taktik saha görünümü**

<video src="https://github.com/user-attachments/assets/a1dda86d-07d1-4e90-ba1f-71374ad8c45a" controls width="30%"></video>

---

## Özellikler

- **Çoklu nesne takibi** — RF-DETR tespitleri MCByte'a beslenir; SAM + Cutie maske propagasyonu ile temas ve örtüşme anlarında kimlikler korunur
- **Forma numarası OCR** — PARSeq, kırpılmış forma bölgelerinden numara okur; oylama bankası kimlik kararlılığını sağlar
- **Taktik projeksiyon** — saha keypoint'leri + RANSAC homography ile her oyuncunun ayak konumu ölçeklenmiş saha görseline anlık yansıtılır
- **Duplicate baskılama** — örtüşen tespitlerden kaynaklanan hayalet ve tekrar track'ler özel son-işleme ile elenir
- **İki çıktı videosu** — işaretlenmiş tracking görünümü ve ayrı bir taktik saha videosu

---

## Nasıl Çalışır

```
Giriş videosu
    │
    ▼
RF-DETR tespiti            (oyuncu, hakem, forma bölgesi — Roboflow)
    │
    ▼
MCByte takibi              (maske destekli çoklu nesne takibi)
 + SAM / Cutie             (segmentasyon maskeleri ile daha temiz association)
    │
    ▼
Forma OCR                  (PARSeq + oylama bankası)
    │
    ▼
Saha keypoint'leri         (Roboflow YOLO-Pose → RANSAC homography)
    │
    ▼
İşaretlenmiş video + Taktik saha görünümü
```

---

## Modeller ve Kütüphaneler

| Bileşen | Görev | Link |
|---|---|---|
| RF-DETR | Roboflow Inference üzerinden oyuncu, hakem ve forma bölgesi tespiti. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | Lokal inference runtime — modeller ilk çalıştırmada otomatik indirilir. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Duplicate baskılama ve kayıp track yeniden kullanımı ile genişletilmiş çoklu nesne tracker'ı. | [MCByte](https://github.com/tstanczyk95/McByte) |
| Segment Anything (SAM) | Dedektör kutularından ilk segmentasyon maskelerini üretir. | [SAM](https://github.com/facebookresearch/segment-anything) |
| Cutie | Oyuncu maskelerini frame'ler arasında temporal tutarlılık için yayar. | [Cutie](https://github.com/hkchengrex/Cutie) |
| PARSeq | Kırpılmış forma bölgelerinden sahne metni tanıma ile numara okur. | [PARSeq](https://github.com/baudm/parseq) |
| OpenCV | Homography tahmini, perspektif dönüşüm, video okuma/yazma, çizim. | [OpenCV](https://opencv.org/) |
| PyTorch | SAM, Cutie, PARSeq ve YOLO modellerinin runtime'ı. | [PyTorch](https://pytorch.org/) |

> Model ID'leri, güven eşikleri ve tracker parametreleri gibi gelişmiş ayarlar `APP/helpers/config.py` üzerinden yapılandırılır.

---

## Kurulum

### 1. Gereksinimler

- Python 3.11+
- CUDA destekli GPU (önerilir)
- PyTorch 2.x
- Ücretsiz bir [Roboflow](https://roboflow.com) hesabı ve API key

### 2. Klonla ve yükle

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. API key'ini ayarla

```bash
cp .env.example .env
```

`.env` dosyasını açıp şunu ayarla:

```
ROBOFLOW_API_KEY=your_key_here
```

### 4. Harici repoları klonla

```bash
# MCByte tracker
git clone https://github.com/tstanczyk95/McByte.git external/mcbyte

# Cutie (video nesne segmentasyonu — MCByte içine)
git clone https://github.com/hkchengrex/Cutie.git external/mcbyte/mask_propagation/Cutie

# PARSeq (forma OCR)
git clone https://github.com/baudm/parseq.git parseq
pip install -r parseq/requirements/core.txt
```

### 5. Model ağırlıklarını indir

```bash
# SAM ViT-B ağırlıkları
mkdir -p external/mcbyte/sam_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O external/mcbyte/sam_models/sam_vit_b_01ec64.pth

# Cutie ağırlıkları
mkdir -p external/mcbyte/mask_propagation/Cutie/weights
wget https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth \
     -O external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth
```

Tespit ve keypoint modelleri ilk çalıştırmada Roboflow Inference tarafından otomatik indirilir.

---

## Kullanım

```bash
python -m APP \
  --input  videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda \
  --max-frames 500
```

Şu çıktıları üretir:

```
videos/output/result.mp4           ← işaretlenmiş tracking videosu
videos/output/result_tactical.mp4  ← taktik saha görünümü
videos/output/LOG.log              ← tam çalışma logu
```

### CLI Parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--input` | — | Giriş videosu yolu (zorunlu) |
| `--output` | — | Çıktı videosu yolu (zorunlu) |
| `--max-frames` | 300 | İşlenecek maksimum frame sayısı |
| `--start` | 0.8 | Başlangıç ofseti (saniye) |
| `--frame-skip` | 1 | Her N frame'de bir işleme |
| `--device` | `cuda` | `cuda`, `cuda:0` veya `cpu` |
| `--confidence` | config'den | Dedektör güven eşiği |
| `--debug` | kapalı | Tanı loglarını etkinleştir (tracking, mask, keypoint) |

---

## Proje Yapısı

```
APP/
  __main__.py              CLI giriş noktası
  assets/
    basketball_court.png   Taktik saha şablonu
  helpers/
    config.py              Tüm parametreler burada — ayarlamak için düzenle
    pipeline.py            Uçtan uca pipeline orkestrasyon
    mcbyte_tracker.py      MCByte wrapper (maske + baskılama mantığı)
    rfdetr_detector.py     RF-DETR / Roboflow Inference wrapper
    jersey_detector.py     PARSeq OCR ve forma oylama bankası
    court_utils.py         Homography, keypoint ayrıştırma, taktik rendering
```

---

## Notlar

- Tüm inference Roboflow Inference üzerinden lokalde çalışır — ilk çalıştırmadan sonra frame'ler buluta gönderilmez.
- Taktik görünüm, saha keypoint'leri kısmen görünür olduğunda bile çalışmaya devam eder: pipeline son güvenilir homography'yi taşır ve büyük geometrik sapmalara yol açan güncellemeleri reddeder.

---

## Roadmap

- Takım possession ve event seviyesinde analiz
- Şut tespiti ve top durum entegrasyonu
- Oyuncu başına hız ve mesafe metrikleri
- Export edilebilir maç raporları

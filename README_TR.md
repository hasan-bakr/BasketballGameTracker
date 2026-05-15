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

<video src="https://github.com/user-attachments/assets/0730f5e0-849e-4947-8b2b-817f860640a8" controls width="75%"></video>

**Taktik saha görünümü**

<video src="https://github.com/user-attachments/assets/cab1b58e-5c63-43d7-bd57-4b6f2614854a" controls width="30%"></video>

---

## Özellikler

- **Çoklu nesne takibi** — RF-DETR tespitleri MCByte'a beslenir; SAM + Cutie maske propagasyonu ile temas ve örtüşme anlarında kimlikler korunur
- **Forma numarası OCR** — PARSeq, kırpılmış forma bölgelerinden numara okur; oylama bankası kimlik kararlılığını sağlar
- **Taktik projeksiyon** — YOLO-Pose saha keypoint'leri + RANSAC homography ile her oyuncunun ayak konumu ölçeklenmiş saha görseline anlık olarak yansıtılır
- **Duplicate baskılama** — örtüşen tespitlerden kaynaklanan hayalet ve tekrar track'ler özel son-işleme katmanı ile elenir
- **İki çıktı videosu** — işaretlenmiş tracking görünümü ve ayrı bir taktik saha videosu

---

## Nasıl Çalışır

```
Giriş videosu
    │
    ▼
RF-DETR tespiti            (oyuncu, hakem, forma bölgesi)
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
İşaretlenmiş video + Taktik saha videosu
```

---

## Modeller ve Kütüphaneler

### Tespit ve Takip

| Bileşen | Görev | Link |
|---|---|---|
| RF-DETR | Roboflow Inference üzerinden oyuncu, hakem ve forma bölgesi tespiti. Varsayılan model: `basketball-player-detection-3-ycjdo/4`. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | RF-DETR için lokal inference runtime. `ROBOFLOW_API_KEY` gerektirir. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Duplicate baskılama, maske izolasyonu ve kayıp track yeniden kullanımı ile genişletilmiş çoklu nesne tracker'ı. | [MCByte](https://github.com/tstanczyk95/McByte) · [makale](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.html) |
| ByteTrack | MCByte'ın kullandığı tracking-by-detection association tabanı. | [ByteTrack](https://github.com/FoundationVision/ByteTrack) |
| Segment Anything (SAM) | Dedektör kutularından ilk maskeleri üretir. | [SAM](https://github.com/facebookresearch/segment-anything) |
| Cutie | Oyuncu maskelerini frame'ler arasında temporal tutarlılık için yayar. | [Cutie](https://github.com/hkchengrex/Cutie) |

### Forma OCR

| Bileşen | Görev | Link |
|---|---|---|
| PARSeq | Kırpılmış forma bölgelerinden numara okur. | [PARSeq](https://github.com/baudm/parseq) |

### Saha ve Taktik Görünüm

| Bileşen | Görev | Link |
|---|---|---|
| Roboflow YOLO-Pose | Roboflow Inference üzerinden frame başına 33 saha keypoint'i tespit eder. Varsayılan model: `basketball-court-detection-2/19`. | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| OpenCV | Homography tahmini, perspektif dönüşüm, video okuma/yazma, çizim. | [OpenCV](https://opencv.org/) |

### Destekleyici Kütüphaneler

| Bileşen | Görev |
|---|---|
| PyTorch | SAM, Cutie, PARSeq ve YOLO modellerinin runtime'ı. |
| NumPy / SciPy | Dizi işlemleri ve MCByte'ın Kalman filter matematiği. |
| Supervision | Annotation yardımcıları ve tespit utility'leri. |
| Pillow | Forma crop'larını PARSeq için PIL image'e çevirir. |
| python-dotenv | `ROBOFLOW_API_KEY`'i `.env` dosyasından yükler. |
| LAP / FilterPy | MCByte için assignment ve filtering yardımcıları. |
| Hydra / OmegaConf | Cutie'nin konfigürasyon sistemi. |

---

## Gereksinimler

- Python 3.11+
- CUDA destekli GPU (önerilir)
- PyTorch 2.x
- Roboflow API key (ücretsiz tier yeterli)

Aşağıdaki dosyaların çalıştırmadan önce lokal olarak yerleştirilmesi gerekir — Git'e eklenmezler:

```
external/mcbyte/                                                      ← MCByte repo
external/mcbyte/sam_models/sam_vit_b_01ec64.pth                      ← SAM ağırlıkları
external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth    ← Cutie ağırlıkları
parseq/                                                               ← PARSeq repo
```

---

## Kurulum

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Environment dosyası oluştur ve Roboflow API key'ini ekle:

```bash
cp .env.example .env
# .env dosyasını düzenle:
ROBOFLOW_API_KEY=your_key_here
```

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

### CLI Referansı

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--input` | — | Giriş videosu yolu (zorunlu) |
| `--output` | — | İşaretlenmiş çıktı videosu yolu (zorunlu) |
| `--max-frames` | 300 | İşlenecek maksimum frame sayısı |
| `--start` | 0.8 | Başlangıç ofseti (saniye) |
| `--frame-skip` | 1 | Her N frame'de bir işleme |
| `--device` | config'den | `cuda`, `cuda:0` veya `cpu` |
| `--confidence` | config'den | Dedektör güven eşiği |
| `--detector-iou` | config'den | RF-DETR NMS IoU eşiği |
| `--track-thresh` | config'den | MCByte yüksek skor takip eşiği |
| `--new-track-thresh` | config'den | Yeni track başlatma minimum skoru |
| `--keypoint-backend` | config'den | `roboflow` veya `local` |
| `--keypoint-model-id` | config'den | Roboflow keypoint model ID override |
| `--keypoint-confidence` | config'den | Keypoint tespit güven eşiği |
| `--debug` | kapalı | Tracking, mask ve keypoint tanı logları |
| `--debug-lifecycle` | kapalı | Frame başına track lifecycle logları |
| `--debug-suppression` | kapalı | Duplicate baskılama kararları logu |
| `--tracking-report-json` | — | Lifecycle özetini JSON olarak yazar |

---

## Proje Yapısı

```
APP/
  __main__.py              CLI giriş noktası
  assets/
    basketball_court.png   Taktik saha şablonu
  helpers/
    config.py              Merkezi konfigürasyon dataclass'ları
    pipeline.py            Uçtan uca pipeline orkestrasyon
    mcbyte_tracker.py      MCByte wrapper (maske + baskılama mantığı)
    rfdetr_detector.py     RF-DETR / Roboflow Inference wrapper
    jersey_detector.py     PARSeq OCR ve forma oylama bankası
    court_utils.py         Homography, keypoint ayrıştırma, taktik rendering
    team_detector.py       Takım rengi kümeleme yardımcıları
```

---

## Notlar

- Tüm tespit ve keypoint inference'ı Roboflow Inference üzerinden lokalde çalışır — model cache'lendikten sonra frame'ler buluta gönderilmez.
- Taktik görünüm, saha keypoint'leri kısmen görünür olduğunda bile çalışmaya devam eder: pipeline son güvenilir homography'yi taşır ve büyük geometrik sapmalara yol açan güncellemeleri reddeder.
- Giriş/çıktı videoları, model ağırlıkları, API key'ler ve harici repo'lar Git tarafından ignore edilir.

---

## Roadmap

- Takım possession ve event seviyesinde analiz
- Şut tespiti ve top durum entegrasyonu
- Oyuncu başına hız ve mesafe metrikleri
- Export edilebilir maç raporları

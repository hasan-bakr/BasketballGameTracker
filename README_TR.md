# Basketball Game Tracker

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-önerilir-76B900)
![Durum](https://img.shields.io/badge/Durum-Aktif-brightgreen)

Basketbol maç videoları için bilgisayarlı görü pipeline'ı. Oyuncu ve hakemleri algılar, kalabalık pozisyonlarda oyuncu kimliklerini korumaya çalışır, forma numaralarını okur ve oyuncu konumlarını 2D taktik saha görünümüne aktarır.

English documentation: [README.md](README.md)

## Demo

**Ana video tracking görünümü**

<video src="https://github.com/user-attachments/assets/0730f5e0-849e-4947-8b2b-817f860640a8" controls width="75%"></video>

**Taktik saha görünümü**

<video src="https://github.com/user-attachments/assets/cab1b58e-5c63-43d7-bd57-4b6f2614854a" controls width="30%"></video>

## Öne Çıkanlar

- RF-DETR ile oyuncu, hakem ve forma bölgesi algılama
- MCByte ile çoklu oyuncu takibi
- SAM + Cutie ile maske propagasyonu ve maske destekli association
- Occlusion anlarında duplicate ve phantom track baskılama
- PARSeq OCR ve voting tabanlı forma numarası bankası
- YOLO-Pose saha keypoint algılama ve RANSAC homography
- Keypoint carry, sağ/sol saha geçişi kontrolü ve smoothing ile daha stabil tactical view
- Ana video ve tactical view için ayrı çıktı videoları

## Modeller, Tracker'lar ve Temel Kütüphaneler

| Bileşen | Nerede Kullanılıyor | Link |
|---|---|---|
| RF-DETR | Oyuncu, hakem, top ve forma-numara bölgesi detection'ları için kullanılır. Varsayılan proje model id'si `basketball-player-detection-3-ycjdo/4`. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | RF-DETR modelini çalıştıran local/remote inference runtime. `ROBOFLOW_API_KEY` değeri `.env` dosyasından okunur. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Ana multi-object tracker. Proje `external/mcbyte/` checkout'unu kullanır ve mask-aware association akışına duplicate suppression, mask isolation ve lost-track reuse mantığı ekler. | [MCByte](https://github.com/tstanczyk95/McByte), [makale](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.html) |
| ByteTrack | MCByte içindeki tracking-by-detection association tabanı. | [ByteTrack](https://github.com/FoundationVision/ByteTrack) |
| Segment Anything (SAM) | Detector box'larından ilk maskeleri üretir. Beklenen varsayılan ağırlık `external/mcbyte/sam_models/sam_vit_b_01ec64.pth`. | [Segment Anything](https://github.com/facebookresearch/segment-anything) |
| Cutie | Oyuncu maskelerini frame'ler arasında propagate eden video object segmentation modeli. Beklenen varsayılan ağırlık `external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth`. | [Cutie](https://github.com/hkchengrex/Cutie) |
| PARSeq | Local `parseq/` checkout'u ve STRHub modülleri üzerinden forma numarası OCR'ı. | [PARSeq](https://github.com/baudm/parseq) |
| Ultralytics YOLO Pose | Homography ve tactical view projection için saha keypoint detection. Beklenen custom ağırlık `models/keypoints/yolo26l-fine-tuned.pt`. | [Ultralytics](https://github.com/ultralytics/ultralytics), [pose dokümanı](https://docs.ultralytics.com/tasks/pose/) |
| OpenCV | Homography estimation, perspective transform, video okuma/yazma, çizim ve tactical view render işlemleri. | [OpenCV](https://opencv.org/) |
| Supervision | Görsel output pipeline'ındaki annotation helper'ları, renkler ve detection utility'leri. | [Supervision](https://github.com/roboflow/supervision) |
| PyTorch | Detector, tracker bağımlılıkları, SAM, Cutie, PARSeq ve YOLO modellerinin deep-learning runtime'ı. | [PyTorch](https://pytorch.org/) |

Destekleyici runtime ve utility bağımlılıkları:

| Bileşen | Nerede Kullanılıyor | Link |
|---|---|---|
| NumPy / SciPy | Array işlemleri, geometri hesapları ve MCByte Kalman-filter matematiği. | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Pillow | Forma crop'larını PARSeq preprocessing öncesinde PIL image'e çevirir. | [Pillow](https://python-pillow.org/) |
| python-dotenv | `ROBOFLOW_API_KEY` gibi `.env` değerlerini secret commit etmeden yükler. | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| LAP / FilterPy | MCByte tracking stack'inde assignment ve filtering yardımcıları. | [lap](https://github.com/gatagat/lap), [FilterPy](https://github.com/rlabbe/filterpy) |
| Hydra / OmegaConf | Cutie mask-propagation kodunun konfigürasyon sistemi. | [Hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/) |
| h5py / scikit-image / Hugging Face Hub | External model bağımlılıklarının model/checkpoint ve image-processing utility'leri. | [h5py](https://www.h5py.org/), [scikit-image](https://scikit-image.org/), [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) |
| tqdm | External model utility ve script'lerinde progress bar. | [tqdm](https://github.com/tqdm/tqdm) |

## Pipeline

```text
Input video
    |
    v
RF-DETR detection
    |
    v
MCByte tracking + SAM/Cutie mask propagation
    |
    v
Forma OCR ve ID stabilizasyonu
    |
    v
Saha keypoint'leri + homography
    |
    v
Annotated video + tactical court video
```

## Proje Yapısı

```text
APP/
  __main__.py              CLI giriş noktası
  assets/
    basketball_court.png   Tactical court görseli
  helpers/
    config.py              Merkezi konfigürasyon
    pipeline.py            Uçtan uca pipeline
    mcbyte_tracker.py      MCByte ve maske wrapper'ı
    rfdetr_detector.py     RF-DETR detector wrapper'ı
    jersey_detector.py     PARSeq OCR ve forma bankası
    court_utils.py         Homography ve tactical rendering
    team_detector.py       Takım rengi yardımcıları
```

## Gereksinimler

Önerilen ortam:

- Python 3.11+
- CUDA destekli GPU
- PyTorch 2.x
- RF-DETR inference için Roboflow API key

Büyük model dosyaları ve harici research repo'ları Git'e eklenmez. Full pipeline çalışmadan önce lokal olarak yerleştirilmelidir.

Beklenen lokal asset'ler:

```text
models/keypoints/yolo26l-fine-tuned.pt
external/mcbyte/
external/mcbyte/sam_models/sam_vit_b_01ec64.pth
external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth
parseq/
```

## Kurulum

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Environment dosyasını oluştur:

```bash
cp .env.example .env
```

Sonra API key'i gir:

```bash
ROBOFLOW_API_KEY=your_key_here
```

## Kullanım

```bash
python -m APP \
  --input videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda \
  --max-frames 300
```

Komut şu çıktıları üretir:

```text
videos/output/result.mp4
videos/output/result_tactical.mp4
videos/output/LOG.log
```

## Faydalı CLI Parametreleri

| Parametre | Açıklama |
|---|---|
| `--input` | Input video yolu |
| `--output` | Annotated output video yolu |
| `--max-frames` | İşlenecek maksimum frame sayısı |
| `--start` | Başlangıç zamanı, saniye |
| `--frame-skip` | Her N frame'de bir işleme |
| `--device` | `cuda`, `cuda:0` veya `cpu` |
| `--confidence` | Detector confidence override |
| `--detector-iou` | RF-DETR internal NMS IoU override |
| `--track-thresh` | MCByte high-score threshold |
| `--new-track-thresh` | Yeni track başlatma skoru |
| `--debug` | Tracking, mask ve keypoint debug logları |
| `--debug-lifecycle` | Track lifecycle logları |
| `--debug-suppression` | Duplicate suppression ve maske karar logları |
| `--tracking-report-json` | Lifecycle summary JSON çıktısı |

## Notlar

- Input videolar, output videolar, model weight'leri, API key'ler ve harici lokal repo'lar Git tarafından ignore edilir.
- Tactical view saha keypoint'lerine bağlıdır. Pipeline ani homography zıplamaları riskini azaltmak için son iyi keypoint'leri taşır ve kararsız H güncellemelerini reddeder.

## Roadmap

- Takım possession ve event seviyesinde analiz
- Şut ve top durumu entegrasyonu
- Hız ve mesafe metrikleri
- Export edilebilir maç raporları

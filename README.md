# Basketball Game Tracker

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)
![SAM2](https://img.shields.io/badge/SAM2-Segmentation-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

An AI-powered computer vision pipeline for analyzing basketball game footage — detecting players, tracking them across frames, reading jersey numbers, and projecting positions onto a 2D tactical map.

## Demo

**Player Segmentation & Tracking (SAM2)**

![Segmentation Demo](videos/sam2_robust_2_output_compressed.gif)

**Tactical Bird's-eye View**

![Tactical Map Demo](videos/sam2_robust_2_output_tactical.gif)

## Features

| Feature | Status |
|---|---|
| Player & ball detection (YOLO) | ✅ |
| Instance segmentation & tracking (SAM2) | ✅ |
| Jersey number OCR (PARSeq) | ✅ |
| Player Re-ID via jersey voting | ✅ |
| Court keypoint detection (YOLO-Pose, 18 pts) | ✅ |
| Camera-to-court homography | ✅ |
| 2D tactical view projection | ✅ |
| Speed / distance calculation | 🔲 |

## Architecture

```
Input Video
    │
    ▼
YOLO Detection ──────────────────────────────────────┐
(players, ball, rim, jersey numbers)                 │
    │                                                │
    ▼                                                │
SAM2 Video Propagation                               │
(pixel-level segmentation, memory bank)              │
    │                                                │
    ▼                                                ▼
IoU Re-ID (Hungarian algorithm)        Jersey OCR (PARSeq)
    │                                  (async, per player)
    │                                                │
    └──────────────────┬─────────────────────────────┘
                       ▼
          Court Keypoint Detection (YOLO-Pose)
          + Homography (RANSAC, temporal blend)
                       │
                       ▼
              Tactical Bird's-eye View
```

## Project Structure

```
APP/
├── __main__.py               # CLI entry point  →  python -m APP
├── helpers/
│   ├── robust_sam2_tracker.py  # Main tracker pipeline
│   ├── yolo_detector.py        # YOLO wrapper (basketball classes)
│   ├── jersey_detector.py      # PARSeq OCR + JerseyReIDBank
│   ├── court_utils.py          # Court constants, homography, tactical view
│   ├── visualization.py        # Mask overlay drawing
│   ├── court_keypoint_detector.py
│   ├── homography_transformer.py
│   ├── generate_tactical_view.py
│   ├── sam_helper.py           # SAM2 ONNX (single-frame)
│   ├── manual_court_selector.py
│   ├── rfdetr_detector.py
│   └── download_court_model.py
└── assets/
    └── basketball_court.png    # Court template for tactical view

videos/
├── sam2_robust_2_output_compressed.gif  # Tracking demo
└── sam2_robust_2_output_tactical.gif    # Tactical view demo
```

## Setup

### Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- Conda

```bash
git clone https://github.com/hasan-bakr/BasketballGameTracker.git
cd BasketballGameTracker

conda create -n dsci python=3.12
conda activate dsci

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Models

Place model weights in `models/`:
- `best_detection.pt` — Custom YOLO for basketball (players, ball, rim, jersey number)
- `sam2.1_hiera_base_plus.pt` — SAM2.1 checkpoint
- `models/keypoints/best.pt` — Court keypoint YOLO-Pose model

### Run

```bash
python -m APP --input videos/input/game.mp4 --output videos/output/result.mp4
```

**All options:**
```
--input / -i       Input video path (required)
--output / -o      Output video path (required)
--max-frames       Frames to process (default: 300)
--batch-size       SAM2 batch size (default: 150)
--confidence       YOLO threshold (default: 0.5)
--device           cuda or cpu (default: cuda)
--sam2-checkpoint  Path to SAM2 .pt file
--yolo-model       Path to YOLO .pt file
--no-amp           Disable FP16 mixed precision
```

## Tech Stack

- **Detection:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (custom trained)
- **Segmentation & Tracking:** [SAM2](https://github.com/facebookresearch/sam2) (Meta AI)
- **Jersey OCR:** [PARSeq](https://github.com/baudm/parseq) (scene text recognition)
- **Court Geometry:** YOLO-Pose → RANSAC Homography
- **Re-ID:** IoU matching + Hungarian algorithm + jersey number voting

---

## Proje Özeti (Türkçe)

Basketbol maçı görüntülerini analiz eden yapay zeka tabanlı bir bilgisayarlı görü sistemi. Ham video girdisinden oyuncu tespiti, piksel hassasiyetinde takip, forma numarası tanıma ve 2D taktik harita çıktısı üretir.

**Ana modüller:** YOLO (tespit) → SAM2 (segmentasyon & takip) → PARSeq (OCR) → Homografi → Taktik görünüm

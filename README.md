# Basketball Game Tracker

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![RF--DETR](https://img.shields.io/badge/RF--DETR-Detection-orange)
![BoT--SORT](https://img.shields.io/badge/BoT--SORT-Tracking-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Computer-vision pipeline for basketball footage analysis:
- player/referee detection
- multi-object tracking with stable IDs
- optional jersey OCR
- court homography and tactical bird's-eye view

## Current Pipeline

```text
Input Video
   ->
RF-DETR (players, referees, optional ball/rim classes)
   ->
BoT-SORT tracking (player/referee trackers separated)
   ->
Memory ReID (DINOv2 embedding bank, ID stabilization)
   ->
Jersey OCR (optional YOLO jersey boxes + PARSeq)
   ->
Court keypoints + homography
   ->
Annotated video + tactical view video
```

## Features

| Feature | Status |
|---|---|
| Player/referee detection (RF-DETR) | ✅ |
| Multi-object tracking (BoT-SORT) | ✅ |
| Memory-bank ReID stabilization (DINOv2) | ✅ |
| Jersey OCR (optional PARSeq flow) | ✅ |
| Court keypoint detection + homography | ✅ |
| Tactical bird's-eye projection | ✅ |
| Speed / distance analytics | 🔲 |

## Project Structure

```text
APP/
├── __main__.py                    # CLI entry point: python -m APP
├── assets/
│   └── basketball_court.png
└── helpers/
    ├── botsort_pipeline.py        # End-to-end pipeline
    ├── botsort_tracker.py         # BoT-SORT wrapper
    ├── rfdetr_detector.py         # Roboflow RF-DETR wrapper
    ├── memory_reid.py             # DINOv2 memory-bank ReID
    ├── yolo_detector.py           # Optional jersey-box detector
    ├── jersey_detector.py         # PARSeq OCR + jersey bank
    ├── court_utils.py             # Keypoints, homography, tactical view
    ├── team_detector.py           # Team-color helper
    └── visualization.py           # Drawing helpers
```

## Setup

### Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- Conda (recommended)

```bash
git clone https://github.com/hasan-bakr/BasketballGameTracker.git
cd BasketballGameTracker

conda create -n dsci python=3.11 -y
conda activate dsci

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Required Environment Variable

RF-DETR runs via Roboflow Inference SDK:

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

If `inference-gpu` is missing:

```bash
pip uninstall inference -y
pip install inference-gpu
```

### Model Assets

- `models/keypoints/yolo26l-fine-tuned.pt` (court keypoint model)
- Optional jersey detector model for `--yolo-model`

## Run

```bash
python -m APP \
  --input videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda
```

### CLI Options

```text
--input / -i               Input video path (required)
--output / -o              Annotated output path (required)
--max-frames               Frames to process (default: 300)
--start                    Start time in seconds (default: 0.8)
--confidence               Detection threshold (default: 0.4)
--device                   cuda or cpu (default: cuda)
--rfdetr-model-id          Roboflow RF-DETR model ID
--yolo-model               Optional YOLO model path (jersey boxes)
--frame-skip               Process 1 frame out of N (default: 1)
--log-file                 Verbose run log path
```

## Outputs

One run produces:
- `result.mp4` (annotated source view)
- `result_tactical.mp4` (2D tactical projection)
- `log.txt` (verbose run metadata and frame stats)

---

## Proje Özeti (Türkçe)

Bu proje basketbol videosundan oyuncu/hakem tespiti, takip, ID stabilizasyonu,
opsiyonel forma numarası OCR ve taktik görünüm üretir.

Ana akış:
**RF-DETR -> BoT-SORT -> DINOv2 memory bank ReID -> homografi -> taktik harita**.

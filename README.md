# Basketball Game Tracker

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A computer vision pipeline for basketball footage analysis. Detects and tracks players and referees across frames with MCByte mask-assisted association, reads jersey numbers, and projects positions onto a 2D tactical court map.

---

## Demo

**Tracking view**

![Tracking demo](videos/demo_tracking.gif)

**Tactical bird's-eye view**

![Tactical demo](videos/demo_tactical.gif)

---

## Pipeline

```
Input Video
    ↓
RF-DETR  ──  player / referee / jersey detection
    ↓
MCByte  ──  multi-object tracking with mask-assisted association
    ↓
Jersey OCR (optional)  ──  YOLO jersey crops → PARSeq number reading
    ↓
Court keypoints + RANSAC homography
    ↓
Annotated video  +  Tactical view video
```

---

## Features

| Feature | Status |
|---|---|
| Player / referee detection (RF-DETR) | Done |
| Multi-object tracking (MCByte) | Done |
| Mask-assisted identity association (SAM + Cutie) | Done |
| Jersey number OCR (PARSeq) | Done |
| Court keypoint detection (fine-tuned YOLO-Pose) | Done |
| RANSAC homography + tactical projection | Done |
| Speed / distance analytics | Planned |

---

## Project Structure

```
APP/
├── __main__.py              # CLI entry point: python -m APP
├── assets/
│   └── basketball_court.png
└── helpers/
    ├── config.py            # Central pipeline configuration
    ├── pipeline.py          # End-to-end pipeline orchestration
    ├── mcbyte_tracker.py    # MCByte wrapper
    ├── rfdetr_detector.py   # RF-DETR detection via Roboflow Inference
    ├── jersey_detector.py   # PARSeq OCR + jersey number bank
    ├── court_utils.py       # Keypoints, homography, tactical drawing
    └── team_detector.py     # Team-color classifier
```

---

## Setup

**Requirements:** Python 3.11+, CUDA-capable GPU (recommended), Conda

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**RF-DETR runs via the Roboflow Inference SDK. Set your API key:**

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

If the GPU inference package is missing:

```bash
pip uninstall inference -y
pip install inference-gpu
```

**Required model asset:**

- `models/keypoints/yolo26l-fine-tuned.pt` — fine-tuned YOLO-Pose court keypoint model

---

## Run

```bash
python -m APP \
  --input  videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda
```

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Input video path |
| `--output` / `-o` | required | Annotated output video path |
| `--max-frames` | 300 | Number of frames to process |
| `--start` | 0.8 | Start offset in seconds |
| `--confidence` | 0.4 | Detection confidence threshold |
| `--device` | cuda | `cuda` or `cpu` |
| `--rfdetr-model-id` | see below | Roboflow RF-DETR model ID |
| `--frame-skip` | 1 | Process 1 out of every N frames |
| `--log-file` | auto | Verbose log path |

Default model ID: `basketball-player-detection-3-ycjdo/4`

---

## Outputs

Each run produces three files:

| File | Contents |
|---|---|
| `result.mp4` | Annotated source-view video |
| `result_tactical.mp4` | 2D tactical projection video |
| `log.txt` | Frame-level stats and run metadata |

---

## Branches

| Branch | Description |
|---|---|
| `main` | Current pipeline: RF-DETR + MCByte + SAM/Cutie masks |
| `sam2tracker` | Earlier experiment: SAM2-based instance tracking |

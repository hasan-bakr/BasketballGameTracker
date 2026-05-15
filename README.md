# Basketball Game Tracker

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/CUDA-recommended-76B900" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

A computer vision pipeline for basketball game footage. It detects and tracks players and referees frame-by-frame, reads jersey numbers, and projects every player's position onto a real-time 2D tactical court view.

Turkish documentation: [README_TR.md](README_TR.md)

---

## Demo

**Player tracking**

<video src="https://github.com/user-attachments/assets/0730f5e0-849e-4947-8b2b-817f860640a8" controls width="75%"></video>

**Tactical court view**

<video src="https://github.com/user-attachments/assets/cab1b58e-5c63-43d7-bd57-4b6f2614854a" controls width="30%"></video>

---

## Features

- **Multi-object tracking** — RF-DETR detections fed into MCByte with SAM + Cutie mask propagation for stable player identities through contact and occlusion
- **Jersey number OCR** — PARSeq reads jersey numbers from cropped regions; a voting bank stabilizes IDs across frames
- **Tactical projection** — court keypoints + RANSAC homography maps each player's foot position onto a scaled court image in real time
- **Duplicate suppression** — custom post-processing removes phantom and duplicate tracks caused by overlapping detections
- **Two output videos** — annotated tracking view and a separate tactical court view

---

## How It Works

```
Input video
    │
    ▼
RF-DETR detection          (players, referees, jersey regions — Roboflow)
    │
    ▼
MCByte tracking            (multi-object tracking with mask-aware association)
 + SAM / Cutie             (segmentation masks for cleaner association)
    │
    ▼
Jersey OCR                 (PARSeq on cropped jersey regions, voting bank)
    │
    ▼
Court keypoints            (Roboflow YOLO-Pose model → RANSAC homography)
    │
    ▼
Annotated video + Tactical court view
```

---

## Models and Libraries

| Component | Role | Link |
|---|---|---|
| RF-DETR | Detects players, referees, and jersey regions via Roboflow Inference. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | Local inference runtime — models are cached on first run. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Multi-object tracker extended with duplicate suppression and lost-track reuse. | [MCByte](https://github.com/tstanczyk95/McByte) |
| Segment Anything (SAM) | Generates initial segmentation masks from detector boxes. | [SAM](https://github.com/facebookresearch/segment-anything) |
| Cutie | Propagates player masks across frames for temporal consistency. | [Cutie](https://github.com/hkchengrex/Cutie) |
| PARSeq | Reads jersey numbers from cropped regions via scene text recognition. | [PARSeq](https://github.com/baudm/parseq) |
| OpenCV | Homography estimation, perspective transforms, video I/O, drawing. | [OpenCV](https://opencv.org/) |
| PyTorch | Runtime for SAM, Cutie, PARSeq, and YOLO models. | [PyTorch](https://pytorch.org/) |

> Advanced settings (model IDs, confidence thresholds, tracker parameters) are configured in `APP/helpers/config.py`.

---

## Setup

### 1. Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- PyTorch 2.x
- A free [Roboflow](https://roboflow.com) account and API key

### 2. Clone and install

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Set your API key

```bash
cp .env.example .env
```

Open `.env` and set:

```
ROBOFLOW_API_KEY=your_key_here
```

### 4. Clone external repos

```bash
# MCByte tracker
git clone https://github.com/tstanczyk95/McByte.git external/mcbyte

# Cutie (video object segmentation — goes inside MCByte)
git clone https://github.com/hkchengrex/Cutie.git external/mcbyte/mask_propagation/Cutie

# PARSeq (jersey OCR)
git clone https://github.com/baudm/parseq.git parseq
pip install -r parseq/requirements/core.txt
```

### 5. Download model weights

```bash
# SAM ViT-B weights
mkdir -p external/mcbyte/sam_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O external/mcbyte/sam_models/sam_vit_b_01ec64.pth

# Cutie weights
mkdir -p external/mcbyte/mask_propagation/Cutie/weights
wget https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth \
     -O external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth
```

Detection and keypoint models are downloaded automatically by Roboflow Inference on first run.

---

## Usage

```bash
python -m APP \
  --input  videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda \
  --max-frames 500
```

This produces:

```
videos/output/result.mp4           ← annotated tracking video
videos/output/result_tactical.mp4  ← tactical court view
videos/output/LOG.log              ← full run log
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Input video path (required) |
| `--output` | — | Output video path (required) |
| `--max-frames` | 300 | Maximum frames to process |
| `--start` | 0.8 | Start offset in seconds |
| `--frame-skip` | 1 | Process every Nth frame |
| `--device` | `cuda` | `cuda`, `cuda:0`, or `cpu` |
| `--confidence` | from config | Detector confidence threshold |
| `--debug` | off | Enable diagnostics (tracking, masks, keypoints) |

---

## Repository Structure

```
APP/
  __main__.py              CLI entry point
  assets/
    basketball_court.png   Tactical court template image
  helpers/
    config.py              All configuration — edit this to tune parameters
    pipeline.py            End-to-end pipeline orchestration
    mcbyte_tracker.py      MCByte wrapper with mask and suppression logic
    rfdetr_detector.py     RF-DETR / Roboflow Inference wrapper
    jersey_detector.py     PARSeq OCR and jersey voting bank
    court_utils.py         Homography, keypoint parsing, tactical rendering
```

---

## Notes

- All inference runs locally via Roboflow Inference — no frames are sent to the cloud after the model is cached on first run.
- The tactical view degrades gracefully when court keypoints are partially visible: the pipeline carries the last reliable homography and rejects updates that would cause large geometric jumps.

---

## Roadmap

- Team possession and event-level analytics
- Shot detection and ball-state integration
- Speed and distance metrics per player
- Exportable match reports

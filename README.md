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
- **Tactical projection** — YOLO-Pose court keypoints + RANSAC homography maps each player's foot position onto a scaled court image in real time
- **Duplicate suppression** — custom post-processing removes phantom and duplicate tracks caused by overlapping detections
- **Two output videos** — annotated tracking view and a separate tactical court view

---

## How It Works

```
Input video
    │
    ▼
RF-DETR detection          (players, referees, jersey regions)
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
Annotated video + Tactical court video
```

---

## Models and Libraries

### Detection and Tracking

| Component | Role | Link |
|---|---|---|
| RF-DETR | Detects players, referees, and jersey regions via Roboflow Inference. Default model: `basketball-player-detection-3-ycjdo/4`. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | Local inference runtime for RF-DETR. Requires `ROBOFLOW_API_KEY`. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Multi-object tracker extended with duplicate suppression, mask isolation, and lost-track reuse. | [MCByte](https://github.com/tstanczyk95/McByte) · [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.html) |
| ByteTrack | Tracking-by-detection association base used by MCByte. | [ByteTrack](https://github.com/FoundationVision/ByteTrack) |
| Segment Anything (SAM) | Generates initial masks from detector boxes. | [SAM](https://github.com/facebookresearch/segment-anything) |
| Cutie | Propagates player masks across frames for temporal consistency. | [Cutie](https://github.com/hkchengrex/Cutie) |

### Jersey OCR

| Component | Role | Link |
|---|---|---|
| PARSeq | Reads jersey numbers from cropped regions. | [PARSeq](https://github.com/baudm/parseq) |

### Court and Tactical View

| Component | Role | Link |
|---|---|---|
| Roboflow YOLO-Pose | Detects 33 court keypoints per frame via Roboflow Inference. Default model: `basketball-court-detection-2/19`. | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| OpenCV | Homography estimation, perspective transforms, video I/O, drawing. | [OpenCV](https://opencv.org/) |

### Supporting Libraries

| Component | Role |
|---|---|
| PyTorch | Runtime for SAM, Cutie, PARSeq, and YOLO models. |
| NumPy / SciPy | Array operations and Kalman-filter math used by MCByte. |
| Supervision | Annotation helpers and detection utilities. |
| Pillow | Converts jersey crops to PIL images for PARSeq. |
| python-dotenv | Loads `ROBOFLOW_API_KEY` from `.env`. |
| LAP / FilterPy | Assignment and filtering helpers for MCByte. |
| Hydra / OmegaConf | Configuration system for Cutie. |

---

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- PyTorch 2.x
- A Roboflow API key (free tier is sufficient)

The following files must be placed locally before running — they are not committed to Git:

```
external/mcbyte/                                                  ← MCByte repo
external/mcbyte/sam_models/sam_vit_b_01ec64.pth                  ← SAM weights
external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth  ← Cutie weights
parseq/                                                           ← PARSeq repo
```

---

## Installation

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create an environment file and add your Roboflow API key:

```bash
cp .env.example .env
# then edit .env:
ROBOFLOW_API_KEY=your_key_here
```

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

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input` | — | Input video path (required) |
| `--output` | — | Annotated output video path (required) |
| `--max-frames` | 300 | Maximum frames to process |
| `--start` | 0.8 | Start offset in seconds |
| `--frame-skip` | 1 | Process every Nth frame |
| `--device` | from config | `cuda`, `cuda:0`, or `cpu` |
| `--confidence` | from config | Detector confidence threshold |
| `--detector-iou` | from config | RF-DETR NMS IoU threshold |
| `--track-thresh` | from config | MCByte high-score tracking threshold |
| `--new-track-thresh` | from config | Minimum score to start a new track |
| `--keypoint-backend` | from config | `roboflow` or `local` |
| `--keypoint-model-id` | from config | Roboflow keypoint model ID override |
| `--keypoint-confidence` | from config | Keypoint detection confidence |
| `--debug` | off | Enable tracking, mask, and keypoint diagnostics |
| `--debug-lifecycle` | off | Log per-frame track lifecycle events |
| `--debug-suppression` | off | Log duplicate suppression decisions |
| `--tracking-report-json` | — | Write lifecycle summary to JSON |

---

## Repository Structure

```
APP/
  __main__.py              CLI entry point
  assets/
    basketball_court.png   Tactical court template image
  helpers/
    config.py              Central configuration dataclasses
    pipeline.py            End-to-end pipeline orchestration
    mcbyte_tracker.py      MCByte wrapper with mask and suppression logic
    rfdetr_detector.py     RF-DETR / Roboflow Inference wrapper
    jersey_detector.py     PARSeq OCR and jersey voting bank
    court_utils.py         Homography, keypoint parsing, tactical rendering
    team_detector.py       Team color clustering utilities
```

---

## Notes

- All detection and keypoint inference runs through Roboflow Inference locally — no frames are sent to the cloud after the model is cached.
- The tactical view degrades gracefully when court keypoints are partially visible: the pipeline carries the last reliable homography and rejects updates that cause large geometric jumps.
- Input/output videos, model weights, API keys, and external repos are all Git-ignored.

---

## Roadmap

- Team possession and event-level analytics
- Shot detection and ball-state integration
- Speed and distance metrics per player
- Exportable match reports

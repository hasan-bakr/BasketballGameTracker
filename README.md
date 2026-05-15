# Basketball Game Tracker

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-recommended-76B900)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Computer vision pipeline for basketball game footage. It detects players and referees, keeps player identities stable through crowded possessions, reads jersey numbers, and projects player positions onto a 2D tactical court view.

Turkish documentation: [README_TR.md](README_TR.md)

## Demo

**Main tracking view**

<video src="https://github.com/user-attachments/assets/0730f5e0-849e-4947-8b2b-817f860640a8" controls width="75%"></video>

**Tactical court view**

<video src="https://github.com/user-attachments/assets/cab1b58e-5c63-43d7-bd57-4b6f2614854a" controls width="30%"></video>

## Highlights

- RF-DETR based player, referee, and jersey-region detection
- MCByte multi-object tracking for stable player identities
- SAM + Cutie mask propagation for mask-assisted association
- Duplicate/phantom track suppression during occlusions
- Jersey number OCR with PARSeq and a voting based Re-ID bank
- YOLO-Pose court keypoints with RANSAC homography
- Stabilized tactical projection with keypoint carry, side-switch handling, and motion smoothing
- Separate annotated source video and tactical video outputs

## Models, Trackers, And Core Libraries

| Component | Where It Is Used | Link |
|---|---|---|
| RF-DETR | Player, referee, ball, and jersey-region detections through the configured Roboflow model id. The default project id is `basketball-player-detection-3-ycjdo/4`. | [RF-DETR](https://github.com/roboflow/rf-detr) |
| Roboflow Inference | Local/remote inference runtime for the RF-DETR detector. The `ROBOFLOW_API_KEY` value is loaded from `.env`. | [Roboflow Inference](https://github.com/roboflow/inference) |
| MCByte | Main multi-object tracker. The project uses the local `external/mcbyte/` checkout and extends the mask-aware association flow with duplicate suppression, mask isolation, and lost-track reuse logic. | [MCByte](https://github.com/tstanczyk95/McByte), [paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Stanczyk_No_Train_Yet_Gain_Towards_Generic_Multi-Object_Tracking_in_Sports_CVPRW_2025_paper.html) |
| ByteTrack | Baseline tracking-by-detection association used by MCByte. | [ByteTrack](https://github.com/FoundationVision/ByteTrack) |
| Segment Anything (SAM) | Creates initial masks from detector boxes before temporal propagation. The expected default weight is `external/mcbyte/sam_models/sam_vit_b_01ec64.pth`. | [Segment Anything](https://github.com/facebookresearch/segment-anything) |
| Cutie | Video object segmentation model that propagates player masks across frames. The expected default weight is `external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth`. | [Cutie](https://github.com/hkchengrex/Cutie) |
| PARSeq | Jersey-number OCR through the local `parseq/` checkout and STRHub modules. | [PARSeq](https://github.com/baudm/parseq) |
| Ultralytics YOLO Pose | Court keypoint detection for homography and tactical-view projection. The expected custom weight is `models/keypoints/yolo26l-fine-tuned.pt`. | [Ultralytics](https://github.com/ultralytics/ultralytics), [pose docs](https://docs.ultralytics.com/tasks/pose/) |
| OpenCV | Homography estimation, perspective transforms, video I/O, drawing, and tactical-view rendering. | [OpenCV](https://opencv.org/) |
| Supervision | Annotation helpers, colors, and detection utilities used by the visual output pipeline. | [Supervision](https://github.com/roboflow/supervision) |
| PyTorch | Deep-learning runtime used by the detector, tracker dependencies, SAM, Cutie, PARSeq, and YOLO models. | [PyTorch](https://pytorch.org/) |

Supporting runtime and utility dependencies:

| Component | Where It Is Used | Link |
|---|---|---|
| NumPy / SciPy | Array operations, geometry calculations, and MCByte Kalman-filter math. | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Pillow | Converts jersey crops to PIL images before PARSeq preprocessing. | [Pillow](https://python-pillow.org/) |
| python-dotenv | Loads `.env` values such as `ROBOFLOW_API_KEY` without committing secrets. | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| LAP / FilterPy | Assignment and filtering helpers used by the MCByte tracking stack. | [lap](https://github.com/gatagat/lap), [FilterPy](https://github.com/rlabbe/filterpy) |
| Hydra / OmegaConf | Configuration system used by the Cutie mask-propagation code. | [Hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/) |
| h5py / scikit-image / Hugging Face Hub | Model/checkpoint and image-processing utilities used by external model dependencies. | [h5py](https://www.h5py.org/), [scikit-image](https://scikit-image.org/), [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) |
| tqdm | Progress bars used by external model utilities and scripts. | [tqdm](https://github.com/tqdm/tqdm) |

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
Jersey OCR and ID stabilization
    |
    v
Court keypoints + homography
    |
    v
Annotated video + tactical court video
```

## Repository Structure

```text
APP/
  __main__.py              CLI entry point
  assets/
    basketball_court.png   Tactical court image
  helpers/
    config.py              Central configuration
    pipeline.py            End-to-end orchestration
    mcbyte_tracker.py      MCByte and mask wrapper
    rfdetr_detector.py     RF-DETR detector wrapper
    jersey_detector.py     PARSeq OCR and jersey bank
    court_utils.py         Homography and tactical rendering
    team_detector.py       Team color utilities
```

## Requirements

Recommended environment:

- Python 3.11+
- CUDA capable GPU
- PyTorch 2.x
- Roboflow API key for RF-DETR inference

Large model files and external research repos are not committed to Git. Place them locally before running the full pipeline.

Expected local assets:

```text
models/keypoints/yolo26l-fine-tuned.pt
external/mcbyte/
external/mcbyte/sam_models/sam_vit_b_01ec64.pth
external/mcbyte/mask_propagation/Cutie/weights/cutie-base-mega.pth
parseq/
```

## Installation

```bash
git clone https://github.com/hasan-bakr/Basketball-Game-Tracker.git
cd Basketball-Game-Tracker

conda create -n bgt python=3.11 -y
conda activate bgt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Then set:

```bash
ROBOFLOW_API_KEY=your_key_here
```

## Usage

```bash
python -m APP \
  --input videos/input/game.mp4 \
  --output videos/output/result.mp4 \
  --device cuda \
  --max-frames 300
```

The command creates:

```text
videos/output/result.mp4
videos/output/result_tactical.mp4
videos/output/LOG.log
```

## Useful CLI Flags

| Flag | Description |
|---|---|
| `--input` | Input video path |
| `--output` | Annotated output video path |
| `--max-frames` | Maximum processed frames |
| `--start` | Start offset in seconds |
| `--frame-skip` | Process every Nth frame |
| `--device` | `cuda`, `cuda:0`, or `cpu` |
| `--confidence` | Detector confidence override |
| `--detector-iou` | RF-DETR internal NMS IoU override |
| `--track-thresh` | MCByte high-score threshold |
| `--new-track-thresh` | Minimum score for starting a new track |
| `--debug` | Enable tracking, mask, and keypoint diagnostics |
| `--debug-lifecycle` | Log track lifecycle events |
| `--debug-suppression` | Log duplicate suppression and mask decisions |
| `--tracking-report-json` | Write lifecycle summary JSON |

## Notes

- Input videos, output videos, model weights, API keys, and local external repos are ignored by Git.
- The tactical view depends on court keypoints. The pipeline carries recent keypoints and rejects unstable homography jumps to reduce visual teleporting.

## Roadmap

- Team possession and event-level analytics
- Shot and ball-state integration
- Speed and distance metrics
- Exportable match reports

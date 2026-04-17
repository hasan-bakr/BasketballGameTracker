# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda create -n dsci python=3.12
conda activate dsci
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

PARSeq must be cloned alongside this repo (sister directory `parseq/`) — `jersey_detector.py` appends `../parseq` to `sys.path` at import time.

## Model Weights

Place under `models/` (project root, not inside `APP/`):
- `models/yolo/best_detection.pt` — custom YOLO (10 classes: ball, rim, number, player×5, referee, rim2)
- `models/sam2.1_hiera_base_plus.pt` — SAM2.1 checkpoint
- `models/keypoints/test_keypoint.pt` — YOLO-Pose court keypoint model (18 points)

## Run Commands

**Full pipeline:**
```bash
python -m APP --input videos/input/game.mp4 --output videos/output/result.mp4
```

Produces two output videos: `result.mp4` (annotated) and `result_tactical.mp4` (bird's-eye view).

**Debug YOLO detections only (no SAM2, fast):**
```bash
python -m APP.debug_detections \
    --input  videos/input/game.mp4 \
    --output videos/output/debug_det.mp4 \
    --start 150 --max-frames 150 --frame-skip 2
```

Useful for tuning court filter / size filter thresholds without waiting for SAM2.

**All main app options:**
```
--max-frames N      Frames to process (default: 300)
--start S           Start time in seconds (default: 0)
--batch-size N      SAM2 propagation batch size (default: 150)
--confidence F      YOLO threshold (default: 0.5)
--device            cuda or cpu (default: cuda)
--frame-skip N      Process 1 in N frames (default: 1)
--no-amp            Disable FP16 mixed precision
--log-file PATH     Verbose run log path (default: output_dir/log.txt)
```

**Logging behavior (`python -m APP`):**
- `log.txt` is truncated on every run (fresh log each execution).
- Log includes run metadata, output file paths, and batch/frame ranges:
  - `Run metadata: ...`
  - `Output files: annotated=... | tactical=...`
  - `Batch: processed frames A–B (source frames X–Y)`
- Terminal mirrors key run lines + SAM2 progress bar, while suppressing the noisy
  `Not enough SMs to use max_autotune_gemm mode` warning from terminal output.
- Annotated output HUD shows both processed frame index (`Frame: ...`) and source
  frame index (`Source Frame: ...`).

## Architecture Overview

The pipeline is orchestrated by `RobustSAM2Tracker` (in `APP/helpers/robust_sam2_tracker.py`), which is assembled from three mixins:

```
RobustSAM2Tracker
  ├── CourtFilterMixin    (court_filter.py)     — perspective-aware in/out-of-court bbox checks
  ├── SAM2PipelineMixin   (sam2_pipeline.py)    — batch extract → SAM2 init/propagate → mask conflict resolution
  └── PlayerDetectionMixin (player_detection.py) — keypoint detection, jersey OCR dispatch, smart re-detection
```

**Two-pass batch processing** (`process_video`):
1. Pass 1 — SAM2 propagation (uninterrupted GPU run, forward + `reverse=True`) → `batch_masks[frame_idx][obj_id] = mask`
2. Pass 2 — per-frame annotation: draw masks, jersey labels, keypoints, HUD text, write to MP4

**Double-buffer extraction**: While GPU runs Pass 1, the next batch of frames is extracted to `/dev/shm` in a background thread, hiding disk I/O latency.

## Key Design Decisions

**YOLO class IDs** (custom model, not COCO):
| ID | Class |
|----|-------|
| 0  | ball |
| 1  | rim |
| 2  | jersey number |
| 3–7 | player (A–E, different team/kit variants) |
| 8  | referee |
| 9  | rim2 |

`PLAYER_CLASSES = [3,4,5,6,7]`, `NUMBER_CLASS = 2`, `REFEREE_CLASS = 8`

**Court keypoints** — 18 points mapped to real-world court positions in `TACTICAL_KEYPOINTS` (court_utils.py). FT inner points (indices 8, 9, 16, 17) are excluded from boundary filtering. Points at frame borders (within `KEYPOINT_BORDER_MARGIN=30px`) are zeroed out.

**Homography** — RANSAC (`cv2.findHomography`) computed from ≥4 valid keypoints → projects camera coords → 300×161 px tactical court image. Temporally smoothed (α=0.6) with camera-transition detection (>50% of keypoints jump >150px).

**Jersey Re-ID flow**:
1. YOLO detects `class_id=2` (number bbox) in the frame
2. The center pixel of that bbox is checked against each SAM2 player mask
3. Matched mask → crop expanded by `JERSEY_EXPAND=1.5×` → PARSeq OCR
4. `JerseyReIDBank` votes: a number must appear ≥`JERSEY_LOCK_VOTES=3` times before it's committed; after `JERSEY_LOCK_VOTES` votes the ID is locked (no more OCR)

**SAM2 batching** — SAM2 processes video from JPEG frames on disk. Frames are capped at 1024px on the long edge before writing to disk. Between batches, previous masks are fed as prompts (`add_new_mask`) for continuity; degraded objects (mask area < 30% of initial) receive a fresh YOLO point prompt instead. Each batch is propagated in both directions: a normal forward pass plus a backward pass with `reverse=True` so frames before the seed frame can also receive masks.

**Mask conflict resolution** — pixel-wise argmax over logit score maps → non-overlapping masks. Masks filtered by: area < `MIN_MASK_AREA=100`, aspect ratio H/W < 0.30 (eliminates floor segments), area > 20% of frame (eliminates large background segments).

## SAM2 Propagation & Tracking Table

| Stage | Algorithm / Logic | Default params | Effect |
|---|---|---|---|
| Batch extraction | Frames are written as JPEGs for SAM2 input; long edge is resized to fit SAM2-friendly disk input size | `batch_size=150` (CLI), `frame_skip=1`, max long edge `1024px` | Reduces SAM2 input size and lets the predictor read a JPEG sequence from disk |
| Initial seed frame search | Scan first `10` frames, extend to `60` if jump-ball style clustering is detected; prefer frames with valid court keypoints, more in-court players, and larger player spread | `CLUSTER_TRIGGER_COUNT=3`, `CLUSTER_MIN_SPREAD_PX=100`, `CLUSTER_MAX_SCAN_FRAMES=60`, keypoint conf `0.3` | Chooses a cleaner initialization frame so propagation starts from a less crowded setup |
| SAM2 prompt creation | Each detection is converted into a single foreground point prompt at bbox center | point = `(cx, cy)` | Uses bbox centroid as the SAM2 seed point for both initialization and re-prompts |
| Player seed filtering | YOLO detections are filtered to player classes and checked with perspective-aware court boundaries before seeding | `PLAYER_CLASSES=[3,4,5,6,7]`, `PLAYER_MIN_CONF=0.8`, `KEYPOINT_BORDER_MARGIN=30px` | Avoids seeding spectators / bench / out-of-court boxes into SAM2 |
| Batch-to-batch continuation | Previous batch masks are resized to current SAM2 resolution and passed back with `add_new_mask(frame_idx=0)` | mask prompt for all non-degraded tracks | Preserves identity continuity across batches without reinitializing every object from scratch |
| Degradation detection | Track quality is monitored via mask area shrinkage relative to the first propagated area | `DEGRADATION_THRESHOLD=0.30`, `MIN_MASK_AREA=100` | Marks drifting / collapsing tracks before they poison the next batch |
| Degraded re-prompt | For degraded objects, nearest YOLO player detection to last mask centroid is used as a fresh point prompt; old mask is fallback | `REPROMPT_MAX_DIST_PX=100`, `PLAYER_MIN_CONF=0.8` | Re-centers SAM2 on the actual player and reduces drift accumulation |
| Propagation | SAM2 runs `propagate_in_video` twice over the batch: forward first, then backward with `reverse=True`; reverse results only fill missing object masks for a frame | mixed precision enabled unless `--no-amp`; `use_vos_optimized=True` in constructor | Covers frames on both sides of the seed prompt while keeping forward masks as the primary result |
| Conflict resolution | Competing object logits are merged by pixel-wise argmax ownership | `SAM2_MASK_LOGIT_THRESHOLD=0.00` | Produces mutually exclusive masks from overlapping SAM2 outputs |
| Mask geometry filtering | Post-argmax masks are dropped if too small, too horizontal, or too large relative to frame area | `MIN_MASK_AREA=100`, `MASK_MIN_ASPECT=0.30`, `MAX_MASK_AREA_RATIO=0.20` | Removes floor/background segments and obvious bad masks |
| Duplicate mask cleanup | Pairwise mask IoU pruning keeps only one mask when two tracks overlap too much | overlap threshold `IoU > 0.5`, lower object ID kept | Prevents duplicate tracks from surviving the same frame |
| New-player recovery | After each batch, YOLO checks the last frame for valid in-court players not already covered by current masks | overlap ratio `>0.3` means already tracked, `PLAYER_MIN_HEIGHT_PX=40`, `PLAYER_MIN_AREA_PX=900` | Adds players that entered late or were missed at initialization |

## Tactical Court Dimensions

`TACTICAL_WIDTH=300`, `TACTICAL_HEIGHT=161` (scaled from real court 28m × 15m). The `basketball_court.png` template in `APP/assets/` is the base image; if missing, a plain dark rectangle is used as fallback.

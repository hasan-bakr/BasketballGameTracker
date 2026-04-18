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
- `models/sam2.1_hiera_base_plus.pt` — default SAM2.1 checkpoint
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

**Debug simple SAM2 (eski mimari — court filter/memory yok):**
```bash
python -m APP.debug_sam2_simple \
    --input  videos/input/game.mp4 \
    --output videos/output/simple_out.mp4 \
    --max-frames 300 --batch-size 50 --start 70
```

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
- Log includes run metadata, output file paths, and batch/frame ranges.
- Terminal mirrors key run lines + SAM2 progress bar, while suppressing the noisy
  `Not enough SMs to use max_autotune_gemm mode` warning.
- Annotated output HUD shows both processed frame index (`Frame: ...`) and source
  frame index (`Source Frame: ...`).

## Architecture Overview

The pipeline is orchestrated by `RobustSAM2Tracker` (in `APP/helpers/robust_sam2_tracker.py`), assembled from three mixins:

```
RobustSAM2Tracker
  ├── CourtFilterMixin     (court_filter.py)      — perspective-aware in/out-of-court bbox checks
  ├── SAM2PipelineMixin    (sam2_pipeline.py)     — batch extract → SAM2 init/propagate → mask conflict resolution
  └── PlayerDetectionMixin (player_detection.py) — keypoint detection, jersey OCR dispatch, smart re-detection
```

**Two-pass batch processing** (`process_video`):
1. Pass 1 — SAM2 propagation (uninterrupted GPU run) → `batch_masks[frame_idx][obj_id] = mask`
2. Pass 2 — per-frame annotation: draw masks, jersey labels, keypoints, HUD text, write to MP4

**Double-buffer extraction**: While GPU runs Pass 1, the next batch of frames is extracted to `/dev/shm` in a background thread, hiding disk I/O latency.

## All Parameters (current values in code)

### Detection & Prompt Gating

| Parameter | Value | Description |
|---|---|---|
| `NEW_PROMPT_MIN_CONF` | `0.70` | YOLO min confidence to seed a new SAM2 object |
| `NEW_PROMPT_NMS_IOU` | `0.30` | NMS IoU threshold between new seeds in same frame |
| `NEW_PROMPT_OVERLAP_EXISTING` | `0.30` | Block new prompt if overlap with existing mask > this |
| `PLAYER_MIN_CONF` | `0.80` | Min YOLO conf for player class (general use) |
| `REFEREE_MIN_CONF` | `0.85` | Min YOLO conf for referee class |
| `PLAYER_EDGE_MARGIN_PX` | `60` | New prompts blocked if bbox is within 60px of any frame edge (prevents ghost tracks from players exiting frame) |
| `PLAYER_MIN_HEIGHT_PX` | `40` | Min bbox height to be seeded |
| `PLAYER_MIN_AREA_PX` | `900` | Min bbox area (w×h) to be seeded |

### Prompt Point Placement

| Parameter | Value | Description |
|---|---|---|
| chest-level point | `y = y1 + 0.30*h` | SAM2 foreground point placed at 30% from top of bbox (chest), avoids feet/floor bias |

### Mask Filtering (in `_resolve_mask_conflicts`)

| Parameter | Value | Description |
|---|---|---|
| `SAM2_MASK_LOGIT_THRESHOLD` | `0.15` | Pixels with logit score below this are not included in any mask |
| `MAX_MASK_AREA_RATIO` | `0.20` | Mask discarded if it covers >20% of frame area |
| aspect ratio check | **REMOVED** | H/W aspect ratio filter was removed — it incorrectly killed head-only occlusion masks |

> **Note:** `MASK_MIN_ASPECT = 0.30` constant still exists but is only used in new-prompt bbox gating, not in mask resolution.

### Jersey / Re-ID

| Parameter | Value | Description |
|---|---|---|
| `JERSEY_EXPAND` | `1.5` | Crop expansion multiplier around jersey bbox before OCR |
| `JERSEY_MIN_SIZE` | `10` | Min pixel size of jersey crop |
| `JERSEY_MIN_CONF` | `0.4` | Min PARSeq OCR confidence |
| `JERSEY_LOCK_VOTES` | `3` | Votes needed to commit a jersey number; locked after that |
| `JERSEY_SWAP_MAX_DIST_PX` | `140` | Block jersey ID swap if distance > 140px (implausible) |
| `JERSEY_UPDATE_INTERVAL` | `3` | Run jersey OCR every N frames |
| `DETECTION_UPDATE_INTERVAL` | `2` | Run new-player detection every N frames |

### Court Keypoints

| Parameter | Value | Description |
|---|---|---|
| `KEYPOINT_BORDER_MARGIN` | `30` | Zero out keypoints within 30px of frame border |
| `KEYPOINT_EDGE_BAND_RATIO` | `0.30` | Edge band ratio for court boundary projection |
| `KEYPOINT_STILL_PX` | `6` | Keypoint movement below this = "still" (no camera transition) |
| `KEYPOINT_STILL_FRAMES` | `5` | Frames of stillness required |

### Cross-Batch Memory

| Parameter | Value | Description |
|---|---|---|
| `CROSS_BATCH_MEMORY_BATCHES` | `3` | Number of past batches whose `maskmem_features` + `maskmem_pos_enc` tokens are kept on CPU and injected into the new batch at negative frame indices |

### Track Continuity (currently unused / gating disabled)

| Parameter | Value | Description |
|---|---|---|
| `MAX_TRACK_JUMP_PX` | `140` | Max allowed centroid jump in px between batches |
| `MAX_TRACK_JUMP_SCALE` | `1.8` | Max jump relative to object size |
| `PLAYER_FOOT_BAND_MARGIN` | `12` | Foot-band margin for bottom-boundary check |

### Init Frame Selection (jump-ball / clustering)

| Parameter | Value | Description |
|---|---|---|
| `CLUSTER_MIN_SPREAD_PX` | `100` | Avg pairwise distance below this → clustering detected |
| `CLUSTER_TRIGGER_COUNT` | `3` | Min players in tight cluster to trigger extended scan |
| `CLUSTER_MAX_SCAN_FRAMES` | `60` | Extended scan window when clustering detected |

## SAM2 Propagation & Tracking

| Stage | Algorithm / Logic | Default params | Effect |
|---|---|---|---|
| Batch extraction | Frames written as JPEGs to `/dev/shm`; long edge capped at 1024px | `batch_size=150`, `frame_skip=1` | SAM2 reads JPEG sequence from RAM disk |
| Initial seed frame | Scan first 10 frames (up to 60 if clustering); prefer frames with valid keypoints, more in-court players, larger spread | `CLUSTER_TRIGGER_COUNT=3`, `CLUSTER_MAX_SCAN_FRAMES=60` | Avoids seeding in crowded/jump-ball frames |
| SAM2 prompt | Single foreground point at chest level per detection | `point=(cx, y1+0.30*h)` | Better body anchoring vs feet-level |
| New prompt gating | High-conf + low-overlap check; edge margin check | `NEW_PROMPT_MIN_CONF=0.70`, `NEW_PROMPT_OVERLAP_EXISTING=0.30`, `PLAYER_EDGE_MARGIN_PX=60` | Prevents duplicate seeds, ghost tracks from frame-edge players |
| Cross-batch memory | Last 3 batches' maskmem tokens injected at negative frame indices | `CROSS_BATCH_MEMORY_BATCHES=3` | SAM2 has history context at batch start instead of cold-starting |
| Propagation | `propagate_in_video` forward pass with AMP | `use_vos_optimized=True`, FP16 unless `--no-amp` | Single clean forward pass |
| Conflict resolution | Pixel-wise argmax over logit maps → non-overlapping masks | `SAM2_MASK_LOGIT_THRESHOLD=0.15` | Mutually exclusive masks |
| Mask area filter | Masks < 100px or > 20% of frame discarded | `MAX_MASK_AREA_RATIO=0.20` | Removes noise and background bleed |
| Aspect ratio filter | **Disabled in mask resolution** (was killing occlusion masks) | removed | SAM2's natural tracking handles shape changes |
| New-player recovery | YOLO checks last frame for in-court players not covered by any mask | `NEW_PROMPT_OVERLAP_EXISTING=0.30`, `PLAYER_MIN_HEIGHT_PX=40` | Picks up late-entering or initially missed players |

## Disabled / Removed Systems

| System | Status | Reason |
|---|---|---|
| `_remove_overlapping_masks` | Disabled — returns input unchanged | SAM2's conflict resolution is sufficient; IoU-based removal caused false positives |
| `_apply_track_sanity_gating` | Disabled (call site commented out) | Was incorrectly rejecting valid masks during occlusion |
| `_detect_degraded_objects` / `_reprompt_degraded` | Removed entirely | Re-prompting degraded objects caused more drift than it fixed |
| Mask aspect ratio filter | Removed from `_resolve_mask_conflicts` | Was killing head-only / occluded player masks |

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

**Court keypoints** — 18 points mapped to real-world court positions in `TACTICAL_KEYPOINTS` (court_utils.py). FT inner points (indices 8, 9, 16, 17) excluded from boundary filtering. Points at frame borders (within `KEYPOINT_BORDER_MARGIN=30px`) are zeroed out.

**Keypoint groups:**
- `LEFT_KP_INDICES = [0,1,2,3,4,5]` — left sideline ("/" shape in camera view)
- `RIGHT_KP_INDICES = [10,11,12,13,14,15]` — right sideline ("\" shape)
- `TOP_KP_INDICES = [0,7,15]` — far baseline
- `BOTTOM_KP_INDICES = [5,6,10]` — near baseline

**Homography** — RANSAC (`cv2.findHomography`) from ≥4 valid keypoints → 300×161px tactical court image. Temporally smoothed (α=0.6) with camera-transition detection (>50% keypoints jump >150px).

**Jersey Re-ID flow**:
1. YOLO detects `class_id=2` (number bbox)
2. Center pixel checked against each SAM2 player mask
3. Matched mask → crop ×`JERSEY_EXPAND=1.5` → PARSeq OCR
4. `JerseyReIDBank` votes ≥`JERSEY_LOCK_VOTES=3` times → commit and lock

**SAM2 batching** — SAM2 reads JPEG frames from `/dev/shm`. Frames capped at 1024px long edge. Between batches, previous masks fed as `add_new_mask` prompts at frame 0 for continuity.

**Edge-exit ghost track prevention** — New prompts are rejected if the YOLO bbox touches within `PLAYER_EDGE_MARGIN_PX=60px` of any frame edge. This prevents SAM2 from tracking floor/background after a player exits the frame.

## Tactical Court Dimensions

`TACTICAL_WIDTH=300`, `TACTICAL_HEIGHT=161` (scaled from real court 28m × 15m). The `basketball_court.png` template in `APP/assets/` is the base image; plain dark rectangle fallback if missing.

## Debug Scripts

| Script | Command | Purpose |
|---|---|---|
| `APP.debug_detections` | `python -m APP.debug_detections --input ... --output ...` | YOLO-only, no SAM2. Shows court filter pass/fail per bbox. Fast. |
| `APP.debug_sam2_simple` | `python -m APP.debug_sam2_simple --input ... --output ...` | Original (Jan 2026) simple SAM2: YOLO bbox init, no court filter, no memory bank. Baseline comparison. |
| `APP.debug_prompt` | `python -m APP.debug_prompt --input ... --output ...` | Shows prompt events and keypoint overlay. |

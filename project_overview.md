---
name: Project Overview
description: BasketballGameTracker - AI basketball video analysis with player tracking, jersey OCR, court mapping, tactical view
type: project
---

Basketball game video analysis pipeline. CLI entry point: `python -m APP --input ... --output ...`

**Pipeline:** Input Video → YOLO Detection (custom classes: player, ball, rim, number) → SAM2 Video Propagation (memory bank, batch processing) → IoU Re-ID → Jersey OCR (PARSeq, async) → Court Keypoint Detection (YOLO-Pose, 18 points) → Homography → Tactical Bird's-eye View

**Key modules in APP/:**
- `__main__.py` — CLI entry point (argparse)
- `helpers/robust_sam2_tracker.py` — Main pipeline (RobustSAM2Tracker class, ~650 lines)
- `helpers/yolo_detector.py` — YOLO wrapper (custom `best_detection.pt`, basketball classes)
- `helpers/jersey_detector.py` — PARSeq OCR + JerseyReIDBank (voting system)
- `helpers/court_utils.py` — Court constants, compute_homography, draw_keypoints_on_frame, draw_tactical_view
- `helpers/visualization.py` — draw_masks_with_ids (standalone)
- `helpers/court_keypoint_detector.py` — YOLO-Pose 18-point landmark detection + EMA
- `helpers/homography_transformer.py` — RANSAC homography + temporal blending
- `helpers/sam_helper.py` — SAM2 ONNX segmentation (single-frame, not used in main pipeline)
- `helpers/generate_tactical_view.py` — Standalone tactical view generator
- `helpers/manual_court_selector.py` — Interactive OpenCV GUI for court point selection
- `helpers/rfdetr_detector.py` — RF-DETR ONNX alternative detector
- `helpers/download_court_model.py` — Court model download utility
- `assets/basketball_court.png` — Court template image for tactical view

**Models (in models/, gitignored):**
- `models/yolo/best_detection.pt` — Custom YOLO (players, ball, rim, jersey number)
- `models/sam2.1_hiera_tiny.pt` / `_small.pt` / `_base_plus.pt` — SAM2.1 checkpoints
- `models/keypoints/test_keypoint.pt` — Court keypoint YOLO-Pose

**Known issues / decisions:**
- `vos_optimized=False` — SAM2's internal torch.compile (VOS mode) causes CUDA graph buffer conflict during stateful propagation; disabled for stability
- `torch.compile` removed from tracker entirely (CUDA graph incompatibility with stateful SAM2 API)
- SAM2 video propagation (memory bank) cannot be ONNX-exported; main pipeline runs pure PyTorch

**Repo state:** Clean, public-ready. GitHub: hasan-bakr/BasketballGameTracker

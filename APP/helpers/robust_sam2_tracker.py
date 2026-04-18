"""
robust_sam2_tracker.py
======================
End-to-end basketball tracking pipeline:
  - YOLO detection (players, ball, rim, jersey numbers)
  - SAM2 video propagation with memory bank
  - IoU-based re-identification across batches
  - Jersey number OCR via PARSeq (async)
  - Court keypoint detection + RANSAC homography
  - Tactical bird's-eye view generation

Modül yapısı:
  court_filter.py      → CourtFilterMixin   (saha içi filtre + geometry helpers)
  sam2_pipeline.py     → SAM2PipelineMixin  (batch extract/init/propagate)
  player_detection.py  → PlayerDetectionMixin (keypoint, referee, jersey, re-detect)
  robust_sam2_tracker.py → RobustSAM2Tracker (orchestration + __init__ + process_video)
"""

import os
import gc
import shutil
import sys

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

try:
    torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from sam2.build_sam import build_sam2_video_predictor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from APP.helpers.yolo_detector import YoloDetector
from APP.helpers.jersey_detector import JerseyDetector, JerseyReIDBank
from APP.helpers.court_utils import (
    TACTICAL_WIDTH, TACTICAL_HEIGHT,
    DEFAULT_KEYPOINT_MODEL, DEFAULT_COURT_IMAGE,
    compute_homography, draw_keypoints_on_frame, draw_tactical_view,
)
from APP.helpers.visualization import draw_masks_with_ids
from APP.helpers.court_filter     import CourtFilterMixin
from APP.helpers.sam2_pipeline    import SAM2PipelineMixin
from APP.helpers.player_detection import PlayerDetectionMixin


# ── Referee visualization helper ──────────────────────────────────────────────

def _draw_referee_masks(frame: np.ndarray, ref_masks: Dict[int, np.ndarray]) -> np.ndarray:
    """SAM2 hakem maskelerini sarı-turuncu renk ve 'REF {id}' etiketi ile çizer."""
    if not ref_masks:
        return frame
    REF_COLOR = (0, 200, 255)    # BGR: turuncu-sarı
    ALPHA     = 0.45

    result = frame.copy()
    overlay = frame.copy()

    for obj_id, mask in ref_masks.items():
        if mask is None or not mask.any():
            continue
        overlay[mask] = REF_COLOR

        # Etiket — maskenin üst noktasına
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.min()) - 6

        label = f"REF {obj_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx = cx - tw // 2
        ly = max(cy, th + 4)
        cv2.rectangle(result, (lx - 2, ly - th - 3), (lx + tw + 2, ly + 1), REF_COLOR, -1)
        cv2.putText(result, label, (lx, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, ALPHA, result, 1 - ALPHA, 0, result)
    return result



class RobustSAM2Tracker(CourtFilterMixin, SAM2PipelineMixin, PlayerDetectionMixin):
    """SAM2 video tracker with IoU re-ID, jersey OCR, and tactical view."""

    # ── YOLO class IDs ────────────────────────────────────────────────────────
    PLAYER_CLASSES = [3, 4, 5, 6, 7]
    NUMBER_CLASS   = 2
    REFEREE_CLASS  = 8
    RIM_CLASS      = 9

    # ── Thresholds & intervals ────────────────────────────────────────────────
    JERSEY_EXPAND             = 1.5
    JERSEY_MIN_SIZE           = 10
    JERSEY_MIN_CONF           = 0.4
    JERSEY_LOCK_VOTES         = 3
    JERSEY_SWAP_MAX_DIST_PX   = 140
    PLAYER_MIN_CONF           = 0.8
    NEW_PROMPT_MIN_CONF       = 0.70
    NEW_PROMPT_NMS_IOU        = 0.30
    NEW_PROMPT_OVERLAP_EXISTING = 0.30
    REFEREE_MIN_CONF          = 0.85
    JERSEY_UPDATE_INTERVAL    = 3
    DETECTION_UPDATE_INTERVAL = 2
    SAM2_MASK_LOGIT_THRESHOLD = 0.15
    MAX_MASK_AREA_RATIO       = 0.20   # mask frame alanının %20'sinden büyük olamaz
    MASK_MIN_ASPECT           = 0.30   # mask bbox H/W oranı — zemin segm. yatay (düşük) oyuncu dikey
    KEYPOINT_BORDER_MARGIN    = 30
    KEYPOINT_EDGE_BAND_RATIO  = 0.30
    KEYPOINT_STILL_PX         = 6
    KEYPOINT_STILL_FRAMES     = 5
    PLAYER_MIN_HEIGHT_PX      = 40
    PLAYER_MIN_AREA_PX        = 900
    PLAYER_EDGE_MARGIN_PX     = 60   # bbox kenardan bu kadar px içeride olmalı (seed gating)
    PLAYER_FOOT_BAND_MARGIN   = 12
    MAX_TRACK_JUMP_PX         = 140
    MAX_TRACK_JUMP_SCALE      = 1.8

    # ── Clustering / jump-ball detection ─────────────────────────────────────
    # Başlangıçta oyuncular top atışı için bir aradaysa daha iyi bir frame aranır.
    CLUSTER_MIN_SPREAD_PX   = 100  # avg pairwise mesafe bu değerin altındaysa cluster var
    CLUSTER_TRIGGER_COUNT   = 3    # kaç oyuncu aynı bölgedeyse cluster sayılır
    CLUSTER_MAX_SCAN_FRAMES = 60   # clustering varsa tarama penceresini bu kadar genişlet

    # ── Keypoint indices for each court boundary ──────────────────────────────
    # FT inner keypoints (8,9,16,17) excluded from all boundary checks.
    LEFT_KP_INDICES   = [0, 1, 2, 3, 4, 5]        # taktik x=0   → sol baseline
    RIGHT_KP_INDICES  = [10, 11, 12, 13, 14, 15]   # taktik x=MAX → sağ baseline
    TOP_KP_INDICES    = [0, 7, 15]                  # taktik y=0   → uzak baseline
    BOTTOM_KP_INDICES = [5, 6, 10]                  # taktik y=MAX → yakın baseline

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        sam2_checkpoint: str = "models/sam2.1_hiera_base_plus.pt",
        yolo_path: str = "models/yolo/best_detection.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        redetect_interval: int = 30,
        use_amp: bool = True,
        use_vos_optimized: bool = True,
        keypoint_model_path: str = None,
        court_image_path: str = None,
    ):
        print("Loading Robust SAM2 Tracker...")
        print(f"  device={device} | AMP={'ON (FP16)' if use_amp else 'OFF (FP32)'} "
              f"| VOS-opt={'ON' if use_vos_optimized else 'OFF'} | conf={confidence_threshold}")

        self.device               = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.redetect_interval    = redetect_interval
        self.use_amp              = use_amp
        self.use_vos_optimized    = use_vos_optimized

        # ── Models ───────────────────────────────────────────────────────────
        self.yolo = YoloDetector(model_path=yolo_path, device=device)

        self.predictor = build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=device,
            vos_optimized=use_vos_optimized,
        )

        self.jersey_detector = JerseyDetector(device=device)
        self.jersey_bank     = JerseyReIDBank()

        kp_path = keypoint_model_path or DEFAULT_KEYPOINT_MODEL
        self.kp_model = None
        if os.path.exists(kp_path):
            self.kp_model = YOLO(kp_path)
        else:
            print(f"Warning: keypoint model not found at {kp_path}")

        # Court base image for tactical view
        court_path = court_image_path or DEFAULT_COURT_IMAGE
        if os.path.exists(court_path):
            self.court_img = cv2.imread(court_path)
            self.court_img = cv2.resize(self.court_img, (TACTICAL_WIDTH, TACTICAL_HEIGHT))
        else:
            self.court_img = np.ones((TACTICAL_HEIGHT, TACTICAL_WIDTH, 3), dtype=np.uint8) * 40
            cv2.rectangle(self.court_img, (0, 0),
                          (TACTICAL_WIDTH - 1, TACTICAL_HEIGHT - 1), (255, 255, 255), 1)
            cv2.line(self.court_img,
                     (TACTICAL_WIDTH // 2, 0), (TACTICAL_WIDTH // 2, TACTICAL_HEIGHT),
                     (255, 255, 255), 1)

        # ── Keypoint / homography state ───────────────────────────────────────
        self.last_good_keypoints      = None
        self.last_H                   = None
        self.homography_success_count = 0

        # ── Tracking state ────────────────────────────────────────────────────
        self.tracked_objects: Dict[int, dict] = {}
        self.next_obj_id       = 1
        self.frame_size        = None
        self.locked_jersey_ids: set = set()

        np.random.seed(42)
        self.colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(100)]

        # ── Caches ────────────────────────────────────────────────────────────
        self._last_referee_dets: List[dict] = []
        self._last_keypoints = (
            np.zeros((18, 2), dtype=np.float32),
            np.zeros(18, dtype=np.float32),
            None,
        )
        self._prev_detected_keypoints = np.zeros((18, 2), dtype=np.float32)
        self._keypoint_stationary_counts = np.zeros(18, dtype=np.int32)

        # ── Async jersey OCR ──────────────────────────────────────────────────
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._jersey_future: Optional[Future] = None

        # ── Debug prompt logging (activated via process_video debug_prompts=True) ──
        self._debug_prompt_events: Optional[list] = None
        self._current_batch_offset: int = 0

        self._warmup_models()
        print("Robust SAM2 Tracker ready.")

    # ── Warmup ────────────────────────────────────────────────────────────────

    @staticmethod
    def _swap_ids_in_remaining_batch(
        batch_masks: Dict[int, Dict[int, np.ndarray]],
        start_frame_idx: int,
        id_a: int,
        id_b: int,
    ) -> None:
        """Swap two object IDs from the current frame to the end of the batch."""
        for frame_idx in sorted(k for k in batch_masks.keys() if k >= start_frame_idx):
            frame_masks = batch_masks.get(frame_idx, {})
            mask_a = frame_masks.pop(id_a, None)
            mask_b = frame_masks.pop(id_b, None)
            if mask_b is not None:
                frame_masks[id_a] = mask_b
            if mask_a is not None:
                frame_masks[id_b] = mask_a

    def _sync_last_masks_from_batch(
        self,
        batch_masks: Dict[int, Dict[int, np.ndarray]],
        obj_ids: List[int],
    ) -> None:
        """Refresh tracked_objects.last_mask from the last available frame in batch."""
        if not batch_masks:
            return
        last_frame_idx = max(batch_masks.keys())
        last_masks = batch_masks.get(last_frame_idx, {})
        for obj_id in obj_ids:
            if obj_id in self.tracked_objects and obj_id in last_masks:
                self.tracked_objects[obj_id]["last_mask"] = last_masks[obj_id]

    def _apply_track_sanity_gating(
        self,
        frame_masks: Dict[int, np.ndarray],
        keypoints_xy: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Dict[int, np.ndarray]:
        """Reject implausible player masks and preserve the last known good mask."""
        filtered_masks: Dict[int, np.ndarray] = {}

        for obj_id, mask in frame_masks.items():
            obj = self.tracked_objects.get(obj_id)
            if obj is None:
                filtered_masks[obj_id] = mask
                continue

            if obj.get("is_referee", False):
                filtered_masks[obj_id] = mask
                continue

            centroid = self._mask_centroid(mask)
            bbox = self._mask_to_bbox(mask)
            if centroid is None or bbox is None:
                continue

            last_good_centroid = obj.get("last_good_centroid")
            last_good_mask = obj.get("last_good_mask")
            last_good_bbox = self._mask_to_bbox(last_good_mask) if last_good_mask is not None else None

            reject_reason = None
            if last_good_centroid is not None:
                dx = centroid[0] - last_good_centroid[0]
                dy = centroid[1] - last_good_centroid[1]
                jump = float((dx * dx + dy * dy) ** 0.5)
                last_h = (
                    max(last_good_bbox[3] - last_good_bbox[1], 1)
                    if last_good_bbox is not None else self.PLAYER_MIN_HEIGHT_PX
                )
                max_jump = max(self.MAX_TRACK_JUMP_PX, last_h * self.MAX_TRACK_JUMP_SCALE)
                if jump > max_jump:
                    reject_reason = f"teleport({jump:.0f}px)"

            if reject_reason:
                obj["last_mask"] = last_good_mask
                print(f"  [gating] ID {obj_id}: rejected {reject_reason}")
                continue

            obj["last_good_mask"] = mask
            obj["last_good_centroid"] = centroid
            obj["last_mask"] = mask
            filtered_masks[obj_id] = mask

        return filtered_masks

    # ── Debug prompt logging ──────────────────────────────────────────────────

    def _record_prompt(self, local_frame_idx: int, obj_id: int, bbox, source: str):
        """Prompt olayını debug log'una ekle. Debug modu kapalıysa no-op."""
        if self._debug_prompt_events is None:
            return
        abs_frame = self._current_batch_offset + local_frame_idx
        if bbox is not None:
            x1, y1, x2, y2 = [float(c) for c in bbox]
            pt       = [round((x1 + x2) / 2, 1), round(y1 + 0.55 * (y2 - y1), 1)]
            bbox_int = [int(x1), int(y1), int(x2), int(y2)]
        else:
            pt       = None
            bbox_int = None
        self._debug_prompt_events.append({
            "abs_frame": abs_frame,
            "obj_id":    obj_id,
            "source":    source,
            "bbox":      bbox_int,
            "point":     pt,
        })

    def _draw_prompt_events(self, frame: np.ndarray, abs_idx: int) -> np.ndarray:
        """Verilen frame'deki tüm prompt olaylarını frame üzerine çiz."""
        if not self._debug_prompt_events:
            return frame
        SOURCE_COLORS = {
            "init":              (255, 255, 255),   # beyaz  — ilk seed
            "init_ref":          (255, 200, 255),   # açık pembe — hakem seed
            "continue_mask":     (200, 200, 255),   # açık mavi — batch devam mask
            "reprompt":          (0,   140, 255),   # turuncu — degrade re-prompt
            "reprompt_fallback": (80,  80,  255),   # kırmızımsı — fallback
        }
        for ev in self._debug_prompt_events:
            if ev["abs_frame"] != abs_idx:
                continue
            color  = SOURCE_COLORS.get(ev["source"], (180, 180, 180))
            bbox   = ev.get("bbox")
            pt     = ev.get("point")
            obj_id = ev["obj_id"]
            src    = ev["source"]
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            if pt:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(frame, (px, py), 10, color, -1)
                cv2.circle(frame, (px, py), 10, (0, 0, 0), 2)
                label = f"ID{obj_id}[{src}]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (px + 14, py - th - 4),
                              (px + 16 + tw, py + 2), (0, 0, 0), -1)
                cv2.putText(frame, label, (px + 16, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return frame

    def _warmup_models(self):
        """Run a dummy inference on each model to JIT-compile CUDA kernels."""
        if self.device != 'cuda':
            return
        print("  Warming up models on GPU...", end=" ", flush=True)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.detect(dummy, confidence_threshold=0.5)
        if self.kp_model is not None:
            self.kp_model.predict(dummy, conf=0.5, verbose=False,
                                  half=True, device=self.device)
        print("done.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: int = 300,
        batch_size: int = 60,
        start_sec: float = 0.0,
        frame_skip: int = 1,
        show_preview: bool = False,
        debug_prompts: bool = False,
    ):
        """Process a video and write annotated output + tactical view video.

        Args:
            video_path:  Path to input video file.
            output_path: Path for the annotated output video.
            max_frames:  Maximum number of frames to process.
            batch_size:  Frames per SAM2 propagation batch.
            start_sec:   Start time in seconds (skip intro etc.).
            frame_skip:  Process 1 out of every N frames.
        """
        print(f"\nProcessing: {video_path}")

        _debug_frames_dir: Optional[str] = None
        if debug_prompts:
            self._debug_prompt_events = []
            self._current_batch_offset = 0
            _debug_frames_dir = os.path.splitext(output_path)[0] + "_prompt_frames"
            os.makedirs(_debug_frames_dir, exist_ok=True)
            print(f"  [debug] Prompt event logging ACTIVE → frames: {_debug_frames_dir}")

        cap             = None
        writer          = None
        tactical_writer = None
        inference_state = None
        interrupted     = False

        # ── Double-buffer temp dirs (GPU propagation ile frame extraction'ı parallel ──
        def _make_temp(suffix):
            base = "/dev/shm" if os.path.exists("/dev/shm") else "."
            return f"{base}/sam2_tracker_{os.getpid()}_{suffix}"

        use_double_buffer = batch_size <= 60
        temp_dirs = [_make_temp("a"), _make_temp("b")] if use_double_buffer else [_make_temp("a")]
        cur = 0   # aktif buffer index

        base, ext = os.path.splitext(output_path)
        tactical_output_path = f"{base}_tactical{ext}"

        tactical_display_w = 600
        tactical_display_h = int(TACTICAL_HEIGHT * (tactical_display_w / TACTICAL_WIDTH))

        frame_idx = 0

        try:
            cap = cv2.VideoCapture(video_path)
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = source_fps / frame_skip

            source_start_frame = int(start_sec * source_fps)
            if source_start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)

            total_frames = min(
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - source_start_frame,
                max_frames,
            )
            if source_start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)

            ret, first_frame = cap.read()
            if not ret:
                print("Error: could not read video.")
                return
            self.frame_size = first_frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)

            planned_processed = (total_frames + frame_skip - 1) // frame_skip
            print(
                "Run metadata: "
                f"source_fps={source_fps:.3f}, output_fps={fps:.3f}, "
                f"source_start_frame={source_start_frame}, source_end_frame~={source_start_frame + total_frames}, "
                f"max_source_frames={total_frames}, frame_skip={frame_skip}, "
                f"planned_processed_frames~={planned_processed}"
            )
            print(f"Output files: annotated={output_path} | tactical={tactical_output_path}")

            # ── İlk batch'i senkron olarak çek ──────────────────────────────
            frames = self._extract_batch(
                cap, temp_dirs[cur], batch_size, frame_skip=frame_skip
            )
            if not frames:
                return

            # Arka plan extraction future (başlangıçta yok)
            import concurrent.futures as _cf
            next_future = None

            while frame_idx < total_frames:
                batch_end = min(frame_idx + batch_size, total_frames)
                source_batch_start = source_start_frame + (frame_idx * frame_skip)
                source_batch_end = source_start_frame + ((batch_end - 1) * frame_skip)
                print(
                    f"\nBatch: processed frames {frame_idx}–{batch_end - 1} "
                    f"(source frames {source_batch_start}–{source_batch_end})"
                )

                # Mevcut batch temp_dir
                temp_dir = temp_dirs[cur]
                nxt      = 1 - cur if use_double_buffer else cur

                # Arka plan: bir sonraki batch frame'lerini çek (GPU propagation sırasında)
                # next_future burada planlanır; sonucu propagation bittikten sonra alınır.


                if writer is None:
                    h, w   = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    tactical_writer = cv2.VideoWriter(
                        tactical_output_path, fourcc, fps,
                        (tactical_display_w, tactical_display_h),
                    )

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                    inference_state = self.predictor.init_state(
                        video_path=temp_dir,
                        offload_video_to_cpu=True,
                        async_loading_frames=True,
                    )

                self._current_batch_offset = frame_idx
                if frame_idx == 0:
                    self._initialize_objects(inference_state, frames, temp_dir)
                else:
                    self._continue_tracking(inference_state, temp_dir,
                                            first_frame=frames[0])

                # ── Pre-scan: propagasyondan önce batch boyunca yeni oyuncu ekle ──
                # Batch=15 ile test edildi; init_state overhead nedeniyle yavaş.
                # Büyük batch'lerde prescan faydalı olabilir — gerekirse aktif et.
                # self._prescan_batch(inference_state, frames, temp_dir, scan_interval=15)

                # ── Arka plan: bir sonraki batch frame'lerini çek ────────────
                # GPU propagation başlamadan HEMEN önce başlatılır; disk I/O ile
                # GPU işi örtüşür → batch arası bekleme ortadan kalkar.
                if use_double_buffer:
                    next_future = self._executor.submit(
                        self._extract_batch,
                        cap, temp_dirs[nxt], batch_size, frame_skip
                    )
                else:
                    next_future = None

                # ── Pass 1: SAM2 propagation (uninterrupted GPU run) ─────────
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                    batch_masks = self._propagate_batch(inference_state, frames)


                # ── Pass 2: per-frame annotation & writing ───────────────────
                for i, frame in enumerate(frames):
                    masks   = batch_masks.get(i, {})
                    abs_idx = frame_idx + i

                    if abs_idx % self.DETECTION_UPDATE_INTERVAL == 0:
                        keypoints_xy, confidences, H = self._detect_keypoints(frame)
                        self._last_keypoints = (keypoints_xy, confidences, H)
                    else:
                        keypoints_xy, confidences, H = self._last_keypoints

                    # _apply_track_sanity_gating devre dışı
                    batch_masks[i] = masks

                    if abs_idx % self.JERSEY_UPDATE_INTERVAL == 0:
                        # Jersey OCR sadece player (non-referee) maskeleri için
                        player_masks_for_ocr = {
                            oid: m for oid, m in masks.items()
                            if not self.tracked_objects.get(oid, {}).get("is_referee", False)
                        }
                        if player_masks_for_ocr:
                            swap_pairs = self._detect_jerseys(
                                frame.copy(),
                                player_masks_for_ocr.copy(),
                            )
                            for id_a, id_b in swap_pairs:
                                self._swap_ids_in_remaining_batch(batch_masks, i, id_a, id_b)
                                self._sync_last_masks_from_batch(batch_masks, [id_a, id_b])

                    n_detected = int((keypoints_xy > 0).all(axis=1).sum())

                    # Maskeleri player / referee olarak ayır
                    player_masks = {
                        oid: m for oid, m in masks.items()
                        if not self.tracked_objects.get(oid, {}).get("is_referee", False)
                    }
                    ref_masks = {
                        oid: m for oid, m in masks.items()
                        if self.tracked_objects.get(oid, {}).get("is_referee", False)
                    }

                    result = (
                        draw_masks_with_ids(frame, player_masks, self.colors, self.jersey_bank)
                        if player_masks else frame.copy()
                    )
                    result = _draw_referee_masks(result, ref_masks)
                    result = draw_keypoints_on_frame(result, keypoints_xy, confidences)
                    if debug_prompts:
                        result = self._draw_prompt_events(result, abs_idx)
                        if _debug_frames_dir and any(
                            ev["abs_frame"] == abs_idx
                            for ev in self._debug_prompt_events
                        ):
                            sources = "_".join(sorted({
                                ev["source"] for ev in self._debug_prompt_events
                                if ev["abs_frame"] == abs_idx
                            }))
                            fname = f"frame_{abs_idx:05d}_{sources}.jpg"
                            cv2.imwrite(
                                os.path.join(_debug_frames_dir, fname),
                                result,
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )

                    player_feet, player_jerseys = self._masks_to_feet(masks)

                    cv2.putText(result, f"Frame: {abs_idx}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    source_abs_idx = source_start_frame + (abs_idx * frame_skip)
                    cv2.putText(result, f"Source Frame: {source_abs_idx}", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(result,
                                f"Objects: {len(self.tracked_objects)} | KP: {n_detected}/18",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(result,
                                f"Homography: {'OK' if H is not None else 'FAIL'}",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0) if H is not None else (0, 0, 255), 2)
                    cv2.putText(result,
                                f"Jerseys: {len(self.jersey_bank.obj_to_jersey)}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    writer.write(result)

                    tactical = draw_tactical_view(
                        self.court_img, keypoints_xy, H, player_feet, player_jerseys
                    )
                    tactical_writer.write(
                        cv2.resize(tactical, (tactical_display_w, tactical_display_h))
                    )

                    if show_preview:
                        try:
                            max_w   = 1280
                            display = (
                                cv2.resize(
                                    result,
                                    (max_w, int(result.shape[0] * max_w / result.shape[1]))
                                )
                                if result.shape[1] > max_w else result
                            )
                            cv2.imshow("SAM2 Tracker", display)
                            cv2.imshow("Tactical View",
                                       cv2.resize(tactical, (tactical_display_w, tactical_display_h)))
                            cv2.waitKey(1)
                        except cv2.error:
                            pass

                if interrupted:
                    break

                self._smart_redetection(frames[-1], batch_masks.get(len(frames) - 1, {}))

                self._save_batch_memory(inference_state, len(frames))
                self.predictor.reset_state(inference_state)
                inference_state = None
                frame_idx += len(frames)

                # ── Bir sonraki batch sonucunu al ────────────────────────────
                if use_double_buffer:
                    if next_future is not None:
                        frames = next_future.result()
                        next_future = None
                        if not frames:
                            break
                        cur = nxt   # buffer'ları değiştir
                    else:
                        break
                else:
                    if frame_idx < total_frames:
                        frames = self._extract_batch(
                            cap, temp_dir, batch_size, frame_skip=frame_skip
                        )
                        if not frames:
                            break
                    else:
                        break

                gc.collect()
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted by user. Cleaning up safely...")

        finally:
            if inference_state is not None:
                try:
                    self.predictor.reset_state(inference_state)
                except Exception as e:
                    print(f"  Warning: predictor state reset failed: {e}")

            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if tactical_writer is not None:
                tactical_writer.release()

            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=2)

        if not interrupted:
            print(f"\nDone! Output:   {output_path}")
            print(f"     Tactical:  {tactical_output_path}")
            print(f"     Objects tracked:    {len(self.tracked_objects)}")
            print(f"     Homography success: {self.homography_success_count}")

        if debug_prompts and self._debug_prompt_events is not None:
            import json as _json
            base_out = os.path.splitext(output_path)[0]
            json_path = base_out + "_prompt_events.json"
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(self._debug_prompt_events, f, indent=2, ensure_ascii=False)
            print(f"     Prompt events: {json_path}  ({len(self._debug_prompt_events)} events)")
            if _debug_frames_dir:
                n = len({ev["abs_frame"] for ev in self._debug_prompt_events})
                print(f"     Prompt frames: {_debug_frames_dir}/  ({n} JPEGs)")

    # ── Static helpers (mask geometry) ────────────────────────────────────────

    @staticmethod
    def _mask_centroid(mask: np.ndarray):
        """Return (cx, cy) of a boolean mask, or None if empty."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        """Return (x1,y1,x2,y2) tight bbox of a boolean mask, or None."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

"""
botsort_pipeline.py
===================
End-to-end basketball tracking pipeline.

  - RF-DETR detection (players, referees)
    - BotSort multi-object tracking
    - Jersey number OCR via PARSeq (RF-DETR class=2 jersey boxes)
  - Court keypoint detection + RANSAC homography
  - Supervision-based visualization
  - Tactical bird's-eye view
"""

import colorsys
import os
import sys

import cv2
import numpy as np
import supervision as sv
import torch
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from APP.helpers.rfdetr_detector import RFDETRDetector
from APP.helpers.botsort_tracker import BotSortTracker
from APP.helpers.memory_reid import MemoryReIDMatcher
from APP.helpers.court_utils import (
    TACTICAL_WIDTH, TACTICAL_HEIGHT, TACTICAL_KEYPOINTS,
    DEFAULT_KEYPOINT_MODEL, DEFAULT_COURT_IMAGE,
    compute_homography, draw_keypoints_on_frame, draw_tactical_view,
)


# ── Color helpers ──────────────────────────────────────────────────────────────

def _make_sv_palette(n: int = 60) -> sv.ColorPalette:
    """HSV-spread palette with n distinct colors."""
    colors = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append(sv.Color(r=int(r * 255), g=int(g * 255), b=int(b * 255)))
    return sv.ColorPalette(colors=colors)


# ── Geometry helper ────────────────────────────────────────────────────────────

def _interp_x_at_y(pts_sorted_by_y: list, y: float) -> float:
    if not pts_sorted_by_y:
        return 0.0
    if len(pts_sorted_by_y) == 1:
        return pts_sorted_by_y[0][0]
    if y <= pts_sorted_by_y[0][1]:
        p0, p1 = pts_sorted_by_y[0], pts_sorted_by_y[1]
        t = (y - p0[1]) / (p1[1] - p0[1]) if p1[1] != p0[1] else 0
        return p0[0] + t * (p1[0] - p0[0])
    if y >= pts_sorted_by_y[-1][1]:
        p0, p1 = pts_sorted_by_y[-2], pts_sorted_by_y[-1]
        t = (y - p0[1]) / (p1[1] - p0[1]) if p1[1] != p0[1] else 1
        return p0[0] + t * (p1[0] - p0[0])
    for i in range(len(pts_sorted_by_y) - 1):
        p0, p1 = pts_sorted_by_y[i], pts_sorted_by_y[i + 1]
        if p0[1] <= y <= p1[1]:
            t = (y - p0[1]) / (p1[1] - p0[1]) if p1[1] != p0[1] else 0.5
            return p0[0] + t * (p1[0] - p0[0])
    return pts_sorted_by_y[-1][0]


# ── sv.Detections builder ──────────────────────────────────────────────────────

def _tracks_to_sv(tracks: List[Dict]) -> sv.Detections:
    if not tracks:
        return sv.Detections.empty()
    xyxy        = np.array([t["bbox"] for t in tracks], dtype=np.float32)
    tracker_ids = np.array([t["track_id"] for t in tracks], dtype=int)
    class_ids   = np.array([t["class_id"] for t in tracks], dtype=int)
    confidences = np.array([t["confidence"] for t in tracks], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        tracker_id=tracker_ids,
        class_id=class_ids,
        confidence=confidences,
    )


class BotSortPipeline:
    """BotSort-based basketball tracking pipeline."""

    # ── Class IDs (match RFDETRDetector / BotSortTracker) ─────────────────────
    PLAYER_CLASSES = [3, 4, 5, 6, 7]
    REFEREE_CLASS  = 8
    NUMBER_CLASS   = 2  # jersey numbers (RF-DETR class=2)

    # ── Thresholds & intervals ─────────────────────────────────────────────────
    PLAYER_MIN_CONF      = 0.4
    REFEREE_MIN_CONF     = 0.3
    JERSEY_MIN_CONF      = 0.4
    JERSEY_MIN_SIZE      = 10
    JERSEY_EXPAND        = 1.5
    JERSEY_LOCK_VOTES    = 3
    JERSEY_UPDATE_INT    = 3
    KEYPOINT_UPDATE_INT  = 2
    TRAIL_LENGTH         = 30
    MEM_W_APP            = 0.80
    MEM_W_GIOU           = 0.15
    MEM_W_KALMAN         = 0.02
    MEM_SIM_THRESH       = 0.45
    MEM_TTL              = 90
    MEM_EMA_ALPHA        = 0.7
    MEM_BANK_BLEND       = 0.7
    MEM_GATE_SCALE       = 6.0
    STABLE_REID_THRESH   = 0.92
    STABLE_REID_TTL      = 180
    STABLE_REID_SPATIAL_MIN = 120.0
    STABLE_REID_SPATIAL_MAX = 360.0
    STABLE_REID_SPATIAL_DIAG = 3.0
    DETECTOR_IOU_THRESH  = 0.6
    DET_NMS_IOU          = 0.65
    CROSS_ROLE_IOU       = 0.80
    TRACK_NMS_IOU        = 0.88
    ID_DEBUG_EVERY_FRAME = True
    ID_JUMP_PX           = 90.0
    ID_SWAP_PX           = 70.0
    RAW_SWITCH_JUMP_PX   = 90.0
    RAW_SWITCH_COOLDOWN  = 12
    RESURRECT_DIST_PX    = 250.0
    SPLIT_IOU_THRESH     = 0.4
    SPLIT_SIM_THRESH     = 0.85

    # ── Keypoint filtering ─────────────────────────────────────────────────────
    KEYPOINT_BORDER_MARGIN   = 30
    KEYPOINT_EDGE_BAND_RATIO = 0.30
    KEYPOINT_STILL_PX        = 6
    KEYPOINT_STILL_FRAMES    = 5

    # ── Court boundary keypoint indices ────────────────────────────────────────
    LEFT_KP_INDICES   = [0, 1, 2, 3, 4, 5]
    RIGHT_KP_INDICES  = [10, 11, 12, 13, 14, 15]
    TOP_KP_INDICES    = [0, 7, 15]
    BOTTOM_KP_INDICES = [5, 6, 10]

    def __init__(
        self,
        rfdetr_model_id: str = RFDETRDetector.DEFAULT_MODEL_ID,
        device: str = "cuda",
        confidence_threshold: float = 0.4,
        keypoint_model_path: str = None,
        court_image_path: str = None,
    ):
        print("Loading BotSort Pipeline...")
        self.device               = device
        self.confidence_threshold = confidence_threshold

        # Primary detector: RF-DETR for players / referees
        self.player_detector = RFDETRDetector(model_id=rfdetr_model_id, device=device)

        # Jersey OCR: use RF-DETR class=2 number boxes + PARSeq
        from APP.helpers.jersey_detector import JerseyDetector, JerseyReIDBank
        self.jersey_detector = JerseyDetector(device=device)
        self.jersey_bank     = JerseyReIDBank()
        print("  Jersey OCR: RF-DETR class=2 + PARSeq")

        self.tracker = BotSortTracker(
            device=device,
            with_reid=False,
            match_thresh=0.85,
            track_buffer=90,
            fuse_first_associate=True,
            asso_func="giou",
            appearance_thresh=0.35,
            memory_ttl=self.MEM_TTL,
            memory_ema_alpha=self.MEM_EMA_ALPHA,
            w_app=self.MEM_W_APP,
            w_giou=self.MEM_W_GIOU,
            w_kalman=self.MEM_W_KALMAN,
            similarity_threshold=self.MEM_SIM_THRESH,
            bank_blend=self.MEM_BANK_BLEND,
            kalman_gate_scale=self.MEM_GATE_SCALE,
            kalman_only_position=True,
        )
        self.memory_reid = MemoryReIDMatcher(
            device=device,
            similarity_threshold=self.STABLE_REID_THRESH,
            ghost_ttl=self.STABLE_REID_TTL,
            ema_alpha=self.MEM_EMA_ALPHA,
            min_frames_before_match=0,
            extractor=self.tracker.embedding_extractor,
            ghost_only_for_new=True,
            spatial_gate_min=self.STABLE_REID_SPATIAL_MIN,
            spatial_gate_max=self.STABLE_REID_SPATIAL_MAX,
            spatial_gate_diag_mult=self.STABLE_REID_SPATIAL_DIAG,
            raw_switch_jump_thresh=self.RAW_SWITCH_JUMP_PX,
            raw_switch_cooldown=self.RAW_SWITCH_COOLDOWN,
            resurrect_dist_thresh=self.RESURRECT_DIST_PX,
            split_iou_thresh=self.SPLIT_IOU_THRESH,
            split_sim_thresh=self.SPLIT_SIM_THRESH,
        )
        print(
            "  Tracking tuning: "
            f"match_thresh=0.85 track_buffer=90 "
            f"w_app={self.MEM_W_APP:.2f} w_giou={self.MEM_W_GIOU:.2f} "
            f"w_kalman={self.MEM_W_KALMAN:.2f} "
            f"gate_scale={self.MEM_GATE_SCALE:.1f} kalman_only_position=True "
            f"stable_reid_thresh={self.STABLE_REID_THRESH:.2f} "
            f"stable_reid_ttl={self.STABLE_REID_TTL} ghost_only_for_new=True "
            f"stable_spatial=({self.STABLE_REID_SPATIAL_MIN:.0f},"
            f"{self.STABLE_REID_SPATIAL_MAX:.0f},"
            f"{self.STABLE_REID_SPATIAL_DIAG:.1f}) "
            f"id_debug={int(self.ID_DEBUG_EVERY_FRAME)} "
            f"id_jump_px={self.ID_JUMP_PX:.0f} "
            f"id_swap_px={self.ID_SWAP_PX:.0f} "
            f"raw_switch_jump_px={self.RAW_SWITCH_JUMP_PX:.0f} "
            f"raw_switch_cooldown={self.RAW_SWITCH_COOLDOWN} "
            f"resurrect_dist_px={self.RESURRECT_DIST_PX:.0f} "
            f"det_nms_iou={self.DET_NMS_IOU:.2f} "
            f"split_iou={self.SPLIT_IOU_THRESH:.2f} "
            f"split_sim={self.SPLIT_SIM_THRESH:.2f}"
        )

        kp_path = keypoint_model_path or DEFAULT_KEYPOINT_MODEL
        self.kp_model = YOLO(kp_path) if os.path.exists(kp_path) else None
        if self.kp_model is None:
            print(f"  Warning: keypoint model not found at {kp_path}")

        court_path = court_image_path or DEFAULT_COURT_IMAGE
        if os.path.exists(court_path):
            self.court_img = cv2.resize(cv2.imread(court_path), (TACTICAL_WIDTH, TACTICAL_HEIGHT))
        else:
            self.court_img = np.ones((TACTICAL_HEIGHT, TACTICAL_WIDTH, 3), dtype=np.uint8) * 40
            cv2.rectangle(self.court_img, (0, 0), (TACTICAL_WIDTH - 1, TACTICAL_HEIGHT - 1), (255, 255, 255), 1)
            cv2.line(self.court_img, (TACTICAL_WIDTH // 2, 0), (TACTICAL_WIDTH // 2, TACTICAL_HEIGHT), (255, 255, 255), 1)

        # Keypoint / homography state
        self.last_good_keypoints         = None
        self.last_H                      = None
        self.homography_success_count    = 0
        self._last_keypoints             = (np.zeros((18, 2), dtype=np.float32), np.zeros(18), None)
        self._prev_detected_keypoints    = np.zeros((18, 2), dtype=np.float32)
        self._keypoint_stationary_counts = np.zeros(18, dtype=np.int32)

        self.locked_jersey_ids: set = set()
        self._prev_id_debug: Dict[int, Dict] = {}
        self._prev_raw_to_stable_debug: Dict[int, int] = {}

        # ── Supervision annotators ─────────────────────────────────────────────
        self._player_palette = _make_sv_palette(60)
        self._ref_color      = sv.Color.from_hex("#00C8FF")

        self.trace_annotator = sv.TraceAnnotator(
            color=self._player_palette,
            color_lookup=sv.ColorLookup.TRACK,
            trace_length=self.TRAIL_LENGTH,
            thickness=2,
            position=sv.Position.BOTTOM_CENTER,
        )
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=self._player_palette,
            color_lookup=sv.ColorLookup.TRACK,
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self._player_palette,
            color_lookup=sv.ColorLookup.TRACK,
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.TOP_CENTER,
        )
        self.ref_ellipse_annotator = sv.EllipseAnnotator(
            color=self._ref_color,
            thickness=2,
        )
        self.ref_label_annotator = sv.LabelAnnotator(
            color=self._ref_color,
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.TOP_CENTER,
        )

        self._warmup()
        print("BotSort Pipeline ready.")

    # ── Warmup ─────────────────────────────────────────────────────────────────

    def _warmup(self):
        print("  Warming up models...", end=" ", flush=True)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.player_detector.detect(dummy, confidence_threshold=0.5)
        if self.kp_model is not None:
            self.kp_model.predict(dummy, conf=0.5, verbose=False,
                                  half=(self.device == "cuda"), device=self.device)
        print("done.")

    # ── Court keypoint detection ───────────────────────────────────────────────

    def _detect_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.kp_model is None:
            return np.zeros((18, 2), dtype=np.float32), np.zeros(18), self.last_H

        results = self.kp_model.predict(
            frame, conf=0.5, verbose=False,
            half=(self.device == "cuda"), device=self.device,
        )
        keypoints_xy = np.zeros((18, 2), dtype=np.float32)
        confidences  = np.zeros(18, dtype=np.float32)

        if results and results[0].keypoints is not None:
            kp = results[0].keypoints
            if kp.xy is not None and len(kp.xy) > 0:
                xy = kp.xy[0].cpu().numpy()
                keypoints_xy[:min(len(xy), 18)] = xy[:18]
                if kp.conf is not None and len(kp.conf) > 0:
                    conf = kp.conf[0].cpu().numpy()
                    n = min(len(conf), 18)
                    confidences[:n] = conf[:n]

        h, w = frame.shape[:2]
        margin = self.KEYPOINT_BORDER_MARGIN

        def _valid(pt):
            x, y = float(pt[0]), float(pt[1])
            return margin <= x < (w - margin) and margin <= y < (h - margin)

        def _in_edge_band(pt):
            x, y = float(pt[0]), float(pt[1])
            if x <= 0 or y <= 0:
                return False
            bx = w * self.KEYPOINT_EDGE_BAND_RATIO
            by = h * self.KEYPOINT_EDGE_BAND_RATIO
            return x <= bx or x >= (w - bx) or y <= by or y >= (h - by)

        for i in range(18):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        if self.last_good_keypoints is not None:
            valid_pairs = [
                i for i in range(18)
                if _valid(keypoints_xy[i]) and _valid(self.last_good_keypoints[i])
            ]
            if valid_pairs:
                jumps = sum(
                    1 for i in valid_pairs
                    if np.linalg.norm(keypoints_xy[i] - self.last_good_keypoints[i]) > 150
                )
                if jumps / len(valid_pairs) > 0.5:
                    self.last_good_keypoints = None
                    self._prev_detected_keypoints[:] = 0.0
                    self._keypoint_stationary_counts[:] = 0

        for i in range(18):
            pt   = keypoints_xy[i]
            prev = self._prev_detected_keypoints[i]
            if not _valid(pt):
                self._keypoint_stationary_counts[i] = 0
                self._prev_detected_keypoints[i] = 0.0
                continue
            if _in_edge_band(pt) and _valid(prev) and np.linalg.norm(pt - prev) <= self.KEYPOINT_STILL_PX:
                self._keypoint_stationary_counts[i] += 1
            else:
                self._keypoint_stationary_counts[i] = 0
            self._prev_detected_keypoints[i] = pt.copy()
            if self._keypoint_stationary_counts[i] >= self.KEYPOINT_STILL_FRAMES:
                keypoints_xy[i] = 0.0
                confidences[i]  = 0.0

        high_conf = np.where(
            (confidences >= 0.5).reshape(-1, 1) & (keypoints_xy > 0),
            keypoints_xy, 0.0,
        ).astype(np.float32)
        pre_H = compute_homography(high_conf) if (high_conf > 0).any() else None
        if pre_H is not None:
            for i in range(18):
                if keypoints_xy[i][0] <= 0:
                    continue
                dst = cv2.perspectiveTransform(np.array([[keypoints_xy[i]]], dtype=np.float32), pre_H)
                tx, ty = dst[0][0]
                ex, ey = TACTICAL_KEYPOINTS[i]
                if np.sqrt((tx - ex) ** 2 + (ty - ey) ** 2) > 50:
                    keypoints_xy[i] = 0.0

        ALPHA = 0.6
        if self.last_good_keypoints is not None:
            for i in range(18):
                has_det  = _valid(keypoints_xy[i])
                has_hist = _valid(self.last_good_keypoints[i])
                if has_det and has_hist:
                    keypoints_xy[i] = ALPHA * keypoints_xy[i] + (1 - ALPHA) * self.last_good_keypoints[i]
                elif not has_det and has_hist:
                    keypoints_xy[i] = self.last_good_keypoints[i].copy()

        for i in range(18):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        self.last_good_keypoints = keypoints_xy.copy()
        H = compute_homography(keypoints_xy)
        if H is not None:
            self.homography_success_count += 1
            self.last_H = H
        else:
            H = self.last_H

        return keypoints_xy, confidences, H

    # ── Court boundary filter ──────────────────────────────────────────────────

    def _is_valid_player_bbox(self, bbox: list, keypoints_xy: np.ndarray, frame_shape: Tuple) -> bool:
        if not any(kp[0] > 0 and kp[1] > 0 for kp in keypoints_xy):
            return True
        INNER = {8, 9, 16, 17}

        def _vis(indices):
            return sorted(
                [(float(keypoints_xy[i][0]), float(keypoints_xy[i][1]))
                 for i in indices if i < len(keypoints_xy)
                 and keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0],
                key=lambda p: p[1],
            )

        x1, y1, x2, y2 = [float(c) for c in bbox]
        cy = (y1 + y2) * 0.5

        left_pts = _vis(self.LEFT_KP_INDICES)
        if left_pts and x2 < _interp_x_at_y(left_pts, cy):
            return False

        right_pts = _vis(self.RIGHT_KP_INDICES)
        if right_pts and x1 > _interp_x_at_y(right_pts, cy):
            return False

        top_pts  = _vis(self.TOP_KP_INDICES)
        edge_pts = _vis([i for i in range(18) if i not in INNER])
        ref_top  = top_pts or edge_pts
        if ref_top and y2 < min(p[1] for p in ref_top):
            return False

        bottom_pts = _vis(self.BOTTOM_KP_INDICES)
        if bottom_pts and y2 > max(p[1] for p in bottom_pts):
            return False

        return True

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    def _dedupe_target_detections(self, dets: List[Dict]) -> List[Dict]:
        """Suppress near-duplicate player/referee detections before tracking."""
        target_classes = set(self.PLAYER_CLASSES + [self.REFEREE_CLASS])
        target = [d for d in dets if d["class_id"] in target_classes]
        other = [d for d in dets if d["class_id"] not in target_classes]
        target.sort(key=lambda d: float(d["confidence"]), reverse=True)

        kept: List[Dict] = []
        for det in target:
            is_ref = det["class_id"] == self.REFEREE_CLASS
            keep = True
            for prev in kept:
                prev_is_ref = prev["class_id"] == self.REFEREE_CLASS
                iou = self._bbox_iou(det["bbox"], prev["bbox"])
                if is_ref == prev_is_ref and iou >= self.DET_NMS_IOU:
                    keep = False
                    break
                if is_ref != prev_is_ref and iou >= self.CROSS_ROLE_IOU:
                    keep = False
                    break
            if keep:
                kept.append(det)
        return kept + other

    def _dedupe_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """Remove overlapping duplicate tracks and role-conflict overlays."""
        if not tracks:
            return tracks
        ordered = sorted(tracks, key=lambda t: float(t.get("confidence", 0.0)), reverse=True)
        kept: List[Dict] = []
        for tr in ordered:
            keep = True
            for prev in kept:
                iou = self._bbox_iou(tr["bbox"], prev["bbox"])
                if tr["is_referee"] == prev["is_referee"] and iou >= self.TRACK_NMS_IOU:
                    keep = False
                    break
                if tr["is_referee"] != prev["is_referee"] and iou >= self.CROSS_ROLE_IOU:
                    keep = False
                    break
            if keep:
                kept.append(tr)
        return kept

    @staticmethod
    def _bbox_center_xy(bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @staticmethod
    def _point_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _log_id_debug(self, tracks: List[Dict], frame_idx: int) -> None:
        """Frame-level ID diagnostics for raw/stable mapping and possible swaps."""
        player_tracks = [t for t in tracks if not t.get("is_referee", False)]
        current: Dict[int, Dict] = {}
        current_raw_to_stable: Dict[int, int] = {}

        for tr in sorted(player_tracks, key=lambda t: int(t.get("track_id", -1))):
            sid = int(tr.get("stable_track_id", tr.get("track_id", -1)))
            rid = int(tr.get("raw_track_id", tr.get("track_id", sid)))
            cx, cy = self._bbox_center_xy(tr["bbox"])
            current[sid] = {
                "raw": rid,
                "bbox": tr["bbox"],
                "center": (cx, cy),
                "conf": float(tr.get("confidence", 0.0)),
            }
            current_raw_to_stable[rid] = sid

            prev_raw_sid = self._prev_raw_to_stable_debug.get(rid)
            if prev_raw_sid is not None and prev_raw_sid != sid:
                print(
                    f"    [id-map-change] frame={frame_idx} "
                    f"raw={rid} stable_old={prev_raw_sid} stable_new={sid}"
                )

            prev = self._prev_id_debug.get(sid)
            if prev is not None:
                jump = self._point_dist(prev["center"], (cx, cy))
                raw_changed = int(prev["raw"]) != rid
                if raw_changed or jump >= self.ID_JUMP_PX:
                    print(
                        f"    [id-track] frame={frame_idx} stable={sid} raw={rid} "
                        f"prev_raw={int(prev['raw'])} cx={cx:.1f} cy={cy:.1f} "
                        f"jump={jump:.1f} raw_changed={int(raw_changed)} "
                        f"bbox={tr['bbox']}"
                    )
            else:
                print(
                    f"    [id-new] frame={frame_idx} stable={sid} raw={rid} "
                    f"cx={cx:.1f} cy={cy:.1f} bbox={tr['bbox']}"
                )

        prev_ids = set(self._prev_id_debug.keys())
        current_ids = set(current.keys())
        for sid in sorted(prev_ids - current_ids):
            prev = self._prev_id_debug[sid]
            px, py = prev["center"]
            print(
                f"    [id-lost] frame={frame_idx} stable={sid} raw={int(prev['raw'])} "
                f"last_cx={px:.1f} last_cy={py:.1f}"
            )

        for sid_a in sorted(prev_ids & current_ids):
            prev_a = self._prev_id_debug[sid_a]
            cur_a = current[sid_a]
            for sid_b in sorted((prev_ids & current_ids) - {sid_a}):
                if sid_b <= sid_a:
                    continue
                prev_b = self._prev_id_debug[sid_b]
                cur_b = current[sid_b]
                cross_ab = self._point_dist(prev_a["center"], cur_b["center"])
                cross_ba = self._point_dist(prev_b["center"], cur_a["center"])
                self_a = self._point_dist(prev_a["center"], cur_a["center"])
                self_b = self._point_dist(prev_b["center"], cur_b["center"])
                if (
                    cross_ab + cross_ba + self.ID_SWAP_PX
                    < self_a + self_b
                ):
                    print(
                        f"    [possible-swap] frame={frame_idx} "
                        f"a={sid_a} raw_a={int(cur_a['raw'])} "
                        f"b={sid_b} raw_b={int(cur_b['raw'])} "
                        f"self=({self_a:.1f},{self_b:.1f}) "
                        f"cross=({cross_ab:.1f},{cross_ba:.1f})"
                    )

        self._prev_id_debug = current
        self._prev_raw_to_stable_debug = current_raw_to_stable

    # ── Jersey OCR ────────────────────────────────────────────────────────────

    def _detect_jerseys(self, frame: np.ndarray, tracks: List[Dict], dets: List[Dict]) -> None:
        if self.jersey_detector is None:
            return
        h, w = frame.shape[:2]
        player_tracks = {t["track_id"]: t for t in tracks if not t["is_referee"]}
        if not player_tracks:
            return

        number_boxes = [
            d for d in dets
            if d["class_id"] == self.NUMBER_CLASS
            and float(d.get("confidence", 0.0)) >= self.JERSEY_MIN_CONF
            and (d["bbox"][2] - d["bbox"][0]) >= self.JERSEY_MIN_SIZE
            and (d["bbox"][3] - d["bbox"][1]) >= self.JERSEY_MIN_SIZE
        ]
        if not number_boxes:
            return

        for det in number_boxes:
            nx1, ny1, nx2, ny2 = [int(c) for c in det["bbox"]]
            cx, cy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
            bw, bh = nx2 - nx1, ny2 - ny1

            matched_id = None
            for tid, track in player_tracks.items():
                tx1, ty1, tx2, ty2 = track["bbox"]
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    matched_id = tid
                    break

            if matched_id is None or matched_id in self.locked_jersey_ids:
                continue

            ew  = int(bw * self.JERSEY_EXPAND)
            eh  = int(bh * self.JERSEY_EXPAND)
            ex1 = max(0, cx - ew // 2)
            ey1 = max(0, cy - eh // 2)
            ex2 = min(w, cx + ew // 2)
            ey2 = min(h, cy + eh // 2)

            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            number = self._ocr_jersey(crop)
            if not number:
                continue

            self.jersey_bank.register(matched_id, number)
            counts     = self.jersey_bank.detection_counts.get(matched_id, {})
            best_count = max(counts.values(), default=0)
            if best_count >= self.JERSEY_LOCK_VOTES:
                self.locked_jersey_ids.add(matched_id)
                confirmed = self.jersey_bank.get_jersey(matched_id)
                print(f"  Jersey locked: ID {matched_id} → #{confirmed}")

    def _ocr_jersey(self, crop: np.ndarray) -> str:
        try:
            text, conf = self.jersey_detector._ocr(crop)
            number = self.jersey_detector._text_to_number(text)
            return number if conf > 0.3 else ""
        except Exception:
            return ""

    # ── Feet positions for tactical view ──────────────────────────────────────

    @staticmethod
    def _tracks_to_feet(
        tracks: List[Dict], jersey_bank
    ) -> Tuple[List[List[float]], List[Optional[str]]]:
        feet, jerseys = [], []
        for t in tracks:
            if t["is_referee"]:
                continue
            x1, y1, x2, y2 = t["bbox"]
            feet.append([float((x1 + x2) / 2), float(y2)])
            jerseys.append(jersey_bank.get_jersey(t["track_id"]))
        return feet, jerseys

    # ── Supervision annotation ─────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        kp_xy: np.ndarray,
        kp_conf: np.ndarray,
        H: Optional[np.ndarray],
        frame_idx: int,
        src_idx: int,
    ) -> np.ndarray:
        result = frame.copy()

        player_tracks = [t for t in tracks if not t["is_referee"]]
        ref_tracks    = [t for t in tracks if t["is_referee"]]

        if player_tracks:
            p_dets  = _tracks_to_sv(player_tracks)
            p_labels = [
                f"#{self.jersey_bank.get_jersey(t['track_id'])}"
                if self.jersey_bank.get_jersey(t["track_id"])
                else str(t["track_id"])
                for t in player_tracks
            ]
            result = self.trace_annotator.annotate(scene=result, detections=p_dets)
            result = self.ellipse_annotator.annotate(scene=result, detections=p_dets)
            result = self.label_annotator.annotate(scene=result, detections=p_dets, labels=p_labels)

        if ref_tracks:
            r_dets   = _tracks_to_sv(ref_tracks)
            r_labels = ["REF"] * len(ref_tracks)
            result   = self.ref_ellipse_annotator.annotate(scene=result, detections=r_dets)
            result   = self.ref_label_annotator.annotate(scene=result, detections=r_dets, labels=r_labels)

        result = draw_keypoints_on_frame(result, kp_xy, kp_conf)

        n_kp = int((kp_xy > 0).all(axis=1).sum())
        cv2.putText(result, f"Frame: {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result, f"Source: {src_idx}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(result, f"Tracks: {len(tracks)} | KP: {n_kp}/18", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result,
                    f"Homography: {'OK' if H is not None else 'FAIL'}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if H is not None else (0, 0, 255), 2)
        cv2.putText(result, f"Jerseys: {len(self.jersey_bank.obj_to_jersey)}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return result

    # ── Main video loop ────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: int = 300,
        start_sec: float = 0.0,
        frame_skip: int = 1,
        show_preview: bool = False,
    ):
        print(f"\nProcessing: {video_path}")

        cap     = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = src_fps / frame_skip

        start_frame = int(start_sec * src_fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame, max_frames)

        tactical_w = 600
        tactical_h = int(TACTICAL_HEIGHT * (tactical_w / TACTICAL_WIDTH))
        base, ext  = os.path.splitext(output_path)
        tactical_path = f"{base}_tactical{ext}"

        writer          = None
        tactical_writer = None
        frame_idx       = 0
        interrupted     = False

        print(
            f"Run metadata: src_fps={src_fps:.3f}, out_fps={out_fps:.3f}, "
            f"start_frame={start_frame}, max_frames={total}, frame_skip={frame_skip}"
        )
        print(f"Output files: annotated={output_path} | tactical={tactical_path}")

        try:
            while frame_idx < total:
                for _ in range(frame_skip - 1):
                    cap.grab()

                ret, frame = cap.read()
                if not ret:
                    break

                dets = self.player_detector.detect(
                    frame,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.DETECTOR_IOU_THRESH,
                )

                if frame_idx % self.KEYPOINT_UPDATE_INT == 0:
                    kp_xy, kp_conf, H = self._detect_keypoints(frame)
                    self._last_keypoints = (kp_xy, kp_conf, H)
                else:
                    kp_xy, kp_conf, H = self._last_keypoints

                filtered_dets = [
                    d for d in dets
                    if d["class_id"] not in self.PLAYER_CLASSES
                    or self._is_valid_player_bbox(d["bbox"], kp_xy, frame.shape[:2])
                ]
                filtered_dets = self._dedupe_target_detections(filtered_dets)

                tracks = self.tracker.update(filtered_dets, frame)
                tracks = self.memory_reid.update(tracks, frame, frame_idx)
                reid_frame_dbg = self.memory_reid.get_debug_summary()
                if (
                    reid_frame_dbg.get("remapped", 0.0) > 0
                    and reid_frame_dbg.get("ghost_matches", 0.0) > 0
                ):
                    print(
                        f"    [stable-reid-event] frame={frame_idx} "
                        f"new_raw={int(reid_frame_dbg.get('new_raw', 0))} "
                        f"ghost_match={int(reid_frame_dbg.get('ghost_matches', 0))} "
                        f"remap={int(reid_frame_dbg.get('remapped', 0))} "
                        f"max_ghost_sim={float(reid_frame_dbg.get('max_ghost_sim', 0.0)):.3f} "
                        f"ghost_spatial_rej={int(reid_frame_dbg.get('ghost_spatial_rejects', 0))} "
                        f"ghosts={int(reid_frame_dbg.get('ghosts', 0))}"
                    )
                if reid_frame_dbg.get("switch_blocks", 0.0) > 0:
                    print(
                        f"    [raw-switch-block] frame={frame_idx} "
                        f"blocks={int(reid_frame_dbg.get('switch_blocks', 0))} "
                        f"tracks={int(reid_frame_dbg.get('tracks', 0))}"
                    )
                if reid_frame_dbg.get("fresh_allocs", 0.0) > 0:
                    print(
                        f"    [fresh-alloc] frame={frame_idx} "
                        f"count={int(reid_frame_dbg.get('fresh_allocs', 0))} "
                        f"resurrect={int(reid_frame_dbg.get('resurrect_blocks', 0))}"
                    )
                tracks = self._dedupe_tracks(tracks)
                if self.ID_DEBUG_EVERY_FRAME:
                    self._log_id_debug(tracks, frame_idx)

                if frame_idx % self.JERSEY_UPDATE_INT == 0 and tracks:
                    self._detect_jerseys(frame, tracks, dets)

                src_idx = start_frame + frame_idx * frame_skip
                result  = self._annotate_frame(frame, tracks, kp_xy, kp_conf, H, frame_idx, src_idx)

                if writer is None:
                    h, w = result.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
                    tactical_writer = cv2.VideoWriter(
                        tactical_path, fourcc, out_fps, (tactical_w, tactical_h)
                    )

                writer.write(result)

                player_feet, player_jerseys = self._tracks_to_feet(tracks, self.jersey_bank)
                tactical = draw_tactical_view(
                    self.court_img, kp_xy, H, player_feet, player_jerseys, {}
                )
                tactical_writer.write(cv2.resize(tactical, (tactical_w, tactical_h)))

                if show_preview:
                    try:
                        max_w   = 1280
                        display = (
                            cv2.resize(result, (max_w, int(result.shape[0] * max_w / result.shape[1])))
                            if result.shape[1] > max_w else result
                        )
                        cv2.imshow("BotSort Tracker", display)
                        cv2.imshow("Tactical View", cv2.resize(tactical, (tactical_w, tactical_h)))
                        cv2.waitKey(1)
                    except cv2.error:
                        pass

                n_kp = int((kp_xy > 0).all(axis=1).sum())
                if frame_idx % 30 == 0:
                    print(f"  frame {frame_idx:4d}  tracks={len(tracks):2d}  kp={n_kp}/18  jersey={len(self.jersey_bank.obj_to_jersey)}")
                    dbg = self.tracker.get_debug_summary()
                    p = dbg.get("player", {})
                    r = dbg.get("referee", {})
                    reid_dbg = self.memory_reid.get_debug_summary()
                    if p:
                        p_pairs = float(p.get("pairs", 0.0))
                        p_gate_pct = 100.0 * float(p.get("gate_blocked", 0.0)) / p_pairs if p_pairs else 0.0
                        p_finite_pct = 100.0 * float(p.get("finite_pairs", 0.0)) / p_pairs if p_pairs else 0.0
                        print(
                            "    [assoc-player] "
                            f"pairs={int(p.get('pairs', 0))} "
                            f"gate={int(p.get('gate_blocked', 0))} "
                            f"gate_pct={p_gate_pct:.1f} "
                            f"app_pen={int(p.get('app_penalty', 0))} "
                            f"finite={int(p.get('finite_pairs', 0))} "
                            f"finite_pct={p_finite_pct:.1f} "
                            f"matches={int(p.get('matches', 0))} "
                            f"u_t={int(p.get('u_track', 0))} "
                            f"u_d={int(p.get('u_det', 0))} "
                            f"m_app={float(p.get('match_app', 0.0)):.3f} "
                            f"m_giou={float(p.get('match_giou', 0.0)):.3f} "
                            f"m_motion={float(p.get('match_motion', 0.0)):.3f} "
                            f"m_cost={float(p.get('match_cost', 0.0)):.3f}"
                        )
                    if reid_dbg:
                        print(
                            "    [stable-reid]  "
                            f"tracks={int(reid_dbg.get('tracks', 0))} "
                            f"new_raw={int(reid_dbg.get('new_raw', 0))} "
                            f"locked={int(reid_dbg.get('locked', 0))} "
                            f"eligible={int(reid_dbg.get('eligible', 0))} "
                            f"cand={int(reid_dbg.get('candidates', 0))} "
                            f"remap={int(reid_dbg.get('remapped', 0))} "
                            f"ghost_match={int(reid_dbg.get('ghost_matches', 0))} "
                            f"ghost_cand={int(reid_dbg.get('ghost_candidates', 0))} "
                            f"ghost_spatial_rej={int(reid_dbg.get('ghost_spatial_rejects', 0))} "
                            f"switch_blocks={int(reid_dbg.get('switch_blocks', 0))} "
                            f"fresh_allocs={int(reid_dbg.get('fresh_allocs', 0))} "
                            f"resurrect_blocks={int(reid_dbg.get('resurrect_blocks', 0))} "
                            f"split_drops={int(reid_dbg.get('split_drops', 0))} "
                            f"max_sim={float(reid_dbg.get('max_match_sim', 0.0)):.3f} "
                            f"max_ghost_sim={float(reid_dbg.get('max_ghost_sim', 0.0)):.3f} "
                            f"active_bank={int(reid_dbg.get('active_bank', 0))} "
                            f"ghosts={int(reid_dbg.get('ghosts', 0))}"
                        )
                    if r:
                        r_pairs = float(r.get("pairs", 0.0))
                        r_gate_pct = 100.0 * float(r.get("gate_blocked", 0.0)) / r_pairs if r_pairs else 0.0
                        r_finite_pct = 100.0 * float(r.get("finite_pairs", 0.0)) / r_pairs if r_pairs else 0.0
                        print(
                            "    [assoc-ref]    "
                            f"pairs={int(r.get('pairs', 0))} "
                            f"gate={int(r.get('gate_blocked', 0))} "
                            f"gate_pct={r_gate_pct:.1f} "
                            f"app_pen={int(r.get('app_penalty', 0))} "
                            f"finite={int(r.get('finite_pairs', 0))} "
                            f"finite_pct={r_finite_pct:.1f} "
                            f"matches={int(r.get('matches', 0))} "
                            f"u_t={int(r.get('u_track', 0))} "
                            f"u_d={int(r.get('u_det', 0))} "
                            f"m_app={float(r.get('match_app', 0.0)):.3f} "
                            f"m_giou={float(r.get('match_giou', 0.0)):.3f} "
                            f"m_motion={float(r.get('match_motion', 0.0)):.3f} "
                            f"m_cost={float(r.get('match_cost', 0.0)):.3f}"
                        )

                frame_idx += 1

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted.")

        finally:
            cap.release()
            if writer:
                writer.release()
            if tactical_writer:
                tactical_writer.release()
            if show_preview:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        if not interrupted:
            print(f"\nDone! Output:   {output_path}")
            print(f"     Tactical:  {tactical_path}")
            print(f"     Homography success: {self.homography_success_count}")

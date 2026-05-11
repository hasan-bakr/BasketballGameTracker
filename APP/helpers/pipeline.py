"""
pipeline.py
===================
End-to-end basketball tracking pipeline.

  - RF-DETR detection (players, referees)
    - MCByte multi-object tracking
    - Jersey number OCR via PARSeq (RF-DETR class=2 jersey boxes)
  - Court keypoint detection + RANSAC homography
  - Supervision-based visualization
  - Tactical bird's-eye view
"""

import colorsys
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from typing import Deque, Dict, List, Optional, Tuple
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from APP.helpers.rfdetr_detector import RFDETRDetector
from APP.helpers.mcbyte_tracker import McByteBasketballTracker
from APP.helpers.config import PipelineConfig
from APP.helpers.court_utils import (
    TACTICAL_WIDTH, TACTICAL_HEIGHT, TACTICAL_KEYPOINTS,
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


def _track_color_bgr(track_id: int) -> Tuple[int, int, int]:
    h = ((int(track_id) % 60) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


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

def _tracks_to_sv(tracks: List[Dict], masks: Optional[np.ndarray] = None) -> sv.Detections:
    if not tracks:
        return sv.Detections.empty()
    xyxy        = np.array([t["bbox"] for t in tracks], dtype=np.float32)
    tracker_ids = np.array([t["track_id"] for t in tracks], dtype=int)
    class_ids   = np.array([t["class_id"] for t in tracks], dtype=int)
    confidences = np.array([t["confidence"] for t in tracks], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        tracker_id=tracker_ids,
        class_id=class_ids,
        confidence=confidences,
    )


class BasketballTrackingPipeline:
    """End-to-end basketball tracking and visualization pipeline."""

    def __init__(
        self,
        rfdetr_model_id: str | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        keypoint_model_path: str = None,
        court_image_path: str = None,
        config: PipelineConfig | None = None,
    ):
        print("Loading Basketball Tracking Pipeline...")
        self.config = config or PipelineConfig()
        if device is not None:
            self.config.device = device
        if rfdetr_model_id is not None:
            self.config.detector.rfdetr_model_id = rfdetr_model_id
        if confidence_threshold is not None:
            self.config.detector.confidence = confidence_threshold
        if keypoint_model_path is not None:
            self.config.keypoints.model_path = keypoint_model_path
        if court_image_path is not None:
            self.config.keypoints.court_image_path = court_image_path

        self.device = self.config.device
        self.confidence_threshold = self.config.detector.confidence

        # Primary detector: RF-DETR for players / referees
        self.player_detector = RFDETRDetector(model_id=self.config.detector.rfdetr_model_id, device=self.device)

        # Jersey OCR: use RF-DETR class=2 number boxes + PARSeq
        from APP.helpers.jersey_detector import JerseyDetector, JerseyReIDBank
        self.jersey_detector = JerseyDetector(device=self.device)
        self.jersey_bank     = JerseyReIDBank(verbose=self.config.debug.enabled)
        print("  Jersey OCR: RF-DETR class=2 + PARSeq")

        self.tracker = McByteBasketballTracker(
            device=self.device,
            fps=self.config.tracker.fps,
            track_thresh=self.config.tracker.track_thresh,
            new_track_thresh=self.config.tracker.new_track_thresh,
            track_buffer=self.config.tracker.track_buffer,
            cmc_method=self.config.tracker.cmc_method,
            assoc1_thresh=self.config.tracker.assoc1_thresh,
            assoc2_thresh=self.config.tracker.assoc2_thresh,
            unconfirmed_assoc_thresh=self.config.tracker.unconfirmed_assoc_thresh,
            mask_duplicate_min_fill=self.config.tracker.mask_duplicate_min_fill,
            mask_bbox_expand=self.config.tracker.mask_bbox_expand,
            mask_min_area_px=self.config.tracker.mask_min_area_px,
            mask_min_bbox_inside_ratio=self.config.tracker.mask_min_bbox_inside_ratio,
            ref_player_conflict_iou=self.config.tracker.ref_player_conflict_iou,
            ref_player_conflict_mask_fill=self.config.tracker.ref_player_conflict_mask_fill,
            ref_player_conflict_conf_margin=self.config.tracker.ref_player_conflict_conf_margin,
            use_player_masks=self.config.tracker.use_player_masks,
            use_referee_masks=self.config.tracker.use_referee_masks,
            require_cuda=self.config.tracker.require_cuda,
            sam_model_type=self.config.tracker.sam_model_type,
            sam_checkpoint=self.config.tracker.sam_checkpoint,
            cutie_weights=self.config.tracker.cutie_weights,
            cutie_max_internal_size=self.config.tracker.cutie_max_internal_size,
            debug_masks=self.config.debug.masks,
            debug_association=self.config.debug.tracking,
            debug_lifecycle=self.config.debug.lifecycle,
            debug_suppression=self.config.debug.suppression,
            switch_proxy_max_dist=self.config.tracker.switch_proxy_max_dist,
            switch_proxy_max_dt=self.config.tracker.switch_proxy_max_dt,
            switch_proxy_min_source_life=self.config.tracker.switch_proxy_min_source_life,
            duplicate_new_track_max_age=self.config.tracker.duplicate_new_track_max_age,
            duplicate_existing_min_age=self.config.tracker.duplicate_existing_min_age,
            duplicate_new_track_iou=self.config.tracker.duplicate_new_track_iou,
            duplicate_new_track_center_px=self.config.tracker.duplicate_new_track_center_px,
            mask_reinit_missing_frames=self.config.tracker.mask_reinit_missing_frames,
            mask_reinit_area_fail_frames=self.config.tracker.mask_reinit_area_fail_frames,
            mask_reinit_inside_fail_frames=self.config.tracker.mask_reinit_inside_fail_frames,
            mask_reinit_inside_ratio=self.config.tracker.mask_reinit_inside_ratio,
            mask_reinit_cooldown=self.config.tracker.mask_reinit_cooldown,
            mask_assignment_near_iou=self.config.tracker.mask_assignment_near_iou,
            mask_assignment_min_center_px=self.config.tracker.mask_assignment_min_center_px,
            mask_assignment_center_scale=self.config.tracker.mask_assignment_center_scale,
            mask_reuse_min_fill=self.config.tracker.mask_reuse_min_fill,
        )
        self._prev_tracks: List[Dict] = []
        self._supp_det_pairs: Dict = {}
        self._supp_det_duplicate_pairs: Dict = {}
        self._track_birth_frame: Dict[int, int] = {}
        self._track_birth_center: Dict[int, Tuple[float, float]] = {}
        print(
            "  Tracking: MCByte "
            f"device={self.tracker.device} "
            f"det_conf={self.config.detector.confidence:.2f} "
            f"det_model_iou={self.config.detector.iou_threshold:.2f} "
            f"track_thresh={self.config.tracker.track_thresh:.2f} "
            f"new_track_thresh={self.config.tracker.new_track_thresh:.2f} "
            f"track_buffer={self.config.tracker.track_buffer} "
            f"cmc={self.config.tracker.cmc_method} "
            f"assoc=({self.config.tracker.assoc1_thresh:.2f},"
            f"{self.config.tracker.assoc2_thresh:.2f},"
            f"{self.config.tracker.unconfirmed_assoc_thresh:.2f}) "
            f"mask_dup_fill={self.config.tracker.mask_duplicate_min_fill:.2f} "
            f"mask_guard=(expand={self.config.tracker.mask_bbox_expand:.2f},"
            f"inside>={self.config.tracker.mask_min_bbox_inside_ratio:.2f},"
            f"area>={self.config.tracker.mask_min_area_px}) "
            f"mask_iso=(iou>={self.config.tracker.mask_assignment_near_iou:.2f},"
            f"d<={self.config.tracker.mask_assignment_min_center_px:.0f}px) "
            f"young_dup=(age<={self.config.tracker.duplicate_new_track_max_age},"
            f"old>={self.config.tracker.duplicate_existing_min_age},"
            f"iou>={self.config.tracker.duplicate_new_track_iou:.2f}) "
            f"ref_conflict=(iou>={self.config.tracker.ref_player_conflict_iou:.2f},"
            f"fill>={self.config.tracker.ref_player_conflict_mask_fill:.2f},"
            f"margin={self.config.tracker.ref_player_conflict_conf_margin:.2f}) "
            f"player_masks={int(self.config.tracker.use_player_masks)} "
            f"referee_masks={int(self.config.tracker.use_referee_masks)} "
            f"sam={self.config.tracker.sam_model_type} "
            f"cutie_size={self.config.tracker.cutie_max_internal_size} "
            f"debug={int(self.config.debug.enabled)} "
            f"id_jump_px={self.config.debug.id_jump_px:.0f} "
            f"id_swap_px={self.config.debug.id_swap_px:.0f} "
            f"det_nms_iou={self.config.detector.detection_nms_iou:.2f} "
            f"track_aware_supp={int(self.config.detector.track_aware_suppression)} "
            f"match_iou={self.config.detector.track_match_iou:.2f} "
            f"lost_match=(iou>={self.config.detector.lost_track_match_iou:.2f},"
            f"dist<={self.config.detector.lost_track_match_center_px:.0f}) "
            f"court_mask_activation={int(self.config.detector.activate_tracks_by_court_mask)} "
            f"court_mask=(overlap>={self.config.detector.court_mask_min_overlap:.2f},"
            f"area>={self.config.detector.court_mask_min_area_px}) "
        )

        kp_path = self.config.keypoints.model_path
        self.kp_model = YOLO(kp_path) if os.path.exists(kp_path) else None
        if self.kp_model is None:
            print(f"  Warning: keypoint model not found at {kp_path}")

        court_path = self.config.keypoints.court_image_path
        if os.path.exists(court_path):
            self.court_img = cv2.resize(cv2.imread(court_path), (TACTICAL_WIDTH, TACTICAL_HEIGHT))
        else:
            self.court_img = np.ones((TACTICAL_HEIGHT, TACTICAL_WIDTH, 3), dtype=np.uint8) * 40
            cv2.rectangle(self.court_img, (0, 0), (TACTICAL_WIDTH - 1, TACTICAL_HEIGHT - 1), (255, 255, 255), 1)
            cv2.line(self.court_img, (TACTICAL_WIDTH // 2, 0), (TACTICAL_WIDTH // 2, TACTICAL_HEIGHT), (255, 255, 255), 1)

        # Keypoint / homography state
        self.last_good_keypoints         = None
        self.last_H                      = None
        self._court_side: Optional[str]  = None
        self.homography_success_count    = 0
        self._last_keypoints             = (np.zeros((18, 2), dtype=np.float32), np.zeros(18), None)
        self._prev_detected_keypoints    = np.zeros((18, 2), dtype=np.float32)
        self._keypoint_stationary_counts = np.zeros(18, dtype=np.int32)
        self._keypoint_missing_counts    = np.full(18, self.config.keypoints.carry_missing_updates + 1, dtype=np.int32)

        self.locked_jersey_ids: set = set()
        self._track_id_alias: Dict[int, int] = {}
        self._lifecycle_event_cursor: int = 0
        self._prev_id_debug: Dict[int, Dict] = {}
        self._prev_raw_to_stable_debug: Dict[int, int] = {}
        self._last_player_tracks: List[Dict] = []
        self._court_active_track_ids: set[int] = set()
        self._tactical_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.config.visualization.tactical_trail_length)
        )
        self._tactical_smoothed: Dict[int, Tuple[float, float]] = {}

        # ── Supervision annotators ─────────────────────────────────────────────
        self._player_palette = _make_sv_palette(60)
        self._ref_color      = sv.Color.from_hex("#00C8FF")

        self.trace_annotator = sv.TraceAnnotator(
            color=self._player_palette,
            color_lookup=sv.ColorLookup.TRACK,
            trace_length=max(1, self.config.visualization.trail_length),
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
        print("Basketball Tracking Pipeline ready.")

    # ── Warmup ─────────────────────────────────────────────────────────────────

    def _warmup(self):
        print("  Warming up models...", end=" ", flush=True)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.player_detector.detect(dummy, confidence_threshold=0.5)
        if self.kp_model is not None:
            self.kp_model.predict(dummy, conf=self.config.keypoints.confidence, verbose=False,
                                  half=(self.device == "cuda"), device=self.device)
        print("done.")

    # ── Court keypoint detection ───────────────────────────────────────────────

    def _detect_keypoints(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        src_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.kp_model is None:
            return np.zeros((18, 2), dtype=np.float32), np.zeros(18), self.last_H

        results = self.kp_model.predict(
            frame, conf=self.config.keypoints.confidence, verbose=False,
            half=(self.device == "cuda"), device=self.device,
        )
        keypoints_xy = np.zeros((18, 2), dtype=np.float32)
        confidences  = np.zeros(18, dtype=np.float32)
        debug_boxes = 0
        debug_instances = 0

        if results and results[0].keypoints is not None:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                debug_boxes = len(boxes)
            kp = results[0].keypoints
            if kp.xy is not None and len(kp.xy) > 0:
                debug_instances = len(kp.xy)
                xy = kp.xy[0].cpu().numpy()
                keypoints_xy[:min(len(xy), 18)] = xy[:18]
                if kp.conf is not None and len(kp.conf) > 0:
                    conf = kp.conf[0].cpu().numpy()
                    n = min(len(conf), 18)
                    confidences[:n] = conf[:n]

        raw_valid_count = int(((keypoints_xy > 0).all(axis=1)).sum())
        raw_conf_01 = int((confidences >= 0.1).sum())
        raw_conf_05 = int((confidences >= 0.5).sum())

        h, w = frame.shape[:2]
        margin = self.config.keypoints.border_margin

        def _valid(pt):
            x, y = float(pt[0]), float(pt[1])
            return margin <= x < (w - margin) and margin <= y < (h - margin)

        def _in_edge_band(pt):
            x, y = float(pt[0]), float(pt[1])
            if x <= 0 or y <= 0:
                return False
            bx = w * self.config.keypoints.edge_band_ratio
            by = h * self.config.keypoints.edge_band_ratio
            return x <= bx or x >= (w - bx) or y <= by or y >= (h - by)

        for i in range(18):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        observed_side, side_counts = self._detected_court_side(keypoints_xy, confidences)
        side_switched = False
        if observed_side is not None and self._court_side is not None and observed_side != self._court_side:
            side_switched = True
            old_side = self._court_side
            self._court_side = None
            self.last_good_keypoints = None
            self.last_H = None
            self._prev_detected_keypoints[:] = 0.0
            self._keypoint_stationary_counts[:] = 0
            self._keypoint_missing_counts[:] = self.config.keypoints.carry_missing_updates + 1
            self._tactical_history.clear()
            self._tactical_smoothed.clear()
            if self.config.debug.keypoints:
                frame_label = "-" if frame_idx is None else str(frame_idx)
                src_label = "-" if src_idx is None else str(src_idx)
                print(
                    f"    [kp-side-switch] frame={frame_label} src={src_label} "
                    f"{old_side}->{observed_side} "
                    f"left={side_counts[0]} center={side_counts[1]} right={side_counts[2]}"
                )

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
                    self._keypoint_missing_counts[:] = self.config.keypoints.carry_missing_updates + 1

        for i in range(18):
            pt   = keypoints_xy[i]
            prev = self._prev_detected_keypoints[i]
            if not _valid(pt):
                self._keypoint_stationary_counts[i] = 0
                self._prev_detected_keypoints[i] = 0.0
                continue
            if _in_edge_band(pt) and _valid(prev) and np.linalg.norm(pt - prev) <= self.config.keypoints.still_px:
                self._keypoint_stationary_counts[i] += 1
            else:
                self._keypoint_stationary_counts[i] = 0
            self._prev_detected_keypoints[i] = pt.copy()
            if self._keypoint_stationary_counts[i] >= self.config.keypoints.still_frames:
                keypoints_xy[i] = 0.0
                confidences[i]  = 0.0

        high_conf = np.where(
            (confidences >= self.config.keypoints.geometry_confidence).reshape(-1, 1) & (keypoints_xy > 0),
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

        detected_this_update = np.array([_valid(keypoints_xy[i]) for i in range(18)], dtype=bool)
        for i in range(18):
            if detected_this_update[i]:
                self._keypoint_missing_counts[i] = 0
            else:
                self._keypoint_missing_counts[i] += 1

        ALPHA = 0.6
        carried_count = 0
        if self.last_good_keypoints is not None:
            for i in range(18):
                has_det  = _valid(keypoints_xy[i])
                has_hist = _valid(self.last_good_keypoints[i])
                if has_det and has_hist:
                    keypoints_xy[i] = ALPHA * keypoints_xy[i] + (1 - ALPHA) * self.last_good_keypoints[i]
                elif (
                    not has_det
                    and has_hist
                    and self._keypoint_missing_counts[i] <= self.config.keypoints.carry_missing_updates
                ):
                    keypoints_xy[i] = self.last_good_keypoints[i].copy()
                    carried_count += 1

        for i in range(18):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0
                confidences[i] = 0.0

        candidate_keypoints = keypoints_xy.copy()
        final_valid_count = int(((keypoints_xy > 0).all(axis=1)).sum())
        candidate_H = compute_homography(candidate_keypoints)
        H = candidate_H
        h_status = "fail"
        if H is not None and self._accept_homography_update(H, frame.shape, final_valid_count):
            self.homography_success_count += 1
            self.last_good_keypoints = candidate_keypoints.copy()
            self.last_H = H
            if observed_side is not None:
                self._court_side = observed_side
            h_status = "ok"
        else:
            H = self.last_H
            h_status = "fallback" if H is not None else "fail"
            if H is not None and self.last_good_keypoints is not None:
                keypoints_xy = self.last_good_keypoints.copy()

        if self.config.debug.keypoints:
            frame_label = "-" if frame_idx is None else str(frame_idx)
            src_label = "-" if src_idx is None else str(src_idx)
            print(
                "    [kp-debug] "
                f"frame={frame_label} src={src_label} "
                f"boxes={debug_boxes} instances={debug_instances} "
                f"raw_valid={raw_valid_count}/18 "
                f"conf>=0.1={raw_conf_01} conf>=0.5={raw_conf_05} "
                f"final_valid={final_valid_count}/18 "
                f"carried={carried_count} "
                f"side={observed_side or self._court_side or '-'} "
                f"side_counts=(L{side_counts[0]},C{side_counts[1]},R{side_counts[2]}) "
                f"side_switched={int(side_switched)} "
                f"H={h_status}"
            )

        return keypoints_xy, confidences, H

    def _detected_court_side(
        self,
        keypoints_xy: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[Optional[str], Tuple[int, int, int]]:
        valid = (keypoints_xy > 0).all(axis=1)
        confident = confidences >= self.config.keypoints.confidence
        left_indices = set(self.config.keypoints.left_indices) | {8, 9}
        center_indices = set(self.config.keypoints.center_indices)
        right_indices = set(self.config.keypoints.right_indices) | {16, 17}

        left_count = sum(1 for idx in left_indices if idx < len(valid) and valid[idx] and confident[idx])
        center_count = sum(1 for idx in center_indices if idx < len(valid) and valid[idx] and confident[idx])
        right_count = sum(1 for idx in right_indices if idx < len(valid) and valid[idx] and confident[idx])
        min_count = self.config.keypoints.side_switch_min_keypoints
        center_min = self.config.keypoints.center_switch_min_keypoints
        margin = self.config.keypoints.side_switch_margin

        if right_count >= min_count and right_count >= left_count + margin:
            return "right", (left_count, center_count, right_count)
        if left_count >= min_count and left_count >= right_count + margin:
            return "left", (left_count, center_count, right_count)
        if center_count >= center_min:
            return "center", (left_count, center_count, right_count)
        return None, (left_count, center_count, right_count)

    def _accept_homography_update(
        self,
        H: np.ndarray,
        frame_shape: Tuple[int, int, int],
        valid_keypoints: int,
    ) -> bool:
        if self.last_H is None:
            return True
        if valid_keypoints < self.config.keypoints.homography_min_update_keypoints:
            return False

        h, w = frame_shape[:2]
        sample = np.array(
            [
                [0.0, 0.0],
                [w * 0.5, 0.0],
                [w - 1.0, 0.0],
                [0.0, h * 0.5],
                [w * 0.5, h * 0.5],
                [w - 1.0, h * 0.5],
                [0.0, h - 1.0],
                [w * 0.5, h - 1.0],
                [w - 1.0, h - 1.0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            prev = cv2.perspectiveTransform(sample, self.last_H).reshape(-1, 2)
            cur = cv2.perspectiveTransform(sample, H).reshape(-1, 2)
        except cv2.error:
            return False

        if not np.isfinite(prev).all() or not np.isfinite(cur).all():
            return False
        shifts = np.linalg.norm(cur - prev, axis=1)
        mean_shift = float(np.mean(shifts))
        max_shift = float(np.max(shifts))
        return (
            mean_shift <= self.config.keypoints.homography_max_mean_shift
            and max_shift <= self.config.keypoints.homography_max_point_shift
        )

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

        left_pts = _vis(self.config.keypoints.left_indices)
        if left_pts and x2 < _interp_x_at_y(left_pts, cy):
            return False

        right_pts = _vis(self.config.keypoints.right_indices)
        if right_pts and x1 > _interp_x_at_y(right_pts, cy):
            return False

        top_pts  = _vis(self.config.keypoints.top_indices)
        edge_pts = _vis([i for i in range(18) if i not in INNER])
        ref_top  = top_pts or edge_pts
        if ref_top and y2 < min(p[1] for p in ref_top):
            return False

        bottom_pts = _vis(self.config.keypoints.bottom_indices)
        if bottom_pts and y2 > max(p[1] for p in bottom_pts):
            return False

        return True

    def _matches_existing_player_track(self, bbox: List[int]) -> bool:
        for track in self._last_player_tracks:
            iou = self._bbox_iou(bbox, track["bbox"])
            if iou >= self.config.detector.outside_court_existing_track_iou:
                return True
            det_center = self._bbox_center_xy(bbox)
            track_center = self._bbox_center_xy(track["bbox"])
            if self._point_dist(det_center, track_center) <= self.config.detector.outside_court_existing_track_center_px:
                return True
        return False

    def _keep_detection_for_court_filter(
        self,
        det: Dict,
        keypoints_xy: np.ndarray,
        frame_shape: Tuple,
    ) -> bool:
        if det["class_id"] not in self.config.classes.player_classes:
            return True
        if self._is_valid_player_bbox(det["bbox"], keypoints_xy, frame_shape):
            return True
        return self._matches_existing_player_track(det["bbox"])

    def _mask_court_overlap(self, mask: np.ndarray, H: Optional[np.ndarray]) -> Tuple[float, int]:
        if H is None or mask is None or not np.any(mask):
            return 0.0, 0

        ys, xs = np.where(mask.astype(bool))
        total = int(len(xs))
        if total == 0:
            return 0.0, 0

        max_samples = 2500
        step = max(1, total // max_samples)
        pts = np.stack([xs[::step], ys[::step]], axis=1).astype(np.float32).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        except cv2.error:
            return 0.0, 0

        inside = (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < TACTICAL_WIDTH)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < TACTICAL_HEIGHT)
        )
        ratio = float(np.mean(inside)) if inside.size else 0.0
        inside_area = int(round(total * ratio))
        return ratio, inside_area

    def _bbox_court_overlap(self, bbox: List[int], H: Optional[np.ndarray]) -> Tuple[float, int]:
        if H is None:
            return 0.0, 0

        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        pts = np.array(
            [
                [x1 + 0.50 * w, y2],
                [x1 + 0.25 * w, y2],
                [x1 + 0.75 * w, y2],
                [x1 + 0.50 * w, y1 + 0.80 * h],
                [x1 + 0.50 * w, y1 + 0.65 * h],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        except cv2.error:
            return 0.0, 0

        inside = (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < TACTICAL_WIDTH)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < TACTICAL_HEIGHT)
        )
        inside_count = int(np.count_nonzero(inside))
        return float(inside_count) / float(len(pts)), inside_count

    def _activate_tracks_by_court_mask(
        self,
        tracks: List[Dict],
        H: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        frame_idx: int,
    ) -> List[Dict]:
        if not self.config.detector.activate_tracks_by_court_mask:
            return tracks

        player_tracks = [t for t in tracks if not t["is_referee"]]
        ref_tracks = [t for t in tracks if t["is_referee"]]
        if not player_tracks:
            return ref_tracks

        mask_tracks, masks = self.tracker.get_player_masks_for_tracks(player_tracks, frame_shape)
        mask_by_id = {}
        if mask_tracks and masks is not None:
            mask_by_id = {
                int(track["track_id"]): mask.astype(bool)
                for track, mask in zip(mask_tracks, masks)
            }

        kept_players: List[Dict] = []
        for track in player_tracks:
            track_id = int(track["track_id"])
            if track_id in self._court_active_track_ids:
                kept_players.append(track)
                continue

            mask = mask_by_id.get(track_id)
            overlap, inside_area = self._mask_court_overlap(mask, H) if mask is not None else (0.0, 0)
            mask_activates = (
                overlap >= self.config.detector.court_mask_min_overlap
                and inside_area >= self.config.detector.court_mask_min_area_px
            )
            bbox_overlap, bbox_inside = self._bbox_court_overlap(track["bbox"], H)
            bbox_activates = bbox_overlap >= 0.40 and bbox_inside >= 2
            should_activate = mask_activates or bbox_activates

            if should_activate:
                self._court_active_track_ids.add(track_id)
                kept_players.append(track)
                if self.config.debug.tracking or self.config.debug.suppression:
                    source = "mask" if mask_activates else "bbox"
                    print(
                        f"    [court-mask-activate] frame={frame_idx} track={track_id} "
                        f"source={source} mask_overlap={overlap:.3f} "
                        f"mask_area={inside_area} bbox_overlap={bbox_overlap:.2f} "
                        f"bbox_inside={bbox_inside} bbox={track['bbox']}"
                    )
            elif self.config.debug.suppression and (mask is None or bbox_overlap > 0.0 or overlap > 0.0):
                print(
                    f"    [court-mask-pending] frame={frame_idx} track={track_id} "
                    f"has_mask={int(mask is not None)} mask_overlap={overlap:.3f} "
                    f"mask_area={inside_area} bbox_overlap={bbox_overlap:.2f} "
                    f"bbox_inside={bbox_inside} bbox={track['bbox']}"
                )

        return kept_players + ref_tracks

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

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _current_track_candidates(self) -> List[Dict]:
        if hasattr(self.tracker, "get_track_candidates"):
            candidates = self.tracker.get_track_candidates()
        else:
            candidates = []
        if candidates:
            return candidates
        return [
            {
                "track_id": int(track["track_id"]),
                "bbox": track["bbox"],
                "is_referee": bool(track.get("is_referee", False)),
                "state": "active",
                "age": 0,
            }
            for track in self._prev_tracks
        ]

    def _track_candidate_matches(self, bbox: List[float], is_referee: Optional[bool] = None) -> List[Dict]:
        bx, by = self._bbox_center(bbox)
        matches: List[Dict] = []
        for candidate in self._current_track_candidates():
            if is_referee is not None and bool(candidate.get("is_referee", False)) != is_referee:
                continue
            iou = self._bbox_iou(bbox, candidate["bbox"])
            cx, cy = self._bbox_center(candidate["bbox"])
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            state = str(candidate.get("state", "active"))
            if state == "lost":
                matched = (
                    iou >= self.config.detector.lost_track_match_iou
                    or dist <= self.config.detector.lost_track_match_center_px
                )
            else:
                matched = iou >= self.config.detector.track_match_iou
            if not matched:
                continue
            score = iou
            if state == "active":
                score += 0.25
            score -= min(dist, 500.0) / 5000.0
            matches.append({
                "track_id": int(candidate["track_id"]),
                "state": state,
                "iou": iou,
                "dist": dist,
                "score": score,
            })
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches

    def _nearest_track_match(self, bbox: List[float], is_referee: Optional[bool] = None) -> Tuple[Optional[int], float]:
        matches = self._track_candidate_matches(bbox, is_referee)
        if not matches:
            return None, 0.0
        return int(matches[0]["track_id"]), float(matches[0]["iou"])

    def _nearest_track_id(self, bbox: List[float], is_referee: Optional[bool] = None) -> Optional[int]:
        tid, _ = self._nearest_track_match(bbox, is_referee)
        return tid

    def _associated_track_age(
        self,
        bbox: List[float],
        is_referee: Optional[bool],
        frame_idx: int,
    ) -> int:
        matches = self._track_candidate_matches(bbox, is_referee)
        if not matches:
            return 0
        birth_frame = self._track_birth_frame.get(int(matches[0]["track_id"]))
        if birth_frame is None:
            return 0
        return max(0, int(frame_idx) - int(birth_frame))

    def _update_track_births(self, tracks: List[Dict], frame_idx: int) -> None:
        for track in tracks:
            tid = int(track["track_id"])
            if tid in self._track_birth_frame:
                continue
            self._track_birth_frame[tid] = frame_idx
            self._track_birth_center[tid] = self._bbox_center(track["bbox"])

    def _different_track_births(self, a: Dict, b: Dict) -> bool:
        a_tid = int(a["track_id"])
        b_tid = int(b["track_id"])
        if a_tid == b_tid:
            return False
        a_frame = self._track_birth_frame.get(a_tid)
        b_frame = self._track_birth_frame.get(b_tid)
        a_center = self._track_birth_center.get(a_tid)
        b_center = self._track_birth_center.get(b_tid)
        if a_frame is None or b_frame is None or a_center is None or b_center is None:
            return False

        dt = abs(a_frame - b_frame)
        dist = ((a_center[0] - b_center[0]) ** 2 + (a_center[1] - b_center[1]) ** 2) ** 0.5
        return (
            dt > self.config.detector.duplicate_birth_max_dt
            or dist > self.config.detector.duplicate_birth_max_dist
        )

    def _track_age(self, track: Dict, frame_idx: int) -> int:
        birth_frame = self._track_birth_frame.get(int(track["track_id"]))
        if birth_frame is None:
            return 0
        return max(0, int(frame_idx) - int(birth_frame))

    def _young_established_pair(
        self,
        a: Dict,
        b: Dict,
        frame_idx: int,
    ) -> Tuple[Optional[Dict], Optional[Dict], int, int]:
        a_age = self._track_age(a, frame_idx)
        b_age = self._track_age(b, frame_idx)
        young_max = self.config.tracker.duplicate_new_track_max_age
        established_min = self.config.tracker.duplicate_existing_min_age

        if a_age <= young_max and b_age >= established_min:
            return a, b, a_age, b_age
        if b_age <= young_max and a_age >= established_min:
            return b, a, b_age, a_age
        return None, None, a_age, b_age

    def _candidate_age(self, candidate: Dict, frame_idx: int) -> int:
        birth_frame = self._track_birth_frame.get(int(candidate["track_id"]))
        if birth_frame is None:
            return 0
        return max(0, int(frame_idx) - int(birth_frame))

    @staticmethod
    def _format_track_candidates(candidates: List[Dict]) -> str:
        if not candidates:
            return "[]"
        return "[" + ",".join(
            f"{int(c['track_id'])}:{c['state']}:iou={float(c['iou']):.2f}:d={float(c['dist']):.0f}"
            for c in candidates[:5]
        ) + "]"

    @staticmethod
    def _distinct_candidate_pair(a_candidates: List[Dict], b_candidates: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
        best_pair = None
        best_score = -1e9
        for a in a_candidates:
            for b in b_candidates:
                if int(a["track_id"]) == int(b["track_id"]):
                    continue
                score = float(a["score"]) + float(b["score"])
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)
        return best_pair

    def _det_nms_bypass(
        self,
        det: Dict,
        kept_det: Dict,
        frame_idx: int,
        det_iou: float,
    ) -> bool:
        if not self.config.detector.track_aware_suppression:
            return False
        is_ref = det["class_id"] == self.config.classes.referee_class
        kept_is_ref = kept_det["class_id"] == self.config.classes.referee_class
        if is_ref != kept_is_ref:
            return False
        dropped_candidates = self._track_candidate_matches(det["bbox"], is_ref)
        kept_candidates = self._track_candidate_matches(kept_det["bbox"], kept_is_ref)
        if self.config.debug.suppression:
            print(
                f"    [nms:candidates det f={frame_idx}] "
                f"drop={self._format_track_candidates(dropped_candidates)} "
                f"kept={self._format_track_candidates(kept_candidates)}"
            )
        pair = self._distinct_candidate_pair(dropped_candidates, kept_candidates)
        if pair is None:
            return False
        dropped_match, kept_match = pair
        dropped_age = self._candidate_age(dropped_match, frame_idx)
        kept_age = self._candidate_age(kept_match, frame_idx)
        young_max = self.config.tracker.duplicate_new_track_max_age
        established_min = self.config.tracker.duplicate_existing_min_age
        young_overlap = (
            dropped_age <= young_max
            and kept_age >= established_min
        ) or (
            kept_age <= young_max
            and dropped_age >= established_min
        )
        if young_overlap and det_iou >= self.config.detector.detection_nms_iou:
            if self.config.debug.suppression:
                print(
                    f"    [nms:hold det f={frame_idx}] "
                    f"tid={int(dropped_match['track_id'])}({dropped_age}f) "
                    f"vs tid={int(kept_match['track_id'])}({kept_age}f) "
                    f"det_iou={det_iou:.2f} reason=young_overlap"
                )
            return False
        if self.config.debug.suppression:
            print(
                f"    [nms:bypass det f={frame_idx}] "
                f"tid={int(dropped_match['track_id'])}({dropped_match['state']}) "
                f"vs tid={int(kept_match['track_id'])}({kept_match['state']}) "
                f"det_iou={det_iou:.2f} "
                f"match=({float(dropped_match['iou']):.2f},{float(kept_match['iou']):.2f}) "
                f"reason=different_active_or_lost_tracks"
            )
        return True

    def _track_nms_bypass(self, track: Dict, kept_track: Dict, frame_idx: int, track_iou: float) -> bool:
        if not self.config.detector.track_aware_suppression:
            return False
        if bool(track.get("is_referee", False)) != bool(kept_track.get("is_referee", False)):
            return False
        young, established, young_age, established_age = self._young_established_pair(
            track,
            kept_track,
            frame_idx,
        )
        if young is not None and track_iou >= self.config.tracker.duplicate_new_track_iou:
            if self.config.debug.suppression:
                print(
                    f"    [nms:hold track f={frame_idx}] "
                    f"young={int(young['track_id'])}({young_age}f) "
                    f"established={int(established['track_id'])}({established_age}f) "
                    f"iou={track_iou:.2f} reason=young_overlap"
                )
            return False
        if not self._different_track_births(track, kept_track):
            return False
        if self.config.debug.suppression:
            print(
                f"    [nms:bypass track f={frame_idx}] "
                f"tid={int(track['track_id'])} vs tid={int(kept_track['track_id'])} "
                f"iou={track_iou:.2f} reason=different_birth"
            )
        return True

    def _dedupe_target_detections(self, dets: List[Dict], frame_idx: int = 0) -> List[Dict]:
        """Suppress near-duplicate player/referee detections before tracking."""
        target_classes = set(self.config.classes.player_classes + [self.config.classes.referee_class])
        target = [d for d in dets if d["class_id"] in target_classes]
        other = [d for d in dets if d["class_id"] not in target_classes]
        target.sort(
            key=lambda d: (
                min(
                    self._associated_track_age(
                        d["bbox"],
                        d["class_id"] == self.config.classes.referee_class,
                        frame_idx,
                    ),
                    60,
                ),
                float(d["confidence"]),
            ),
            reverse=True,
        )

        kept: List[Dict] = []
        for det in target:
            is_ref = det["class_id"] == self.config.classes.referee_class
            keep = True
            suppress_by = None
            suppress_iou = 0.0
            for prev in kept:
                prev_is_ref = prev["class_id"] == self.config.classes.referee_class
                iou = self._bbox_iou(det["bbox"], prev["bbox"])
                if is_ref == prev_is_ref and iou >= self.config.detector.detection_nms_iou:
                    if self._det_nms_bypass(det, prev, frame_idx, iou):
                        continue
                    keep = False
                    suppress_by = prev
                    suppress_iou = iou
                    break
                if is_ref != prev_is_ref and iou >= self.config.detector.cross_role_iou:
                    keep = False
                    suppress_by = prev
                    suppress_iou = iou
                    break
            if keep:
                kept.append(det)
            elif self.config.debug.suppression and suppress_by is not None:
                dropped_tid = self._nearest_track_id(det["bbox"], is_ref)
                kept_tid = self._nearest_track_id(
                    suppress_by["bbox"],
                    suppress_by["class_id"] == self.config.classes.referee_class,
                )
                print(
                    f"    [supp:det f={frame_idx}] drop near_tid={dropped_tid} "
                    f"cls={det['class_id']} conf={det['confidence']:.2f} "
                    f"iou={suppress_iou:.2f} vs kept near_tid={kept_tid} conf={suppress_by['confidence']:.2f}"
                )
                pair = (dropped_tid, kept_tid)
                if dropped_tid is not None and dropped_tid == kept_tid:
                    self._supp_det_duplicate_pairs.setdefault(pair, []).append(frame_idx)
                    print(
                        f"    [nms:drop duplicate f={frame_idx}] "
                        f"tid={dropped_tid} iou={suppress_iou:.2f} reason=same_prev_track"
                    )
                else:
                    self._supp_det_pairs.setdefault(pair, []).append(frame_idx)
                    print(
                        f"    [nms:drop duplicate f={frame_idx}] "
                        f"drop_tid={dropped_tid} kept_tid={kept_tid} "
                        f"iou={suppress_iou:.2f} reason=no_distinct_candidate"
                    )
        return kept + other

    def _dedupe_tracks(self, tracks: List[Dict], frame_idx: int = 0) -> List[Dict]:
        """Remove overlapping duplicate tracks and role-conflict overlays."""
        if not tracks:
            return tracks
        ordered = sorted(
            tracks,
            key=lambda t: (
                min(self._track_age(t, frame_idx), 60),
                float(t.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        kept: List[Dict] = []
        for tr in ordered:
            keep = True
            suppress_by = None
            suppress_iou = 0.0
            for prev in kept:
                iou = self._bbox_iou(tr["bbox"], prev["bbox"])
                if tr["is_referee"] == prev["is_referee"] and iou >= self.config.detector.track_nms_iou:
                    if self._track_nms_bypass(tr, prev, frame_idx, iou):
                        continue
                    keep = False
                    suppress_by = prev
                    suppress_iou = iou
                    break
                if tr["is_referee"] != prev["is_referee"] and iou >= self.config.detector.cross_role_iou:
                    keep = False
                    suppress_by = prev
                    suppress_iou = iou
                    break
            if keep:
                kept.append(tr)
            elif self.config.debug.suppression and suppress_by is not None:
                print(
                    f"    [supp:track f={frame_idx}] drop tid={int(tr['track_id'])} "
                    f"conf={tr.get('confidence', 0.0):.2f} iou={suppress_iou:.2f} "
                    f"vs kept tid={int(suppress_by['track_id'])} conf={suppress_by.get('confidence', 0.0):.2f}"
                )
                if tr["is_referee"] == suppress_by["is_referee"]:
                    print(
                        f"    [nms:drop duplicate f={frame_idx}] "
                        f"tid={int(tr['track_id'])} vs tid={int(suppress_by['track_id'])} "
                        f"iou={suppress_iou:.2f} reason=same_birth"
                    )
        return kept

    def _canonical_track_id(self, track_id: int) -> int:
        tid = int(track_id)
        # Keep this one-hop: a switch should preserve the ID that was visible
        # immediately before the raw tracker changed, not collapse old aliases.
        return int(self._track_id_alias.get(tid, tid))

    def _relink_track_id(self, old_track_id: int, new_track_id: int, frame_idx: int, reason: str) -> None:
        old_tid = int(old_track_id)
        new_tid = int(new_track_id)
        stable = old_tid
        if new_tid == stable:
            return
        self._track_id_alias[new_tid] = stable
        if self.config.debug.lifecycle or self.config.debug.suppression:
            print(
                f"    [id-alias f={frame_idx}] raw={new_tid} -> stable={stable} "
                f"source={old_tid} reason={reason}"
            )

    def _consume_lifecycle_alias_events(self, frame_idx: int) -> None:
        if not hasattr(self.tracker, "get_lifecycle_events"):
            return
        events_by_role = self.tracker.get_lifecycle_events()
        player_events = events_by_role.get("player", [])
        new_events = player_events[self._lifecycle_event_cursor:]
        self._lifecycle_event_cursor = len(player_events)
        for event in new_events:
            if event.get("type") != "switch":
                continue
            new_tid = event.get("new")
            if new_tid is None:
                continue
            source_tid = event.get("lost", event.get("rm"))
            if source_tid is None:
                continue
            event_frame = int(event.get("frame", frame_idx))
            self._relink_track_id(int(source_tid), int(new_tid), event_frame, "switch")

    def _apply_canonical_track_ids(self, tracks: List[Dict]) -> List[Dict]:
        canonical_tracks: List[Dict] = []
        for track in tracks:
            if track.get("is_referee", False):
                canonical_tracks.append(track)
                continue
            raw_display_id = int(track["track_id"])
            stable_id = self._canonical_track_id(raw_display_id)
            if stable_id == raw_display_id:
                canonical_tracks.append(track)
                continue
            updated = dict(track)
            updated["raw_display_track_id"] = raw_display_id
            updated["stable_track_id"] = stable_id
            updated["track_id"] = stable_id
            canonical_tracks.append(updated)
        return canonical_tracks

    def _draw_zebra_masks(
        self,
        frame: np.ndarray,
        mask_tracks: List[Dict],
        masks: np.ndarray,
    ) -> np.ndarray:
        result = frame.copy()
        cfg = self.config.visualization
        h, w = result.shape[:2]

        for track, mask in zip(mask_tracks, masks):
            mask = mask.astype(bool)
            if not np.any(mask):
                continue

            color = _track_color_bgr(int(track["track_id"]))
            fill = np.zeros_like(result)
            fill[mask] = color
            result[mask] = (
                result[mask].astype(np.float32) * (1.0 - cfg.mask_alpha)
                + fill[mask].astype(np.float32) * cfg.mask_alpha
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(result, contours, -1, color, cfg.mask_border_thickness)

            ys, xs = np.where(mask)
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            jersey = self.jersey_bank.get_jersey(track["track_id"])
            center_label = f"#{jersey}" if jersey else f"ID {track['track_id']}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = cfg.mask_center_text_scale
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(center_label, font, scale, thickness)
            x1 = max(0, min(w - tw - 10, cx - tw // 2 - 5))
            y1 = max(th + 10, min(h - baseline - 8, cy + th // 2))
            label_bg = tuple(int(c * 0.65) for c in color)
            cv2.rectangle(
                result,
                (x1 - 5, y1 - th - baseline - 5),
                (x1 + tw + 5, y1 + baseline + 5),
                label_bg,
                -1,
            )
            cv2.putText(result, center_label, (x1, y1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return result

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
            jersey = self.jersey_bank.get_jersey(sid) or "-"
            jersey_label = f"#{jersey}" if jersey != "-" else "-"
            cx, cy = self._bbox_center_xy(tr["bbox"])
            current[sid] = {
                "raw": rid,
                "jersey": jersey_label,
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
                if raw_changed or jump >= self.config.debug.id_jump_px:
                    print(
                        f"    [id-track] frame={frame_idx} stable={sid} raw={rid} jersey={jersey_label} "
                        f"prev_raw={int(prev['raw'])} cx={cx:.1f} cy={cy:.1f} "
                        f"jump={jump:.1f} raw_changed={int(raw_changed)} "
                        f"bbox={tr['bbox']}"
                    )
            else:
                print(
                    f"    [id-new] frame={frame_idx} stable={sid} raw={rid} jersey={jersey_label} "
                    f"cx={cx:.1f} cy={cy:.1f} bbox={tr['bbox']}"
                )

        prev_ids = set(self._prev_id_debug.keys())
        current_ids = set(current.keys())
        for sid in sorted(prev_ids - current_ids):
            prev = self._prev_id_debug[sid]
            px, py = prev["center"]
            jersey = prev.get("jersey", "-")
            print(
                f"    [id-lost] frame={frame_idx} stable={sid} raw={int(prev['raw'])} jersey={jersey} "
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
                    cross_ab + cross_ba + self.config.debug.id_swap_px
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
            if d["class_id"] == self.config.classes.jersey_number_class
            and float(d.get("confidence", 0.0)) >= self.config.jersey.min_confidence
            and (d["bbox"][2] - d["bbox"][0]) >= self.config.jersey.min_size
            and (d["bbox"][3] - d["bbox"][1]) >= self.config.jersey.min_size
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

            ew  = int(bw * self.config.jersey.expand)
            eh  = int(bh * self.config.jersey.expand)
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
            if best_count >= self.config.jersey.lock_votes:
                self.locked_jersey_ids.add(matched_id)
                confirmed = self.jersey_bank.get_jersey(matched_id)
                if self.config.debug.enabled:
                    print(f"  Jersey locked: ID {matched_id} → #{confirmed}")

    def _ocr_jersey(self, crop: np.ndarray) -> str:
        try:
            text, conf = self.jersey_detector._ocr(crop)
            number = self.jersey_detector._text_to_number(text)
            return number if conf > 0.3 else ""
        except Exception:
            return ""

    # ── Feet positions for tactical view ──────────────────────────────────────

    def _update_tactical_history(
        self,
        player_ids: List[int],
        player_feet: List[List[float]],
        H: Optional[np.ndarray],
    ) -> None:
        if H is None:
            return

        active_ids = set(player_ids)
        for track_id, foot_pt in zip(player_ids, player_feet):
            try:
                dst = cv2.perspectiveTransform(np.array([[foot_pt]], dtype=np.float32), H)
                px = float(dst[0][0][0])
                py = float(dst[0][0][1])
            except cv2.error:
                continue

            px = float(np.clip(px, 0, TACTICAL_WIDTH - 1))
            py = float(np.clip(py, 0, TACTICAL_HEIGHT - 1))

            previous = self._tactical_smoothed.get(track_id)
            if previous is None:
                smoothed = (px, py)
            else:
                max_step = self.config.visualization.tactical_max_step_px
                dx = px - previous[0]
                dy = py - previous[1]
                step = float((dx * dx + dy * dy) ** 0.5)
                if max_step > 0 and step > max_step:
                    scale = max_step / step
                    px = previous[0] + dx * scale
                    py = previous[1] + dy * scale
                alpha = self.config.visualization.tactical_smoothing
                smoothed = (
                    previous[0] * alpha + px * (1.0 - alpha),
                    previous[1] * alpha + py * (1.0 - alpha),
                )
            self._tactical_smoothed[track_id] = smoothed
            self._tactical_history[track_id].append(smoothed)

        stale_ids = set(self._tactical_smoothed.keys()) - active_ids
        for track_id in stale_ids:
            self._tactical_smoothed.pop(track_id, None)

    @staticmethod
    def _tracks_to_feet(
        tracks: List[Dict], jersey_bank
    ) -> Tuple[List[int], List[List[float]], List[Optional[str]]]:
        ids, feet, jerseys = [], [], []
        for t in tracks:
            if t["is_referee"]:
                continue
            x1, y1, x2, y2 = t["bbox"]
            ids.append(int(t["track_id"]))
            feet.append([float((x1 + x2) / 2), float(y2)])
            jerseys.append(jersey_bank.get_jersey(t["track_id"]))
        return ids, feet, jerseys

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
            if self.config.visualization.draw_masks and hasattr(self.tracker, "get_player_masks_for_tracks"):
                mask_tracks, masks = self.tracker.get_player_masks_for_tracks(player_tracks, result.shape[:2])
                if mask_tracks and masks is not None:
                    result = self._draw_zebra_masks(result, mask_tracks, masks)

            p_dets  = _tracks_to_sv(player_tracks)
            p_labels = [
                f"#{self.jersey_bank.get_jersey(t['track_id'])}"
                if self.jersey_bank.get_jersey(t["track_id"])
                else str(t["track_id"])
                for t in player_tracks
            ]
            if self.config.visualization.trail_length > 0:
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
                    iou_threshold=self.config.detector.iou_threshold,
                )

                src_idx = start_frame + frame_idx * frame_skip

                if frame_idx % self.config.keypoints.update_interval == 0:
                    kp_xy, kp_conf, H = self._detect_keypoints(frame, frame_idx=frame_idx, src_idx=src_idx)
                    self._last_keypoints = (kp_xy, kp_conf, H)
                else:
                    kp_xy, kp_conf, H = self._last_keypoints

                if self.config.detector.activate_tracks_by_court_mask:
                    court_dets = list(dets)
                else:
                    court_dets = [
                        d for d in dets
                        if self._keep_detection_for_court_filter(d, kp_xy, frame.shape[:2])
                    ]
                filtered_dets = self._dedupe_target_detections(court_dets, frame_idx)

                raw_players = sum(1 for d in dets if d["class_id"] in self.config.classes.player_classes)
                raw_refs = sum(1 for d in dets if d["class_id"] == self.config.classes.referee_class)
                court_players = sum(1 for d in court_dets if d["class_id"] in self.config.classes.player_classes)
                court_refs = sum(1 for d in court_dets if d["class_id"] == self.config.classes.referee_class)
                filt_players = sum(1 for d in filtered_dets if d["class_id"] in self.config.classes.player_classes)
                filt_refs = sum(1 for d in filtered_dets if d["class_id"] == self.config.classes.referee_class)
                dropped_players = raw_players - filt_players
                if self.config.debug.tracking and dropped_players > 0:
                    print(
                        f"    [det-debug] frame={frame_idx} src={src_idx} "
                        f"players={raw_players}->{filt_players}(dropped={dropped_players}) "
                        f"refs={raw_refs}->{filt_refs}"
                    )

                tracks = self.tracker.update(filtered_dets, frame)
                self._update_track_births(tracks, frame_idx)
                tracker_player_count = sum(1 for t in tracks if not t.get("is_referee", False))
                tracker_ref_count = sum(1 for t in tracks if t.get("is_referee", False))
                tracks = self._dedupe_tracks(tracks, frame_idx)
                dedupe_player_count = sum(1 for t in tracks if not t.get("is_referee", False))
                dedupe_ref_count = sum(1 for t in tracks if t.get("is_referee", False))
                tracks = self._activate_tracks_by_court_mask(
                    tracks,
                    H,
                    frame.shape[:2],
                    frame_idx,
                )
                self._prev_tracks = list(tracks)
                self._consume_lifecycle_alias_events(frame_idx)
                tracks = self._apply_canonical_track_ids(tracks)
                self._last_player_tracks = [t for t in tracks if not t["is_referee"]]
                if self.config.debug.suppression:
                    active_players = len(self._last_player_tracks)
                    active_refs = sum(1 for t in tracks if t.get("is_referee", False))
                    if (
                        raw_players < 10
                        or filt_players < 10
                        or active_players < 10
                        or frame_idx % self.config.debug.progress_interval == 0
                    ):
                        print(
                            f"    [det-flow] frame={frame_idx} src={src_idx} "
                            f"players raw={raw_players} court={court_players} nms={filt_players} "
                            f"tracker={tracker_player_count} dedupe={dedupe_player_count} "
                            f"active={active_players} refs raw={raw_refs} court={court_refs} "
                            f"nms={filt_refs} tracker={tracker_ref_count} "
                            f"dedupe={dedupe_ref_count} active={active_refs}"
                        )
                if self.config.debug.tracking:
                    self._log_id_debug(tracks, frame_idx)

                if frame_idx % self.config.jersey.update_interval == 0 and tracks:
                    self._detect_jerseys(frame, tracks, dets)

                result  = self._annotate_frame(frame, tracks, kp_xy, kp_conf, H, frame_idx, src_idx)

                if writer is None:
                    h, w = result.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
                    tactical_writer = cv2.VideoWriter(
                        tactical_path, fourcc, out_fps, (tactical_w, tactical_h)
                    )

                writer.write(result)

                player_ids, player_feet, player_jerseys = self._tracks_to_feet(tracks, self.jersey_bank)
                self._update_tactical_history(player_ids, player_feet, H)
                tactical = draw_tactical_view(
                    self.court_img,
                    kp_xy,
                    H,
                    player_ids=player_ids,
                    player_feet=player_feet,
                    player_jerseys=player_jerseys,
                    position_history=self._tactical_history,
                    point_radius=self.config.visualization.tactical_point_radius,
                    trail_thickness=self.config.visualization.tactical_trail_thickness,
                )
                tactical_writer.write(cv2.resize(tactical, (tactical_w, tactical_h)))

                if show_preview:
                    try:
                        max_w   = 1280
                        display = (
                            cv2.resize(result, (max_w, int(result.shape[0] * max_w / result.shape[1])))
                            if result.shape[1] > max_w else result
                        )
                        cv2.imshow("MCByte Tracker", display)
                        cv2.imshow("Tactical View", cv2.resize(tactical, (tactical_w, tactical_h)))
                        cv2.waitKey(1)
                    except cv2.error:
                        pass

                n_kp = int((kp_xy > 0).all(axis=1).sum())
                if self.config.debug.enabled and frame_idx % self.config.debug.progress_interval == 0:
                    print(f"  frame {frame_idx:4d}  tracks={len(tracks):2d}  kp={n_kp}/18  jersey={len(self.jersey_bank.obj_to_jersey)}")
                    dbg = self.tracker.get_debug_summary()
                    p = dbg.get("player", {})
                    r = dbg.get("referee", {})
                    if p:
                        print(
                            "    [mcbyte-player] "
                            f"tracked={int(p.get('tracked', 0))} "
                            f"lost={int(p.get('lost', 0))} "
                            f"removed={int(p.get('removed', 0))} "
                            f"masks={int(p.get('masks', 0))}"
                        )
                    if r:
                        print(
                            "    [mcbyte-ref]    "
                            f"tracked={int(r.get('tracked', 0))} "
                            f"lost={int(r.get('lost', 0))} "
                            f"removed={int(r.get('removed', 0))} "
                            f"masks={int(r.get('masks', 0))}"
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

        summary = self.tracker.get_lifecycle_summary()
        for role, s in summary.items():
            if not s.get("new") and not s.get("removed"):
                continue
            print(
                f"\n  [{role}] lifecycle: "
                f"new={s['new']} lost={s['lost']} rec={s['recovered']} "
                f"rm={s['removed']} switches={s['switches']} | "
                f"life avg={s['avg_life']:.0f}f med={s['median_life']}f p90={s['p90_life']}f"
            )
            if s["shortest_lived"]:
                print(f"    shortest: {s['shortest_lived']}")
            if s["most_recovered"]:
                print(f"    most_recovered: {s['most_recovered']}")

        if self.config.tracking_report_json:
            import json
            with open(self.config.tracking_report_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Lifecycle report: {self.config.tracking_report_json}")

        if self.config.debug.suppression and self._supp_det_pairs:
            print("\n  [hold] Detection suppression pairs (≥5 consecutive or ≥10 total frames):")
            reported = []
            for (dropped, kept), frames in self._supp_det_pairs.items():
                frames_sorted = sorted(set(frames))
                max_run, cur_run, cur_start = 0, 1, frames_sorted[0]
                run_start = run_end = frames_sorted[0]
                for i in range(1, len(frames_sorted)):
                    if frames_sorted[i] == frames_sorted[i - 1] + 1:
                        cur_run += 1
                    else:
                        if cur_run > max_run:
                            max_run, run_start, run_end = cur_run, cur_start, frames_sorted[i - 1]
                        cur_run, cur_start = 1, frames_sorted[i]
                if cur_run > max_run:
                    max_run, run_start, run_end = cur_run, cur_start, frames_sorted[-1]
                if max_run >= 5 or len(frames) >= 10:
                    reported.append((dropped, kept, len(frames), max_run, run_start, run_end))
            if reported:
                reported.sort(key=lambda x: -x[3])
                for dropped, kept, total, streak, sf, ef in reported:
                    print(
                        f"    (dropped~tid={dropped}, kept~tid={kept}): "
                        f"total={total}f max_streak={streak}f (f={sf}-{ef})"
                    )
            else:
                print("    (no persistent pairs found)")

        if self.config.debug.suppression and self._supp_det_duplicate_pairs:
            print("\n  [duplicate] Same-track detection suppressions:")
            reported = []
            for (dropped, kept), frames in self._supp_det_duplicate_pairs.items():
                frames_sorted = sorted(set(frames))
                reported.append((dropped, kept, len(frames), frames_sorted[0], frames_sorted[-1]))
            reported.sort(key=lambda x: -x[2])
            for dropped, kept, total, sf, ef in reported[:10]:
                print(
                    f"    (tid={dropped}, kept={kept}): "
                    f"total={total}f span=f{sf}-{ef}"
                )

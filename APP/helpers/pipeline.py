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
from typing import Any, Deque, Dict, List, Optional, Tuple
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from APP.helpers.rfdetr_detector import RFDETRDetector
from APP.helpers.mcbyte_tracker import McByteBasketballTracker
from APP.helpers.config import PipelineConfig
from APP.helpers.court_utils import (
    TACTICAL_WIDTH, TACTICAL_HEIGHT, TACTICAL_KEYPOINTS,
    compute_homography, draw_keypoints_on_frame, draw_tactical_view,
    build_minimal_court,
    KP_NAMES, ROBOFLOW_COURT_KEYPOINT_LABELS, ROBOFLOW_LABEL_TO_TACTICAL_INDEX,
    ROBOFLOW_TACTICAL_KEYPOINTS, ROBOFLOW_TO_TACTICAL_INDEX,
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


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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

        self.kp_backend = str(self.config.keypoints.backend).lower()
        self._roboflow_native_keypoints = self.kp_backend == "roboflow"
        if self._roboflow_native_keypoints:
            self._kp_len = len(ROBOFLOW_TACTICAL_KEYPOINTS)
            self._kp_left_indices = [0, 1, 2, 3, 4, 5]
            self._kp_right_indices = [27, 28, 29, 30, 31, 32]
            self._kp_center_indices = [15, 16, 17]
            self._kp_top_indices = [0, 15, 27]
            self._kp_bottom_indices = [5, 17, 32]
            self._kp_left_inner_indices = [9, 11]
            self._kp_right_inner_indices = [21, 23]
            self._kp_inner_indices = set(
                self._kp_left_inner_indices + self._kp_right_inner_indices
            )
            self._base_tactical_dst = np.array(ROBOFLOW_TACTICAL_KEYPOINTS, dtype=np.float32)
        else:
            self._kp_len = len(TACTICAL_KEYPOINTS)
            self._kp_left_indices = list(self.config.keypoints.left_indices)
            self._kp_right_indices = list(self.config.keypoints.right_indices)
            self._kp_center_indices = list(self.config.keypoints.center_indices)
            self._kp_top_indices = list(self.config.keypoints.top_indices)
            self._kp_bottom_indices = list(self.config.keypoints.bottom_indices)
            self._kp_left_inner_indices = [8, 9]
            self._kp_right_inner_indices = [16, 17]
            self._kp_inner_indices = set(
                self._kp_left_inner_indices + self._kp_right_inner_indices
            )
            self._base_tactical_dst = np.array(TACTICAL_KEYPOINTS, dtype=np.float32)

        self.kp_model = None
        if self.kp_backend == "roboflow":
            self.kp_model = self._load_roboflow_keypoint_model()
        else:
            kp_path = self.config.keypoints.model_path
            self.kp_model = YOLO(kp_path) if kp_path and os.path.exists(kp_path) else None
            if self.kp_model is None:
                print(f"  Warning: keypoint model not found at {kp_path}")
            else:
                print(f"  Court keypoints: local YOLO ({kp_path})")

        court_path = self.config.keypoints.court_image_path
        if court_path and os.path.exists(court_path):
            loaded = cv2.imread(court_path)
            self.court_img = cv2.resize(loaded, (TACTICAL_WIDTH, TACTICAL_HEIGHT)) if loaded is not None else build_minimal_court()
        else:
            self.court_img = build_minimal_court()

        # Keypoint / homography state
        self.last_good_keypoints         = None
        self.last_H                      = None
        self._court_side: Optional[str]  = None
        self.homography_success_count    = 0
        self._id_switch_count            = 0
        self._last_keypoints             = (
            np.zeros((self._kp_len, 2), dtype=np.float32),
            np.zeros(self._kp_len),
            None,
        )
        self._prev_detected_keypoints    = np.zeros((self._kp_len, 2), dtype=np.float32)
        self._keypoint_stationary_counts = np.zeros(self._kp_len, dtype=np.int32)
        self._keypoint_missing_counts    = np.full(
            self._kp_len,
            self.config.keypoints.carry_missing_updates + 1,
            dtype=np.int32,
        )

        # Side-switch hysteresis and transition state
        self._pending_side: Optional[str]  = None
        self._pending_side_count: int      = 0
        self._side_transition_frames: int  = 0
        self._h_fallback_streak: int       = 0
        self._last_center_kp_x: Optional[float] = None
        self._force_homography_reseed: bool = False
        self._last_H_mode: Optional[str] = None
        self._tactical_dst_normal = self._base_tactical_dst.copy()
        self._tactical_dst_mirrored = self._tactical_dst_normal.copy()
        self._tactical_dst_mirrored[:, 0] = TACTICAL_WIDTH - self._tactical_dst_mirrored[:, 0]
        self._prev_pan_keypoints = np.zeros((self._kp_len, 2), dtype=np.float32)
        self._prev_pan_confidences = np.zeros(self._kp_len, dtype=np.float32)
        self._prev_pan_frame_idx: Optional[int] = None
        self._tactical_pan_prior_dx: float = 0.0
        self._tactical_pan_prior_conf: float = 0.0
        self._tactical_pan_prior_frame: int = -1
        self._last_layout_side: Optional[str] = None
        self._last_layout_slope: float = 0.0
        self._last_layout_pairs: int = 0
        self._rf_kp_schema_logged: bool = False

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
        self._tactical_last_seen: Dict[int, int] = {}


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

    def _load_roboflow_keypoint_model(self):
        if not os.getenv("ROBOFLOW_API_KEY"):
            raise SystemExit("ROBOFLOW_API_KEY env var is not set.")

        for var in (
            "PALIGEMMA_ENABLED", "FLORENCE2_ENABLED", "QWEN_2_5_ENABLED",
            "QWEN_3_ENABLED", "CORE_MODEL_SAM_ENABLED", "CORE_MODEL_SAM3_ENABLED",
            "CORE_MODEL_GAZE_ENABLED", "SMOLVLM2_ENABLED", "DEPTH_ESTIMATION_ENABLED",
            "MOONDREAM2_ENABLED", "CORE_MODEL_TROCR_ENABLED", "CORE_MODEL_GROUNDINGDINO_ENABLED",
        ):
            os.environ.setdefault(var, "False")

        try:
            from inference import get_model
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency. Run: pip uninstall inference -y && pip install inference-gpu"
            ) from exc

        model_id = self.config.keypoints.roboflow_model_id
        print(
            "  Court keypoints: Roboflow "
            f"{model_id} conf={self.config.keypoints.roboflow_confidence:.2f} "
            f"anchor>={self.config.keypoints.anchor_confidence:.2f}"
        )
        return get_model(model_id=model_id)

    def _warmup(self):
        print("  Warming up models...", end=" ", flush=True)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.player_detector.detect(dummy, confidence_threshold=0.5)
        if self.kp_model is not None and self.kp_backend == "roboflow":
            self.kp_model.infer(dummy, confidence=self.config.keypoints.roboflow_confidence)
        elif self.kp_model is not None:
            self.kp_model.predict(dummy, conf=self.config.keypoints.confidence, verbose=False,
                                  half=(self.device == "cuda"), device=self.device)
        print("done.")

    def _homography_shift_stats(
        self,
        H: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, float]:
        if self.last_H is None:
            return 0.0, 0.0
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
            return float("inf"), float("inf")
        if not np.isfinite(prev).all() or not np.isfinite(cur).all():
            return float("inf"), float("inf")
        shifts = np.linalg.norm(cur - prev, axis=1)
        return float(np.mean(shifts)), float(np.max(shifts))

    @staticmethod
    def _homography_reprojection_error(
        keypoints_xy: np.ndarray,
        H: np.ndarray,
        dst_keypoints: np.ndarray,
    ) -> float:
        valid_indices = [
            i for i in range(len(keypoints_xy))
            if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0
        ]
        if len(valid_indices) < 4:
            return float("inf")
        src = np.array([keypoints_xy[i] for i in valid_indices], dtype=np.float32)
        dst = np.array([dst_keypoints[i] for i in valid_indices], dtype=np.float32)
        try:
            projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        except cv2.error:
            return float("inf")
        if not np.isfinite(projected).all():
            return float("inf")
        residuals = np.linalg.norm(projected - dst, axis=1)
        return float(np.median(residuals))

    @staticmethod
    def _candidate_destination_side(
        keypoints_xy: np.ndarray,
        dst_keypoints: np.ndarray,
    ) -> Optional[str]:
        xs = [
            float(dst_keypoints[i][0])
            for i in range(len(keypoints_xy))
            if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0
        ]
        if not xs:
            return None
        mean_x = sum(xs) / len(xs)
        if mean_x < TACTICAL_WIDTH * 0.43:
            return "left"
        if mean_x > TACTICAL_WIDTH * 0.57:
            return "right"
        return "center"

    def _select_homography_candidate(
        self,
        keypoints_xy: np.ndarray,
        observed_side: Optional[str],
        frame_shape: Tuple[int, int, int],
        max_reproj_px: float,
        use_stability: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, float | str]]:
        candidates = []
        for mode, dst_template in (
            ("normal", self._tactical_dst_normal),
            ("mirrored", self._tactical_dst_mirrored),
        ):
            H = compute_homography(
                keypoints_xy,
                max_reproj_px=max_reproj_px,
                dst_keypoints=dst_template,
            )
            if H is None:
                continue

            reproj = self._homography_reprojection_error(keypoints_xy, H, dst_template)
            mean_shift, max_shift = self._homography_shift_stats(H, frame_shape)
            candidate_side = self._candidate_destination_side(keypoints_xy, dst_template)

            score = reproj
            if use_stability and self.last_H is not None:
                score += mean_shift * self.config.keypoints.homography_stability_weight
                score += max_shift * self.config.keypoints.homography_max_shift_weight
                if (
                    self._last_H_mode is not None
                    and mode != self._last_H_mode
                    and not self._force_homography_reseed
                ):
                    score += self.config.keypoints.homography_mode_switch_penalty
            if observed_side in ("left", "right") and candidate_side in ("left", "right"):
                if candidate_side != observed_side:
                    score += self.config.keypoints.homography_side_penalty

            candidates.append(
                {
                    "H": H,
                    "mode": mode,
                    "score": float(score),
                    "reproj": float(reproj),
                    "mean_shift": float(mean_shift),
                    "max_shift": float(max_shift),
                    "side": candidate_side or "-",
                }
            )

        if not candidates:
            return None, None, {}
        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        meta = {
            "score": best["score"],
            "reproj": best["reproj"],
            "mean_shift": best["mean_shift"],
            "max_shift": best["max_shift"],
            "side": best["side"],
        }
        return best["H"], str(best["mode"]), meta

    def _update_keypoint_pan_prior(
        self,
        keypoints_xy: np.ndarray,
        confidences: np.ndarray,
        frame_width: int,
        frame_idx: Optional[int],
    ) -> Tuple[float, float, int, float]:
        cfg = self.config.visualization
        if not cfg.tactical_pan_prior_enabled:
            return 0.0, 0.0, 0, 0.0

        current_frame = -1 if frame_idx is None else int(frame_idx)
        valid = (
            (keypoints_xy > 0).all(axis=1)
            & (confidences >= self.config.keypoints.geometry_confidence)
        )
        prev_valid = (
            (self._prev_pan_keypoints > 0).all(axis=1)
            & (self._prev_pan_confidences >= self.config.keypoints.geometry_confidence)
        )
        matched = valid & prev_valid
        matched_count = int(np.count_nonzero(matched))

        prior_dx = 0.0
        prior_conf = 0.0
        median_dx = 0.0
        mad_dx = 0.0

        if matched_count >= cfg.tactical_pan_prior_min_keypoints and self._prev_pan_frame_idx is not None:
            dt = max(1, current_frame - int(self._prev_pan_frame_idx))
            dxs = keypoints_xy[matched, 0] - self._prev_pan_keypoints[matched, 0]
            median_dx = float(np.median(dxs))
            mad_dx = float(np.median(np.abs(dxs - median_dx)))

            if abs(median_dx) >= cfg.tactical_pan_prior_min_image_px:
                median_sign = 1.0 if median_dx > 0 else -1.0
                agreement = float(np.mean(np.sign(dxs) == median_sign))
                mad_quality = max(0.0, 1.0 - mad_dx / max(cfg.tactical_pan_prior_max_mad_px, 1e-6))
                prior_conf = agreement * mad_quality
                if mad_dx <= cfg.tactical_pan_prior_max_mad_px and prior_conf >= cfg.tactical_pan_prior_min_conf:
                    image_dx_per_frame = median_dx / dt
                    court_scale = TACTICAL_WIDTH / max(float(frame_width), 1.0)
                    prior_dx = -image_dx_per_frame * court_scale * cfg.tactical_pan_prior_gain

        if prior_conf >= cfg.tactical_pan_prior_min_conf:
            self._tactical_pan_prior_dx = (
                self._tactical_pan_prior_dx * 0.55 + prior_dx * 0.45
                if self._tactical_pan_prior_conf >= cfg.tactical_pan_prior_min_conf
                else prior_dx
            )
            self._tactical_pan_prior_conf = prior_conf
            self._tactical_pan_prior_frame = current_frame
        else:
            self._tactical_pan_prior_dx *= 0.75
            self._tactical_pan_prior_conf *= 0.75
            if abs(self._tactical_pan_prior_dx) < 0.05:
                self._tactical_pan_prior_dx = 0.0
                self._tactical_pan_prior_conf = 0.0

        self._prev_pan_keypoints = keypoints_xy.copy()
        self._prev_pan_confidences = confidences.copy()
        self._prev_pan_frame_idx = current_frame

        return self._tactical_pan_prior_dx, self._tactical_pan_prior_conf, matched_count, mad_dx

    def _current_tactical_dst_template(
        self,
        keypoints_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        target_side = self._last_layout_side or self._court_side
        if keypoints_xy is not None and target_side in ("left", "right"):
            normal_side = self._candidate_destination_side(keypoints_xy, self._tactical_dst_normal)
            mirrored_side = self._candidate_destination_side(keypoints_xy, self._tactical_dst_mirrored)
            if normal_side == target_side:
                return self._tactical_dst_normal
            if mirrored_side == target_side:
                return self._tactical_dst_mirrored
        return self._tactical_dst_mirrored if self._last_H_mode == "mirrored" else self._tactical_dst_normal

    def _relative_tactical_projection(
        self,
        foot_pt: List[float],
        keypoints_xy: Optional[np.ndarray],
        confidences: Optional[np.ndarray],
    ) -> Tuple[Optional[Tuple[float, float]], int]:
        cfg = self.config.visualization
        if not cfg.tactical_relative_projection or keypoints_xy is None:
            return None, 0

        valid_indices = []
        for idx in range(len(keypoints_xy)):
            if keypoints_xy[idx][0] <= 0 or keypoints_xy[idx][1] <= 0:
                continue
            valid_indices.append(idx)

        if len(valid_indices) < cfg.tactical_relative_min_keypoints:
            return None, len(valid_indices)

        src = np.array([keypoints_xy[i] for i in valid_indices], dtype=np.float32)
        dst_template = self._current_tactical_dst_template(keypoints_xy)
        dst = np.array([dst_template[i] for i in valid_indices], dtype=np.float32)
        foot = np.array(foot_pt, dtype=np.float32)

        dists = np.linalg.norm(src - foot.reshape(1, 2), axis=1)
        order = np.argsort(dists)
        nearest = order[:max(1, min(cfg.tactical_relative_nearest_keypoints, len(order)))]

        nearest_dists = dists[nearest]
        weights = 1.0 / np.power(nearest_dists + 1.0, cfg.tactical_relative_power)

        if confidences is not None and len(confidences) >= len(keypoints_xy):
            conf = np.array([float(confidences[valid_indices[i]]) for i in nearest], dtype=np.float32)
            conf = np.clip(conf, cfg.tactical_relative_conf_floor, 1.0)
            weights *= conf

        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            return None, len(valid_indices)

        projected = np.sum(dst[nearest] * weights.reshape(-1, 1), axis=0) / weight_sum
        px = float(np.clip(projected[0], 0, TACTICAL_WIDTH - 1))
        py = float(np.clip(projected[1], 0, TACTICAL_HEIGHT - 1))
        return (px, py), len(valid_indices)

    def _project_player_to_tactical(
        self,
        foot_pt: List[float],
        H: Optional[np.ndarray],
        keypoints_xy: Optional[np.ndarray],
        confidences: Optional[np.ndarray],
    ) -> Optional[Dict[str, float | str | int]]:
        cfg = self.config.visualization
        h_point = None
        h_inside = False
        raw_px = raw_py = 0.0

        if H is not None:
            try:
                dst = cv2.perspectiveTransform(np.array([[foot_pt]], dtype=np.float32), H)
                raw_px = float(dst[0][0][0])
                raw_py = float(dst[0][0][1])
                h_inside = (
                    -cfg.tactical_out_of_bounds_margin_px <= raw_px <= TACTICAL_WIDTH + cfg.tactical_out_of_bounds_margin_px
                    and -cfg.tactical_out_of_bounds_margin_px <= raw_py <= TACTICAL_HEIGHT + cfg.tactical_out_of_bounds_margin_px
                )
                h_point = (
                    float(np.clip(raw_px, 0, TACTICAL_WIDTH - 1)),
                    float(np.clip(raw_py, 0, TACTICAL_HEIGHT - 1)),
                )
            except cv2.error:
                h_point = None

        rel_point, rel_count = self._relative_tactical_projection(foot_pt, keypoints_xy, confidences)
        if h_point is None and rel_point is None:
            return None

        if h_point is None:
            px, py = rel_point
            raw_px, raw_py = px, py
            source = "relative"
        elif rel_point is None:
            if not h_inside:
                return None
            px, py = h_point
            source = "H"
        elif h_inside:
            px, py = h_point
            source = "H"
        else:
            blend = cfg.tactical_relative_outside_blend
            px = h_point[0] * (1.0 - blend) + rel_point[0] * blend
            py = h_point[1] * (1.0 - blend) + rel_point[1] * blend
            px = float(np.clip(px, 0, TACTICAL_WIDTH - 1))
            py = float(np.clip(py, 0, TACTICAL_HEIGHT - 1))
            source = "relative-guard"

        return {
            "x": float(px),
            "y": float(py),
            "raw_x": float(raw_px),
            "raw_y": float(raw_py),
            "source": source,
            "relative_keypoints": int(rel_count),
        }

    @staticmethod
    def _normalise_inference_result(result: Any) -> Any:
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    @staticmethod
    def _prediction_keypoints(prediction: Any) -> List[Any]:
        keypoints = _get_field(prediction, "keypoints", None)
        if keypoints is None:
            return []
        if isinstance(keypoints, dict):
            for nested_key in ("items", "predictions", "keypoints"):
                nested = keypoints.get(nested_key)
                if isinstance(nested, list):
                    return nested
            return list(keypoints.values())
        try:
            return list(keypoints)
        except TypeError:
            nested = _get_field(keypoints, "keypoints", None)
            if nested is not None and nested is not keypoints:
                try:
                    return list(nested)
                except TypeError:
                    return []
            return []

    def _keypoint_index(self, keypoint: Any, fallback: int) -> int:
        raw_idx = _get_field(keypoint, "class_id", None)
        if raw_idx is None:
            raw_idx = _get_field(keypoint, "class", None)
        if raw_idx is None:
            raw_idx = _get_field(keypoint, "class_name", None)

        idx = None
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            if isinstance(raw_idx, str):
                digits = "".join(ch for ch in raw_idx if ch.isdigit())
                if digits:
                    try:
                        idx = int(digits)
                    except ValueError:
                        idx = None

        if idx is None:
            idx = fallback
        else:
            idx -= int(self.config.keypoints.keypoint_index_base)
        return int(idx)

    @staticmethod
    def _field_as_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _confidence_value(keypoint: Any) -> float:
        confidence = _get_field(keypoint, "confidence", None)
        if confidence is None:
            confidence = _get_field(keypoint, "conf", 1.0)
        try:
            return float(confidence)
        except (TypeError, ValueError):
            return 0.0

    def _roboflow_raw_keypoint_index(self, keypoint: Any, fallback: int) -> int:
        raw_idx = _get_field(keypoint, "class_id", None)
        if raw_idx is None:
            raw_idx = _get_field(keypoint, "class", None)
        if raw_idx is None:
            raw_idx = _get_field(keypoint, "keypoint_id", None)

        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            idx = fallback
        return idx - int(self.config.keypoints.keypoint_index_base)

    def _roboflow_keypoint_label(self, keypoint: Any, fallback_idx: int) -> Optional[str]:
        for field in ("class_name", "name", "label"):
            label = self._field_as_str(_get_field(keypoint, field, None))
            if label is not None:
                if label.isdigit():
                    return label.zfill(2)
                digits = "".join(ch for ch in label if ch.isdigit())
                if digits:
                    return digits.zfill(2)
        raw_idx = self._roboflow_raw_keypoint_index(keypoint, fallback_idx)
        if 0 <= raw_idx < len(ROBOFLOW_COURT_KEYPOINT_LABELS):
            return ROBOFLOW_COURT_KEYPOINT_LABELS[raw_idx]
        return None

    def _roboflow_native_index(self, keypoint: Any, fallback_idx: int) -> int:
        label = self._roboflow_keypoint_label(keypoint, fallback_idx)
        if label in ROBOFLOW_COURT_KEYPOINT_LABELS:
            return ROBOFLOW_COURT_KEYPOINT_LABELS.index(label)
        raw_idx = self._roboflow_raw_keypoint_index(keypoint, fallback_idx)
        if 0 <= raw_idx < len(ROBOFLOW_COURT_KEYPOINT_LABELS):
            return raw_idx
        return -1

    def _roboflow_tactical_index(self, keypoint: Any, fallback_idx: int) -> int:
        label = self._roboflow_keypoint_label(keypoint, fallback_idx)
        if label in ROBOFLOW_LABEL_TO_TACTICAL_INDEX:
            return int(ROBOFLOW_LABEL_TO_TACTICAL_INDEX[label])
        raw_idx = self._roboflow_raw_keypoint_index(keypoint, fallback_idx)
        if 0 <= raw_idx < len(ROBOFLOW_TO_TACTICAL_INDEX):
            return int(ROBOFLOW_TO_TACTICAL_INDEX[raw_idx])
        return -1

    def _log_roboflow_keypoint_schema(self, raw_keypoints: List[Any]) -> None:
        if self._rf_kp_schema_logged or not self.config.debug.keypoints:
            return
        self._rf_kp_schema_logged = True
        print(
            "    [rf-kp-schema] "
            f"raw={len(raw_keypoints)} "
            f"labels={','.join(ROBOFLOW_COURT_KEYPOINT_LABELS)}"
        )
        for fallback_idx, keypoint in enumerate(raw_keypoints[:len(ROBOFLOW_COURT_KEYPOINT_LABELS)]):
            raw_idx = self._roboflow_raw_keypoint_index(keypoint, fallback_idx)
            native_idx = self._roboflow_native_index(keypoint, fallback_idx)
            label = self._roboflow_keypoint_label(keypoint, fallback_idx) or "-"
            legacy_idx = self._roboflow_tactical_index(keypoint, fallback_idx)
            legacy_name = KP_NAMES[legacy_idx] if 0 <= legacy_idx < len(KP_NAMES) else "-"
            x = _get_field(keypoint, "x", None)
            y = _get_field(keypoint, "y", None)
            try:
                x_text = f"{float(x):.1f}"
                y_text = f"{float(y):.1f}"
            except (TypeError, ValueError):
                x_text = "-"
                y_text = "-"
            print(
                "    [rf-kp-schema] "
                f"raw_idx={raw_idx:02d} label={label} "
                f"xy=({x_text},{y_text}) conf={self._confidence_value(keypoint):.3f} "
                f"native={native_idx} legacy={legacy_idx}:{legacy_name}"
            )

    def _parse_roboflow_keypoints_direct(self, result: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        result = self._normalise_inference_result(result)
        predictions = _get_field(result, "predictions", [])
        if isinstance(predictions, dict):
            nested = predictions.get("predictions") or predictions.get("items")
            predictions = nested if nested is not None else [predictions]
        predictions = list(predictions) if predictions is not None else []

        best_xy = np.zeros((self._kp_len, 2), dtype=np.float32)
        best_conf = np.zeros(self._kp_len, dtype=np.float32)
        best_score = -1.0
        best_debug = {"boxes": len(predictions), "instances": 0, "raw_keypoints": 0}

        for prediction in predictions:
            raw_keypoints = self._prediction_keypoints(prediction)
            if not raw_keypoints:
                continue
            self._log_roboflow_keypoint_schema(raw_keypoints)
            xy = np.zeros((self._kp_len, 2), dtype=np.float32)
            conf = np.zeros(self._kp_len, dtype=np.float32)
            for fallback_idx, keypoint in enumerate(raw_keypoints):
                if self._roboflow_native_keypoints:
                    idx = self._roboflow_native_index(keypoint, fallback_idx)
                else:
                    idx = self._roboflow_tactical_index(keypoint, fallback_idx)
                if idx < 0 or idx >= self._kp_len:
                    continue
                x = _get_field(keypoint, "x", None)
                y = _get_field(keypoint, "y", None)
                if x is None or y is None:
                    continue
                conf_value = self._confidence_value(keypoint)
                xy[idx] = (float(x), float(y))
                conf[idx] = conf_value

            valid_count = int(((xy > 0).all(axis=1) & (conf >= self.config.keypoints.geometry_confidence)).sum())
            score = valid_count + float(conf.sum()) * 0.01
            if score > best_score:
                best_score = score
                best_xy = xy
                best_conf = conf
                best_debug = {
                    "boxes": len(predictions),
                    "instances": 1,
                    "raw_keypoints": len(raw_keypoints),
                }

        return best_xy, best_conf, best_debug

    def _parse_roboflow_keypoints_supervision(self, result: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        result = self._normalise_inference_result(result)
        try:
            key_points = sv.KeyPoints.from_inference(result)
        except Exception:
            return np.zeros((self._kp_len, 2), dtype=np.float32), np.zeros(self._kp_len, dtype=np.float32), {
                "boxes": 0,
                "instances": 0,
                "raw_keypoints": 0,
            }

        xy_all = np.asarray(getattr(key_points, "xy", []), dtype=np.float32)
        conf_all = getattr(key_points, "confidence", None)
        if xy_all.size == 0:
            return np.zeros((self._kp_len, 2), dtype=np.float32), np.zeros(self._kp_len, dtype=np.float32), {
                "boxes": 0,
                "instances": 0,
                "raw_keypoints": 0,
            }

        if xy_all.ndim == 2:
            xy_all = xy_all.reshape(1, xy_all.shape[0], xy_all.shape[1])
        if conf_all is None:
            conf_all = np.ones(xy_all.shape[:2], dtype=np.float32)
        else:
            conf_all = np.asarray(conf_all, dtype=np.float32)
            if conf_all.ndim == 1:
                conf_all = conf_all.reshape(1, conf_all.shape[0])

        n_scored = min(len(ROBOFLOW_TO_TACTICAL_INDEX), conf_all.shape[1])
        mapped_raw = np.array(
            [idx for idx, mapped in enumerate(ROBOFLOW_TO_TACTICAL_INDEX[:n_scored]) if mapped >= 0],
            dtype=np.int32,
        )
        if self._roboflow_native_keypoints:
            scores = np.sum(conf_all >= self.config.keypoints.geometry_confidence, axis=1)
        elif len(mapped_raw) > 0:
            scores = np.sum(conf_all[:, mapped_raw] >= self.config.keypoints.geometry_confidence, axis=1)
        else:
            scores = np.sum(conf_all >= self.config.keypoints.geometry_confidence, axis=1)
        best_idx = int(np.argmax(scores))
        raw_xy = xy_all[best_idx]
        raw_conf = conf_all[best_idx]

        xy = np.zeros((self._kp_len, 2), dtype=np.float32)
        conf = np.zeros(self._kp_len, dtype=np.float32)
        if self._roboflow_native_keypoints:
            n = min(self._kp_len, len(raw_xy), len(raw_conf))
            xy[:n] = raw_xy[:n]
            conf[:n] = raw_conf[:n]
        else:
            n = min(len(ROBOFLOW_TO_TACTICAL_INDEX), len(raw_xy), len(raw_conf))
            for raw_idx in range(n):
                tactical_idx = ROBOFLOW_TO_TACTICAL_INDEX[raw_idx]
                if tactical_idx < 0 or tactical_idx >= self._kp_len:
                    continue
                xy[tactical_idx] = raw_xy[raw_idx]
                conf[tactical_idx] = raw_conf[raw_idx]
        return xy, conf, {
            "boxes": int(len(xy_all)),
            "instances": int(len(xy_all)),
            "raw_keypoints": int(len(raw_xy)),
        }

    def _infer_roboflow_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        result = self.kp_model.infer(frame, confidence=self.config.keypoints.roboflow_confidence)
        xy, conf, debug = self._parse_roboflow_keypoints_direct(result)
        if int((xy > 0).all(axis=1).sum()) > 0:
            return xy, conf, debug
        return self._parse_roboflow_keypoints_supervision(result)

    # ── Court keypoint detection ───────────────────────────────────────────────

    def _detect_keypoints(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        src_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.kp_model is None:
            return (
                np.zeros((self._kp_len, 2), dtype=np.float32),
                np.zeros(self._kp_len, dtype=np.float32),
                self.last_H,
            )

        keypoints_xy = np.zeros((self._kp_len, 2), dtype=np.float32)
        confidences = np.zeros(self._kp_len, dtype=np.float32)
        debug_boxes = 0
        debug_instances = 0
        debug_raw_keypoints = 0

        if self.kp_backend == "roboflow":
            keypoints_xy, confidences, kp_debug = self._infer_roboflow_keypoints(frame)
            debug_boxes = int(kp_debug.get("boxes", 0))
            debug_instances = int(kp_debug.get("instances", 0))
            debug_raw_keypoints = int(kp_debug.get("raw_keypoints", 0))
        else:
            results = self.kp_model.predict(
                frame, conf=self.config.keypoints.confidence, verbose=False,
                half=(self.device == "cuda"), device=self.device,
            )

            if results and results[0].keypoints is not None:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None:
                    debug_boxes = len(boxes)
                kp = results[0].keypoints
                if kp.xy is not None and len(kp.xy) > 0:
                    debug_instances = len(kp.xy)
                    xy = kp.xy[0].cpu().numpy()
                    debug_raw_keypoints = len(xy)
                    n = min(len(xy), self._kp_len)
                    keypoints_xy[:n] = xy[:n]
                    if kp.conf is not None and len(kp.conf) > 0:
                        conf = kp.conf[0].cpu().numpy()
                        n = min(len(conf), self._kp_len)
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

        for i in range(self._kp_len):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        observed_side, side_counts = self._detected_court_side(keypoints_xy, confidences, frame.shape[1])
        side_switched = False

        # Hysteresis: require N consecutive frames before accepting a side change
        if observed_side is None or observed_side == "center" or observed_side == self._court_side:
            self._pending_side = None
            self._pending_side_count = 0
        elif observed_side == self._pending_side:
            self._pending_side_count += 1
        else:
            self._pending_side = observed_side
            self._pending_side_count = 1

        if (
            self._court_side is not None
            and self._pending_side is not None
            and self._pending_side_count >= self.config.keypoints.side_switch_hysteresis_frames
        ):
            side_switched = True
            old_side = self._court_side
            confirmed_side = self._pending_side
            self._pending_side = None
            self._pending_side_count = 0
            self._court_side = confirmed_side

            # Soft reset: only wipe the departing side's keypoints; keep the rest
            left_ft = set(self._kp_left_indices) | set(self._kp_left_inner_indices)
            right_ft = set(self._kp_right_indices) | set(self._kp_right_inner_indices)
            wipe_idx = left_ft if old_side == "left" else right_ft if old_side == "right" else set()
            carry = self.config.keypoints.carry_missing_updates + 1
            if self.last_good_keypoints is not None:
                for i in wipe_idx:
                    self.last_good_keypoints[i] = 0.0
            for i in wipe_idx:
                self._prev_detected_keypoints[i] = 0.0
                self._keypoint_stationary_counts[i] = 0
                self._keypoint_missing_counts[i] = carry

            # Mark transition so _accept_homography_update uses stricter thresholds
            self._side_transition_frames = self.config.keypoints.side_transition_frames
            self._force_homography_reseed = True

            if self.config.debug.keypoints:
                frame_label = "-" if frame_idx is None else str(frame_idx)
                src_label = "-" if src_idx is None else str(src_idx)
                print(
                    f"    [kp-side-switch] frame={frame_label} src={src_label} "
                    f"{old_side}->{confirmed_side} "
                    f"left={side_counts[0]} center={side_counts[1]} right={side_counts[2]}"
                )

        if self.last_good_keypoints is not None:
            valid_pairs = [
                i for i in range(self._kp_len)
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

        for i in range(self._kp_len):
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

        max_reproj = self.config.keypoints.compute_homography_max_reproj_px
        high_conf = np.where(
            (confidences >= self.config.keypoints.geometry_confidence).reshape(-1, 1) & (keypoints_xy > 0),
            keypoints_xy, 0.0,
        ).astype(np.float32)
        pre_H = None
        pre_mode = None
        if (high_conf > 0).any():
            pre_H, pre_mode, _ = self._select_homography_candidate(
                high_conf,
                observed_side,
                frame.shape,
                max_reproj,
                use_stability=False,
            )
        if pre_H is not None:
            pre_dst = self._tactical_dst_mirrored if pre_mode == "mirrored" else self._tactical_dst_normal
            for i in range(self._kp_len):
                if keypoints_xy[i][0] <= 0:
                    continue
                dst = cv2.perspectiveTransform(np.array([[keypoints_xy[i]]], dtype=np.float32), pre_H)
                tx, ty = dst[0][0]
                ex, ey = pre_dst[i]
                if np.sqrt((tx - ex) ** 2 + (ty - ey) ** 2) > 50:
                    keypoints_xy[i] = 0.0

        pan_dx, pan_conf, pan_matches, pan_mad = self._update_keypoint_pan_prior(
            keypoints_xy,
            confidences,
            frame.shape[1],
            frame_idx,
        )

        detected_this_update = np.array([_valid(keypoints_xy[i]) for i in range(self._kp_len)], dtype=bool)
        for i in range(self._kp_len):
            if detected_this_update[i]:
                self._keypoint_missing_counts[i] = 0
            else:
                self._keypoint_missing_counts[i] += 1

        ALPHA = 0.6
        carried_count = 0
        if self.last_good_keypoints is not None:
            for i in range(self._kp_len):
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

        for i in range(self._kp_len):
            if not _valid(keypoints_xy[i]):
                keypoints_xy[i] = 0.0
                confidences[i] = 0.0

        candidate_keypoints = keypoints_xy.copy()
        final_valid_count = int(((keypoints_xy > 0).all(axis=1)).sum())
        candidate_H, candidate_mode, candidate_meta = self._select_homography_candidate(
            candidate_keypoints,
            observed_side,
            frame.shape,
            max_reproj,
            use_stability=True,
        )
        H = candidate_H
        h_status = "fail"
        reseed_requested = self._force_homography_reseed
        if H is not None and self._accept_homography_update(
            H,
            frame.shape,
            final_valid_count,
            candidate_meta,
        ):
            self.homography_success_count += 1
            self.last_good_keypoints = candidate_keypoints.copy()
            self.last_H = H
            self._last_H_mode = candidate_mode
            self._force_homography_reseed = False
            if observed_side in ("left", "right"):
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
            h_det = float(np.linalg.det(H)) if H is not None else 0.0
            n_kp_used = int((candidate_keypoints > 0).all(axis=1).sum())
            print(
                "    [kp-debug] "
                f"frame={frame_label} src={src_label} "
                f"backend={self.kp_backend} "
                f"boxes={debug_boxes} instances={debug_instances} "
                f"raw_kp={debug_raw_keypoints} "
                f"raw_valid={raw_valid_count}/{self._kp_len} "
                f"conf>=0.1={raw_conf_01} conf>=0.5={raw_conf_05} "
                f"final_valid={final_valid_count}/{self._kp_len} "
                f"carried={carried_count} "
                f"side={observed_side or self._court_side or '-'} "
                f"side_counts=(L{side_counts[0]},C{side_counts[1]},R{side_counts[2]}) "
                f"layout={self._last_layout_side or '-'} "
                f"layout_slope={self._last_layout_slope:.3f} "
                f"layout_pairs={self._last_layout_pairs} "
                f"side_switched={int(side_switched)} "
                f"H={h_status} mode={candidate_mode or self._last_H_mode or '-'} "
                f"cand_side={candidate_meta.get('side', '-')} "
                f"score={float(candidate_meta.get('score', -1.0)):.1f} "
                f"reproj={float(candidate_meta.get('reproj', -1.0)):.1f} "
                f"shift=({float(candidate_meta.get('mean_shift', 0.0)):.1f},"
                f"{float(candidate_meta.get('max_shift', 0.0)):.1f}) "
                f"pan_dx={pan_dx:.2f} pan_conf={pan_conf:.2f} "
                f"pan_match={pan_matches} pan_mad={pan_mad:.1f} "
                f"reseed={int(reseed_requested)} det={h_det:.4f} kp_used={n_kp_used}"
            )

        self._side_transition_frames = max(0, self._side_transition_frames - 1)
        self._h_fallback_streak = 0 if h_status == "ok" else self._h_fallback_streak + 1

        return keypoints_xy, confidences, H

    def _detected_court_side(
        self,
        keypoints_xy: np.ndarray,
        confidences: np.ndarray,
        frame_width: int,
    ) -> Tuple[Optional[str], Tuple[int, int, int]]:
        valid = (keypoints_xy > 0).all(axis=1)
        confident = confidences >= self.config.keypoints.confidence
        left_indices = set(self._kp_left_indices) | set(self._kp_left_inner_indices)
        center_indices = set(self._kp_center_indices)
        right_indices = set(self._kp_right_indices) | set(self._kp_right_inner_indices)

        left_count = sum(1 for idx in left_indices if idx < len(valid) and valid[idx] and confident[idx])
        center_count = sum(1 for idx in center_indices if idx < len(valid) and valid[idx] and confident[idx])
        right_count = sum(1 for idx in right_indices if idx < len(valid) and valid[idx] and confident[idx])
        counts = (left_count, center_count, right_count)

        layout_side, layout_slope, layout_pairs = self._layout_court_side(keypoints_xy, confidences)
        self._last_layout_side = layout_side
        self._last_layout_slope = layout_slope
        self._last_layout_pairs = layout_pairs
        if layout_side is not None:
            return layout_side, counts

        # Primary signal: position of center keypoints (half-court line) in camera frame
        # If center line is in the left portion → camera shows right side of court (and vice versa)
        center_xs = [
            float(keypoints_xy[idx][0])
            for idx in center_indices
            if idx < len(valid) and valid[idx] and confident[idx]
        ]
        band = self.config.keypoints.center_side_band_ratio
        if center_xs:
            center_x = sum(center_xs) / len(center_xs)
            self._last_center_kp_x = center_x
            ratio = center_x / max(frame_width, 1)
            if ratio < band:
                return "right", counts
            if ratio > (1.0 - band):
                return "left", counts
            return "center", counts

        # Center disappeared — infer from last known position
        if self._last_center_kp_x is not None:
            ratio = self._last_center_kp_x / max(frame_width, 1)
            return ("right" if ratio < 0.5 else "left"), counts

        # No center signal ever — fall back to keypoint count
        min_count = self.config.keypoints.side_switch_min_keypoints
        margin = self.config.keypoints.side_switch_margin
        if right_count >= min_count and right_count >= left_count + margin:
            return "right", counts
        if left_count >= min_count and left_count >= right_count + margin:
            return "left", counts
        return None, counts

    def _layout_court_side(
        self,
        keypoints_xy: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[Optional[str], float, int]:
        kp_cfg = self.config.keypoints
        min_conf = kp_cfg.side_layout_min_confidence
        valid = (keypoints_xy > 0).all(axis=1) & (confidences >= min_conf)
        vertical_groups = [
            self._kp_left_indices,
            self._kp_center_indices,
            self._kp_left_inner_indices,
            self._kp_right_indices,
            self._kp_right_inner_indices,
        ]

        slopes: List[float] = []
        for group in vertical_groups:
            usable = [idx for idx in group if idx < len(valid) and valid[idx]]
            if len(usable) < 2:
                continue
            pts = sorted((keypoints_xy[idx] for idx in usable), key=lambda p: float(p[1]))
            for a_i in range(len(pts)):
                for b_i in range(a_i + 1, len(pts)):
                    p0, p1 = pts[a_i], pts[b_i]
                    dy = float(p1[1] - p0[1])
                    if abs(dy) < 18.0:
                        continue
                    dx = float(p1[0] - p0[0])
                    slopes.append(dx / dy)

        if len(slopes) < kp_cfg.side_layout_min_pairs:
            return None, 0.0, len(slopes)

        slope = float(np.median(slopes))
        if abs(slope) < kp_cfg.side_layout_min_slope:
            return None, slope, len(slopes)

        # In image coordinates, "/" has negative dx/dy because x decreases as y goes down.
        side = "left" if slope < 0 else "right"
        return side, slope, len(slopes)

    def _accept_homography_update(
        self,
        H: np.ndarray,
        frame_shape: Tuple[int, int, int],
        valid_keypoints: int,
        candidate_meta: Optional[Dict[str, float | str]] = None,
    ) -> bool:
        # Basic sanity: finite values and non-degenerate matrix
        if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-8:
            return False

        kp_cfg = self.config.keypoints
        min_kp = kp_cfg.homography_min_update_keypoints

        if self.last_H is None:
            # Cold start: require minimum keypoints but skip shift comparison
            return valid_keypoints >= max(min_kp, kp_cfg.cold_start_min_keypoints)

        if valid_keypoints < min_kp:
            return False

        reproj = float("inf")
        if candidate_meta is not None:
            try:
                reproj = float(candidate_meta.get("reproj", float("inf")))
            except (TypeError, ValueError):
                reproj = float("inf")

        if self._force_homography_reseed:
            return (
                valid_keypoints >= max(min_kp, kp_cfg.transition_min_keypoints)
                and reproj <= kp_cfg.homography_recover_reproj_px
            )

        if (
            valid_keypoints >= max(min_kp, kp_cfg.transition_min_keypoints)
            and reproj <= kp_cfg.homography_good_reproj_px
        ):
            return True

        if (
            self._h_fallback_streak >= kp_cfg.homography_recover_fallback_streak
            and valid_keypoints >= min_kp
            and reproj <= kp_cfg.homography_recover_reproj_px
        ):
            return True

        # During transition: stricter keypoint and shift thresholds
        if self._side_transition_frames > 0:
            if valid_keypoints < kp_cfg.transition_min_keypoints:
                return False
            s = kp_cfg.transition_strictness
            max_mean = kp_cfg.homography_max_mean_shift * s
            max_pt   = kp_cfg.homography_max_point_shift * s
        else:
            max_mean = kp_cfg.homography_max_mean_shift
            max_pt   = kp_cfg.homography_max_point_shift

        mean_shift, max_shift = self._homography_shift_stats(H, frame_shape)
        if not np.isfinite(mean_shift) or not np.isfinite(max_shift):
            return False
        return mean_shift <= max_mean and max_shift <= max_pt

    # ── Court boundary filter ──────────────────────────────────────────────────

    def _is_valid_player_bbox(self, bbox: list, keypoints_xy: np.ndarray, frame_shape: Tuple) -> bool:
        if not any(kp[0] > 0 and kp[1] > 0 for kp in keypoints_xy):
            return True

        def _vis(indices):
            return sorted(
                [(float(keypoints_xy[i][0]), float(keypoints_xy[i][1]))
                 for i in indices if i < len(keypoints_xy)
                 and keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0],
                key=lambda p: p[1],
            )

        x1, y1, x2, y2 = [float(c) for c in bbox]
        cy = (y1 + y2) * 0.5

        left_pts = _vis(self._kp_left_indices)
        if left_pts and x2 < _interp_x_at_y(left_pts, cy):
            return False

        right_pts = _vis(self._kp_right_indices)
        if right_pts and x1 > _interp_x_at_y(right_pts, cy):
            return False

        top_pts  = _vis(self._kp_top_indices)
        edge_pts = _vis([i for i in range(len(keypoints_xy)) if i not in self._kp_inner_indices])
        ref_top  = top_pts or edge_pts
        if ref_top and y2 < min(p[1] for p in ref_top):
            return False

        bottom_pts = _vis(self._kp_bottom_indices)
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
            h_unreliable = (
                H is None
                or self._h_fallback_streak >= self.config.detector.h_fallback_unreliable_streak
            )
            mask_absent = mask is None or inside_area == 0
            if mask_absent and h_unreliable:
                if self.config.debug.suppression:
                    print(
                        f"    [court-mask-defer] frame={frame_idx} track={track_id} "
                        f"reason=no_mask_and_stale_H streak={self._h_fallback_streak} "
                        f"bbox={track['bbox']}"
                    )
                continue
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
            self._id_switch_count += 1

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
        keypoints_xy: Optional[np.ndarray] = None,
        keypoint_confidences: Optional[np.ndarray] = None,
        frame_idx: int = -1,
    ) -> None:
        if H is None and keypoints_xy is None:
            return

        active_ids = set(player_ids)
        cfg = self.config.visualization
        alpha = cfg.tactical_smoothing
        prior_age = frame_idx - self._tactical_pan_prior_frame if frame_idx >= 0 else 0
        pan_prior_dx = self._tactical_pan_prior_dx
        pan_prior_conf = self._tactical_pan_prior_conf
        if (
            not cfg.tactical_pan_prior_enabled
            or prior_age > cfg.tactical_pan_prior_max_age_frames
            or pan_prior_conf < cfg.tactical_pan_prior_min_conf
        ):
            pan_prior_dx = 0.0
            pan_prior_conf = 0.0

        for track_id, foot_pt in zip(player_ids, player_feet):
            projected = self._project_player_to_tactical(foot_pt, H, keypoints_xy, keypoint_confidences)
            if projected is None:
                if self.config.debug.tactical:
                    print(
                        f"    [tactical-skip f={frame_idx} tid={track_id} "
                        f"reason=no_projection]"
                    )
                continue

            px = float(projected["x"])
            py = float(projected["y"])
            raw_px = float(projected["raw_x"])
            raw_py = float(projected["raw_y"])
            source = str(projected["source"])
            rel_count = int(projected["relative_keypoints"])

            previous = self._tactical_smoothed.get(track_id)
            last_seen = self._tactical_last_seen.get(track_id)
            reset_gap = (
                frame_idx >= 0
                and last_seen is not None
                and frame_idx - last_seen > cfg.tactical_reset_gap_frames
            )
            if previous is None or reset_gap:
                smoothed = (px, py)
                if reset_gap:
                    self._tactical_history[track_id].clear()
            else:
                dx = px - previous[0]
                dy = py - previous[1]
                step = float(np.hypot(dx, dy))

                if abs(pan_prior_dx) >= 0.1:
                    prior_weight = cfg.tactical_pan_prior_weight * pan_prior_conf
                    predicted_x = float(np.clip(previous[0] + pan_prior_dx, 0, TACTICAL_WIDTH - 1))
                    if abs(dx) >= 0.5 and dx * pan_prior_dx < 0:
                        damp = cfg.tactical_pan_prior_opposite_damping
                        px = float(np.clip(previous[0] + dx * damp + pan_prior_dx * (1.0 - damp), 0, TACTICAL_WIDTH - 1))
                    elif prior_weight > 0:
                        px = float(np.clip(px * (1.0 - prior_weight) + predicted_x * prior_weight, 0, TACTICAL_WIDTH - 1))

                    dx = px - previous[0]
                    dy = py - previous[1]
                    step = float(np.hypot(dx, dy))

                max_step = cfg.tactical_max_step_px
                if max_step > 0 and step > max_step:
                    scale = max_step / max(step, 1e-6)
                    px = previous[0] + dx * scale
                    py = previous[1] + dy * scale
                smoothed = (
                    previous[0] * alpha + px * (1.0 - alpha),
                    previous[1] * alpha + py * (1.0 - alpha),
                )

            self._tactical_smoothed[track_id] = smoothed
            self._tactical_last_seen[track_id] = frame_idx
            self._tactical_history[track_id].append(smoothed)

            if self.config.debug.tactical:
                print(
                    f"    [tactical f={frame_idx} tid={track_id} "
                    f"foot=({foot_pt[0]:.0f},{foot_pt[1]:.0f}) "
                    f"raw=({raw_px:.1f},{raw_py:.1f}) "
                    f"target=({px:.1f},{py:.1f}) "
                    f"src={source} rel_kp={rel_count} "
                    f"smooth=({smoothed[0]:.1f},{smoothed[1]:.1f}) "
                    f"pan_prior=({pan_prior_dx:.2f},{pan_prior_conf:.2f})]"
                )

        stale_ids = set(self._tactical_smoothed.keys()) - active_ids
        for track_id in stale_ids:
            last_seen = self._tactical_last_seen.get(track_id, frame_idx)
            if frame_idx < 0 or frame_idx - last_seen > cfg.tactical_reset_gap_frames:
                self._tactical_smoothed.pop(track_id, None)
                self._tactical_last_seen.pop(track_id, None)
                self._tactical_history.pop(track_id, None)

    @staticmethod
    def _mask_foot_point(mask: np.ndarray) -> Optional[List[float]]:
        if mask is None:
            return None
        mask = mask.astype(bool)
        if not np.any(mask):
            return None
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        y_cut = float(np.percentile(ys, 88))
        lower = ys >= y_cut
        if np.count_nonzero(lower) < 4:
            lower = ys >= max(float(ys.max()) - 3.0, 0.0)
        foot_x = float(np.median(xs[lower]))
        foot_y = float(np.percentile(ys[lower], 96))
        return [foot_x, foot_y]

    def _tracks_to_feet(
        self,
        tracks: List[Dict],
        jersey_bank,
        mask_by_id: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[List[int], List[List[float]], List[Optional[str]]]:
        ids, feet, jerseys = [], [], []
        inset = self.config.visualization.tactical_foot_inset_px
        for t in tracks:
            if t["is_referee"]:
                continue
            x1, y1, x2, y2 = t["bbox"]
            track_id = int(t["track_id"])
            ids.append(track_id)
            mask_foot = self._mask_foot_point(mask_by_id.get(track_id)) if mask_by_id else None
            if mask_foot is not None:
                feet.append(mask_foot)
            else:
                foot_y = max(float(y1), float(y2) - inset)
                feet.append([float((x1 + x2) / 2), foot_y])
            jerseys.append(jersey_bank.get_jersey(track_id))
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
        src_fps: float = 30.0,
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

        if self.config.visualization.draw_keypoints:
            result = draw_keypoints_on_frame(
                result,
                kp_xy,
                kp_conf,
                min_confidence=self.config.keypoints.geometry_confidence,
                draw_labels=False,
            )

        h_img, w_img = result.shape[:2]
        font      = cv2.FONT_HERSHEY_DUPLEX
        scale     = 0.52
        thickness = 1
        pad_x, pad_y = 14, 10
        line_h    = 22

        # Top-left stats panel
        stats = [
            f"FPS  {src_fps:.0f}",
            f"Players  {len(player_tracks)}",
            f"ID switches  {self._id_switch_count}",
            f"Homography  {'OK' if H is not None else 'FAIL'}",
        ]
        panel_w = max(cv2.getTextSize(s, font, scale, thickness)[0][0] for s in stats) + pad_x * 2
        panel_h = line_h * len(stats) + pad_y * 2
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, result, 0.55, 0, result)
        for i, text in enumerate(stats):
            y = pad_y + (i + 1) * line_h
            color = (255, 255, 255) if "FAIL" not in text else (80, 80, 255)
            cv2.putText(result, text, (pad_x, y), font, scale, color, thickness, cv2.LINE_AA)

        # Top-right github label
        gh_text  = "github.com/hasan-bakr"
        gh_scale = 0.42
        (tw, th), _ = cv2.getTextSize(gh_text, font, gh_scale, thickness)
        gx = w_img - tw - pad_x
        gy = th + pad_y
        cv2.putText(result, gh_text, (gx, gy), font, gh_scale, (200, 200, 200), thickness, cv2.LINE_AA)

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

                result  = self._annotate_frame(frame, tracks, kp_xy, kp_conf, H, frame_idx, src_idx, src_fps)

                if writer is None:
                    h, w = result.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
                    tactical_writer = cv2.VideoWriter(
                        tactical_path, fourcc, out_fps, (tactical_w, tactical_h)
                    )

                writer.write(result)

                tactical_mask_by_id: Dict[int, np.ndarray] = {}
                tactical_player_tracks = [t for t in tracks if not t["is_referee"]]
                if tactical_player_tracks and hasattr(self.tracker, "get_player_masks_for_tracks"):
                    mask_tracks, masks = self.tracker.get_player_masks_for_tracks(
                        tactical_player_tracks,
                        frame.shape[:2],
                    )
                    if mask_tracks and masks is not None:
                        tactical_mask_by_id = {
                            int(track["track_id"]): mask.astype(bool)
                            for track, mask in zip(mask_tracks, masks)
                        }

                player_ids, player_feet, player_jerseys = self._tracks_to_feet(
                    tracks,
                    self.jersey_bank,
                    tactical_mask_by_id,
                )
                self._update_tactical_history(player_ids, player_feet, H, kp_xy, kp_conf, frame_idx)
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
                    label_scale=self.config.visualization.tactical_label_scale,
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
                    print(f"  frame {frame_idx:4d}  tracks={len(tracks):2d}  kp={n_kp}/{self._kp_len}  jersey={len(self.jersey_bank.obj_to_jersey)}")
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

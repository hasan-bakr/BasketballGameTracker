"""Central configuration objects for the basketball tracking pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from APP.helpers.court_utils import DEFAULT_COURT_IMAGE, DEFAULT_KEYPOINT_MODEL
from APP.helpers.rfdetr_detector import RFDETRDetector


@dataclass
class ClassConfig:
    player_classes: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    referee_class: int = 8
    jersey_number_class: int = 2


@dataclass
class DetectorConfig:
    rfdetr_model_id: str = RFDETRDetector.DEFAULT_MODEL_ID
    confidence: float = 0.3
    iou_threshold: float = 0.8
    detection_nms_iou: float = 0.65
    cross_role_iou: float = 0.80
    track_nms_iou: float = 0.88
    track_aware_suppression: bool = True
    track_match_iou: float = 0.30
    lost_track_match_iou: float = 0.20
    lost_track_match_center_px: float = 160.0
    duplicate_birth_max_dt: int = 3
    duplicate_birth_max_dist: float = 40.0
    outside_court_existing_track_iou: float = 0.10
    outside_court_existing_track_center_px: float = 140.0
    activate_tracks_by_court_mask: bool = True
    court_mask_min_overlap: float = 0.12
    court_mask_min_area_px: int = 0
    h_fallback_unreliable_streak: int = 3


@dataclass
class TrackerConfig:
    fps: int = 30
    track_thresh: float = 0.5
    new_track_thresh: float = 0.7
    track_buffer: int = 120
    cmc_method: str = "sparseOptFlow"
    assoc1_thresh: float = 0.80
    assoc2_thresh: float = 0.5
    unconfirmed_assoc_thresh: float = 0.7
    mask_duplicate_min_fill: float = 0.45
    mask_bbox_expand: float = 0.20
    mask_min_area_px: int = 80
    mask_min_bbox_inside_ratio: float = 0.35
    ref_player_conflict_iou: float = 0.45
    ref_player_conflict_mask_fill: float = 0.55
    ref_player_conflict_conf_margin: float = 0.05
    use_player_masks: bool = True
    use_referee_masks: bool = False
    require_cuda: bool = True
    sam_model_type: str = "vit_b"
    sam_checkpoint: Optional[str] = None
    cutie_weights: Optional[str] = None
    cutie_max_internal_size: int = 720
    switch_proxy_max_dist: float = 80.0
    switch_proxy_max_dt: int = 60
    switch_proxy_min_source_life: int = 10
    duplicate_new_track_max_age: int = 5
    duplicate_existing_min_age: int = 8
    duplicate_new_track_iou: float = 0.85
    duplicate_new_track_center_px: float = 45.0
    mask_reinit_missing_frames: int = 6
    mask_reinit_area_fail_frames: int = 4
    mask_reinit_inside_fail_frames: int = 10
    mask_reinit_inside_ratio: float = 0.12
    mask_reinit_cooldown: int = 20
    mask_assignment_near_iou: float = 0.08
    mask_assignment_min_center_px: float = 55.0
    mask_assignment_center_scale: float = 0.30
    mask_reuse_min_fill: float = 0.35


@dataclass
class JerseyConfig:
    min_confidence: float = 0.4
    min_size: int = 10
    expand: float = 1.5
    lock_votes: int = 3
    update_interval: int = 3


@dataclass
class KeypointConfig:
    backend: str = "roboflow"
    roboflow_model_id: str = "basketball-court-detection-2/19"
    roboflow_confidence: float = 0.30
    anchor_confidence: float = 0.10
    keypoint_index_base: int = 0
    model_path: Optional[str] = DEFAULT_KEYPOINT_MODEL
    court_image_path: Optional[str] = None
    confidence: float = 0.05
    geometry_confidence: float = 0.1
    update_interval: int = 2
    border_margin: int = 10
    edge_band_ratio: float = 0.30
    still_px: float = 6.0
    still_frames: int = 45
    carry_missing_updates: int = 45
    homography_min_update_keypoints: int = 6
    homography_max_mean_shift: float = 35.0
    homography_max_point_shift: float = 80.0
    side_switch_min_keypoints: int = 3
    side_switch_margin: int = 2
    side_layout_min_pairs: int = 1
    side_layout_min_slope: float = 0.08
    side_layout_min_confidence: float = 0.10
    center_side_band_ratio: float = 0.4
    center_switch_min_keypoints: int = 1
    side_switch_hysteresis_frames: int = 6
    side_transition_frames: int = 8
    cold_start_min_keypoints: int = 6
    transition_min_keypoints: int = 7
    transition_strictness: float = 0.5
    compute_homography_max_reproj_px: float = 60.0
    homography_good_reproj_px: float = 8.0
    homography_recover_reproj_px: float = 14.0
    homography_recover_fallback_streak: int = 2
    homography_stability_weight: float = 0.35
    homography_max_shift_weight: float = 0.05
    homography_side_penalty: float = 80.0
    homography_mode_switch_penalty: float = 25.0
    left_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    right_indices: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 15])
    center_indices: List[int] = field(default_factory=lambda: [6, 7])
    top_indices: List[int] = field(default_factory=lambda: [0, 7, 15])
    bottom_indices: List[int] = field(default_factory=lambda: [5, 6, 10])


@dataclass
class DebugConfig:
    enabled: bool = True
    tracking: bool = True
    masks: bool = True
    keypoints: bool = True
    progress_interval: int = 30
    id_jump_px: float = 90.0
    id_swap_px: float = 70.0
    lifecycle: bool = True
    suppression: bool = True
    tactical: bool = True


@dataclass
class VisualizationConfig:
    trail_length: int = 0
    tactical_trail_length: int = 18
    tactical_smoothing: float = 0.45
    tactical_max_step_px: float = 18.0
    tactical_out_of_bounds_margin_px: float = 25.0
    tactical_reset_gap_frames: int = 12
    tactical_pan_prior_enabled: bool = False
    tactical_pan_prior_min_keypoints: int = 4
    tactical_pan_prior_min_image_px: float = 2.0
    tactical_pan_prior_max_mad_px: float = 18.0
    tactical_pan_prior_min_conf: float = 0.35
    tactical_pan_prior_weight: float = 0.30
    tactical_pan_prior_opposite_damping: float = 0.15
    tactical_pan_prior_gain: float = 1.0
    tactical_pan_prior_max_age_frames: int = 6
    tactical_relative_projection: bool = True
    tactical_relative_min_keypoints: int = 2
    tactical_relative_nearest_keypoints: int = 6
    tactical_relative_power: float = 1.35
    tactical_relative_blend: float = 0.0
    tactical_relative_outside_blend: float = 0.85
    tactical_relative_conf_floor: float = 0.25
    tactical_foot_inset_px: float = 3.0
    tactical_point_radius: int = 7
    tactical_trail_thickness: int = 2
    tactical_label_scale: float = 0.32
    draw_keypoints: bool = False
    draw_masks: bool = True
    mask_alpha: float = 0.45
    mask_border_thickness: int = 4
    mask_center_text_scale: float = 0.55


@dataclass
class PipelineConfig:
    device: str = "cuda"
    classes: ClassConfig = field(default_factory=ClassConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    jersey: JerseyConfig = field(default_factory=JerseyConfig)
    keypoints: KeypointConfig = field(default_factory=KeypointConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    tracking_report_json: Optional[str] = None

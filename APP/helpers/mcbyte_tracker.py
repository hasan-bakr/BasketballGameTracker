"""MCByte tracker wrapper for the basketball pipeline."""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
MCBYTE_DIR = ROOT_DIR / "external" / "mcbyte"
if str(MCBYTE_DIR) not in sys.path:
    sys.path.insert(0, str(MCBYTE_DIR))

from mask_propagation.mask_manager import MaskManager
from yolox.tracker.mcbyte_tracker import McByteTracker as _McByteTracker


PLAYER_CLASSES = [3, 4, 5, 6, 7]
REFEREE_CLASS = 8


@dataclass
class McByteTrackerConfig:
    device: str = "cuda:0"
    fps: int = 30
    track_thresh: float = 0.6
    new_track_thresh: float = 0.7
    track_buffer: int = 90
    cmc_method: str = "orb"
    assoc1_thresh: float = 0.8
    assoc2_thresh: float = 0.5
    unconfirmed_assoc_thresh: float = 0.7
    mask_duplicate_min_fill: float = 0.45
    mask_bbox_expand: float = 0.20
    mask_min_area_px: int = 80
    mask_min_bbox_inside_ratio: float = 0.35
    ref_player_conflict_iou: float = 0.45
    ref_player_conflict_mask_fill: float = 0.55
    ref_player_conflict_conf_margin: float = 0.05
    save_folder: Optional[str] = None
    use_player_masks: bool = True
    use_referee_masks: bool = False
    require_cuda: bool = True
    sam_checkpoint: Optional[str] = None
    cutie_weights: Optional[str] = None
    debug_masks: bool = False
    debug_association: bool = False


class _McByteStream:
    def __init__(
        self,
        *,
        config: McByteTrackerConfig,
        class_ids: List[int],
        is_referee: bool,
        track_id_offset: int = 0,
        use_masks: bool = False,
    ):
        self.config = config
        self.class_ids = set(class_ids)
        self.is_referee = is_referee
        self.track_id_offset = track_id_offset
        self.use_masks = use_masks

        save_folder = config.save_folder or tempfile.mkdtemp(prefix="mcbyte_")
        os.makedirs(save_folder, exist_ok=True)
        args = SimpleNamespace(
            track_thresh=config.track_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.track_buffer,
            cmc_method=config.cmc_method,
            assoc1_thresh=config.assoc1_thresh,
            assoc2_thresh=config.assoc2_thresh,
            unconfirmed_assoc_thresh=config.unconfirmed_assoc_thresh,
            debug_association=config.debug_association,
            debug_name="referee" if is_referee else "player",
        )
        self.tracker = _McByteTracker(args, frame_rate=config.fps, save_folder=save_folder)
        self.mask_manager = (
            MaskManager(
                device=config.device,
                sam_checkpoint=config.sam_checkpoint,
                cutie_weights=config.cutie_weights,
            )
            if use_masks
            else None
        )

        self.frame_id = 0
        self.prev_img_info: Optional[Dict] = None
        self.online_tlwhs: List[np.ndarray] = []
        self.online_ids: List[int] = []
        self.new_tracks: List = []
        self.removed_track_ids: List[int] = []
        self.prediction_mask = None
        self.prediction_colors_preserved = None
        self.tracklet_mask_dict = {}
        self.mask_avg_prob_dict = {}
        self.mask_duplicate_min_fill = config.mask_duplicate_min_fill

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        self.frame_id += 1
        stream_dets = [d for d in detections if int(d["class_id"]) in self.class_ids]
        img_info = {
            "height": frame.shape[0],
            "width": frame.shape[1],
            "raw_img": frame,
        }

        if self.mask_manager is not None and self.prev_img_info is not None:
            (
                self.prediction_mask,
                self.tracklet_mask_dict,
                self.mask_avg_prob_dict,
                self.prediction_colors_preserved,
            ) = self.mask_manager.get_updated_masks(
                img_info,
                self.prev_img_info,
                self.frame_id,
                self.online_tlwhs,
                self.online_ids,
                self.new_tracks,
                self.removed_track_ids,
            )

        output_results = _detections_to_mcbyte_array(stream_dets)
        online_targets, removed_ids, new_tracks, _, _ = self.tracker.update(
            output_results,
            [img_info["height"], img_info["width"]],
            [img_info["height"], img_info["width"]],
            prediction_mask=self.prediction_mask,
            tracklet_mask_dict=self.tracklet_mask_dict,
            mask_avg_prob_dict=self.mask_avg_prob_dict,
            frame_img=frame,
            vis_type="no_vis",
            dets_from_file=True,
        )

        online_targets = self._suppress_duplicate_mask_targets(online_targets)

        self.online_tlwhs = []
        self.online_ids = []
        tracks: List[Dict] = []
        for target in online_targets:
            x, y, w, h = [float(v) for v in target.last_det_tlwh]
            raw_id = int(target.track_id)
            class_id, confidence = _match_source_detection(stream_dets, (x, y, w, h), float(target.score))
            tracks.append(
                {
                    "track_id": raw_id + self.track_id_offset,
                    "raw_track_id": raw_id,
                    "bbox": [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))],
                    "class_id": class_id,
                    "confidence": confidence,
                    "is_referee": self.is_referee,
                }
            )
            self.online_tlwhs.append(target.last_det_tlwh)
            self.online_ids.append(raw_id)

        kept_ids = {int(target.track_id) for target in online_targets}
        self.removed_track_ids = [int(tid) for tid in removed_ids]
        self.new_tracks = [track for track in new_tracks if int(track.track_id) in kept_ids]
        self.prev_img_info = img_info
        return tracks

    def _suppress_duplicate_mask_targets(self, online_targets: List) -> List:
        if self.prediction_mask is None or len(online_targets) <= 1:
            return list(online_targets)

        grouped: Dict[int, List[Tuple[object, float]]] = {}
        for target in online_targets:
            mask_id, fill_ratio = self._dominant_mask_for_tlwh(target.last_det_tlwh)
            if mask_id is None or fill_ratio < self.mask_duplicate_min_fill:
                continue
            grouped.setdefault(mask_id, []).append((target, fill_ratio))

        suppressed_ids = set()
        for mask_id, candidates in grouped.items():
            if len(candidates) <= 1:
                continue
            candidates.sort(
                key=lambda item: (
                    int(getattr(item[0], "start_frame", self.frame_id)),
                    int(item[0].track_id),
                    -float(item[1]),
                )
            )
            keep = candidates[0][0]
            for duplicate, fill_ratio in candidates[1:]:
                suppressed_ids.add(int(duplicate.track_id))
                if self.config.debug_masks:
                    print(
                        f"    [mask-duplicate] frame={self.frame_id} "
                        f"mask={mask_id} keep={int(keep.track_id)} "
                        f"drop={int(duplicate.track_id)} "
                        f"keep_start={int(getattr(keep, 'start_frame', self.frame_id))} "
                        f"drop_start={int(getattr(duplicate, 'start_frame', self.frame_id))} "
                        f"drop_fill={fill_ratio:.3f}"
                    )

        if not suppressed_ids:
            return list(online_targets)
        return [target for target in online_targets if int(target.track_id) not in suppressed_ids]

    def _dominant_mask_for_tlwh(self, tlwh) -> Tuple[Optional[int], float]:
        if self.prediction_mask is None:
            return None, 0.0
        x, y, w, h = [int(round(float(v))) for v in tlwh]
        img_h, img_w = self.prediction_mask.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + max(1, w))
        y2 = min(img_h, y + max(1, h))
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        crop = self.prediction_mask[y1:y2, x1:x2]
        positive = crop[crop > 0]
        if positive.size == 0:
            return None, 0.0

        mask_ids, counts = np.unique(positive, return_counts=True)
        best_idx = int(np.argmax(counts))
        mask_id = int(mask_ids[best_idx])
        fill_ratio = float(counts[best_idx]) / float(max(1, crop.size))

        full_mask = self.prediction_mask == mask_id
        x1, y1, x2, y2 = self._expanded_bbox_bounds([x, y, x + w, y + h], self.prediction_mask.shape[:2])
        clipped_area = float(np.count_nonzero(full_mask[y1:y2, x1:x2]))
        total_area = float(np.count_nonzero(full_mask))
        inside_ratio = clipped_area / max(1.0, total_area)
        if clipped_area < self.config.mask_min_area_px:
            return None, 0.0
        if inside_ratio < self.config.mask_min_bbox_inside_ratio:
            return None, 0.0

        return mask_id, fill_ratio

    def _expanded_bbox_bounds(self, bbox: List[int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * self.config.mask_bbox_expand
        pad_y = bh * self.config.mask_bbox_expand
        return (
            max(0, int(round(x1 - pad_x))),
            max(0, int(round(y1 - pad_y))),
            min(frame_w, int(round(x2 + pad_x))),
            min(frame_h, int(round(y2 + pad_y))),
        )

    def _clip_mask_to_track_bbox(
        self,
        mask: np.ndarray,
        bbox: List[int],
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        if mask is None or not np.any(mask):
            return None, {"area": 0.0, "inside_ratio": 0.0}

        x1, y1, x2, y2 = self._expanded_bbox_bounds(bbox, mask.shape[:2])
        if x2 <= x1 or y2 <= y1:
            return None, {"area": 0.0, "inside_ratio": 0.0}

        total_area = float(np.count_nonzero(mask))
        clipped = np.zeros_like(mask, dtype=bool)
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        clipped_area = float(np.count_nonzero(clipped))
        inside_ratio = clipped_area / max(1.0, total_area)

        stats = {"area": clipped_area, "inside_ratio": inside_ratio}
        if clipped_area < self.config.mask_min_area_px:
            return None, stats
        if inside_ratio < self.config.mask_min_bbox_inside_ratio:
            return None, stats
        return clipped, stats

    def get_debug_summary(self) -> Dict[str, float]:
        return {
            "tracked": float(len(self.tracker.tracked_stracks)),
            "lost": float(len(self.tracker.lost_stracks)),
            "removed": float(len(self.tracker.removed_stracks)),
            "masks": float(len(self.tracklet_mask_dict or {})),
        }

    def get_visualization_mask(self):
        return self.prediction_colors_preserved

    def get_masks_for_tracks(self, tracks: List[Dict], frame_shape: Tuple[int, int]) -> Tuple[List[Dict], Optional[np.ndarray]]:
        if self.prediction_mask is None or not self.tracklet_mask_dict:
            return [], None

        frame_h, frame_w = frame_shape
        mask_map = self.prediction_mask
        if mask_map.shape[:2] != (frame_h, frame_w):
            mask_map = cv2.resize(
                mask_map.astype(np.uint8),
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_tracks: List[Dict] = []
        masks: List[np.ndarray] = []
        for track in tracks:
            raw_id = int(track.get("raw_track_id", track["track_id"] - self.track_id_offset))
            mask_id = self.tracklet_mask_dict.get(raw_id)
            if mask_id is None:
                continue
            mask = mask_map == int(mask_id)
            mask, stats = self._clip_mask_to_track_bbox(mask, track["bbox"])
            if mask is None:
                if self.config.debug_masks:
                    print(
                        f"    [mask-drift] frame={self.frame_id} track={int(track['track_id'])} "
                        f"mask={int(mask_id)} area={stats['area']:.0f} "
                        f"inside={stats['inside_ratio']:.3f}"
                    )
                continue
            mask_tracks.append(track)
            masks.append(mask)

        if not masks:
            return [], None
        return mask_tracks, np.asarray(masks, dtype=bool)

    def dominant_mask_for_bbox(self, bbox: List[int]) -> Tuple[Optional[int], float]:
        if self.prediction_mask is None:
            return None, 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return self._dominant_mask_for_tlwh((x1, y1, x2 - x1, y2 - y1))

    def clear_visualization_mask(self, mask_id: int) -> None:
        if self.prediction_mask is not None:
            self.prediction_mask[self.prediction_mask == mask_id] = 0
        if self.prediction_colors_preserved is not None:
            self.prediction_colors_preserved[self.prediction_colors_preserved == mask_id] = 0


class McByteBasketballTracker:
    """Basketball-specific MCByte tracker wrapper."""

    def __init__(
        self,
        device: str = "cuda",
        fps: int = 30,
        track_thresh: float = 0.6,
        new_track_thresh: float = 0.7,
        track_buffer: int = 90,
        cmc_method: str = "orb",
        assoc1_thresh: float = 0.8,
        assoc2_thresh: float = 0.5,
        unconfirmed_assoc_thresh: float = 0.7,
        mask_duplicate_min_fill: float = 0.45,
        mask_bbox_expand: float = 0.20,
        mask_min_area_px: int = 80,
        mask_min_bbox_inside_ratio: float = 0.35,
        ref_player_conflict_iou: float = 0.45,
        ref_player_conflict_mask_fill: float = 0.55,
        ref_player_conflict_conf_margin: float = 0.05,
        save_folder: Optional[str] = None,
        use_player_masks: bool = True,
        use_referee_masks: bool = False,
        require_cuda: bool = True,
        sam_checkpoint: Optional[str] = None,
        cutie_weights: Optional[str] = None,
        debug_masks: bool = False,
        debug_association: bool = False,
    ):
        resolved_device = _resolve_device(device, require_cuda=require_cuda)
        config = McByteTrackerConfig(
            device=resolved_device,
            fps=fps,
            track_thresh=track_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            cmc_method=cmc_method,
            assoc1_thresh=assoc1_thresh,
            assoc2_thresh=assoc2_thresh,
            unconfirmed_assoc_thresh=unconfirmed_assoc_thresh,
            mask_duplicate_min_fill=mask_duplicate_min_fill,
            mask_bbox_expand=mask_bbox_expand,
            mask_min_area_px=mask_min_area_px,
            mask_min_bbox_inside_ratio=mask_min_bbox_inside_ratio,
            ref_player_conflict_iou=ref_player_conflict_iou,
            ref_player_conflict_mask_fill=ref_player_conflict_mask_fill,
            ref_player_conflict_conf_margin=ref_player_conflict_conf_margin,
            save_folder=save_folder,
            use_player_masks=use_player_masks,
            use_referee_masks=use_referee_masks,
            require_cuda=require_cuda,
            sam_checkpoint=sam_checkpoint,
            cutie_weights=cutie_weights,
            debug_masks=debug_masks,
            debug_association=debug_association,
        )
        self.device = resolved_device
        self.player_stream = _McByteStream(
            config=config,
            class_ids=PLAYER_CLASSES,
            is_referee=False,
            use_masks=use_player_masks,
        )
        self.referee_stream = _McByteStream(
            config=config,
            class_ids=[REFEREE_CLASS],
            is_referee=True,
            track_id_offset=1000,
            use_masks=use_referee_masks,
        )

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        player_tracks = self.player_stream.update(detections, frame)
        ref_tracks = self.referee_stream.update(detections, frame)
        player_tracks = self._suppress_referee_player_mask_conflicts(player_tracks, ref_tracks)
        return player_tracks + ref_tracks

    def get_debug_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            "player": self.player_stream.get_debug_summary(),
            "referee": self.referee_stream.get_debug_summary(),
        }

    def get_visualization_mask(self):
        return self.player_stream.get_visualization_mask()

    def get_player_masks_for_tracks(self, tracks: List[Dict], frame_shape: Tuple[int, int]) -> Tuple[List[Dict], Optional[np.ndarray]]:
        return self.player_stream.get_masks_for_tracks(tracks, frame_shape)

    def _suppress_referee_player_mask_conflicts(
        self,
        player_tracks: List[Dict],
        ref_tracks: List[Dict],
    ) -> List[Dict]:
        if not player_tracks or not ref_tracks:
            return player_tracks

        kept = []
        for player in player_tracks:
            mask_id, mask_fill = self.player_stream.dominant_mask_for_bbox(player["bbox"])
            drop = False
            matched_ref = None
            for ref in ref_tracks:
                iou = _bbox_iou(player["bbox"], ref["bbox"])
                ref_conf = float(ref.get("confidence", 0.0))
                player_conf = float(player.get("confidence", 0.0))
                ref_conf_advantage = ref_conf >= player_conf + self.player_stream.config.ref_player_conflict_conf_margin
                if (
                    mask_id is not None
                    and mask_fill >= self.player_stream.config.ref_player_conflict_mask_fill
                    and iou >= self.player_stream.config.ref_player_conflict_iou
                    and ref_conf_advantage
                ):
                    drop = True
                    matched_ref = (ref, iou)
                    break

            if drop:
                ref, iou = matched_ref
                self.player_stream.clear_visualization_mask(int(mask_id))
                if self.player_stream.config.debug_masks:
                    print(
                        f"    [ref-player-mask-conflict] "
                        f"drop_player={int(player['track_id'])} "
                        f"keep_ref={int(ref['track_id'])} "
                        f"mask={int(mask_id)} fill={mask_fill:.3f} iou={iou:.3f} "
                        f"player_conf={float(player.get('confidence', 0.0)):.3f} "
                        f"ref_conf={float(ref.get('confidence', 0.0)):.3f} "
                        f"player_bbox={player['bbox']} ref_bbox={ref['bbox']}"
                    )
                continue
            kept.append(player)

        return kept


def _resolve_device(device: str, require_cuda: bool) -> str:
    if device == "cuda":
        device = "cuda:0"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA requested for MCByte, but torch.cuda.is_available() is False.")
        return "cpu"
    if require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"MCByte is configured to require CUDA, got device={device!r}.")
    return device


def _detections_to_mcbyte_array(detections: List[Dict]) -> np.ndarray:
    if not detections:
        return np.empty((0, 5), dtype=np.float32)
    rows = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        rows.append([x1, y1, x2, y2, float(det.get("confidence", 0.0))])
    return np.asarray(rows, dtype=np.float32)


def _match_source_detection(
    detections: List[Dict],
    tlwh: Tuple[float, float, float, float],
    fallback_confidence: float,
) -> Tuple[int, float]:
    if not detections:
        return PLAYER_CLASSES[0], fallback_confidence
    x, y, w, h = tlwh
    target = [x, y, x + w, y + h]
    best_det = max(detections, key=lambda det: _bbox_iou(target, det["bbox"]))
    return int(best_det["class_id"]), float(best_det.get("confidence", fallback_confidence))


def _bbox_iou(a: List[float], b: List[float]) -> float:
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

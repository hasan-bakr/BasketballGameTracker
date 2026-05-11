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
    sam_model_type: str = "vit_b"
    sam_checkpoint: Optional[str] = None
    cutie_weights: Optional[str] = None
    cutie_max_internal_size: int = 720
    debug_masks: bool = False
    debug_association: bool = False
    debug_lifecycle: bool = False
    debug_suppression: bool = False
    switch_proxy_max_dist: float = 150.0
    switch_proxy_max_dt: int = 60


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
            debug_association=False,
            debug_name="referee" if is_referee else "player",
        )
        self.tracker = _McByteTracker(args, frame_rate=config.fps, save_folder=save_folder)
        self.mask_manager = (
            MaskManager(
                device=config.device,
                sam_model_type=config.sam_model_type,
                sam_checkpoint=config.sam_checkpoint,
                cutie_weights=config.cutie_weights,
                cutie_max_internal_size=config.cutie_max_internal_size,
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

        # lifecycle tracking
        self._prev_active_ids: set = set()
        self._track_birth_frame: Dict[int, int] = {}
        self._lost_since: Dict[int, int] = {}
        self._active_positions: Dict[int, Tuple[float, float]] = {}  # offset_tid -> (cx, cy)
        self._lost_positions: Dict[int, Tuple[int, float, float]] = {}  # offset_tid -> (frame, cx, cy)
        self._removed_positions: Dict[int, tuple] = {}  # raw_tid -> (frame, cx, cy)
        self._removed_seen: set = set()
        self.events: List[Dict] = []

    @staticmethod
    def _tlwh_to_xyxy(tlwh) -> List[int]:
        x, y, w, h = [float(v) for v in tlwh]
        return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]

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

        if self.config.debug_association:
            _prev_tracked_ids = {int(s.track_id) + self.track_id_offset for s in self.tracker.tracked_stracks}
            if len(output_results):
                _scores = output_results[:, 4]
                _hi = int((_scores >= self.tracker.args.track_thresh).sum())
                _lo = int(((_scores >= 0.1) & (_scores < self.tracker.args.track_thresh)).sum())
                _smin, _smax = float(_scores.min()), float(_scores.max())
            else:
                _hi = _lo = 0; _smin = _smax = 0.0

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

        if self.config.debug_association:
            role = "ref" if self.is_referee else "player"
            _active_ids = {int(t.track_id) + self.track_id_offset for t in online_targets}
            _rm_ids = {int(tid) + self.track_id_offset for tid in removed_ids}
            _went_lost = sorted(_prev_tracked_ids - _active_ids - _rm_ids)
            _lost_pool = sorted(int(s.track_id) + self.track_id_offset for s in self.tracker.lost_stracks)
            _went_lost_str = f" went_lost={_went_lost}" if _went_lost else ""
            _pool_str = f"[{_lost_pool}]" if _lost_pool else "[]"
            print(
                f"    [assoc:{role} f={self.frame_id}] "
                f"in={len(output_results)}(h={_hi},l={_lo}) s=({_smin:.3f},{_smax:.3f}) "
                f"on={len(online_targets)} pool={_pool_str} "
                f"new={len(new_tracks)} rm={len(removed_ids)}{_went_lost_str}"
            )

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

        self._update_lifecycle(tracks)
        return tracks

    def get_track_candidates(self) -> List[Dict]:
        candidates: List[Dict] = []
        seen = set()
        for state, stracks in (
            ("active", self.tracker.tracked_stracks),
            ("lost", self.tracker.lost_stracks),
        ):
            for strack in stracks:
                raw_id = int(strack.track_id)
                track_id = raw_id + self.track_id_offset
                if track_id in seen:
                    continue
                seen.add(track_id)
                tlwh = getattr(strack, "last_det_tlwh", None)
                if tlwh is None:
                    tlwh = getattr(strack, "tlwh", None)
                if tlwh is None:
                    continue
                last_frame = int(getattr(strack, "frame_id", self.frame_id))
                candidates.append(
                    {
                        "track_id": track_id,
                        "raw_track_id": raw_id,
                        "bbox": self._tlwh_to_xyxy(tlwh),
                        "is_referee": self.is_referee,
                        "state": state,
                        "last_frame": last_frame,
                        "age": max(0, self.frame_id - last_frame),
                    }
                )
        return candidates

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
                dup_iou = self._tlwh_iou(keep.last_det_tlwh, duplicate.last_det_tlwh)
                dup_dist = self._tlwh_center_dist(keep.last_det_tlwh, duplicate.last_det_tlwh)
                if dup_iou < 0.50 and dup_dist > 40.0:
                    if self.config.debug_masks or self.config.debug_suppression:
                        print(
                            f"    [mask-duplicate:bypass f={self.frame_id}] "
                            f"mask={mask_id} keep={int(keep.track_id)} "
                            f"other={int(duplicate.track_id)} "
                            f"iou={dup_iou:.2f} dist={dup_dist:.0f} "
                            f"reason=different_bbox"
                        )
                    continue
                suppressed_ids.add(int(duplicate.track_id))
                if self.config.debug_masks or self.config.debug_suppression:
                    print(
                        f"    [mask-duplicate f={self.frame_id}] "
                        f"mask={mask_id} keep={int(keep.track_id)} "
                        f"drop={int(duplicate.track_id)} "
                        f"keep_start={int(getattr(keep, 'start_frame', self.frame_id))} "
                        f"drop_start={int(getattr(duplicate, 'start_frame', self.frame_id))} "
                        f"drop_fill={fill_ratio:.3f} iou={dup_iou:.2f} dist={dup_dist:.0f}"
                    )

        if not suppressed_ids:
            return list(online_targets)
        return [target for target in online_targets if int(target.track_id) not in suppressed_ids]

    @staticmethod
    def _tlwh_center(tlwh) -> Tuple[float, float]:
        x, y, w, h = [float(v) for v in tlwh]
        return x + w / 2.0, y + h / 2.0

    @classmethod
    def _tlwh_center_dist(cls, a, b) -> float:
        ax, ay = cls._tlwh_center(a)
        bx, by = cls._tlwh_center(b)
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    @staticmethod
    def _tlwh_iou(a, b) -> float:
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1.0, aw * ah)
        area_b = max(1.0, bw * bh)
        return float(inter / max(1.0, area_a + area_b - inter))

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

    def _update_lifecycle(self, tracks: List[Dict]) -> None:
        role = "ref" if self.is_referee else "player"
        current_ids = {t["track_id"] for t in tracks}
        rm_ids = {int(rid) + self.track_id_offset for rid in self.removed_track_ids}
        tid_to_track = {t["track_id"]: t for t in tracks}

        new_ids = current_ids - self._prev_active_ids
        disappeared = self._prev_active_ids - current_ids

        new_born, recovered, went_lost = [], [], []

        for tid in new_ids:
            self._track_birth_frame.setdefault(tid, self.frame_id)
            lost_pos = self._lost_positions.pop(tid, None)
            if tid in self._lost_since:
                lost_for = self.frame_id - self._lost_since.pop(tid)
                recovered.append((tid, lost_for))
                if self.config.debug_lifecycle or self.config.debug_suppression:
                    tr = tid_to_track.get(tid)
                    if tr and lost_pos is not None:
                        cx = (tr["bbox"][0] + tr["bbox"][2]) / 2
                        cy = (tr["bbox"][1] + tr["bbox"][3]) / 2
                        dist = ((cx - lost_pos[1]) ** 2 + (cy - lost_pos[2]) ** 2) ** 0.5
                        if dist > self.config.switch_proxy_max_dist or lost_for > 90:
                            print(
                                f"    [switch:rec {role} f={self.frame_id}] "
                                f"tid={tid} lost@f={lost_pos[0]} -> rec@f={self.frame_id} "
                                f"d={dist:.0f}px lost_for={lost_for}f"
                            )
                            self.events.append({
                                "type": "switch", "frame": self.frame_id,
                                "rec_tid": tid, "lost_frame": lost_pos[0],
                                "dist": dist, "lost_for": lost_for,
                            })
                    elif lost_for > 90:
                        print(
                            f"    [switch:rec {role} f={self.frame_id}] "
                            f"tid={tid} lost@f=? -> rec@f={self.frame_id} "
                            f"d=? lost_for={lost_for}f reason=long_lost_no_pos"
                        )
                        self.events.append({
                            "type": "switch", "frame": self.frame_id,
                            "rec_tid": tid, "lost_frame": None,
                            "dist": None, "lost_for": lost_for,
                            "reason": "long_lost_no_pos",
                        })
            else:
                new_born.append(tid)

        for tid in disappeared:
            if tid not in rm_ids:
                pos = self._active_positions.get(tid)
                if pos:
                    self._lost_positions[tid] = (self.frame_id, pos[0], pos[1])
                self._lost_since[tid] = self.frame_id
                went_lost.append(tid)

        removed = []
        for tid in rm_ids:
            if tid in self._removed_seen:
                continue
            self._removed_seen.add(tid)
            life = self.frame_id - self._track_birth_frame.pop(tid, self.frame_id)
            self._lost_since.pop(tid, None)
            self._lost_positions.pop(tid, None)
            removed.append((tid, life))

        max_dist = self.config.switch_proxy_max_dist
        max_dt = self.config.switch_proxy_max_dt

        # prune stale positions
        stale_rm = [k for k, v in self._removed_positions.items() if self.frame_id - v[0] > max_dt]
        for k in stale_rm:
            del self._removed_positions[k]
        stale_lost = [k for k, v in self._lost_positions.items() if self.frame_id - v[0] > max_dt]
        for k in stale_lost:
            del self._lost_positions[k]

        for tid in new_born:
            tr = tid_to_track.get(tid)
            if tr is None:
                continue
            x1, y1, x2, y2 = tr["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            switch_candidates = []
            # REMOVED→NEW switch
            for rm_raw, (rm_frame, rx, ry) in self._removed_positions.items():
                dt = self.frame_id - rm_frame
                dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                if dist <= max_dist and dt <= max_dt:
                    rm_tid = rm_raw + self.track_id_offset
                    switch_candidates.append(("rm", rm_tid, dist, dt))
            # LOST→NEW switch
            for lost_tid, (lost_frame, lx, ly) in self._lost_positions.items():
                if lost_tid in current_ids:  # recovered normally, not a switch
                    continue
                dt = self.frame_id - lost_frame
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist <= max_dist and dt <= max_dt:
                    switch_candidates.append(("lost", lost_tid, dist, dt))

            if not switch_candidates:
                continue

            switch_candidates.sort(key=lambda item: (item[2], item[3]))
            best = switch_candidates[0]
            second = switch_candidates[1] if len(switch_candidates) > 1 else None
            if second is not None and (second[2] - best[2]) < 20.0:
                if self.config.debug_lifecycle or self.config.debug_suppression:
                    print(
                        f"    [switch:ambiguous {role} f={self.frame_id}] "
                        f"new={tid} best={best[0]}:{best[1]} d={best[2]:.0f}px "
                        f"second={second[0]}:{second[1]} d={second[2]:.0f}px"
                    )
                continue

            source_kind, source_tid, dist, dt = best
            print(
                f"    [switch:{role} f={self.frame_id}] "
                f"{source_kind}={source_tid} -> new={tid} d={dist:.0f}px dt={dt}f"
            )
            event = {
                "type": "switch",
                "frame": self.frame_id,
                source_kind: source_tid,
                "new": tid,
                "dist": dist,
                "dt": dt,
            }
            self.events.append(event)

        # store removed track positions for future REMOVED→NEW detection
        for tid in rm_ids:
            raw = tid - self.track_id_offset
            if raw in self.tracklet_mask_dict:
                continue
            strack = next((s for s in self.tracker.removed_stracks if int(s.track_id) == raw), None)
            if strack is not None:
                x, y, w, h = strack.last_det_tlwh
                self._removed_positions[raw] = (self.frame_id, x + w / 2, y + h / 2)

        # update active positions for next frame
        self._active_positions = {
            tid: ((t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2)
            for tid, t in tid_to_track.items()
        }

        ev = {"frame": self.frame_id, "new": new_born, "recovered": recovered,
              "went_lost": went_lost, "removed": removed}
        self.events.append(ev)

        if self.config.debug_lifecycle and (new_born or recovered or went_lost or removed):
            rec_str = ",".join(f"{t}({f}f)" for t, f in recovered)
            rm_str  = ",".join(f"{t}({f}f)" for t, f in removed)
            print(
                f"    [life:{role} f={self.frame_id}]"
                + (f" NEW={new_born}" if new_born else "")
                + (f" REC=[{rec_str}]" if recovered else "")
                + (f" LOST={went_lost}" if went_lost else "")
                + (f" RM=[{rm_str}]" if removed else "")
            )

        self._prev_active_ids = current_ids

    def get_lifecycle_summary(self) -> Dict:
        from collections import Counter
        lifetimes = []
        recoveries = Counter()
        switches = []
        new_total = lost_total = rm_total = rec_total = 0

        for ev in self.events:
            if ev.get("type") == "switch":
                switches.append(ev)
                continue
            new_total  += len(ev.get("new", []))
            lost_total += len(ev.get("went_lost", []))
            rm_total   += len(ev.get("removed", []))
            rec_total  += len(ev.get("recovered", []))
            for tid, life in ev.get("removed", []):
                lifetimes.append(life)
            for tid, _ in ev.get("recovered", []):
                recoveries[tid] += 1

        lifetimes.sort()
        n = len(lifetimes)
        avg_life = sum(lifetimes) / n if n else 0
        med_life = lifetimes[n // 2] if n else 0
        p90_life = lifetimes[int(n * 0.9)] if n else 0
        shortest = sorted(
            [(tid, life) for ev in self.events if ev.get("type") != "switch"
             for tid, life in ev.get("removed", [])],
            key=lambda x: x[1]
        )[:10]

        return {
            "new": new_total, "lost": lost_total, "recovered": rec_total,
            "removed": rm_total, "switches": len(switches),
            "avg_life": avg_life, "median_life": med_life, "p90_life": p90_life,
            "shortest_lived": shortest,
            "most_recovered": recoveries.most_common(10),
        }

    def get_lifecycle_events(self) -> List[Dict]:
        return list(self.events)

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
                if self.config.debug_masks or self.config.debug_suppression:
                    reason = "area_too_small" if stats["area"] < self.config.mask_min_area_px else "inside_ratio"
                    print(
                        f"    [mask-cut f={self.frame_id}] tid={int(track['track_id'])} "
                        f"reason={reason} inside={stats['inside_ratio']:.2f} area={stats['area']:.0f}"
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
        sam_model_type: str = "vit_b",
        sam_checkpoint: Optional[str] = None,
        cutie_weights: Optional[str] = None,
        cutie_max_internal_size: int = 720,
        debug_masks: bool = False,
        debug_association: bool = False,
        debug_lifecycle: bool = False,
        debug_suppression: bool = False,
        switch_proxy_max_dist: float = 80.0,
        switch_proxy_max_dt: int = 30,
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
            sam_model_type=sam_model_type,
            sam_checkpoint=sam_checkpoint,
            cutie_weights=cutie_weights,
            cutie_max_internal_size=cutie_max_internal_size,
            debug_masks=debug_masks,
            debug_association=debug_association,
            debug_lifecycle=debug_lifecycle,
            debug_suppression=debug_suppression,
            switch_proxy_max_dist=switch_proxy_max_dist,
            switch_proxy_max_dt=switch_proxy_max_dt,
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

    def get_track_candidates(self) -> List[Dict]:
        return self.player_stream.get_track_candidates() + self.referee_stream.get_track_candidates()

    def get_lifecycle_summary(self) -> Dict:
        return {
            "player": self.player_stream.get_lifecycle_summary(),
            "referee": self.referee_stream.get_lifecycle_summary(),
        }

    def get_lifecycle_events(self) -> Dict[str, List[Dict]]:
        return {
            "player": self.player_stream.get_lifecycle_events(),
            "referee": self.referee_stream.get_lifecycle_events(),
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

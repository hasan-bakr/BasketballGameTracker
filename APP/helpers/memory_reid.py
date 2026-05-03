"""
Memory-bank ReID matcher for BoT-SORT track stabilization.

Pipeline:
    BotSort raw tracks -> DINOv2 embeddings -> full-bank matching -> stable IDs

Every frame, every track is compared against the full bank (active + ghost).
Greedy conflict resolution assigns stable IDs by descending similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class _GhostTrack:
    embedding: np.ndarray
    last_seen_frame: int
    is_referee: bool
    bbox: Tuple[float, float, float, float]


class EmbeddingExtractor:
    """DINOv2 embedding extractor for player/referee crops."""

    def __init__(self, device: str = "cuda", model_name: str = "dinov2_vits14"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        self.input_size = 224
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def _clip_bbox(bbox: List[int], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w,     x2))
        y2 = max(0, min(h,     y2))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None
        return x1, y1, x2, y2

    def _prepare_batch(
        self, frame: np.ndarray, bboxes: List[List[int]]
    ) -> Tuple[torch.Tensor, List[int]]:
        h, w = frame.shape[:2]
        crops: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for idx, bbox in enumerate(bboxes):
            clipped = self._clip_bbox(bbox, h, w)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crops.append(tensor)
            valid_indices.append(idx)

        if not crops:
            empty = torch.empty((0, 3, self.input_size, self.input_size), dtype=torch.float32, device=self.device)
            return empty, valid_indices

        batch = torch.stack(crops, dim=0).to(self.device)
        batch = (batch - self._mean) / self._std
        return batch, valid_indices

    def extract(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        results = self.extract_batch(frame, [bbox])
        return results[0] if results else np.zeros(384, dtype=np.float32)

    def extract_batch(self, frame: np.ndarray, bboxes: List[List[int]]) -> List[np.ndarray]:
        batch, valid_indices = self._prepare_batch(frame, bboxes)
        if batch.shape[0] == 0:
            dim = 384
            return [np.zeros(dim, dtype=np.float32) for _ in bboxes]

        with torch.inference_mode():
            feats = self.model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = F.normalize(feats, p=2, dim=1)

        emb_np = feats.detach().cpu().numpy().astype(np.float32)
        dim    = emb_np.shape[1]
        out: List[np.ndarray] = [np.zeros(dim, dtype=np.float32) for _ in bboxes]
        for emb_row, idx in zip(emb_np, valid_indices):
            out[idx] = emb_row
        return out


class AppearanceMemoryBank:
    """Track-level EMA embedding store used during association scoring."""

    def __init__(self, ema_alpha: float = 0.7, ttl_frames: int = 90):
        self.ema_alpha = float(ema_alpha)
        self.ttl_frames = int(ttl_frames)
        self._embeddings: Dict[int, np.ndarray] = {}
        self._last_seen: Dict[int, int] = {}

    @staticmethod
    def _is_valid_embedding(emb: np.ndarray) -> bool:
        return emb.size > 0 and np.linalg.norm(emb) > 0

    def get(self, track_id: int, frame_idx: int) -> Optional[np.ndarray]:
        emb = self._embeddings.get(int(track_id))
        last_seen = self._last_seen.get(int(track_id))
        if emb is None or last_seen is None:
            return None
        if frame_idx - last_seen > self.ttl_frames:
            return None
        return emb

    def update_track(self, track_id: int, embedding: np.ndarray, frame_idx: int) -> None:
        if not self._is_valid_embedding(embedding):
            return
        tid = int(track_id)
        if tid in self._embeddings:
            merged = self.ema_alpha * embedding + (1.0 - self.ema_alpha) * self._embeddings[tid]
            norm = np.linalg.norm(merged)
            self._embeddings[tid] = (merged / norm).astype(np.float32) if norm > 0 else embedding
        else:
            self._embeddings[tid] = embedding.astype(np.float32)
        self._last_seen[tid] = int(frame_idx)

    def update_batch(self, track_ids: List[int], embeddings: List[np.ndarray], frame_idx: int) -> None:
        for tid, emb in zip(track_ids, embeddings):
            self.update_track(int(tid), emb, frame_idx)

    def cleanup(self, frame_idx: int) -> None:
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if frame_idx - last > self.ttl_frames
        ]
        for tid in stale_ids:
            self._last_seen.pop(tid, None)
            self._embeddings.pop(tid, None)


class TrackMemoryBank:
    """Active EMA embeddings + ghost embeddings for lost tracks."""

    def __init__(
        self,
        ema_alpha: float = 0.7,
        ghost_ttl: int = 90,
        spatial_gate_min: float = 120.0,
        spatial_gate_max: float = 360.0,
        spatial_gate_diag_mult: float = 3.0,
        raw_switch_jump_thresh: float = 90.0,
        raw_switch_cooldown: int = 12,
    ):
        self.ema_alpha = float(ema_alpha)
        self.ghost_ttl = int(ghost_ttl)
        self.spatial_gate_min = float(spatial_gate_min)
        self.spatial_gate_max = float(spatial_gate_max)
        self.spatial_gate_diag_mult = float(spatial_gate_diag_mult)
        self.active: Dict[int, np.ndarray]      = {}
        self.active_is_referee: Dict[int, bool] = {}
        self.active_bbox: Dict[int, Tuple[float, float, float, float]] = {}
        self.ghosts: Dict[int, _GhostTrack]     = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _valid_emb(emb: np.ndarray) -> bool:
        return emb.size > 0 and np.linalg.norm(emb) > 0

    @staticmethod
    def _category_ok(is_referee: bool, track_id: int, stored_is_ref: bool) -> bool:
        if stored_is_ref != is_referee:
            return False
        if is_referee and track_id < 1000:
            return False
        if (not is_referee) and track_id >= 1000:
            return False
        return True

    @staticmethod
    def _bbox_tuple(bbox: Optional[List[int]]) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @staticmethod
    def _bbox_diag(bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))

    def _spatial_ok(
        self,
        bbox: Optional[List[int]],
        ghost_bbox: Tuple[float, float, float, float],
    ) -> bool:
        cur_bbox = self._bbox_tuple(bbox)
        if cur_bbox is None:
            return False
        cx, cy = self._bbox_center(cur_bbox)
        gx, gy = self._bbox_center(ghost_bbox)
        dist = float(np.hypot(cx - gx, cy - gy))
        gate = self.spatial_gate_diag_mult * max(self._bbox_diag(cur_bbox), self._bbox_diag(ghost_bbox))
        gate = min(self.spatial_gate_max, max(self.spatial_gate_min, gate))
        return dist <= gate

    # ── write ops ────────────────────────────────────────────────────────────

    def update_track(
        self,
        track_id: int,
        embedding: np.ndarray,
        frame_idx: int,
        is_referee: bool,
        bbox: Optional[List[int]] = None,
    ) -> None:
        if not self._valid_emb(embedding):
            return
        if track_id in self.active:
            merged = self.ema_alpha * embedding + (1.0 - self.ema_alpha) * self.active[track_id]
            n = np.linalg.norm(merged)
            self.active[track_id] = (merged / n).astype(np.float32) if n > 0 else embedding
        else:
            self.active[track_id] = embedding.astype(np.float32)
        self.active_is_referee[track_id] = bool(is_referee)
        bbox_tuple = self._bbox_tuple(bbox)
        if bbox_tuple is not None:
            self.active_bbox[track_id] = bbox_tuple
        self.ghosts.pop(track_id, None)

    def mark_lost(self, track_id: int, frame_idx: int) -> None:
        if track_id not in self.active:
            return
        bbox = self.active_bbox.get(track_id)
        if bbox is None:
            self.active.pop(track_id, None)
            self.active_is_referee.pop(track_id, None)
            self.active_bbox.pop(track_id, None)
            return
        is_ref = self.active_is_referee.get(track_id, track_id >= 1000)
        self.ghosts[track_id] = _GhostTrack(
            embedding=self.active[track_id],
            last_seen_frame=frame_idx,
            is_referee=is_ref,
            bbox=bbox,
        )
        self.active.pop(track_id, None)
        self.active_is_referee.pop(track_id, None)
        self.active_bbox.pop(track_id, None)

    def cleanup_ghosts(self, frame_idx: int) -> None:
        stale = [tid for tid, g in self.ghosts.items()
                 if frame_idx - g.last_seen_frame > self.ghost_ttl]
        for tid in stale:
            del self.ghosts[tid]

    # ── search (no side effects) ──────────────────────────────────────────────

    def find_best(
        self,
        embedding: np.ndarray,
        frame_idx: int,
        is_referee: bool,
        threshold: float = 0.65,
        exclude_ids: Optional[Set[int]] = None,
        ghosts_only: bool = False,
        bbox: Optional[List[int]] = None,
    ) -> Tuple[Optional[int], float, bool, int]:
        """Search active + ghost bank.
        Returns (best_id, similarity, is_ghost). No side effects — caller
        must delete matched ghost explicitly if needed."""
        if not self._valid_emb(embedding):
            return None, 0.0, False, 0

        excl = exclude_ids or set()
        best_id, best_sim, best_is_ghost = None, threshold, False
        spatial_rejects = 0

        if not ghosts_only:
            for tid, stored in self.active.items():
                if tid in excl:
                    continue
                if not self._category_ok(is_referee, tid, self.active_is_referee.get(tid, tid >= 1000)):
                    continue
                sim = float(np.dot(embedding, stored))
                if sim > best_sim:
                    best_sim, best_id, best_is_ghost = sim, tid, False

        for tid, ghost in self.ghosts.items():
            if tid in excl:
                continue
            if frame_idx - ghost.last_seen_frame > self.ghost_ttl:
                continue
            if not self._category_ok(is_referee, tid, ghost.is_referee):
                continue
            if not self._spatial_ok(bbox, ghost.bbox):
                spatial_rejects += 1
                continue
            sim = float(np.dot(embedding, ghost.embedding))
            if sim > best_sim:
                best_sim, best_id, best_is_ghost = sim, tid, True

        return best_id, best_sim, best_is_ghost, spatial_rejects


class MemoryReIDMatcher:
    """Post-processes tracker output and stabilizes track IDs.

    Every frame, every track's DINOv2 embedding is compared against the
    full memory bank (active EMA + ghost). Greedy assignment (sorted by
    descending similarity) resolves conflicts.  Young tracks (age <
    min_frames_before_match) are exempt from bank search and their stable
    IDs are reserved so eligible tracks cannot steal them.
    """

    def __init__(
        self,
        device: str = "cuda",
        similarity_threshold: float = 0.65,
        ghost_ttl: int = 90,
        ema_alpha: float = 0.7,
        extractor: Optional[EmbeddingExtractor] = None,
        ghost_only_for_new: bool = False,
        min_frames_before_match: int = 0,
        resurrect_dist_thresh: float = 35.0,
        recent_ghost_dt: int = 8,
        recent_ghost_sim_thresh: float = 0.86,
        recent_ghost_top_k: int = 0,
        recent_ghost_iou_thresh: float = 0.20,
        recent_ghost_dist_thresh: float = 90.0,
        recent_ghost_ambiguity_margin: float = 0.03,
        spatial_gate_min: float = 120.0,
        spatial_gate_max: float = 360.0,
        spatial_gate_diag_mult: float = 3.0,
        raw_switch_jump_thresh: float = 90.0,
        raw_switch_cooldown: int = 12,
        split_iou_thresh: float = 0.4,
        split_sim_thresh: float = 0.85,
    ):
        self.similarity_threshold    = float(similarity_threshold)
        self.extractor = extractor or EmbeddingExtractor(device=device)
        self.bank      = TrackMemoryBank(
            ema_alpha=ema_alpha,
            ghost_ttl=ghost_ttl,
            spatial_gate_min=spatial_gate_min,
            spatial_gate_max=spatial_gate_max,
            spatial_gate_diag_mult=spatial_gate_diag_mult,
        )
        self.ghost_only_for_new = bool(ghost_only_for_new)
        self.min_frames_before_match = int(min_frames_before_match)
        self.resurrect_dist_thresh = float(resurrect_dist_thresh)

        # Extra recovery path: when a track is about to get a fresh stable_id,
        # first compare it against the most recently lost stable IDs (ghosts).
        # This is a softer check than the full-bank threshold.
        self.recent_ghost_dt = int(recent_ghost_dt)
        self.recent_ghost_sim_thresh = float(recent_ghost_sim_thresh)
        self.recent_ghost_top_k = int(recent_ghost_top_k)
        self.recent_ghost_iou_thresh = float(recent_ghost_iou_thresh)
        self.recent_ghost_dist_thresh = float(recent_ghost_dist_thresh)
        self.recent_ghost_ambiguity_margin = float(recent_ghost_ambiguity_margin)

        self.raw_switch_jump_thresh = float(raw_switch_jump_thresh)
        self.raw_switch_cooldown = int(raw_switch_cooldown)
        self.split_iou_thresh = float(split_iou_thresh)
        self.split_sim_thresh = float(split_sim_thresh)

        self._prev_raw_ids:   Set[int]       = set()
        self._raw_to_stable:  Dict[int, int] = {}
        self._raw_is_referee: Dict[int, bool] = {}
        self._raw_first_seen: Dict[int, int] = {}
        self._stable_last_raw: Dict[int, int] = {}
        self._stable_last_bbox: Dict[int, List[int]] = {}
        self._stable_last_frame: Dict[int, int] = {}
        self._stable_switch_frame: Dict[int, int] = {}
        self._next_fresh_id_player: int = 1
        self._next_fresh_id_ref: int = 2000
        self.last_debug: Dict[str, float] = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0.0 else 0.0

    def _debug_id_assignment(
        self,
        bbox: Optional[List[int]],
        is_referee: bool,
        raw_id: int,
        frame_idx: int,
        outcome: str,
    ) -> None:
        """Diagnostic dump: closest ghost & active track to this bbox,
        with IoU + center distance, so we can see what failed thresholds."""
        if bbox is None:
            return
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        best_g = (None, 0.0, 1e9)
        for gid, g in self.bank.ghosts.items():
            if bool(g.is_referee) != bool(is_referee):
                continue
            iou = self._bbox_iou(bbox, [g.bbox[0], g.bbox[1], g.bbox[2], g.bbox[3]])
            gx = (g.bbox[0] + g.bbox[2]) * 0.5
            gy = (g.bbox[1] + g.bbox[3]) * 0.5
            d = float(np.hypot(cx - gx, cy - gy))
            if iou > best_g[1] or (iou == 0 and d < best_g[2]):
                best_g = (gid, iou, d)
        best_a = (None, 0.0, 1e9)
        for aid, abox in self.bank.active_bbox.items():
            if bool(self.bank.active_is_referee.get(aid, False)) != bool(is_referee):
                continue
            iou = self._bbox_iou(bbox, [abox[0], abox[1], abox[2], abox[3]])
            ax = (abox[0] + abox[2]) * 0.5
            ay = (abox[1] + abox[3]) * 0.5
            d = float(np.hypot(cx - ax, cy - ay))
            if iou > best_a[1] or (iou == 0 and d < best_a[2]):
                best_a = (aid, iou, d)
        print(
            f"    [id-debug] frame={frame_idx} raw={raw_id} cx={cx:.0f} cy={cy:.0f} "
            f"outcome={outcome} "
            f"closest_ghost=(sid={best_g[0]}, iou={best_g[1]:.2f}, dist={best_g[2]:.0f}) "
            f"closest_active=(sid={best_a[0]}, iou={best_a[1]:.2f}, dist={best_a[2]:.0f})"
        )

    def _iou_revive_ghost(
        self,
        bbox: Optional[List[int]],
        is_referee: bool,
        exclude: Optional[Set[int]] = None,
        iou_thresh: float = 0.5,
        dist_thresh: float = 35.0,
    ) -> Optional[int]:
        """Find a same-role ghost matching this bbox by IoU >= iou_thresh
        OR center distance <= dist_thresh. Returns best match (highest IoU,
        ties broken by smaller distance), or None."""
        if bbox is None:
            return None
        excl = exclude or set()
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        best_sid: Optional[int] = None
        best_score = (-1.0, 1e9)  # (iou, dist) larger iou wins, tie-break smaller dist
        for gid, ghost in self.bank.ghosts.items():
            if gid in excl:
                continue
            if bool(ghost.is_referee) != bool(is_referee):
                continue
            gx1, gy1, gx2, gy2 = ghost.bbox
            iou = self._bbox_iou(bbox, [gx1, gy1, gx2, gy2])
            gx = (gx1 + gx2) * 0.5
            gy = (gy1 + gy2) * 0.5
            dist = float(np.hypot(cx - gx, cy - gy))
            if iou >= iou_thresh or dist <= dist_thresh:
                score = (iou, -dist)
                if (score[0], -score[1]) > (best_score[0], best_score[1]):
                    best_score = (iou, dist)
                    best_sid = gid
        return best_sid

    def _stable_is_ref(self, sid: int) -> bool:
        if sid in self.bank.active_is_referee:
            return bool(self.bank.active_is_referee[sid])
        ghost = self.bank.ghosts.get(sid)
        if ghost is not None:
            return bool(ghost.is_referee)
        return sid >= 1000

    def _allocate_fresh_id(self, is_referee: bool = False) -> int:
        """Return a stable_id that does not collide with any past identity
        within the same role (player vs referee)."""
        seen: Set[int] = set()
        seen.update(self.bank.active.keys())
        seen.update(self.bank.ghosts.keys())
        seen.update(self._stable_last_frame.keys())
        seen.update(self._raw_to_stable.values())
        seen.update(self._stable_last_raw.keys())
        seen = {sid for sid in seen if self._stable_is_ref(sid) == is_referee}
        if is_referee:
            ceiling = max(seen) + 1 if seen else 2000
            if self._next_fresh_id_ref <= ceiling:
                self._next_fresh_id_ref = ceiling
            new_id = self._next_fresh_id_ref
            self._next_fresh_id_ref += 1
        else:
            seen = {sid for sid in seen if sid < 1000}
            ceiling = max(seen) + 1 if seen else 1
            if self._next_fresh_id_player <= ceiling:
                self._next_fresh_id_player = ceiling
            new_id = self._next_fresh_id_player
            self._next_fresh_id_player += 1
        return new_id

    def _rescue_recent_ghost(
        self,
        embedding: np.ndarray,
        bbox: Optional[List[int]],
        frame_idx: int,
        is_referee: bool,
        exclude: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, float, int, float, float]]:
        """Try to re-attach a brand-new track to a very recent ghost.

        Returns (stable_id, sim, dt_frames, iou, dist_px) or None.
        """
        if bbox is None:
            return None
        if embedding is None or embedding.size == 0 or np.linalg.norm(embedding) <= 0:
            return None

        excl = exclude or set()
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5

        scored: List[Tuple[float, int, float, float, int]] = []
        considered = 0

        # Scan all ghosts within a small dt window.
        # This avoids missing the right ghost when many ids churn.
        for gid, ghost in self.bank.ghosts.items():
            if gid in excl:
                continue
            if bool(ghost.is_referee) != bool(is_referee):
                continue
            dt = int(frame_idx) - int(ghost.last_seen_frame)
            if dt < 0 or dt > self.recent_ghost_dt:
                continue

            considered += 1
            if self.recent_ghost_top_k > 0 and considered > self.recent_ghost_top_k:
                continue

            # Reuse the same spatial gate used in full-bank matching.
            if not self.bank._spatial_ok(bbox, ghost.bbox):
                continue

            gx1, gy1, gx2, gy2 = ghost.bbox
            iou = self._bbox_iou(bbox, [gx1, gy1, gx2, gy2])
            gx = (gx1 + gx2) * 0.5
            gy = (gy1 + gy2) * 0.5
            dist = float(np.hypot(cx - gx, cy - gy))

            sim = float(np.dot(embedding, ghost.embedding))

            # Allow lower sim when the candidate is very recent and spatial
            # evidence is strong. This is intentionally separate from the
            # global full-bank threshold.
            base = self.recent_ghost_sim_thresh
            strong_spatial = (iou >= 0.30) or (dist <= 70.0)
            very_recent = dt <= 3
            sim_req = base
            if strong_spatial:
                sim_req = max(0.80, base - 0.03)
            if strong_spatial and very_recent:
                sim_req = max(0.80, base - 0.06)
            if sim < sim_req:
                continue
            if iou < self.recent_ghost_iou_thresh and dist > self.recent_ghost_dist_thresh:
                continue
            scored.append((sim, gid, iou, dist, dt))

        if not scored:
            return None

        scored.sort(key=lambda x: (-x[0], x[4]))
        best = scored[0]
        if len(scored) >= 2:
            if (best[0] - scored[1][0]) < self.recent_ghost_ambiguity_margin:
                return None

        sim, gid, iou, dist, dt = best
        return int(gid), float(sim), int(dt), float(iou), float(dist)

    def _probe_best_recent_ghost(
        self,
        embedding: np.ndarray,
        bbox: Optional[List[int]],
        frame_idx: int,
        is_referee: bool,
        exclude: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, float, int, float, float]]:
        """Return best recent ghost by similarity, with minimal filters.

        This is for debugging why a rescue did not happen.
        Returns (stable_id, sim, dt_frames, iou, dist_px) or None.
        """
        if bbox is None:
            return None
        if embedding is None or embedding.size == 0 or np.linalg.norm(embedding) <= 0:
            return None

        excl = exclude or set()
        cx = (float(bbox[0]) + float(bbox[2])) * 0.5
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5

        best: Optional[Tuple[int, float, int, float, float]] = None
        best_sim = -1.0

        for gid, ghost in self.bank.ghosts.items():
            if gid in excl:
                continue
            if bool(ghost.is_referee) != bool(is_referee):
                continue
            dt = int(frame_idx) - int(ghost.last_seen_frame)
            if dt < 0 or dt > self.recent_ghost_dt:
                continue
            if not self.bank._spatial_ok(bbox, ghost.bbox):
                continue

            gx1, gy1, gx2, gy2 = ghost.bbox
            iou = self._bbox_iou(bbox, [gx1, gy1, gx2, gy2])
            gx = (gx1 + gx2) * 0.5
            gy = (gy1 + gy2) * 0.5
            dist = float(np.hypot(cx - gx, cy - gy))
            sim = float(np.dot(embedding, ghost.embedding))

            if sim > best_sim:
                best_sim = sim
                best = (int(gid), float(sim), int(dt), float(iou), float(dist))

        return best

    def _mark_lost_tracks(self, current_raw_ids: Set[int], frame_idx: int) -> None:
        for raw_id in (self._prev_raw_ids - current_raw_ids):
            stable_id = self._raw_to_stable.pop(raw_id, raw_id)
            self.bank.mark_lost(stable_id, frame_idx)
            self._raw_is_referee.pop(raw_id, None)
            self._raw_first_seen.pop(raw_id, None)

    # ── main update ──────────────────────────────────────────────────────────

    def update(self, tracks: List[Dict], frame: np.ndarray, frame_idx: int) -> List[Dict]:
        if not tracks:
            self._mark_lost_tracks(set(), frame_idx)
            self.bank.cleanup_ghosts(frame_idx)
            self._prev_raw_ids = set()
            self.last_debug = {
                "tracks": 0.0,
                "remapped": 0.0,
                "ghost_matches": 0.0,
                "new_raw": 0.0,
                "locked": 0.0,
                "eligible": 0.0,
                "candidates": 0.0,
                "ghost_candidates": 0.0,
                "ghost_spatial_rejects": 0.0,
                "switch_blocks": 0.0,
                "fresh_allocs": 0.0,
                "ghost_revives": 0.0,
                "max_match_sim": 0.0,
                "max_ghost_sim": 0.0,
                "active_bank": float(len(self.bank.active)),
                "ghosts": float(len(self.bank.ghosts)),
            }
            return []

        current_raw_ids = {int(t["track_id"]) for t in tracks}
        new_raw_ids = current_raw_ids - set(self._raw_to_stable.keys())
        self._mark_lost_tracks(current_raw_ids, frame_idx)
        self.bank.cleanup_ghosts(frame_idx)

        raw_ids     = [int(t["track_id"])   for t in tracks]
        bboxes      = [t["bbox"]            for t in tracks]
        is_ref_flags = [bool(t["is_referee"]) for t in tracks]
        current_raw_set = set(raw_ids)

        for raw_id, is_ref in zip(raw_ids, is_ref_flags):
            self._raw_first_seen.setdefault(raw_id, frame_idx)
            self._raw_is_referee[raw_id] = is_ref

        # ── batch embeddings ─────────────────────────────────────────────────
        embeddings = self.extractor.extract_batch(frame, bboxes)

        # ── step 1: keep existing mappings stable, only new raw IDs may recover ghosts ────
        locked_stable:   Dict[int, int] = {}   # track_idx -> stable_id
        reserved_ids:    Set[int]       = set()
        eligible_indices: List[int]     = []
        locked_targets:  Set[int]       = set()
        fresh_allocs = 0
        resurrect_blocks = 0

        ghost_revives = 0
        recent_ghost_rescues = 0
        # ── pre-step: drop brand-new raws that IoU-overlap (≥ split_iou_thresh)
        # with another current track in the same role. The newer raw is a
        # duplicate detection — suppress it so its raw_id can ghost-revive
        # the matched identity once the original raw drops out.
        dup_drop: Set[int] = set()
        for i in range(len(raw_ids)):
            if i in dup_drop:
                continue
            ri = raw_ids[i]
            i_existing = ri in self._raw_to_stable
            i_brand = not i_existing
            for j in range(len(raw_ids)):
                if i == j or j in dup_drop:
                    continue
                if bool(is_ref_flags[i]) != bool(is_ref_flags[j]):
                    continue
                pair_iou = self._bbox_iou(bboxes[i], bboxes[j])
                if pair_iou < self.split_iou_thresh:
                    # Near-miss diagnostic: when one is brand-new and IoU
                    # is in [0.2, threshold), report so we can tune the gate.
                    rj = raw_ids[j]
                    j_existing = rj in self._raw_to_stable
                    j_brand = not j_existing
                    if (i_brand or j_brand) and pair_iou >= 0.2 and i < j:
                        ci = ((bboxes[i][0]+bboxes[i][2])*0.5, (bboxes[i][1]+bboxes[i][3])*0.5)
                        cj = ((bboxes[j][0]+bboxes[j][2])*0.5, (bboxes[j][1]+bboxes[j][3])*0.5)
                        d = float(np.hypot(ci[0]-cj[0], ci[1]-cj[1]))
                        print(
                            f"    [dup-near-miss] frame={frame_idx} "
                            f"raw_i={ri}(brand={i_brand}) raw_j={rj}(brand={j_brand}) "
                            f"iou={pair_iou:.2f} dist={d:.0f}"
                        )
                    continue
                rj = raw_ids[j]
                j_existing = rj in self._raw_to_stable
                j_brand = not j_existing
                drop = None
                if i_brand and not j_brand:
                    drop = i
                elif j_brand and not i_brand:
                    drop = j
                elif i_brand and j_brand:
                    drop = i if self._raw_first_seen[ri] >= self._raw_first_seen[rj] else j
                if drop is not None:
                    drop_rid = raw_ids[drop]
                    keep_rid = raw_ids[i if drop == j else j]
                    self._raw_to_stable.pop(drop_rid, None)
                    print(
                        f"    [dup-suppress] frame={frame_idx} "
                        f"drop_raw={drop_rid} keep_raw={keep_rid} "
                        f"iou={self._bbox_iou(bboxes[i], bboxes[j]):.2f}"
                    )
                    dup_drop.add(drop)
                    if drop == i:
                        break

        for i, raw_id in enumerate(raw_ids):
            if i in dup_drop:
                continue
            if raw_id in self._raw_to_stable:
                s = self._raw_to_stable[raw_id]
                if s in locked_targets:
                    new_s = self._allocate_fresh_id(is_referee=is_ref_flags[i])
                    print(
                        f"    [dedupe-lock] frame={frame_idx} "
                        f"raw={raw_id} dup_stable={s} -> fresh_stable={new_s}"
                    )
                    s = new_s
                    fresh_allocs += 1
                self._raw_to_stable[raw_id] = s
                locked_stable[i] = s
                reserved_ids.add(s)
                locked_targets.add(s)
            else:
                age = int(frame_idx) - int(self._raw_first_seen.get(raw_id, frame_idx))
                if self.min_frames_before_match > 0 and age < self.min_frames_before_match:
                    s = self._allocate_fresh_id(is_referee=is_ref_flags[i])
                    self._raw_to_stable[raw_id] = s
                    locked_stable[i] = s
                    reserved_ids.add(s)
                    locked_targets.add(s)
                    fresh_allocs += 1
                else:
                    eligible_indices.append(i)

        # ── step 2: eligible tracks search full bank (excluding reserved) ────
        # candidates: (track_idx, best_id, sim, is_ghost)
        candidates: List[Tuple[int, int, float, bool]] = []
        ghost_candidate_sims: List[float] = []
        match_sims: List[float] = []
        ghost_spatial_rejects = 0
        switch_blocks = 0

        for i in eligible_indices:
            raw_id = raw_ids[i]
            emb    = embeddings[i]
            is_ref = is_ref_flags[i]
            best_id, best_sim, is_ghost, spatial_rejects = self.bank.find_best(
                emb, frame_idx, is_ref,
                threshold=self.similarity_threshold,
                exclude_ids=reserved_ids,
                ghosts_only=self.ghost_only_for_new,
                bbox=bboxes[i],
            )
            ghost_spatial_rejects += spatial_rejects

            if best_id is not None:
                candidates.append((i, best_id, best_sim, is_ghost))
                match_sims.append(float(best_sim))
                if is_ghost:
                    ghost_candidate_sims.append(float(best_sim))
            else:
                # No bank match: fall back to existing mapping or fresh id.
                fallback = self._raw_to_stable.get(raw_id)
                if fallback is None:
                    rescued = self._rescue_recent_ghost(
                        emb,
                        bboxes[i],
                        frame_idx,
                        is_ref,
                        exclude=reserved_ids,
                    )
                    if rescued is not None:
                        sid, sim, dt, iou, dist = rescued
                        fallback = sid
                        self.bank.ghosts.pop(sid, None)
                        recent_ghost_rescues += 1
                        print(
                            f"    [recent-ghost] frame={frame_idx} "
                            f"raw={raw_id} -> stable={sid} "
                            f"sim={sim:.3f} dt={dt} iou={iou:.2f} dist={dist:.0f}"
                        )
                    else:
                        revived = self._iou_revive_ghost(
                            bboxes[i],
                            is_ref,
                            exclude=reserved_ids,
                        )
                        if revived is not None:
                            fallback = revived
                            self.bank.ghosts.pop(revived, None)
                            ghost_revives += 1
                            print(
                                f"    [iou-revive] frame={frame_idx} "
                                f"raw={raw_id} -> stable={revived} (ghost reclaimed by IoU, eligible)"
                            )
                        else:
                            probe = self._probe_best_recent_ghost(
                                emb,
                                bboxes[i],
                                frame_idx,
                                is_ref,
                                exclude=reserved_ids,
                            )
                            if probe is not None:
                                sid, sim, dt, iou, dist = probe
                                base = self.recent_ghost_sim_thresh
                                strong_spatial = (iou >= 0.30) or (dist <= 70.0)
                                very_recent = dt <= 3
                                sim_req = base
                                if strong_spatial:
                                    sim_req = max(0.80, base - 0.03)
                                if strong_spatial and very_recent:
                                    sim_req = max(0.80, base - 0.06)
                                print(
                                    f"    [recent-ghost-miss] frame={frame_idx} "
                                    f"raw={raw_id} best_sid={sid} sim={sim:.3f} sim_req={sim_req:.3f} "
                                    f"dt={dt} iou={iou:.2f} dist={dist:.0f}"
                                )
                            fallback = self._allocate_fresh_id(is_referee=is_ref)
                            fresh_allocs += 1
                            self._debug_id_assignment(
                                bboxes[i],
                                is_ref,
                                raw_id,
                                frame_idx,
                                f"fresh={fallback}(eligible)",
                            )
                candidates.append((i, fallback, 0.0, False))

        # ── step 3: greedy conflict resolution (highest sim wins) ────────────
        candidates_sorted = sorted(candidates, key=lambda c: -c[2])
        claimed:  Set[int]       = set(reserved_ids)
        assigned: Dict[int, int] = {}    # track_idx → stable_id
        ghosts_to_remove: Set[int] = set()

        for track_idx, best_id, best_sim, is_ghost in candidates_sorted:
            if best_id not in claimed:
                claimed.add(best_id)
                assigned[track_idx] = best_id
                if is_ghost:
                    ghosts_to_remove.add(best_id)
            else:
                # Conflict: fall back to existing mapping, IoU revive, or fresh.
                rid = raw_ids[track_idx]
                rbbox = bboxes[track_idx]
                rref = is_ref_flags[track_idx]
                fallback = self._raw_to_stable.get(rid)
                if fallback is None or fallback in claimed:
                    revived = self._iou_revive_ghost(rbbox, rref, exclude=claimed)
                    if revived is not None:
                        fallback = revived
                        self.bank.ghosts.pop(revived, None)
                    else:
                        fallback = self._allocate_fresh_id(is_referee=rref)
                        print(
                            f"    [resurrect-block] frame={frame_idx} "
                            f"raw={rid} conflict_fallback -> fresh_stable={fallback}"
                        )
                        fresh_allocs += 1
                        resurrect_blocks += 1
                claimed.add(fallback)
                assigned[track_idx] = fallback

        # Remove matched ghosts after all assignments are settled
        for gid in ghosts_to_remove:
            self.bank.ghosts.pop(gid, None)

        # ── step 4: build output and update bank ─────────────────────────────
        stable_tracks: List[Dict] = []
        best_by_stable: Dict[int, Tuple[np.ndarray, bool, float, List[int]]] = {}

        for i, (track, raw_id, emb, is_ref) in enumerate(
            zip(tracks, raw_ids, embeddings, is_ref_flags)
        ):
            if i in dup_drop:
                continue
            stable_id = locked_stable.get(i, assigned.get(i, raw_id))
            prev_raw = self._stable_last_raw.get(stable_id)
            prev_bbox = self._stable_last_bbox.get(stable_id)
            last_switch = self._stable_switch_frame.get(stable_id, -10**9)
            raw_changed = prev_raw is not None and int(prev_raw) != int(raw_id)
            if raw_changed and prev_bbox is not None:
                prev_tuple = TrackMemoryBank._bbox_tuple(prev_bbox)
                cur_tuple = TrackMemoryBank._bbox_tuple(track["bbox"])
                if prev_tuple is not None and cur_tuple is not None:
                    prev_center = TrackMemoryBank._bbox_center(prev_tuple)
                    cur_center = TrackMemoryBank._bbox_center(cur_tuple)
                    jump = float(np.hypot(cur_center[0] - prev_center[0], cur_center[1] - prev_center[1]))
                    prev_raw_active = int(prev_raw) in current_raw_set
                    in_cooldown = frame_idx - last_switch < self.raw_switch_cooldown
                    if prev_raw_active and (jump >= self.raw_switch_jump_thresh or in_cooldown):
                        revived = self._iou_revive_ghost(track["bbox"], is_ref)
                        if revived is not None and revived != stable_id:
                            self.bank.ghosts.pop(revived, None)
                            ghost_revives += 1
                            stable_id = revived
                        else:
                            stable_id = self._allocate_fresh_id(is_referee=is_ref)
                            fresh_allocs += 1
                        switch_blocks += 1
                    else:
                        self._stable_switch_frame[stable_id] = frame_idx
            self._raw_to_stable[raw_id] = stable_id

            out = dict(track)
            out["raw_track_id"] = int(raw_id)
            out["stable_track_id"] = int(stable_id)
            out["track_id"] = int(stable_id)
            stable_tracks.append(out)
            self._stable_last_raw[int(stable_id)] = int(raw_id)
            self._stable_last_bbox[int(stable_id)] = list(track["bbox"])
            self._stable_last_frame[int(stable_id)] = int(frame_idx)

            conf = float(track.get("confidence", 0.0))
            prev = best_by_stable.get(stable_id)
            if prev is None or conf > prev[2]:
                best_by_stable[stable_id] = (emb, is_ref, conf, track["bbox"])

        split_drops = 0

        for stable_id, (emb, is_ref, _, bbox) in best_by_stable.items():
            self.bank.update_track(stable_id, emb, frame_idx, is_referee=is_ref, bbox=bbox)

        remapped = sum(
            1 for track, raw_id in zip(stable_tracks, raw_ids)
            if int(track["track_id"]) != int(raw_id)
        )
        self.last_debug = {
            "tracks": float(len(stable_tracks)),
            "remapped": float(remapped),
            "ghost_matches": float(len(ghosts_to_remove)),
            "new_raw": float(len(new_raw_ids)),
            "locked": float(len(locked_stable)),
            "eligible": float(len(eligible_indices)),
            "candidates": float(len(candidates)),
            "ghost_candidates": float(len(ghost_candidate_sims)),
            "ghost_spatial_rejects": float(ghost_spatial_rejects),
            "switch_blocks": float(switch_blocks),
            "fresh_allocs": float(fresh_allocs),
            "resurrect_blocks": float(resurrect_blocks),
            "split_drops": float(split_drops),
            "ghost_revives": float(ghost_revives),
            "recent_ghost_rescues": float(recent_ghost_rescues),
            "max_match_sim": float(max(match_sims) if match_sims else 0.0),
            "max_ghost_sim": float(max(ghost_candidate_sims) if ghost_candidate_sims else 0.0),
            "active_bank": float(len(self.bank.active)),
            "ghosts": float(len(self.bank.ghosts)),
        }

        self._prev_raw_ids = current_raw_ids
        return stable_tracks

    def get_debug_summary(self) -> Dict[str, float]:
        return dict(self.last_debug)

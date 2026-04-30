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


class TrackMemoryBank:
    """Active EMA embeddings + ghost embeddings for lost tracks."""

    def __init__(self, ema_alpha: float = 0.7, ghost_ttl: int = 90):
        self.ema_alpha = float(ema_alpha)
        self.ghost_ttl = int(ghost_ttl)
        self.active: Dict[int, np.ndarray]      = {}
        self.active_is_referee: Dict[int, bool] = {}
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

    # ── write ops ────────────────────────────────────────────────────────────

    def update_track(self, track_id: int, embedding: np.ndarray, frame_idx: int, is_referee: bool) -> None:
        if not self._valid_emb(embedding):
            return
        if track_id in self.active:
            merged = self.ema_alpha * embedding + (1.0 - self.ema_alpha) * self.active[track_id]
            n = np.linalg.norm(merged)
            self.active[track_id] = (merged / n).astype(np.float32) if n > 0 else embedding
        else:
            self.active[track_id] = embedding.astype(np.float32)
        self.active_is_referee[track_id] = bool(is_referee)
        self.ghosts.pop(track_id, None)

    def mark_lost(self, track_id: int, frame_idx: int) -> None:
        if track_id not in self.active:
            return
        is_ref = self.active_is_referee.get(track_id, track_id >= 1000)
        self.ghosts[track_id] = _GhostTrack(
            embedding=self.active[track_id],
            last_seen_frame=frame_idx,
            is_referee=is_ref,
        )
        self.active.pop(track_id, None)
        self.active_is_referee.pop(track_id, None)

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
    ) -> Tuple[Optional[int], float, bool]:
        """Search active + ghost bank.
        Returns (best_id, similarity, is_ghost). No side effects — caller
        must delete matched ghost explicitly if needed."""
        if not self._valid_emb(embedding):
            return None, 0.0, False

        excl = exclude_ids or set()
        best_id, best_sim, best_is_ghost = None, threshold, False

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
            sim = float(np.dot(embedding, ghost.embedding))
            if sim > best_sim:
                best_sim, best_id, best_is_ghost = sim, tid, True

        return best_id, best_sim, best_is_ghost


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
        min_frames_before_match: int = 5,
    ):
        self.similarity_threshold    = float(similarity_threshold)
        self.min_frames_before_match = int(min_frames_before_match)
        self.extractor = EmbeddingExtractor(device=device)
        self.bank      = TrackMemoryBank(ema_alpha=ema_alpha, ghost_ttl=ghost_ttl)

        self._prev_raw_ids:   Set[int]       = set()
        self._raw_to_stable:  Dict[int, int] = {}
        self._raw_is_referee: Dict[int, bool] = {}
        self._raw_first_seen: Dict[int, int] = {}

    # ── helpers ──────────────────────────────────────────────────────────────

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
            return []

        current_raw_ids = {int(t["track_id"]) for t in tracks}
        self._mark_lost_tracks(current_raw_ids, frame_idx)
        self.bank.cleanup_ghosts(frame_idx)

        raw_ids     = [int(t["track_id"])   for t in tracks]
        bboxes      = [t["bbox"]            for t in tracks]
        is_ref_flags = [bool(t["is_referee"]) for t in tracks]

        for raw_id, is_ref in zip(raw_ids, is_ref_flags):
            self._raw_first_seen.setdefault(raw_id, frame_idx)
            self._raw_is_referee[raw_id] = is_ref

        # ── batch embeddings ─────────────────────────────────────────────────
        embeddings = self.extractor.extract_batch(frame, bboxes)

        # ── step 1: young tracks keep existing mapping, reserve their IDs ────
        young_stable:    Dict[int, int] = {}   # track_idx → stable_id
        reserved_ids:    Set[int]       = set()
        eligible_indices: List[int]     = []

        for i, raw_id in enumerate(raw_ids):
            age = frame_idx - self._raw_first_seen[raw_id]
            if age < self.min_frames_before_match:
                s = self._raw_to_stable.get(raw_id, raw_id)
                young_stable[i] = s
                reserved_ids.add(s)
            else:
                eligible_indices.append(i)

        # ── step 2: eligible tracks search full bank (excluding reserved) ────
        # candidates: (track_idx, best_id, sim, is_ghost)
        candidates: List[Tuple[int, int, float, bool]] = []

        for i in eligible_indices:
            raw_id = raw_ids[i]
            emb    = embeddings[i]
            is_ref = is_ref_flags[i]

            best_id, best_sim, is_ghost = self.bank.find_best(
                emb, frame_idx, is_ref,
                threshold=self.similarity_threshold,
                exclude_ids=reserved_ids,
            )

            if best_id is not None:
                candidates.append((i, best_id, best_sim, is_ghost))
            else:
                # No bank match: fall back to existing mapping or raw_id
                fallback = self._raw_to_stable.get(raw_id, raw_id)
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
                # Conflict: fall back to existing mapping or raw_id
                fallback = self._raw_to_stable.get(raw_ids[track_idx], raw_ids[track_idx])
                # Ensure fallback itself is unclaimed (edge case)
                if fallback in claimed:
                    fallback = raw_ids[track_idx]
                claimed.add(fallback)
                assigned[track_idx] = fallback

        # Remove matched ghosts after all assignments are settled
        for gid in ghosts_to_remove:
            self.bank.ghosts.pop(gid, None)

        # ── step 4: build output and update bank ─────────────────────────────
        stable_tracks: List[Dict] = []
        best_by_stable: Dict[int, Tuple[np.ndarray, bool, float]] = {}

        for i, (track, raw_id, emb, is_ref) in enumerate(
            zip(tracks, raw_ids, embeddings, is_ref_flags)
        ):
            stable_id = young_stable.get(i, assigned.get(i, raw_id))
            self._raw_to_stable[raw_id] = stable_id

            out = dict(track)
            out["track_id"] = int(stable_id)
            stable_tracks.append(out)

            conf = float(track.get("confidence", 0.0))
            prev = best_by_stable.get(stable_id)
            if prev is None or conf > prev[2]:
                best_by_stable[stable_id] = (emb, is_ref, conf)

        for stable_id, (emb, is_ref, _) in best_by_stable.items():
            self.bank.update_track(stable_id, emb, frame_idx, is_referee=is_ref)

        self._prev_raw_ids = current_raw_ids
        return stable_tracks

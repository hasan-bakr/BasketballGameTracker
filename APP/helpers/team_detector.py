"""
Team Detector
=============
Assigns players to one of two teams using K-means clustering on jersey color.

Per-track torso crops are sampled each frame; every UPDATE_INTERVAL frames the
accumulated mean colors are re-clustered (k=2) to produce stable team labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TeamDetector:
    """Unsupervised 2-team classifier based on jersey (torso) color."""

    UPDATE_INTERVAL = 30    # frames between re-cluster
    MIN_TRACK_SAMPLES = 8   # samples needed before a track participates
    MAX_HISTORY = 40        # color samples kept per track
    BBOX_SHRINK = 0.75      # shrink factor applied to bbox before color sampling
    OVERLAP_IOU_THRESH = 0.15  # skip color sample if bbox overlaps another this much

    def __init__(self):
        self._color_history: Dict[int, List[np.ndarray]] = {}
        self._team_labels: Dict[int, int] = {}
        # Will be set after first successful cluster
        self.team_bgr_colors: List[Tuple[int, int, int]] = [
            (255, 80, 50),   # team 0 placeholder
            (50, 80, 255),   # team 1 placeholder
        ]
        self._frame_count = 0
        self._cluster_ready = False

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, tracks: list) -> None:
        """Sample torso colors and periodically re-cluster."""
        self._frame_count += 1

        player_tracks = [t for t in tracks if not t["is_referee"]]
        bboxes = [t["bbox"] for t in player_tracks]

        for i, track in enumerate(player_tracks):
            if self._has_overlap(i, bboxes):
                continue
            color = self._extract_torso_color(frame, track["bbox"])
            if color is None:
                continue
            tid = track["track_id"]
            hist = self._color_history.setdefault(tid, [])
            hist.append(color)
            if len(hist) > self.MAX_HISTORY:
                del hist[: len(hist) - self.MAX_HISTORY]

        if self._frame_count % self.UPDATE_INTERVAL == 0:
            self._recluster()

    # ------------------------------------------------------------------
    def get_team(self, track_id: int) -> Optional[int]:
        """Return 0 or 1, or None if not yet classified."""
        return self._team_labels.get(track_id)

    def get_team_color(
        self, track_id: int, fallback: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        team = self._team_labels.get(track_id)
        return self.team_bgr_colors[team] if team is not None else fallback

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_iou(a: list, b: list) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _has_overlap(self, idx: int, bboxes: list) -> bool:
        for j, bbox in enumerate(bboxes):
            if j == idx:
                continue
            if self._bbox_iou(bboxes[idx], bbox) > self.OVERLAP_IOU_THRESH:
                return True
        return False

    def _extract_torso_color(
        self, frame: np.ndarray, bbox: list
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        bw = x2 - x1
        if bh < 30 or bw < 10:
            return None

        # Shrink bbox to reduce background/neighbour noise
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1 = int(cx - bw * self.BBOX_SHRINK / 2)
        x2 = int(cx + bw * self.BBOX_SHRINK / 2)
        y1 = int(cy - bh * self.BBOX_SHRINK / 2)
        y2 = int(cy + bh * self.BBOX_SHRINK / 2)
        bh = y2 - y1
        bw = x2 - x1

        # Torso band: 20%–65% of bbox height
        ty1 = y1 + int(bh * 0.20)
        ty2 = y1 + int(bh * 0.65)
        crop = frame[ty1:ty2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        # Only remove very dark pixels (shadows, black shorts/shoes bleeding in).
        # White jerseys must NOT be filtered — low saturation != noise.
        valid_px = crop[v > 40].reshape(-1, 3)

        if len(valid_px) < 10:
            valid_px = crop.reshape(-1, 3)

        return valid_px.mean(axis=0).astype(np.float32)  # BGR mean

    # ------------------------------------------------------------------
    def _recluster(self) -> None:
        valid_tids = [
            tid
            for tid, hist in self._color_history.items()
            if len(hist) >= self.MIN_TRACK_SAMPLES
        ]
        if len(valid_tids) < 4:
            return

        # Per-track mean color → feature vector (BGR)
        X = np.array(
            [np.mean(self._color_history[tid], axis=0) for tid in valid_tids],
            dtype=np.float32,
        )

        # Cluster directly on BGR (float32, 0-255 range) — avoids LAB conversion
        # pitfalls with float inputs. White vs blue jerseys are well-separated in BGR.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        labels_flat = labels.flatten()
        for i, tid in enumerate(valid_tids):
            self._team_labels[tid] = int(labels_flat[i])

        self.team_bgr_colors = [
            tuple(int(c) for c in centers[0]),
            tuple(int(c) for c in centers[1]),
        ]
        self._cluster_ready = True
        print(
            f"  [TeamDetector] re-clustered {len(valid_tids)} tracks → "
            f"team0={self.team_bgr_colors[0]}  team1={self.team_bgr_colors[1]}"
        )

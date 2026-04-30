"""
BoT-SORT tracker wrapper.

Converts YoloDetector output dicts → BotSort input array → structured track list.
Output columns from BotSort.update(): [x1, y1, x2, y2, track_id, conf, class_id, det_idx]
"""
from __future__ import annotations

import pathlib
from typing import List, Dict, Optional

import numpy as np
import torch
from boxmot.trackers.botsort.botsort import BotSort

# YOLO class IDs (custom model)
PLAYER_CLASSES = [3, 4, 5, 6, 7]
REFEREE_CLASS = 8
BALL_CLASS = 0
NUMBER_CLASS = 2


class BotSortTracker:
    """Wraps BotSort for basketball player and referee tracking.

    Maintains separate tracker instances for players and referees so their
    track IDs never collide.
    """

    def __init__(
        self,
        device: str = "cuda",
        reid_weights: Optional[pathlib.Path] = None,
        with_reid: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 60,
        match_thresh: float = 0.8,
        min_hits: int = 2,
        frame_rate: int = 30,
        asso_func: str = "giou",
        appearance_thresh: float = 0.35,
        fuse_first_associate: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.with_reid = with_reid
        weights = reid_weights or pathlib.Path("osnet_x0_25_msmt17.pt")

        common_kwargs = dict(
            reid_weights=weights,
            device=self.device,
            half=(str(self.device) != "cpu"),
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
            with_reid=with_reid,
            min_hits=min_hits,
            asso_func=asso_func,
            appearance_thresh=appearance_thresh,
            fuse_first_associate=fuse_first_associate,
        )
        self._player_tracker = BotSort(**common_kwargs)
        self._ref_tracker = BotSort(**common_kwargs)

    # ------------------------------------------------------------------
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Process one frame's detections and return active tracks.

        Args:
            detections: YoloDetector.detect() output list.
            frame: BGR image (H×W×3), used for optional ReID features.

        Returns:
            List of track dicts:
                {track_id, bbox:[x1,y1,x2,y2], class_id, confidence, is_referee}
        """
        player_dets = _to_array([d for d in detections if d["class_id"] in PLAYER_CLASSES])
        ref_dets = _to_array([d for d in detections if d["class_id"] == REFEREE_CLASS])

        tracks: List[Dict] = []

        if player_dets.shape[0] > 0:
            out = self._player_tracker.update(player_dets, frame)
            tracks += _parse_output(out, is_referee=False)

        if ref_dets.shape[0] > 0:
            # Offset referee IDs by 1000 to avoid collision with player IDs
            out = self._ref_tracker.update(ref_dets, frame)
            tracks += _parse_output(out, is_referee=True, track_id_offset=1000)

        return tracks


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_array(dets: List[Dict]) -> np.ndarray:
    """Convert YoloDetector dict list → BotSort input array [N, 6]."""
    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    rows = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rows.append([x1, y1, x2, y2, d["confidence"], d["class_id"]])
    return np.array(rows, dtype=np.float32)


def _parse_output(raw: np.ndarray, is_referee: bool, track_id_offset: int = 0) -> List[Dict]:
    """Parse BotSort output array → list of track dicts."""
    tracks = []
    for row in raw:
        x1, y1, x2, y2, track_id, conf, class_id, *_ = row
        tracks.append({
            "track_id": int(track_id) + track_id_offset,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_id": int(class_id),
            "confidence": float(conf),
            "is_referee": is_referee,
        })
    return tracks

"""BoT-SORT wrapper with memory-bank-first fusion association."""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort import (
    BotSort,
    STrack,
    iou_distance,
    joint_stracks,
    linear_assignment,
)
from boxmot.utils.matching import chi2inv95

from APP.helpers.memory_reid import AppearanceMemoryBank, EmbeddingExtractor

# YOLO class IDs (custom model)
PLAYER_CLASSES = [3, 4, 5, 6, 7]
REFEREE_CLASS = 8
BALL_CLASS = 0
NUMBER_CLASS = 2


def _pairwise_giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return normalized GIoU distance in [0,1]."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)

    tl = np.maximum(b1[:, None, :2], b2[None, :, :2])
    br = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    inter_wh = np.clip(br - tl, a_min=0.0, a_max=None)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = np.clip((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]), 1e-6, None)
    area2 = np.clip((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]), 1e-6, None)
    union = np.clip(area1[:, None] + area2[None, :] - inter, 1e-6, None)
    iou = inter / union

    c_tl = np.minimum(b1[:, None, :2], b2[None, :, :2])
    c_br = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    c_wh = np.clip(c_br - c_tl, a_min=1e-6, a_max=None)
    c_area = c_wh[..., 0] * c_wh[..., 1]

    giou = iou - ((c_area - union) / c_area)
    giou_dist = 1.0 - giou  # [0,2]
    return np.clip(giou_dist / 2.0, 0.0, 1.0).astype(np.float32)


class MemoryBankBotSort(BotSort):
    """BotSort variant where appearance memory bank is part of association."""

    def __init__(
        self,
        *args,
        embedding_extractor: EmbeddingExtractor,
        memory_ttl: int = 90,
        memory_ema_alpha: float = 0.7,
        w_app: float = 0.55,
        w_giou: float = 0.30,
        w_kalman: float = 0.15,
        similarity_threshold: float = 0.55,
        bank_blend: float = 0.7,
        kalman_gate_scale: float = 1.8,
        kalman_only_position: bool = True,
        **kwargs,
    ):
        # Keep BoT-SORT ReID path disabled. Appearance comes from memory bank.
        kwargs["with_reid"] = False
        super().__init__(*args, **kwargs)
        self.embedding_extractor = embedding_extractor
        self.memory_bank = AppearanceMemoryBank(
            ema_alpha=memory_ema_alpha,
            ttl_frames=memory_ttl,
        )
        self.w_app = float(w_app)
        self.w_giou = float(w_giou)
        self.w_kalman = float(w_kalman)
        self.similarity_threshold = float(similarity_threshold)
        self.bank_blend = float(bank_blend)
        self._gating_threshold = chi2inv95[5 if self.is_obb else 4] * float(kalman_gate_scale)
        self.kalman_only_position = bool(kalman_only_position)
        self.last_assoc_debug: Dict[str, float] = {}

    def _compute_detection_embeddings(self, img: np.ndarray, detections: List[STrack]) -> np.ndarray:
        bboxes = [det.xyxy.astype(np.int32).tolist() for det in detections]
        embeddings = self.embedding_extractor.extract_batch(img, bboxes)
        if not embeddings:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    def _compute_track_embeddings(self, tracks: List[STrack], frame_idx: int, emb_dim: int) -> np.ndarray:
        out = np.zeros((len(tracks), emb_dim), dtype=np.float32)
        for i, trk in enumerate(tracks):
            emb = self.memory_bank.get(int(trk.id), frame_idx)
            if emb is not None:
                out[i] = emb
        return out

    def _compute_live_track_embeddings(self, img: np.ndarray, tracks: List[STrack], emb_dim: int) -> np.ndarray:
        if not tracks:
            return np.zeros((0, emb_dim), dtype=np.float32)
        bboxes = [trk.xyxy.astype(np.int32).tolist() for trk in tracks]
        emb_list = self.embedding_extractor.extract_batch(img, bboxes)
        if not emb_list:
            return np.zeros((len(tracks), emb_dim), dtype=np.float32)
        arr = np.asarray(emb_list, dtype=np.float32)
        if arr.shape[1] != emb_dim:
            out = np.zeros((len(tracks), emb_dim), dtype=np.float32)
            out[:, : min(emb_dim, arr.shape[1])] = arr[:, : min(emb_dim, arr.shape[1])]
            return out
        return arr

    def _blend_track_embeddings(self, bank_embs: np.ndarray, live_embs: np.ndarray) -> np.ndarray:
        """Blend memory embedding with live crop embedding for robust association."""
        out = np.zeros_like(bank_embs)
        bank_norm = np.linalg.norm(bank_embs, axis=1) > 0
        live_norm = np.linalg.norm(live_embs, axis=1) > 0
        for i in range(bank_embs.shape[0]):
            if bank_norm[i] and live_norm[i]:
                mixed = self.bank_blend * bank_embs[i] + (1.0 - self.bank_blend) * live_embs[i]
                n = np.linalg.norm(mixed)
                out[i] = mixed / n if n > 0 else mixed
            elif bank_norm[i]:
                out[i] = bank_embs[i]
            elif live_norm[i]:
                out[i] = live_embs[i]
        return out

    def _compute_app_distance(
        self,
        track_embs: np.ndarray,
        det_embs: np.ndarray,
    ) -> np.ndarray:
        if track_embs.size == 0 or det_embs.size == 0:
            return np.zeros((track_embs.shape[0], det_embs.shape[0]), dtype=np.float32)

        sims = np.clip(track_embs @ det_embs.T, -1.0, 1.0)
        app_dist = (1.0 - sims) / 2.0  # [0,1]

        # Missing embeddings should be neutral, not destructive.
        track_valid = (np.linalg.norm(track_embs, axis=1) > 0).astype(np.float32)
        det_valid = (np.linalg.norm(det_embs, axis=1) > 0).astype(np.float32)
        invalid = (track_valid[:, None] * det_valid[None, :]) == 0
        app_dist[invalid] = 0.5
        return app_dist.astype(np.float32)

    def _compute_motion_distance(
        self,
        tracks: List[STrack],
        detections: List[STrack],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not tracks or not detections:
            return (
                np.zeros((len(tracks), len(detections)), dtype=np.float32),
                np.zeros((len(tracks), len(detections)), dtype=bool),
            )
        if self.is_obb:
            measurements = np.asarray([det.xywha for det in detections], dtype=np.float32)
        else:
            measurements = np.asarray([det.xywh for det in detections], dtype=np.float32)

        motion = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        gate_mask = np.zeros((len(tracks), len(detections)), dtype=bool)
        for row, trk in enumerate(tracks):
            gdist = self.kalman_filter.gating_distance(
                trk.mean,
                trk.covariance,
                measurements,
                only_position=self.kalman_only_position,
                metric="maha",
            )
            gate_mask[row] = gdist > self._gating_threshold
            motion[row] = np.clip(gdist / self._gating_threshold, 0.0, 1.0)
        return motion, gate_mask

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        STrack.multi_predict(strack_pool)
        self._apply_camera_motion_compensation(dets, img, strack_pool, unconfirmed)

        if not strack_pool or not detections:
            self.last_assoc_debug = {
                "tracks": float(len(strack_pool)),
                "dets": float(len(detections)),
                "pairs": 0.0,
                "gate_blocked": 0.0,
                "app_penalty": 0.0,
                "finite_pairs": 0.0,
                "matches": 0.0,
                "u_track": float(len(strack_pool)),
                "u_det": float(len(detections)),
                "match_app": 0.0,
                "match_giou": 0.0,
                "match_motion": 0.0,
                "match_cost": 0.0,
            }
            return np.empty((0, 2), dtype=int), tuple(range(len(strack_pool))), tuple(range(len(detections)))

        det_embs = self._compute_detection_embeddings(img, detections)
        emb_dim = det_embs.shape[1] if det_embs.size > 0 else 384
        bank_embs = self._compute_track_embeddings(strack_pool, self.frame_count, emb_dim)
        live_embs = self._compute_live_track_embeddings(img, strack_pool, emb_dim)
        trk_embs = self._blend_track_embeddings(bank_embs, live_embs)

        app_dist = self._compute_app_distance(trk_embs, det_embs)
        giou_dist = _pairwise_giou_distance(
            np.asarray([trk.xyxy for trk in strack_pool], dtype=np.float32),
            np.asarray([det.xyxy for det in detections], dtype=np.float32),
        )
        motion_dist, gate_mask = self._compute_motion_distance(strack_pool, detections)

        final_cost = (
            self.w_app * app_dist
            + self.w_giou * giou_dist
            + self.w_kalman * motion_dist
        )

        # Kalman stays hard-gate. Appearance is soft-prior (not a hard cutoff).
        sim_floor_dist = (1.0 - self.similarity_threshold) / 2.0
        final_cost += np.where(app_dist > sim_floor_dist, 0.10, 0.0).astype(np.float32)
        final_cost[gate_mask] = np.inf

        matches, u_track, u_detection = linear_assignment(
            final_cost, thresh=self.match_thresh
        )
        pairs = float(final_cost.size)
        gate_blocked = float(gate_mask.sum())
        app_penalty = float((app_dist > sim_floor_dist).sum())
        finite_pairs = float(np.isfinite(final_cost).sum())
        if len(matches) > 0:
            m_rows = matches[:, 0]
            m_cols = matches[:, 1]
            match_app = float(np.mean(app_dist[m_rows, m_cols]))
            match_giou = float(np.mean(giou_dist[m_rows, m_cols]))
            match_motion = float(np.mean(motion_dist[m_rows, m_cols]))
            match_cost = float(np.mean(final_cost[m_rows, m_cols]))
        else:
            match_app = match_giou = match_motion = match_cost = 0.0
        self.last_assoc_debug = {
            "tracks": float(len(strack_pool)),
            "dets": float(len(detections)),
            "pairs": pairs,
            "gate_blocked": gate_blocked,
            "app_penalty": app_penalty,
            "finite_pairs": finite_pairs,
            "matches": float(len(matches)),
            "u_track": float(len(u_track)),
            "u_det": float(len(u_detection)),
            "match_app": match_app,
            "match_giou": match_giou,
            "match_motion": match_motion,
            "match_cost": match_cost,
        }

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

            if idet < len(det_embs):
                self.memory_bank.update_track(int(track.id), det_embs[idet], self.frame_count)

        return matches, u_track, u_detection

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        out = super().update(dets, img, embs=None)
        if out.shape[0] > 0 and not self.is_obb:
            bboxes = out[:, :4].astype(np.int32).tolist()
            track_ids = out[:, 4].astype(np.int32).tolist()
            emb_list = self.embedding_extractor.extract_batch(img, bboxes)
            self.memory_bank.update_batch(track_ids, emb_list, self.frame_count)
        self.memory_bank.cleanup(self.frame_count)
        return out

    def get_assoc_debug(self) -> Dict[str, float]:
        return dict(self.last_assoc_debug)


class BotSortTracker:
    """Wraps memory-bank BoT-SORT for player and referee streams."""

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
        memory_ttl: int = 90,
        memory_ema_alpha: float = 0.7,
        w_app: float = 0.55,
        w_giou: float = 0.30,
        w_kalman: float = 0.15,
        similarity_threshold: float = 0.55,
        bank_blend: float = 0.7,
        kalman_gate_scale: float = 1.8,
        kalman_only_position: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.with_reid = with_reid
        weights = reid_weights or pathlib.Path("osnet_x0_25_msmt17.pt")
        self.embedding_extractor = EmbeddingExtractor(device=str(self.device))

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
            with_reid=False,
            min_hits=min_hits,
            asso_func=asso_func,
            appearance_thresh=appearance_thresh,
            fuse_first_associate=fuse_first_associate,
            embedding_extractor=self.embedding_extractor,
            memory_ttl=memory_ttl,
            memory_ema_alpha=memory_ema_alpha,
            w_app=w_app,
            w_giou=w_giou,
            w_kalman=w_kalman,
            similarity_threshold=similarity_threshold,
            bank_blend=bank_blend,
            kalman_gate_scale=kalman_gate_scale,
            kalman_only_position=kalman_only_position,
        )
        self._player_tracker = MemoryBankBotSort(**common_kwargs)
        self._ref_tracker = MemoryBankBotSort(**common_kwargs)

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        player_dets = _to_array([d for d in detections if d["class_id"] in PLAYER_CLASSES])
        ref_dets = _to_array([d for d in detections if d["class_id"] == REFEREE_CLASS])
        tracks: List[Dict] = []

        if player_dets.shape[0] > 0:
            out = self._player_tracker.update(player_dets, frame)
            tracks += _parse_output(out, is_referee=False)

        if ref_dets.shape[0] > 0:
            out = self._ref_tracker.update(ref_dets, frame)
            tracks += _parse_output(out, is_referee=True, track_id_offset=1000)

        return tracks

    def get_debug_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            "player": self._player_tracker.get_assoc_debug(),
            "referee": self._ref_tracker.get_assoc_debug(),
        }


def _to_array(dets: List[Dict]) -> np.ndarray:
    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    rows = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rows.append([x1, y1, x2, y2, d["confidence"], d["class_id"]])
    return np.array(rows, dtype=np.float32)


def _parse_output(raw: np.ndarray, is_referee: bool, track_id_offset: int = 0) -> List[Dict]:
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

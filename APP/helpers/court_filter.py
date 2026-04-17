"""
court_filter.py
===============
Court boundary filtering utilities for the SAM2 tracker.

Provides:
  - CourtFilterMixin  : _bbox_to_mask, _is_valid_player_bbox, _masks_to_feet
  - interp_x_at_y     : standalone interpolation helper (also exposed as static)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Standalone geometry helper ────────────────────────────────────────────────

def interp_x_at_y(pts_sorted_by_y: list, y: float) -> float:
    """Sol/sağ kenar boundary noktaları arasında y'ye göre x interpole eder.

    Perspektif nedeniyle sol kenar "/", sağ kenar "\\" şeklinde olduğundan
    tek bir x değeri yetmez; her y değeri için doğru x hesaplanmalıdır.
    Nokta aralığı dışında lineer extrapolasyon yapılır.
    """
    if not pts_sorted_by_y:
        return 0.0
    if len(pts_sorted_by_y) == 1:
        return pts_sorted_by_y[0][0]

    # Kapsam dışı — en yakın segment extrapolate edilir
    if y <= pts_sorted_by_y[0][1]:
        p0, p1 = pts_sorted_by_y[0], pts_sorted_by_y[1]
        if p1[1] == p0[1]:
            return p0[0]
        t = (y - p0[1]) / (p1[1] - p0[1])
        return p0[0] + t * (p1[0] - p0[0])

    if y >= pts_sorted_by_y[-1][1]:
        p0, p1 = pts_sorted_by_y[-2], pts_sorted_by_y[-1]
        if p1[1] == p0[1]:
            return p1[0]
        t = (y - p0[1]) / (p1[1] - p0[1])
        return p0[0] + t * (p1[0] - p0[0])

    # Çift nokta arası — interpole et
    for i in range(len(pts_sorted_by_y) - 1):
        p0, p1 = pts_sorted_by_y[i], pts_sorted_by_y[i + 1]
        if p0[1] <= y <= p1[1]:
            if p1[1] == p0[1]:
                return (p0[0] + p1[0]) / 2
            t = (y - p0[1]) / (p1[1] - p0[1])
            return p0[0] + t * (p1[0] - p0[0])

    return pts_sorted_by_y[-1][0]


# ── Mixin ─────────────────────────────────────────────────────────────────────

class CourtFilterMixin:
    """Saha içi kontrol ve mask yardımcı metodları.

    Bu mixin'i kullanan sınıf şu attribute'lara sahip olmalıdır:
      self.LEFT_KP_INDICES, self.RIGHT_KP_INDICES,
      self.TOP_KP_INDICES,  self.BOTTOM_KP_INDICES,
      self.frame_size, self.jersey_bank
    """

    # ── Saha içi filtre ───────────────────────────────────────────────────────

    def _bbox_to_mask(self, bbox: List[int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Convert a bounding box to a boolean mask."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return (x1, y1, x2, y2) bbox of a boolean mask, or None if empty."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Return centroid (x, y) of a boolean mask, or None if empty."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _is_valid_player_bbox(
        self, bbox: List[int], keypoints_xy: np.ndarray, frame_shape: Tuple[int, int]
    ) -> bool:
        """Saha içi kontrolü: her kenar için perspektife duyarlı kontrol yapar.

        - LEFT  : Keypoint "/"-çizgisi üzerinde interpole → x2 < sol_sınır → reddet
        - RIGHT : Keypoint "\\"-çizgisi üzerinde interpole → x1 > sağ_sınır → reddet
        - TOP   : ayak noktası (y2) en üst keypoint y'sının ÜSTÜNDE ise → reddet
                  (başı sınırın üstünde olabilir; ayaklar saha içindeyse geçerli oyuncu)
        - BOTTOM: ayak noktası (y2) alt baseline'ın altına inerse → reddet

        Hiç keypoint yoksa True döner (filtreleme yapılamaz).
        """
        if not any(kp[0] > 0 and kp[1] > 0 for kp in keypoints_xy):
            return True

        INNER_KP = {8, 9, 16, 17}

        def _vis(indices):
            return sorted(
                [(float(keypoints_xy[i][0]), float(keypoints_xy[i][1]))
                 for i in indices
                 if i < len(keypoints_xy) and keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0],
                key=lambda p: p[1]
            )

        x1, y1, x2, y2 = [float(c) for c in bbox]
        cy = (y1 + y2) * 0.5

        # ── SOL sınır: "/" şekilli (perspektif) ─────────────────────────────
        left_pts = _vis(self.LEFT_KP_INDICES)
        if left_pts and x2 < interp_x_at_y(left_pts, cy):
            return False

        # ── SAĞ sınır: "\" şekilli (perspektif) ─────────────────────────────
        right_pts = _vis(self.RIGHT_KP_INDICES)
        if right_pts and x1 > interp_x_at_y(right_pts, cy):
            return False

        # ── ÜST sınır: yatay ────────────────────────────────────────────────
        # Referans: y2 (ayak noktası). Oyuncunun gövdesi/başı sınırın üstünde
        # olabilir (saha çizgisine yakın oyuncu), ama ayakları içerideyse geçerli.
        # Fallback: TOP_KP görünmüyorsa tüm edge noktaların min-y'si.
        top_pts  = _vis(self.TOP_KP_INDICES)
        edge_pts = _vis([i for i in range(18) if i not in INNER_KP])
        ref_top  = top_pts if top_pts else edge_pts
        if ref_top and y2 < min(p[1] for p in ref_top):
            return False

        # ── ALT sınır: yatay ────────────────────────────────────────────────
        bottom_pts = _vis(self.BOTTOM_KP_INDICES)
        if bottom_pts and y2 > max(p[1] for p in bottom_pts):
            return False

        return True

    def _is_valid_player_mask(
        self, mask: np.ndarray, keypoints_xy: np.ndarray, frame_shape: Tuple[int, int]
    ) -> bool:
        """Check whether a mask's bbox is plausibly inside the court."""
        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            return False
        return self._is_valid_player_bbox(list(bbox), keypoints_xy, frame_shape)

    # ── Mask → feet ───────────────────────────────────────────────────────────

    def _masks_to_feet(
        self, masks: Dict[int, np.ndarray]
    ) -> Tuple[List[List[float]], List[Optional[str]]]:
        """Extract foot positions (bottom-center of each mask) and jersey numbers."""
        feet, jerseys = [], []
        for obj_id, mask in masks.items():
            if mask.shape[:2] != self.frame_size:
                mask = cv2.resize(mask.astype(np.float32),
                                  (self.frame_size[1], self.frame_size[0])) > 0.5
            ys, xs = np.where(mask)
            feet.append([float(xs.mean()), float(ys.max())] if len(xs) > 0 else [0.0, 0.0])
            jerseys.append(self.jersey_bank.get_jersey(obj_id))
        return feet, jerseys

    # ── Static geometry helpers ───────────────────────────────────────────────

    @staticmethod
    def _interp_x_at_y(pts_sorted_by_y: list, y: float) -> float:
        """Static wrapper around module-level interp_x_at_y."""
        return interp_x_at_y(pts_sorted_by_y, y)

    @staticmethod
    def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """IoU between two boolean masks (same-shape or auto-resize)."""
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2.astype(np.float32),
                               (mask1.shape[1], mask1.shape[0])) > 0.5
        intersection = np.logical_and(mask1, mask2).sum()
        union        = np.logical_or(mask1,  mask2).sum()
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        """IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter  = (ix2 - ix1) * (iy2 - iy1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

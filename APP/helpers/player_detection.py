"""
player_detection.py
===================
Player / referee / jersey detection and smart re-detection pipeline.

Provides:
  - PlayerDetectionMixin : _detect_keypoints, _detect_referees, _draw_referees,
                           _detect_jerseys, _ocr_jersey, _smart_redetection
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from APP.helpers.court_utils import TACTICAL_KEYPOINTS, compute_homography


class PlayerDetectionMixin:
    """YOLO + PARSeq tabanlı tespit ve re-detection metodları.

    Bu mixin'i kullanan sınıf şu attribute'lara sahip olmalıdır:
      self.yolo, self.kp_model, self.jersey_detector, self.jersey_bank,
      self.device, self.use_amp, self.tracked_objects, self.next_obj_id,
      self.last_good_keypoints, self.last_H, self.homography_success_count,
      self._last_keypoints, self._last_referee_dets, self.locked_jersey_ids,
      self.PLAYER_CLASSES, self.REFEREE_CLASS, self.NUMBER_CLASS,
      self.PLAYER_MIN_CONF, self.REFEREE_MIN_CONF, self.JERSEY_MIN_CONF,
      self.JERSEY_MIN_SIZE, self.JERSEY_EXPAND, self.JERSEY_LOCK_VOTES,
      self.KEYPOINT_BORDER_MARGIN, self.PLAYER_MIN_HEIGHT_PX, self.PLAYER_MIN_AREA_PX,
      self.confidence_threshold,
      self._is_valid_player_bbox(), self._bbox_to_mask()
    """

    def _reconcile_masks(self, frame: np.ndarray, current_masks: Dict[int, np.ndarray]) -> None:
        """Mark drifted tracks as degraded by comparing YOLO detections vs SAM2 masks."""
        RECONCILE_IOU_THRESH  = 0.3
        RECONCILE_DIST_THRESH = 50

        all_detections = self.yolo.detect(
            frame,
            confidence_threshold=min(self.confidence_threshold, self.PLAYER_MIN_CONF),
        )
        player_detections = [
            d for d in all_detections if d['class_id'] in self.PLAYER_CLASSES
        ]
        if not player_detections:
            return

        for det in player_detections:
            x1, y1, x2, y2 = det['bbox']
            det_cx, det_cy = (x1 + x2) / 2, (y1 + y2) / 2
            det_mask_r = self._bbox_to_mask(det['bbox'], frame.shape[:2])

            best_oid = None
            best_mask = None
            best_dist = float("inf")
            for oid, obj in self.tracked_objects.items():
                if obj.get("is_referee", False):
                    continue
                mask = current_masks.get(oid)
                if mask is None:
                    mask = obj.get("last_mask")
                if mask is None or mask.sum() < self.MIN_MASK_AREA:
                    continue

                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                mask_cx = float(xs.mean())
                mask_cy = float(ys.mean())

                center_dist = ((det_cx - mask_cx) ** 2 + (det_cy - mask_cy) ** 2) ** 0.5
                if center_dist < best_dist:
                    best_dist = center_dist
                    best_oid = oid
                    best_mask = mask

            if best_oid is None or best_mask is None or best_dist >= RECONCILE_DIST_THRESH:
                continue

            iou = self._mask_iou(det_mask_r, best_mask)
            if iou < RECONCILE_IOU_THRESH:
                self.tracked_objects[best_oid]["degraded"] = True
                print(
                    f"  [reconcile] ID {best_oid}: mask drifted "
                    f"(IoU={iou:.2f}, dist={best_dist:.0f}px)"
                )

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _is_plausible_jersey_swap(
        self,
        matched_id: int,
        owner_id: int,
        masks: Dict[int, np.ndarray],
    ) -> bool:
        """Guard against long-distance accidental ID swaps."""
        matched_mask = masks.get(matched_id)
        owner_mask = masks.get(owner_id)
        if matched_mask is None or owner_mask is None:
            return False
        matched_ctr = self._mask_centroid(matched_mask)
        owner_ctr = self._mask_centroid(owner_mask)
        if matched_ctr is None or owner_ctr is None:
            return False
        dx = matched_ctr[0] - owner_ctr[0]
        dy = matched_ctr[1] - owner_ctr[1]
        dist = (dx * dx + dy * dy) ** 0.5
        return dist <= self.JERSEY_SWAP_MAX_DIST_PX

    # ── Court keypoint detection ──────────────────────────────────────────────

    def _detect_keypoints(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run court keypoint detection with temporal smoothing and homography."""
        if self.kp_model is None:
            return (
                np.zeros((18, 2), dtype=np.float32),
                np.zeros(18, dtype=np.float32),
                self.last_H,
            )

        results = self.kp_model.predict(
            frame, conf=0.5, verbose=False,
            half=self.device == 'cuda', device=self.device
        )
        keypoints_xy = np.zeros((18, 2), dtype=np.float32)
        confidences  = np.zeros(18, dtype=np.float32)

        if results and results[0].keypoints is not None:
            kp_data = results[0].keypoints
            if kp_data.xy is not None and len(kp_data.xy) > 0:
                xy = kp_data.xy[0].cpu().numpy()
                keypoints_xy[:min(len(xy), 18)] = xy[:18]
                if kp_data.conf is not None and len(kp_data.conf) > 0:
                    conf = kp_data.conf[0].cpu().numpy()
                    n = min(len(conf), 18)
                    confidences[:n] = conf[:n]

        SMOOTH_ALPHA    = 0.6
        JUMP_THRESHOLD  = 150
        h_frame, w_frame = frame.shape[:2]

        def _is_valid_kp(pt: np.ndarray) -> bool:
            x, y = float(pt[0]), float(pt[1])
            m = self.KEYPOINT_BORDER_MARGIN
            return m <= x < (w_frame - m) and m <= y < (h_frame - m)

        def _is_in_edge_band(pt: np.ndarray) -> bool:
            x, y = float(pt[0]), float(pt[1])
            if x <= 0 or y <= 0:
                return False
            band_x = w_frame * self.KEYPOINT_EDGE_BAND_RATIO
            band_y = h_frame * self.KEYPOINT_EDGE_BAND_RATIO
            return (
                x <= band_x or x >= (w_frame - band_x)
                or y <= band_y or y >= (h_frame - band_y)
            )

        def _filter_stationary_edge_keypoints() -> None:
            for i in range(18):
                pt = keypoints_xy[i]
                prev_pt = self._prev_detected_keypoints[i]
                if not _is_valid_kp(pt):
                    self._keypoint_stationary_counts[i] = 0
                    self._prev_detected_keypoints[i] = 0.0
                    continue

                if (
                    _is_in_edge_band(pt)
                    and _is_valid_kp(prev_pt)
                    and np.linalg.norm(pt - prev_pt) <= self.KEYPOINT_STILL_PX
                ):
                    self._keypoint_stationary_counts[i] += 1
                else:
                    self._keypoint_stationary_counts[i] = 0

                self._prev_detected_keypoints[i] = pt.copy()

                if self._keypoint_stationary_counts[i] >= self.KEYPOINT_STILL_FRAMES:
                    keypoints_xy[i] = 0.0
                    confidences[i] = 0.0

        # Frame border'da oturan sahte keypoint'leri sıfırla
        for i in range(18):
            if not _is_valid_kp(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        # Kamera geçişi algılama
        is_camera_transition = False
        if self.last_good_keypoints is not None:
            jumps = sum(
                1 for i in range(18)
                if _is_valid_kp(keypoints_xy[i]) and _is_valid_kp(self.last_good_keypoints[i])
                and np.linalg.norm(keypoints_xy[i] - self.last_good_keypoints[i]) > JUMP_THRESHOLD
            )
            valid = sum(
                1 for i in range(18)
                if _is_valid_kp(keypoints_xy[i]) and _is_valid_kp(self.last_good_keypoints[i])
            )
            is_camera_transition = valid > 0 and jumps / valid > 0.5

        if is_camera_transition:
            self.last_good_keypoints = None
            self._prev_detected_keypoints[:] = 0.0
            self._keypoint_stationary_counts[:] = 0

        _filter_stationary_edge_keypoints()

        # Yüksek-güven noktalardan ön-homografi; outlier keypoint'leri eliyor
        high_conf_kp = np.where(
            (confidences >= 0.5).reshape(-1, 1) & (keypoints_xy > 0),
            keypoints_xy, 0.0
        ).astype(np.float32)
        pre_H = compute_homography(high_conf_kp) if (high_conf_kp > 0).any() else None

        if pre_H is not None:
            for i in range(18):
                if keypoints_xy[i][0] <= 0:
                    continue
                dst = cv2.perspectiveTransform(
                    np.array([[keypoints_xy[i]]], dtype=np.float32), pre_H
                )
                tx, ty = dst[0][0]
                ex, ey = TACTICAL_KEYPOINTS[i]
                if np.sqrt((tx - ex) ** 2 + (ty - ey) ** 2) > 50:
                    keypoints_xy[i] = 0.0

        # Temporal smoothing
        if self.last_good_keypoints is not None:
            for i in range(18):
                has_det  = _is_valid_kp(keypoints_xy[i])
                has_hist = _is_valid_kp(self.last_good_keypoints[i])
                if has_det and has_hist:
                    keypoints_xy[i] = (
                        SMOOTH_ALPHA * keypoints_xy[i]
                        + (1 - SMOOTH_ALPHA) * self.last_good_keypoints[i]
                    )
                elif not has_det and has_hist:
                    if _is_valid_kp(self.last_good_keypoints[i]):
                        keypoints_xy[i] = self.last_good_keypoints[i].copy()

        for i in range(18):
            if not _is_valid_kp(keypoints_xy[i]):
                keypoints_xy[i] = 0.0

        self.last_good_keypoints = keypoints_xy.copy()

        H = compute_homography(keypoints_xy)
        if H is not None:
            self.homography_success_count += 1
            self.last_H = H
        else:
            H = self.last_H

        return keypoints_xy, confidences, H

    # ── Referee detection ─────────────────────────────────────────────────────

    def _detect_referees(self, frame: np.ndarray):
        """Run YOLO to find referees (class 8), cache results. Runs every frame."""
        all_dets = self.yolo.detect(frame, confidence_threshold=self.REFEREE_MIN_CONF)
        self._last_referee_dets = [
            d for d in all_dets
            if d['class_id'] == self.REFEREE_CLASS
            and d['confidence'] >= self.REFEREE_MIN_CONF
        ]

    def _draw_referees(self, frame: np.ndarray) -> np.ndarray:
        """Draw referee bboxes (yellow-orange) on the frame."""
        for det in self._last_referee_dets:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            color = (0, 220, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Referee {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        return frame

    # ── Jersey detection / OCR ────────────────────────────────────────────────

    def _detect_jerseys(self, frame: np.ndarray, masks: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """YOLO number bbox'ını lokalizasyon olarak kullanarak jersey numarası oku.

        YOLO class_id=2 → 1.5x expand → PARSeq OCR.
        Kilitlenmiş ID'ler atlanır.
        """
        h_frame, w_frame = frame.shape[:2]
        swap_pairs = set()

        unlocked_masks = {
            oid: mask for oid, mask in masks.items()
            if oid not in self.locked_jersey_ids
        }
        locked_masks = {
            oid: mask for oid, mask in masks.items()
            if oid in self.locked_jersey_ids
        }
        candidate_masks = {**unlocked_masks, **locked_masks}
        if not candidate_masks:
            return []

        det_results = self.yolo.detect(frame, confidence_threshold=self.JERSEY_MIN_CONF)
        number_boxes = [
            d for d in det_results
            if d['class_id'] == self.NUMBER_CLASS
            and (d['bbox'][2] - d['bbox'][0]) >= self.JERSEY_MIN_SIZE
            and (d['bbox'][3] - d['bbox'][1]) >= self.JERSEY_MIN_SIZE
        ]

        if not number_boxes:
            return []

        for det in number_boxes:
            nx1, ny1, nx2, ny2 = [int(c) for c in det['bbox']]
            cx, cy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
            bw, bh = nx2 - nx1, ny2 - ny1

            matched_id = None
            for oid, mask in candidate_masks.items():
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx]:
                        matched_id = oid
                        break

            if matched_id is None:
                continue

            ew  = int(bw * self.JERSEY_EXPAND)
            eh  = int(bh * self.JERSEY_EXPAND)
            ex1 = max(0, cx - ew // 2)
            ey1 = max(0, cy - eh // 2)
            ex2 = min(w_frame, cx + ew // 2)
            ey2 = min(h_frame, cy + eh // 2)

            jersey_crop = frame[ey1:ey2, ex1:ex2]
            if jersey_crop.size == 0:
                continue

            number = self.jersey_detector._text_to_number(self._ocr_jersey(jersey_crop))
            if not number:
                continue

            if matched_id not in self.locked_jersey_ids:
                self.jersey_bank.register(matched_id, number)

            confirmed = self.jersey_bank.get_jersey(matched_id)
            if confirmed:
                counts     = self.jersey_bank.detection_counts.get(matched_id, {})
                best_count = max(counts.values(), default=0)
                if best_count >= self.JERSEY_LOCK_VOTES:
                    self.locked_jersey_ids.add(matched_id)
                    print(f"  Jersey locked: ID {matched_id} → #{confirmed}")

            current_locked = self.jersey_bank.get_jersey(matched_id)
            owner_id = self.jersey_bank.find_by_jersey(number)
            if (
                current_locked
                and current_locked != number
                and owner_id is not None
                and owner_id != matched_id
                and matched_id in self.locked_jersey_ids
                and owner_id in self.locked_jersey_ids
                and owner_id in masks
                and self._is_plausible_jersey_swap(matched_id, owner_id, masks)
            ):
                pair = tuple(sorted((matched_id, owner_id)))
                swap_pairs.add(pair)
                print(
                    f"  Jersey swap detected: mask ID {matched_id} OCR=#{number} "
                    f"but locked as #{current_locked}; swapping with ID {owner_id}"
                )

        return sorted(swap_pairs)

    def _ocr_jersey(self, jersey_crop: np.ndarray) -> str:
        """Run PARSeq OCR on a jersey crop."""
        try:
            text, confidence = self.jersey_detector._ocr(jersey_crop)
            return text if confidence > 0.3 else ""
        except Exception:
            return ""

    # ── Smart re-detection ────────────────────────────────────────────────────

    def _smart_redetection(
        self, frame: np.ndarray, current_masks: Dict[int, np.ndarray]
    ):
        """Detect new players and add them to the tracker.

        1. Keypoint detection — saha sınırları önce belirlenir.
        2. YOLO player detection.
        3. Her detection için:
           a. Saha sınırı filtresi
           b. Boyut filtresi
           c. Zaten takip ediliyorsa atla
        4. Yeni oyuncu kaydı.
        """
        # 1. Keypoint'leri tazele (cache'i de senkronize et)
        court_kp, court_conf, court_H = self._detect_keypoints(frame)
        self._last_keypoints = (court_kp, court_conf, court_H)

        # 2. Player detections
        all_detections = self.yolo.detect(
            frame,
            confidence_threshold=min(self.confidence_threshold, self.PLAYER_MIN_CONF)
        )
        player_detections = [
            d for d in all_detections if d['class_id'] in self.PLAYER_CLASSES
        ]

        if not player_detections:
            return

        tracked_masks = {
            oid: obj["last_mask"]
            for oid, obj in self.tracked_objects.items()
            if obj["last_mask"] is not None and obj["last_mask"].sum() > 100
        }

        for det in player_detections:
            # 3a. Saha sınırı filtresi
            if not self._is_valid_player_bbox(det['bbox'], court_kp, frame.shape[:2]):
                continue

            # 3b. Boyut filtresi
            x1, y1, x2, y2 = det['bbox']
            if (y2 - y1) < self.PLAYER_MIN_HEIGHT_PX:
                continue
            if (x2 - x1) * (y2 - y1) < self.PLAYER_MIN_AREA_PX:
                continue

            det_mask = self._bbox_to_mask(det['bbox'], frame.shape[:2])
            det_area = int(det_mask.sum())

            # 3c. Zaten takip ediliyor mu?
            already_tracked = det_area > 0 and any(
                np.logical_and(det_mask, m).sum() / min(det_area, int(m.sum())) > 0.3
                for m in {**tracked_masks, **current_masks}.values()
                if m.sum() > 0
            )
            if already_tracked:
                continue

            # 4. Yeni oyuncu kaydı
            new_id = self.next_obj_id
            self.next_obj_id += 1
            print(f"  New player detected (re-detection): ID {new_id}")
            self.tracked_objects[new_id] = {
                "last_mask":  det_mask,
                "confidence": det['confidence'],
                "lost_count": 0,
                "is_referee": False,
                "initial_area": int(det_mask.sum()),
                "degraded": False,
            }

        # Batch-end reconciliation: mark drifted tracks for next-batch re-prompt.
        self._reconcile_masks(frame, current_masks)

        # ── Referee re-detection ──────────────────────────────────────────────
        ref_all_dets = self.yolo.detect(frame, confidence_threshold=self.REFEREE_MIN_CONF)
        ref_dets = [d for d in ref_all_dets if d['class_id'] == self.REFEREE_CLASS]

        tracked_ref_masks = {
            oid: obj["last_mask"]
            for oid, obj in self.tracked_objects.items()
            if obj.get("is_referee") and obj["last_mask"] is not None
               and obj["last_mask"].sum() > 100
        }
        all_masks_for_ref = {**tracked_masks, **current_masks, **tracked_ref_masks}

        for det in ref_dets:
            det_mask = self._bbox_to_mask(det['bbox'], frame.shape[:2])
            det_area = int(det_mask.sum())
            already = det_area > 0 and any(
                np.logical_and(det_mask, m).sum() / min(det_area, int(m.sum())) > 0.3
                for m in all_masks_for_ref.values()
                if m.sum() > 0
            )
            if already:
                continue
            new_id = self.next_obj_id
            self.next_obj_id += 1
            print(f"  New referee detected (re-detection): ID {new_id}")
            self.tracked_objects[new_id] = {
                "last_mask":  det_mask,
                "confidence": det['confidence'],
                "lost_count": 0,
                "is_referee": True,
                "initial_area": int(det_mask.sum()),
                "degraded": False,
            }

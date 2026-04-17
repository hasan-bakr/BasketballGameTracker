"""
sam2_pipeline.py
================
SAM2 batch extraction, initialization, propagation and mask conflict resolution.

Provides:
  - SAM2PipelineMixin : _extract_batch, _initialize_objects, _continue_tracking,
                        _propagate_batch, _resolve_mask_conflicts,
                        _remove_overlapping_masks
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


class SAM2PipelineMixin:
    """SAM2 video propagation pipeline helpers.

    Bu mixin'i kullanan sınıf şu attribute'lara sahip olmalıdır:
      self.predictor, self.yolo, self.kp_model, self.device,
      self.tracked_objects, self.next_obj_id,
      self.PLAYER_CLASSES, self.PLAYER_MIN_CONF,
      self.KEYPOINT_BORDER_MARGIN, self.SAM2_MASK_LOGIT_THRESHOLD,
      self.MIN_MASK_AREA, self._is_valid_player_bbox(), _bbox_iou(), _mask_iou()
    """

    # ── SAM2 prompt helper ────────────────────────────────────────────────────

    @staticmethod
    def _make_sam2_points(
        x1: float, y1: float, x2: float, y2: float,
        scale_x: float, scale_y: float,
    ):
        """Bbox'tan SAM2 için tek foreground nokta üret: chest-level (%30)."""
        cx = ((x1 + x2) / 2) * scale_x
        cy = (y1 + 0.30 * (y2 - y1)) * scale_y
        points = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1],        dtype=np.int32)
        return points, labels

    @staticmethod
    def _compute_spread(detections: list) -> float:
        """Detection center'ları arasındaki ortalama pairwise mesafeyi döndür.

        Düşük değer = oyuncular bir arada (jump-ball gibi kalabalık).
        Yüksek değer = oyuncular saha geneline yayılmış.
        """
        if len(detections) < 2:
            return float('inf')
        centers = [
            ((d['bbox'][0] + d['bbox'][2]) / 2,
             (d['bbox'][1] + d['bbox'][3]) / 2)
            for d in detections
        ]
        total, count = 0.0, 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                total += (dx * dx + dy * dy) ** 0.5
                count += 1
        return total / count



    # ── Batch extraction ──────────────────────────────────────────────────────

    def _extract_batch(
        self, cap, temp_dir: str, batch_size: int, frame_skip: int = 1
    ) -> List[np.ndarray]:
        """Read frames from video, optionally subsample, and write to temp_dir for SAM2."""
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        frames = []
        saved_idx = 0

        for i in range(batch_size * frame_skip):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_skip != 0:
                continue
            frames.append(frame)

            # SAM2 JPEG: max dim 1024 (optimal for SAM2)
            h, w = frame.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                frame_sam2 = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_sam2 = frame

            cv2.imwrite(f"{temp_dir}/{saved_idx:05d}.jpg", frame_sam2)
            saved_idx += 1

        return frames

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize_objects(
        self, inference_state, frames: List[np.ndarray], temp_dir: str
    ):
        """Detect players across the first batch and seed SAM2 from the best frame.

        Scans up to the first 10 frames at conf=0.3, picks the one with the most
        in-court player detections (preferring frames that have keypoints), then
        adds those boxes at the corresponding SAM2 frame index.
        """
        print("Detecting players for initialization...")

        h_orig, w_orig = frames[0].shape[:2]
        ref = cv2.imread(os.path.join(temp_dir, "00000.jpg"))
        h_ref, w_ref = ref.shape[:2]
        scale_x, scale_y = w_ref / w_orig, h_ref / h_orig

        best_frame_idx = 0
        best_detections: List[dict] = []
        best_court_kp   = np.zeros((18, 2), dtype=np.float32)
        best_spread      = 0.0
        clustering_found = False
        scan_limit       = 10   # başlangıçta ilk 10 frame

        fi = 0
        while fi < min(len(frames), scan_limit):
            frame = frames[fi]
            # Court keypoints for boundary check
            court_kp = np.zeros((18, 2), dtype=np.float32)
            if self.kp_model is not None:
                kp_results = self.kp_model.predict(
                    frame, conf=0.3, verbose=False,
                    half=self.device == 'cuda', device=self.device
                )
                if kp_results and kp_results[0].keypoints is not None:
                    kp_data = kp_results[0].keypoints
                    if kp_data.xy is not None and len(kp_data.xy) > 0:
                        xy = kp_data.xy[0].cpu().numpy()
                        court_kp[:min(len(xy), 18)] = xy[:18]

            # Frame kenarına yakın sahte keypoint'leri sıfırla
            h_f, w_f = frame.shape[:2]
            m = self.KEYPOINT_BORDER_MARGIN
            for i in range(18):
                x, y = float(court_kp[i][0]), float(court_kp[i][1])
                if not (m <= x < w_f - m and m <= y < h_f - m):
                    court_kp[i] = 0.0

            all_dets = self.yolo.detect(frame, confidence_threshold=self.NEW_PROMPT_MIN_CONF)
            player_dets = [d for d in all_dets if d['class_id'] in self.PLAYER_CLASSES]

            # Filter & prefer keypoint-bearing frames
            in_court = []
            for d in player_dets:
                if not self._is_valid_player_bbox(d['bbox'], court_kp, frame.shape[:2]):
                    continue
                # x1, y1, x2, y2 = d['bbox']
                # h_bb = y2 - y1
                # w_bb = x2 - x1
                # Boyut ve aspect ratio filtreleri — aktif etmek için yorumu kaldır:
                # if h_bb < self.PLAYER_MIN_HEIGHT_PX: continue
                # if h_bb * w_bb < self.PLAYER_MIN_AREA_PX: continue
                # if h_bb / max(w_bb, 1) < self.MASK_MIN_ASPECT: continue
                in_court.append(d)

            # ── Clustering / jump-ball tespiti ────────────────────────────────────
            spread = self._compute_spread(in_court)
            is_clustered = (
                len(in_court) >= self.CLUSTER_TRIGGER_COUNT
                and spread < self.CLUSTER_MIN_SPREAD_PX
            )

            if is_clustered and not clustering_found:
                clustering_found = True
                scan_limit = self.CLUSTER_MAX_SCAN_FRAMES
                print(f"  [init] Clustering detected at frame {fi} "
                      f"(spread={spread:.0f}px) — scan extended to {scan_limit} frames")

            # ── Frame seçim kriteri ─────────────────────────────────────────────────
            # Kalabalık frame > yayılımış frame olsa bile ikincisini tercih et.
            has_kp      = any(kp[0] > 0 and kp[1] > 0 for kp in court_kp)
            best_has_kp = any(kp[0] > 0 and kp[1] > 0 for kp in best_court_kp)

            kp_upgrade   = has_kp and not best_has_kp
            better_count = len(in_court) > len(best_detections)
            better_spread = spread > best_spread and len(in_court) >= max(len(best_detections) - 1, 1)

            is_better = (
                kp_upgrade
                or (has_kp == best_has_kp and better_spread)
                or (has_kp == best_has_kp and better_count and not is_clustered)
            )
            if is_better:
                best_frame_idx  = fi
                best_detections = in_court
                best_court_kp   = court_kp
                best_spread      = spread

            fi += 1

        if not best_detections:
            print("  WARNING: No players found in first 10 frames — SAM2 will propagate empty.")
            return

        # Sort by confidence and deduplicate with bbox NMS
        best_detections = sorted(best_detections, key=lambda d: d['confidence'], reverse=True)
        deduped = []
        for det in best_detections:
            if not any(self._bbox_iou(det['bbox'], d['bbox']) > self.NEW_PROMPT_NMS_IOU for d in deduped):
                deduped.append(det)
        best_detections = deduped

        all_best_frame_dets = self.yolo.detect(
            frames[best_frame_idx], confidence_threshold=self.NEW_PROMPT_MIN_CONF
        )
        skipped = (
            len([d for d in all_best_frame_dets if d['class_id'] in self.PLAYER_CLASSES])
            - len(best_detections)
        )

        for det in best_detections:
            x1, y1, x2, y2 = det['bbox']

            chest_x = int((x1 + x2) / 2)
            chest_y = int(y1 + 0.30 * (y2 - y1))
            prompt_inside_existing = False
            for obj in self.tracked_objects.values():
                m = obj.get("last_mask")
                if m is None:
                    continue
                if 0 <= chest_y < m.shape[0] and 0 <= chest_x < m.shape[1] and m[chest_y, chest_x]:
                    prompt_inside_existing = True
                    break
            if prompt_inside_existing:
                continue

            pts, lbls = self._make_sam2_points(x1, y1, x2, y2, scale_x, scale_y)
            obj_id = self.next_obj_id
            self.next_obj_id += 1
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=best_frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=lbls,
            )
            self.tracked_objects[obj_id] = {
                "last_mask": None, "confidence": 1.0, "lost_count": 0,
                "is_referee": False,
                "initial_area": None,
                "degraded": False,
            }
            self._record_prompt(best_frame_idx, obj_id, det['bbox'], "init")

        # ── Referee seeding ──────────────────────────────────────────────────
        ref_frame = frames[best_frame_idx]
        ref_dets  = self.yolo.detect(ref_frame, confidence_threshold=self.REFEREE_MIN_CONF)
        ref_dets  = [d for d in ref_dets if d['class_id'] == self.REFEREE_CLASS]
        for det in ref_dets:
            x1, y1, x2, y2 = det['bbox']
            pts, lbls = self._make_sam2_points(x1, y1, x2, y2, scale_x, scale_y)
            obj_id = self.next_obj_id
            self.next_obj_id += 1
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=best_frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=lbls,
            )
            self.tracked_objects[obj_id] = {
                "last_mask": None, "confidence": det['confidence'],
                "lost_count": 0, "is_referee": True,
                "initial_area": None,
                "degraded": False,
            }
            self._record_prompt(best_frame_idx, obj_id, det['bbox'], "init_ref")
        if ref_dets:
            print(f"  Initialized {len(ref_dets)} referee object(s) at frame {best_frame_idx}")

        print(
            f"  Initialized {len(best_detections)} player objects at frame {best_frame_idx} "
            f"(skipped {skipped} out-of-court)"
        )

    # ── Pre-propagation batch scan ────────────────────────────────────────────

    def _prescan_batch(
        self,
        inference_state,
        frames: List[np.ndarray],
        temp_dir: str,
        scan_interval: int = 15,
    ):
        """Propagasyondan ÖNCE batch boyunca yeni oyuncuları tespit edip SAM2'ye ekle.

        Her `scan_interval` frame'de YOLO çalışır. Mevcut tracked_objects ile
        örtüşmeyen yeni oyuncular `add_new_points_or_box(frame_idx=i)` ile eklenir.
        SAM2, farklı frame_idx'lerden verilen box'ları hem ileri hem geri propagate eder.

        Parametreler:
            scan_interval: Kaç frame'de bir tarama yapılacağı (default: 15).
                           Batch 150 frame → 150//15 = 10 tarama noktası.
        """
        if not frames:
            return

        h_orig, w_orig = frames[0].shape[:2]
        ref = cv2.imread(os.path.join(temp_dir, "00000.jpg"))
        h_ref, w_ref = ref.shape[:2]
        scale_x, scale_y = w_ref / w_orig, h_ref / h_orig
        m = self.KEYPOINT_BORDER_MARGIN

        # Mevcut tracked bbox'larını topla (zaten-takip kontrolü için)
        current_boxes = []
        for obj_data in self.tracked_objects.values():
            if obj_data.get("last_mask") is not None:
                current_boxes.append(obj_data["last_mask"])

        new_count = 0
        scan_indices = list(range(0, len(frames), scan_interval))
        # Frame 0 _initialize_objects ya da _continue_tracking tarafından zaten ele alındı;
        # ama zararlı değil çünkü already-tracked kontrolü çakışmayı önler.

        for fi in scan_indices:
            frame = frames[fi]
            h_f, w_f = frame.shape[:2]

            # ── Keypoint tespiti (saha filtresi için) ────────────────────────
            court_kp = np.zeros((18, 2), dtype=np.float32)
            if self.kp_model is not None:
                kp_res = self.kp_model.predict(
                    frame, conf=0.3, verbose=False,
                    half=self.device == 'cuda', device=self.device
                )
                if kp_res and kp_res[0].keypoints is not None:
                    kp_data = kp_res[0].keypoints
                    if kp_data.xy is not None and len(kp_data.xy) > 0:
                        xy = kp_data.xy[0].cpu().numpy()
                        court_kp[:min(len(xy), 18)] = xy[:18]
            for i in range(18):
                x, y = float(court_kp[i][0]), float(court_kp[i][1])
                if not (m <= x < w_f - m and m <= y < h_f - m):
                    court_kp[i] = 0.0

            # ── YOLO tespiti ─────────────────────────────────────────────────
            all_dets = self.yolo.detect(frame, confidence_threshold=self.NEW_PROMPT_MIN_CONF)
            player_dets = [d for d in all_dets if d['class_id'] in self.PLAYER_CLASSES]

            for det in player_dets:
                # Saha filtresi
                if not self._is_valid_player_bbox(det['bbox'], court_kp, frame.shape[:2]):
                    continue

                # Boyut filtresi
                x1, y1, x2, y2 = det['bbox']
                if (y2 - y1) < self.PLAYER_MIN_HEIGHT_PX:
                    continue
                if (x2 - x1) * (y2 - y1) < self.PLAYER_MIN_AREA_PX:
                    continue

                # Zaten takip edilen bir nesneyle örtüşüyor mu?
                det_mask = self._bbox_to_mask(det['bbox'], frame.shape[:2])
                det_area = int(det_mask.sum())
                already = det_area > 0 and any(
                    np.logical_and(det_mask, m_existing).sum()
                    / min(det_area, int(m_existing.sum())) > self.NEW_PROMPT_OVERLAP_EXISTING
                    for m_existing in current_boxes
                    if m_existing.sum() > 0
                )
                if already:
                    continue

                # Aynı frame içinde şu an eklenenlerle de çakışma kontrolü
                already2 = any(
                    self._bbox_iou(det['bbox'], added['bbox']) > self.NEW_PROMPT_NMS_IOU
                    for added in (
                        getattr(self, '_prescan_added_this_frame', {}).get(fi, [])
                    )
                )
                if already2:
                    continue

                chest_x = int((x1 + x2) / 2)
                chest_y = int(y1 + 0.30 * (y2 - y1))
                prompt_inside_existing = False
                for obj in self.tracked_objects.values():
                    m = obj.get("last_mask")
                    if m is None:
                        continue
                    if 0 <= chest_y < m.shape[0] and 0 <= chest_x < m.shape[1] and m[chest_y, chest_x]:
                        prompt_inside_existing = True
                        break
                if prompt_inside_existing:
                    continue

                # ── Yeni oyuncuyu SAM2 inference_state'e ekle ───────────────
                pts, lbls = self._make_sam2_points(x1, y1, x2, y2, scale_x, scale_y)
                obj_id = self.next_obj_id
                self.next_obj_id += 1
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=fi,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                )
                self.tracked_objects[obj_id] = {
                    "last_mask": None,
                    "confidence": det['confidence'],
                    "lost_count": 0,
                    "is_referee": False,
                    "initial_area": None,
                    "degraded": False,
                }
                # current_boxes güncelle ki sonraki frame'lerde çakışma olmasın
                current_boxes.append(det_mask)

                # frame bazlı ekleme kaydı (same-frame NMS için)
                if not hasattr(self, '_prescan_added_this_frame'):
                    self._prescan_added_this_frame = {}
                self._prescan_added_this_frame.setdefault(fi, []).append(det)

                new_count += 1
                print(f"  [prescan] frame {fi}: new player ID {obj_id} added to SAM2")

        if new_count:
            print(f"  [prescan] {new_count} new player(s) seeded across batch")

        # Temp state temizle
        self._prescan_added_this_frame = {}



    def _continue_tracking(self, inference_state, temp_dir: str,
                           first_frame: 'np.ndarray | None' = None):
        """Feed the last masks from the previous batch as prompts for the new batch.

        Masks are resized to match the SAM2 input resolution (same as the JPEG
        frames written by _extract_batch), NOT a fixed 256x256.

        Degraded objects (mask area shrank significantly) receive a fresh YOLO
        point prompt instead of the degraded mask — this prevents drift.
        """
        ref = cv2.imread(os.path.join(temp_dir, "00000.jpg"))
        h_ref, w_ref = ref.shape[:2]

        degraded_ids = [oid for oid, obj in self.tracked_objects.items()
                        if obj.get("degraded", False)]

        # Normal objects: mask prompt (unchanged)
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_id in degraded_ids:
                continue
            if obj_data["last_mask"] is not None:
                mask_resized = cv2.resize(
                    obj_data["last_mask"].astype(np.float32), (w_ref, h_ref)
                ) > 0.5
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask_resized,
                )
                bbox = self._mask_to_bbox(obj_data["last_mask"])
                self._record_prompt(0, obj_id, bbox, "continue_mask")

        # Degraded objects: YOLO-backed point prompt
        if degraded_ids and first_frame is not None:
            self._reprompt_degraded(
                inference_state, degraded_ids, first_frame,
                temp_dir, h_ref, w_ref,
            )

    # ── Propagation ───────────────────────────────────────────────────────────

    def _propagate_batch(
        self,
        inference_state,
        frames: List[np.ndarray],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Run SAM2 propagation in both directions and collect per-frame masks."""
        batch_masks: Dict[int, Dict[int, np.ndarray]] = {}
        h_orig, w_orig = frames[0].shape[:2]

        import torch

        def _run_pass(reverse: bool = False) -> None:
            generator = self.predictor.propagate_in_video(
                inference_state,
                reverse=reverse,
            )

            while True:
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                try:
                    out_frame_idx, out_obj_ids, out_mask_logits = next(generator)
                except StopIteration:
                    break

                score_maps: Dict[int, np.ndarray] = {}
                for k, obj_id in enumerate(out_obj_ids):
                    score = out_mask_logits[k].cpu().numpy()
                    if score.ndim == 3:
                        score = score[0]
                    score_maps[obj_id] = cv2.resize(
                        score.astype(np.float32), (w_orig, h_orig)
                    )

                frame_masks = self._resolve_mask_conflicts(score_maps)
                frame_masks = self._remove_overlapping_masks(frame_masks)

                # Forward pass ilk tercih; reverse pass yalnızca eksik objeleri doldursun.
                if out_frame_idx not in batch_masks:
                    batch_masks[out_frame_idx] = frame_masks
                else:
                    merged = batch_masks[out_frame_idx].copy()
                    for obj_id, mask in frame_masks.items():
                        merged.setdefault(obj_id, mask)
                    batch_masks[out_frame_idx] = merged

        _run_pass(reverse=False)
        _run_pass(reverse=True)

        for out_frame_idx in sorted(batch_masks.keys()):
            frame_masks = batch_masks[out_frame_idx]
            for obj_id, mask in frame_masks.items():
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]["last_mask"] = mask
                    # İlk kronolojik maskede initial_area'yı kaydet
                    if self.tracked_objects[obj_id].get("initial_area") is None:
                        self.tracked_objects[obj_id]["initial_area"] = int(mask.sum())

        return batch_masks

    # ── Mask conflict resolution ──────────────────────────────────────────────

    def _resolve_mask_conflicts(
        self, score_maps: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Create non-overlapping masks by pixel-wise max-logit ownership."""
        if not score_maps:
            return {}

        obj_ids = list(score_maps.keys())
        stacked   = np.stack([score_maps[oid] for oid in obj_ids], axis=0)
        best_idx  = np.argmax(stacked, axis=0)
        best_score = np.max(stacked, axis=0)

        resolved: Dict[int, np.ndarray] = {}
        valid_pixels = best_score > self.SAM2_MASK_LOGIT_THRESHOLD
        frame_area   = stacked.shape[1] * stacked.shape[2]   # H × W

        for i, obj_id in enumerate(obj_ids):
            mask = (best_idx == i) & valid_pixels

            if mask.sum() <= self.MIN_MASK_AREA:
                continue

            # ── Geometrik filtreler ──────────────────────────────────────────
            # Zemin/seyirci segmentleri çok büyük ya da yatay olur.
            ys, xs = np.where(mask)
            h_mask = int(ys.max() - ys.min() + 1)
            w_mask = int(xs.max() - xs.min() + 1)

            # Aspect ratio: H/W < MASK_MIN_ASPECT → zemin/yatay segment
            if h_mask / max(w_mask, 1) < self.MASK_MIN_ASPECT:
                continue

            # Max alan: frame'in %MAX_MASK_AREA_RATIO'sundan büyük olamaz
            if mask.sum() > frame_area * self.MAX_MASK_AREA_RATIO:
                continue

            # Density filtresi — aktif etmek için yorumu kaldır:
            # bbox_area = h_mask * w_mask
            # if mask.sum() / max(bbox_area, 1) < 0.15: continue


            resolved[obj_id] = mask

        return resolved


    def _remove_overlapping_masks(
        self, masks: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Remove duplicate masks (IoU > 0.5), keeping the lower (older) ID."""
        if len(masks) <= 1:
            return masks

        obj_ids   = list(masks.keys())
        to_remove: set = set()

        for i, id1 in enumerate(obj_ids):
            if id1 in to_remove:
                continue
            for id2 in obj_ids[i + 1:]:
                if id2 in to_remove:
                    continue
                if self._mask_iou(masks[id1], masks[id2]) > 0.5:
                    to_remove.add(id2 if id1 < id2 else id1)

        for obj_id in to_remove:
            masks.pop(obj_id, None)
            # tracked_objects'a dokunmuyoruz — geçici örtüşme tracking state'i yok etmesin

        return masks

    # ── Degradation detection ─────────────────────────────────────────────────

    DEGRADATION_THRESHOLD     = 0.30   # mask alanı initial'in %30'undan düşükse degrade
    REPROMPT_MAX_DIST_PX      = 100    # YOLO eşleşme mesafesi (piksel)

    def _detect_degraded_objects(self) -> list:
        """Mask alanı initial_area'nın %30'undan düşük olan objeleri degraded işaretle.

        Returns:
            Degraded olarak işaretlenen obj_id listesi.
        """
        degraded_ids = []

        for obj_id, obj_data in self.tracked_objects.items():
            initial = obj_data.get("initial_area")
            mask    = obj_data.get("last_mask")

            if initial is None or initial < self.MIN_MASK_AREA:
                continue

            if mask is None or int(mask.sum()) < self.MIN_MASK_AREA:
                obj_data["degraded"] = True
                degraded_ids.append(obj_id)
                continue

            current_area = int(mask.sum())
            ratio = current_area / initial
            if ratio < self.DEGRADATION_THRESHOLD:
                obj_data["degraded"] = True
                degraded_ids.append(obj_id)
            else:
                obj_data["degraded"] = False

        return degraded_ids

    # ── Re-prompting degraded objects ─────────────────────────────────────────

    def _reprompt_degraded(
        self,
        inference_state,
        degraded_ids: list,
        frame: 'np.ndarray',
        temp_dir: str,
        h_ref: int,
        w_ref: int,
    ):
        """Degraded objeler için YOLO detection ile taze point prompt oluştur.

        Son bilinen mask centroid'ine en yakın YOLO detection eşleştirilir.
        Eşleşme bulunamazsa eski mask ile devam edilir (fallback).
        """
        h_orig, w_orig = frame.shape[:2]
        scale_x = w_ref / w_orig
        scale_y = h_ref / h_orig

        all_dets = self.yolo.detect(frame, confidence_threshold=self.PLAYER_MIN_CONF)
        player_dets = [d for d in all_dets if d['class_id'] in self.PLAYER_CLASSES]

        for obj_id in degraded_ids:
            obj_data = self.tracked_objects[obj_id]
            last_mask = obj_data.get("last_mask")

            # Son bilinen centroid
            if last_mask is not None and last_mask.sum() > 0:
                ys, xs = np.where(last_mask)
                last_cx, last_cy = float(xs.mean()), float(ys.mean())
            else:
                # Mask tamamen kayıp — fallback yok
                print(f"  [re-prompt] ID {obj_id}: mask fully lost, skipping")
                continue

            # En yakın YOLO detection bul
            best_det  = None
            best_dist = float('inf')
            for det in player_dets:
                x1, y1, x2, y2 = det['bbox']
                dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = ((dcx - last_cx) ** 2 + (dcy - last_cy) ** 2) ** 0.5
                if dist < best_dist and dist < self.REPROMPT_MAX_DIST_PX:
                    best_dist = dist
                    best_det  = det

            if best_det is not None:
                x1, y1, x2, y2 = best_det['bbox']
                pts, lbls = self._make_sam2_points(
                    x1, y1, x2, y2, scale_x, scale_y
                )
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                )
                obj_data["degraded"]     = False
                obj_data["initial_area"] = None   # yeni propagasyonda tekrar set edilecek
                self._record_prompt(0, obj_id, best_det['bbox'], "reprompt")
                print(f"  [re-prompt] ID {obj_id}: fresh YOLO point prompt "
                      f"(dist={best_dist:.0f}px)")

                # Bu detection'ı kullanılmış olarak çıkar
                player_dets.remove(best_det)
            else:
                # Fallback: eski mask ile devam et
                if last_mask is not None:
                    mask_resized = cv2.resize(
                        last_mask.astype(np.float32), (w_ref, h_ref)
                    ) > 0.5
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=mask_resized,
                    )
                    bbox = self._mask_to_bbox(last_mask)
                    self._record_prompt(0, obj_id, bbox, "reprompt_fallback")
                print(f"  [re-prompt] ID {obj_id}: no YOLO match, "
                      f"using old mask (fallback)")

"""
SAM2 Robust Video Tracker with IoU Re-ID
=========================================
Özellikler:
- SAM2 Video Propagation (Memory Bank ile)
- IoU tabanlı Re-identification (kayıp nesneleri kurtarma)
- Güven skoru takibi
- Maske görselleştirme + ID gösterimi
"""

import os
import cv2
import numpy as np
import torch
import gc
import shutil
import sys
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# SAM2 Video imports
from sam2.build_sam import build_sam2_video_predictor

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from APP.helpers.yolo_detector import YoloDetector
from APP.helpers.jersey_detector import JerseyDetector, JerseyReIDBank


class RobustSAM2Tracker:
    """
    IoU Re-ID destekli sağlam SAM2 Video Tracker.
    """
    
    # Oyuncu sınıfları (3=player, 4=player-in-possession, 5=player-jump-shot, 6=player-layup-dunk, 7=player-shot-block)
    PLAYER_CLASSES = [3, 4, 5, 6, 7]
    
    def __init__(
        self,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",  # SAM2.1 tiny (faster)
        sam2_checkpoint: str = "models/sam2.1_hiera_tiny.pt",
        yolo_path: str = "models/yolo/best_detection.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        redetect_interval: int = 30,
        use_amp: bool = True  # Enable Automatic Mixed Precision for ~2x speedup
    ):
        print("📦 Loading Robust SAM2 Tracker...")
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.redetect_interval = redetect_interval
        self.use_amp = use_amp
        
        if use_amp:
            print("⚡ AMP (FP16) enabled for ~2x speedup")
        
        # Load models
        self.yolo = YoloDetector(model_path=yolo_path, device="cpu")
        self.predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
        
        # Jersey detector for Re-ID
        self.jersey_detector = JerseyDetector(device=device)
        self.jersey_bank = JerseyReIDBank()
        
        # Tracking state
        self.tracked_objects: Dict[int, dict] = {}  # {obj_id: {"mask": ..., "confidence": ..., "color": ...}}
        self.next_obj_id = 1
        self.frame_size = None
        
        # Color palette
        np.random.seed(42)
        self.colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(100)]
        
        print("✅ Robust SAM2 Tracker Ready!")
    
    # ═══════════════════════════════════════════════════════════════════════
    # IoU Hesaplama
    # ═══════════════════════════════════════════════════════════════════════
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """İki boolean maske arasındaki IoU'yu hesapla."""
        # Boyut uyumu sağla
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2.astype(np.float32), (mask1.shape[1], mask1.shape[0])) > 0.5
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_lost(
        self,
        lost_masks: Dict[int, np.ndarray],
        yolo_detections: List[dict],
        frame: np.ndarray
    ) -> Tuple[Dict[int, dict], List[dict]]:
        """
        Kayıp nesneleri YOLO tespitleriyle IoU ile eşleştir.
        
        Returns:
            matched: {obj_id: yolo_detection}
            unmatched: List of unmatched YOLO detections (new objects)
        """
        if not lost_masks or not yolo_detections:
            return {}, yolo_detections
        
        lost_ids = list(lost_masks.keys())
        
        # IoU matrisi oluştur
        iou_matrix = np.zeros((len(lost_ids), len(yolo_detections)))
        
        for i, obj_id in enumerate(lost_ids):
            lost_mask = lost_masks[obj_id]
            
            for j, det in enumerate(yolo_detections):
                # YOLO bbox'tan maske oluştur (basit dikdörtgen maske)
                det_mask = self._bbox_to_mask(det['bbox'], frame.shape[:2])
                iou_matrix[i, j] = self.calculate_iou(lost_mask, det_mask)
        
        # Hungarian algoritması ile optimal eşleştirme
        cost_matrix = -iou_matrix  # Minimize için negatif
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched = {}
        matched_det_indices = set()
        
        for row, col in zip(row_indices, col_indices):
            iou = iou_matrix[row, col]
            if iou >= self.iou_threshold:
                obj_id = lost_ids[row]
                matched[obj_id] = yolo_detections[col]
                matched_det_indices.add(col)
        
        # Eşleşmeyen tespitler = yeni nesneler
        unmatched = [det for i, det in enumerate(yolo_detections) if i not in matched_det_indices]
        
        return matched, unmatched
    
    def _bbox_to_mask(self, bbox: List[int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Bounding box'tan basit maske oluştur."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask
    
    # ═══════════════════════════════════════════════════════════════════════
    # Görselleştirme
    # ═══════════════════════════════════════════════════════════════════════
    
    def draw_masks_with_ids(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        confidences: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """Maskeleri ve ID'leri frame üzerine çiz."""
        result = frame.copy()
        
        for obj_id, mask in masks.items():
            # Maske boyutunu frame'e uydur
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0])) > 0.5
            
            color = self.colors[obj_id % len(self.colors)]
            
            # Maske overlay (yarı saydam)
            result[mask] = (result[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            
            # Kontur çiz
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)
            
            # Maskenin merkezini bul
            if len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # ID ve jersey numarası yazdır
                    jersey = self.jersey_bank.get_jersey(obj_id)
                    if jersey:
                        text = f"#{jersey}"  # Jersey varsa numara göster
                    else:
                        text = f"ID:{obj_id}"
                    
                    # Arka plan dikdörtgeni
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(result, (cx - tw//2 - 3, cy - th - 5), (cx + tw//2 + 3, cy + 5), color, -1)
                    cv2.putText(result, text, (cx - tw//2, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # Video İşleme
    # ═══════════════════════════════════════════════════════════════════════
    
    def process_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: int = 300,
        batch_size: int = 50
    ):
        """Video işle ve çıktı kaydet."""
        print(f"\n🎬 Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
        
        ret, first_frame = cap.read()
        if not ret:
            print("❌ Video okunamadı!")
            return
        
        self.frame_size = first_frame.shape[:2]  # (H, W)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Video writer (boyutu sonra ayarlanacak)
        writer = None
        temp_dir = "temp_robust_frames"
        
        frame_idx = 0
        
        while frame_idx < total_frames:
            batch_end = min(frame_idx + batch_size, total_frames)
            print(f"\n🔄 Batch: Frames {frame_idx} - {batch_end}")
            
            # Batch frame'leri çıkar
            frames = self._extract_batch(cap, temp_dir, batch_size)
            if not frames:
                break
            
            # Writer'ı başlat
            if writer is None:
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # SAM2 state başlat (AMP ile)
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                inference_state = self.predictor.init_state(video_path=temp_dir)
            
            # İlk batch: YOLO ile tespit
            if frame_idx == 0:
                self._initialize_objects(inference_state, frames[0], temp_dir)
            else:
                # Devam eden takip: Önceki maskeleri prompt olarak kullan
                self._continue_tracking(inference_state, temp_dir)
            
            # Propagate ve sonuçları topla (AMP ile)
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                batch_masks = self._propagate_batch(inference_state, frames, temp_dir)
            
            # Her frame için çiz ve kaydet
            for i, frame in enumerate(frames):
                local_idx = i
                if local_idx in batch_masks:
                    masks = batch_masks[local_idx]
                    
                    # Her 5 frame'de bir jersey tespiti yap (performans için)
                    if (frame_idx + i) % 5 == 0:
                        self._detect_jerseys(frame, masks)
                    
                    result = self.draw_masks_with_ids(frame, masks)
                else:
                    result = frame
                
                cv2.putText(result, f"Frame: {frame_idx + i}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Objects: {len(self.tracked_objects)}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Jersey bank durumunu da göster
                jersey_count = len(self.jersey_bank.obj_to_jersey)
                cv2.putText(result, f"Jerseys: {jersey_count}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                writer.write(result)
            
            # Periyodik YOLO re-detection - Yeni oyuncuları tespit et (jersey ile Re-ID)
            if (frame_idx + batch_size) % self.redetect_interval < batch_size:
                print("🔍 Smart re-detection...")
                self._smart_redetection(frames[-1], batch_masks.get(len(frames)-1, {}))
            
            # Cleanup
            self.predictor.reset_state(inference_state)
            frame_idx += len(frames)
            gc.collect()
            torch.cuda.empty_cache()
        
        cap.release()
        if writer:
            writer.release()
        
        # Temp cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        print(f"\n✅ Done! Output: {output_path}")
        print(f"📊 Total objects tracked: {len(self.tracked_objects)}")
    
    def _extract_batch(self, cap, temp_dir: str, batch_size: int, frame_skip: int = 1) -> List[np.ndarray]:
        """Batch frame'leri çıkar ve diske kaydet.
        
        Args:
            frame_skip: Her kaç frame'de bir alınacak (1 = her frame, 2 = her 2 frame'de 1)
        """
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        frames = []
        saved_idx = 0
        
        for i in range(batch_size * frame_skip):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Her frame_skip'te bir frame al
            if i % frame_skip != 0:
                continue
            
            frames.append(frame)
            
            # Resize for SAM2
            h, w = frame.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_resized = frame
            
            cv2.imwrite(f"{temp_dir}/{saved_idx:05d}.jpg", frame_resized)
            saved_idx += 1
        
        return frames
    
    def _initialize_objects(self, inference_state, first_frame: np.ndarray, temp_dir: str):
        """İlk karede nesneleri YOLO ile algıla ve SAM2'ye ekle."""
        print("🔍 Detecting players in first frame...")
        
        # Sadece oyuncu sınıflarını tespit et (jersey, ball, rim vb. hariç)
        all_detections = self.yolo.detect(first_frame, confidence_threshold=0.3)
        detections = [d for d in all_detections if d['class_id'] in self.PLAYER_CLASSES]
        
        # Frame boyutları
        h_orig, w_orig = first_frame.shape[:2]
        resized_frame = cv2.imread(f"{temp_dir}/00000.jpg")
        h_resized, w_resized = resized_frame.shape[:2]
        scale_x, scale_y = w_resized / w_orig, h_resized / h_orig
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            box = np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)
            
            obj_id = self.next_obj_id
            self.next_obj_id += 1
            
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=box
            )
            
            self.tracked_objects[obj_id] = {
                "last_mask": None,
                "confidence": 1.0,
                "lost_count": 0
            }
        
        print(f"   Initialized {len(detections)} objects")
    
    def _continue_tracking(self, inference_state, temp_dir: str):
        """Önceki batch'ten maskeleri devam ettir."""
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data["last_mask"] is not None:
                # Maskeyi 256x256'ya küçült (SAM2 format)
                mask_resized = cv2.resize(
                    obj_data["last_mask"].astype(np.float32), (256, 256)
                ) > 0.5
                
                self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask_resized
                )
    
    def _propagate_batch(
        self,
        inference_state,
        frames: List[np.ndarray],
        temp_dir: str
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Batch içinde propagate yap ve maskeleri topla."""
        batch_masks = {}
        
        h_orig, w_orig = frames[0].shape[:2]
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame_masks = {}
            
            for i, obj_id in enumerate(out_obj_ids):
                mask_logit = out_mask_logits[i]
                mask = (mask_logit > 0.0).cpu().numpy()
                
                if mask.ndim == 3:
                    mask = mask[0]
                
                # Orijinal boyuta resize
                mask_resized = cv2.resize(mask.astype(np.float32), (w_orig, h_orig)) > 0.5
                
                # Maske boş değilse ekle
                if mask_resized.sum() > 100:  # En az 100 piksel
                    frame_masks[obj_id] = mask_resized
                
                # Son maskeyi kaydet
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]["last_mask"] = mask_resized
            
            # Overlapping mask tespiti: Aynı bölgede birden fazla maske varsa küçüğünü kaldır
            frame_masks = self._remove_overlapping_masks(frame_masks)
            
            batch_masks[out_frame_idx] = frame_masks
        
        return batch_masks
    
    def _remove_overlapping_masks(self, masks: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Birbirine çok benzeyen (overlapping) maskeleri tespit et ve küçüğünü kaldır.
        Bu, occlusion sonrası aynı oyuncuya atanan duplicate ID'leri temizler.
        """
        if len(masks) <= 1:
            return masks
        
        obj_ids = list(masks.keys())
        to_remove = set()
        
        for i, id1 in enumerate(obj_ids):
            if id1 in to_remove:
                continue
            mask1 = masks[id1]
            
            for j, id2 in enumerate(obj_ids[i+1:], i+1):
                if id2 in to_remove:
                    continue
                mask2 = masks[id2]
                
                # IoU hesapla
                iou = self.calculate_iou(mask1, mask2)
                
                if iou > 0.5:  # Çok yüksek overlap = muhtemelen aynı nesne
                    # Küçük ID'yi koru (daha eski = daha güvenilir)
                    if id1 < id2:
                        to_remove.add(id2)
                    else:
                        to_remove.add(id1)
        
        # Çakışan maskeleri kaldır
        for obj_id in to_remove:
            del masks[obj_id]
            if obj_id in self.tracked_objects:
                del self.tracked_objects[obj_id]
        
        return masks
    
    def _detect_jerseys(self, frame: np.ndarray, masks: Dict[int, np.ndarray]):
        """
        YOLO ile jersey (class 2 = 'number') tespit et ve oyunculara eşleştir.
        """
        # YOLO ile jersey tespitleri (class_id=2)
        all_detections = self.yolo.detect(frame, confidence_threshold=0.3)
        jersey_detections = [d for d in all_detections if d['class_id'] == 2]
        
        if not jersey_detections:
            return
        
        # Her jersey tespitini en yakın oyuncu maskesiyle eşleştir
        for jersey_det in jersey_detections:
            jersey_bbox = jersey_det['bbox']
            jersey_mask = self._bbox_to_mask(jersey_bbox, frame.shape[:2])
            
            best_match_id = None
            best_iou = 0.0
            
            for obj_id, player_mask in masks.items():
                # Jersey maskesi ile oyuncu maskesi arasındaki IoU
                iou = self.calculate_iou(jersey_mask, player_mask)
                
                # Jersey oyuncu içinde olmalı, yüksek overlap gerekli değil
                # Jersey genelde oyuncunun %10-30'u kadar alan kaplar
                if iou > best_iou and iou > 0.05:
                    best_iou = iou
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Jersey crop'u al ve OCR yap
                x1, y1, x2, y2 = [int(c) for c in jersey_bbox]
                jersey_crop = frame[y1:y2, x1:x2]
                
                if jersey_crop.size > 0:
                    number = self.jersey_detector._text_to_number(
                        self._ocr_jersey(jersey_crop)
                    )
                    
                    if number:
                        self.jersey_bank.register(best_match_id, number)
    
    def _ocr_jersey(self, jersey_crop: np.ndarray) -> str:
        """Jersey crop'u üzerinde PARSeq OCR çalıştır."""
        try:
            text, confidence = self.jersey_detector._ocr(jersey_crop)
            return text if confidence > 0.3 else ""
        except Exception:
            return ""
    
    def _smart_redetection(self, frame: np.ndarray, current_masks: Dict[int, np.ndarray]):
        """
        Akıllı yeniden tespit:
        1. YOLO ile yeni oyuncuları tespit et
        2. Jersey bank ile dönen oyuncuları tanı (Re-ID)
        3. Sadece gerçekten yeni oyuncuları ekle
        """
        all_detections = self.yolo.detect(frame, confidence_threshold=0.3)
        player_detections = [d for d in all_detections if d['class_id'] in self.PLAYER_CLASSES]
        jersey_detections = [d for d in all_detections if d['class_id'] == 2]  # number class
        
        if not player_detections:
            return
        
        # Mevcut takip edilen maskeler
        tracked_masks = {obj_id: obj["last_mask"] 
                        for obj_id, obj in self.tracked_objects.items() 
                        if obj["last_mask"] is not None and obj["last_mask"].sum() > 100}
        
        new_players = []
        
        for det in player_detections:
            det_bbox = det['bbox']
            det_mask = self._bbox_to_mask(det_bbox, frame.shape[:2])
            
            # 1. Mevcut maskelerle karşılaştır
            is_tracked = False
            for obj_id, existing_mask in tracked_masks.items():
                iou = self.calculate_iou(det_mask, existing_mask)
                if iou > 0.2:  # Zaten takip ediliyor
                    is_tracked = True
                    break
            
            # 2. Current frame maskelerle de kontrol et
            if not is_tracked:
                for obj_id, mask in current_masks.items():
                    iou = self.calculate_iou(det_mask, mask)
                    if iou > 0.2:
                        is_tracked = True
                        break
            
            if not is_tracked:
                new_players.append(det)
        
        # Yeni oyuncular için jersey tespiti ve Re-ID
        for det in new_players:
            det_bbox = det['bbox']
            det_mask = self._bbox_to_mask(det_bbox, frame.shape[:2])
            
            # Jersey numarasını bul
            jersey_number = None
            for jersey_det in jersey_detections:
                j_bbox = jersey_det['bbox']
                j_mask = self._bbox_to_mask(j_bbox, frame.shape[:2])
                iou = self.calculate_iou(j_mask, det_mask)
                
                if iou > 0.05:  # Jersey bu oyuncuya ait
                    x1, y1, x2, y2 = [int(c) for c in j_bbox]
                    jersey_crop = frame[y1:y2, x1:x2]
                    if jersey_crop.size > 0:
                        jersey_number = self.jersey_detector._text_to_number(
                            self._ocr_jersey(jersey_crop)
                        )
                        break
            
            # Re-ID: Bu jersey daha önce görüldü mü?
            existing_id = None
            if jersey_number:
                existing_id = self.jersey_bank.find_by_jersey(jersey_number)
            
            if existing_id and existing_id not in self.tracked_objects:
                # Dönen oyuncu - eski ID'si ile kaydet
                print(f"   🔄 Re-ID: Jersey #{jersey_number} → ID {existing_id}")
                self.tracked_objects[existing_id] = {
                    "last_mask": det_mask,
                    "confidence": 1.0,
                    "lost_count": 0
                }
            else:
                # Gerçekten yeni oyuncu
                new_id = self.next_obj_id
                self.next_obj_id += 1
                
                print(f"   ➕ New player: ID {new_id}" + (f" (jersey #{jersey_number})" if jersey_number else ""))
                
                self.tracked_objects[new_id] = {
                    "last_mask": det_mask,
                    "confidence": 1.0,
                    "lost_count": 0
                }
                
                if jersey_number:
                    self.jersey_bank.register(new_id, jersey_number)
    
    def _periodic_redetection(self, frame: np.ndarray):
        """Periyodik olarak YOLO ile yeni oyuncuları tespit et."""
        all_detections = self.yolo.detect(frame, confidence_threshold=0.5)
        detections = [d for d in all_detections if d['class_id'] in self.PLAYER_CLASSES]
        
        # Mevcut maskelerle karşılaştır
        current_masks = {obj_id: obj["last_mask"] 
                        for obj_id, obj in self.tracked_objects.items() 
                        if obj["last_mask"] is not None}
        
        for det in detections:
            det_mask = self._bbox_to_mask(det['bbox'], frame.shape[:2])
            
            # Mevcut nesnelerle IoU kontrol
            is_new = True
            for obj_id, existing_mask in current_masks.items():
                iou = self.calculate_iou(det_mask, existing_mask)
                if iou > 0.3:  # Zaten takip ediliyor
                    is_new = False
                    break
            
            if is_new:
                print(f"   Found new object, assigning ID: {self.next_obj_id}")
                self.tracked_objects[self.next_obj_id] = {
                    "last_mask": det_mask,
                    "confidence": 1.0,
                    "lost_count": 0
                }
                self.next_obj_id += 1


if __name__ == "__main__":
    video_in = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\court.mp4"
    video_out = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\sam2_robust_2_output.mp4"
    
    tracker = RobustSAM2Tracker(
        confidence_threshold=0.5,
        iou_threshold=0.3,
        redetect_interval=30
    )
    tracker.process_video(video_in, video_out, max_frames=200)

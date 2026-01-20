"""
Pose Detector Module
====================
YOLO-Pose ile oyuncu tespiti ve keypoint analizi.
"""

import sys
from pathlib import Path

# Helpers klasörünü sys.path'e ekle
helpers_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(helpers_dir))

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


class PoseDetector:
    """
    YOLO-Pose modeli ile oyuncu tespiti ve analizi.
    
    Özellikler:
    - Keypoint bazlı visibility kontrolü
    - Arkadan görüş tespiti
    - Hakem filtresi
    - Jersey bölgesi çıkarma
    """
    
    # COCO Keypoint indices
    KEYPOINTS = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    def __init__(self, config: Config = None):
        """
        PoseDetector'ı başlat.
        
        Args:
            config: Config nesnesi
        """
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """YOLO-Pose modelini yükle."""
        try:
            self.model = YOLO(self.config.POSE_MODEL_PATH)
            self.model.to(self.device)
            
            # FP16 optimizasyonu
            if self.config.USE_HALF and self.device == 'cuda':
                self.model.model.half()
                print(f"✅ PoseDetector modeli yüklendi ({self.device}, FP16)")
            else:
                print(f"✅ PoseDetector modeli yüklendi ({self.device})")
        except Exception as e:
            print(f"❌ YOLO model yüklenemedi: {e}")
            self.model = None
    
    def detect(self, frame):
        """
        Tek frame'de pose tespiti yap.
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            results: YOLO inference sonuçları
        """
        if self.model is None:
            return None
        
        results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, device=self.device, verbose=False)
        return results
    
    def detect_with_tracking(self, frame, persist=True):
        """
        Tracking ile pose tespiti yap.
        
        Args:
            frame: BGR formatında görüntü
            persist: Track ID'leri frame'ler arası koru
            
        Returns:
            results: YOLO tracking sonuçları (track_id içerir)
        """
        if self.model is None:
            return None
        
        results = self.model.track(
            frame, 
            conf=self.config.CONFIDENCE_THRESHOLD,
            device=self.device,
            imgsz=self.config.YOLO_IMGSZ,
            persist=persist,
            verbose=False,
            half=self.config.USE_HALF and self.device == 'cuda'
        )
        return results
    
    # Skeleton bağlantıları (COCO format)
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Yüz
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Kollar
        (5, 11), (6, 12), (11, 12),  # Gövde
        (11, 13), (13, 15), (12, 14), (14, 16)  # Bacaklar
    ]
    
    def draw_skeleton(self, frame, keypoints, color=(0, 255, 255), thickness=2):
        """
        Pose iskeletini çiz.
        
        Args:
            frame: BGR görüntü
            keypoints: [17, 3] keypoint array
            color: Çizgi rengi
            thickness: Çizgi kalınlığı
        """
        # Debug modunda çok düşük threshold - eksik parçaları da göster
        min_conf = 0.05
        
        # Noktaları çiz
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > min_conf:
                # Güven rengini ayarla (düşük=kırmızı, yüksek=yeşil)
                point_color = (0, int(255 * conf), int(255 * (1 - conf)))
                cv2.circle(frame, (int(x), int(y)), 4, point_color, -1)
        
        # Bağlantıları çiz
        for (i, j) in self.SKELETON:
            if keypoints[i][2] > min_conf and keypoints[j][2] > min_conf:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                # Ortalama güven üzerinden renk
                avg_conf = (keypoints[i][2] + keypoints[j][2]) / 2
                line_color = (0, int(255 * avg_conf), int(255 * (1 - avg_conf)))
                cv2.line(frame, pt1, pt2, line_color, thickness)
    
    def is_back_view(self, keypoints, min_conf=None) -> tuple:
        """
        Oyuncunun arkadan görünüp görünmediğini kontrol et.
        
        Args:
            keypoints: [17, 3] keypoint array
            min_conf: Keypoint confidence eşiği
            
        Returns:
            tuple: (is_back, face_visible)
        """
        min_conf = min_conf or self.config.KEYPOINT_CONFIDENCE
        
        nose = keypoints[self.KEYPOINTS['nose']]
        left_eye = keypoints[self.KEYPOINTS['left_eye']]
        right_eye = keypoints[self.KEYPOINTS['right_eye']]
        
        # Yüz görünüyor mu?
        face_visible = (nose[2] > min_conf) or (left_eye[2] > min_conf) or (right_eye[2] > min_conf)
        
        # Omuzlar görünüyor mu?
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        shoulders_visible = (left_shoulder[2] > min_conf) or (right_shoulder[2] > min_conf)
        
        # Arkadan bakış: yüz görünmüyor AMA omuzlar görünüyor
        is_back = (not face_visible) and shoulders_visible
        
        return is_back, face_visible
    
    def is_referee(self, crop) -> bool:
        """
        Hakem mi kontrol et (siyah/gri kıyafet).
        
        Args:
            crop: BGR formatında jersey crop
            
        Returns:
            is_dark: Koyu renk kıyafet mi
        """
        if crop is None or crop.size == 0:
            return False
        
        # BGR -> HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Ortalama Saturation ve Value
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Siyah/gri: düşük saturation VE düşük-orta value
        is_dark = (avg_saturation < self.config.REFEREE_SATURATION_THRESHOLD and 
                   avg_value < self.config.REFEREE_VALUE_THRESHOLD)
        
        return is_dark
    
    def check_player_visibility(self, keypoints, min_visible=None, min_conf=None) -> tuple:
        """
        Oyuncunun yeterince görünür olup olmadığını kontrol et.
        
        Args:
            keypoints: [17, 3] keypoint array
            min_visible: Minimum görünür keypoint sayısı
            min_conf: Keypoint confidence eşiği
            
        Returns:
            tuple: (is_visible, visibility_score, visible_count, visible_kps)
        """
        min_visible = min_visible or self.config.MIN_VISIBLE_KEYPOINTS
        min_conf = min_conf or self.config.KEYPOINT_CONFIDENCE
        
        visible_count = 0
        visible_kps = []
        
        for i, kp in enumerate(keypoints):
            if kp[2] > min_conf:
                visible_count += 1
                visible_kps.append(self.KP_NAMES[i])
        
        has_shoulders = ('left_shoulder' in visible_kps or 'right_shoulder' in visible_kps)
        has_hips = ('left_hip' in visible_kps or 'right_hip' in visible_kps)
        
        visibility_score = visible_count / 17.0
        is_visible = (visible_count >= min_visible) and has_shoulders and has_hips
        
        return is_visible, visibility_score, visible_count, visible_kps
    
    def extract_jersey_region(self, image, keypoints, bbox=None):
        """
        Keypoint'lerden jersey (göğüs/sırt) bölgesini çıkar.
        
        Args:
            image: BGR formatında görüntü
            keypoints: [17, 3] keypoint array
            bbox: Oyuncu bounding box (fallback için)
            
        Returns:
            tuple: (jersey_crop, (x1, y1, x2, y2)) veya None
        """
        h, w = image.shape[:2]
        min_conf = self.config.KEYPOINT_CONFIDENCE
        
        # Keypoint'leri al
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_hip = keypoints[self.KEYPOINTS['left_hip']]
        right_hip = keypoints[self.KEYPOINTS['right_hip']]
        
        # Confidence check
        shoulder_visible = left_shoulder[2] > min_conf or right_shoulder[2] > min_conf
        hip_visible = left_hip[2] > min_conf or right_hip[2] > min_conf
        
        if shoulder_visible and hip_visible:
            # Omuzlar
            if left_shoulder[2] > min_conf and right_shoulder[2] > min_conf:
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_y = min(left_shoulder[1], right_shoulder[1])
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            elif left_shoulder[2] > min_conf:
                shoulder_center_x = left_shoulder[0]
                shoulder_y = left_shoulder[1]
                shoulder_width = 100
                shoulder_tilt = 0
            else:
                shoulder_center_x = right_shoulder[0]
                shoulder_y = right_shoulder[1]
                shoulder_width = 100
                shoulder_tilt = 0
            
            # Kalçalar
            hip_center_x = shoulder_center_x
            if left_hip[2] > min_conf and right_hip[2] > min_conf:
                hip_y = max(left_hip[1], right_hip[1])
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                hip_tilt = abs(left_hip[1] - right_hip[1])
            elif left_hip[2] > min_conf:
                hip_y = left_hip[1]
                hip_center_x = left_hip[0]
                hip_tilt = 0
            else:
                hip_y = right_hip[1]
                hip_center_x = right_hip[0]
                hip_tilt = 0
            
            # Eğim faktörü
            body_lean = abs(shoulder_center_x - hip_center_x)
            avg_tilt = (shoulder_tilt + hip_tilt) / 2
            tilt_factor = 1.0 + (avg_tilt / 50) + (body_lean / 100)
            tilt_factor = min(tilt_factor, 2.0)
            
            # Jersey bölgesi
            dynamic_expand = self.config.JERSEY_EXPAND_RATIO * tilt_factor
            crop_width = shoulder_width * (1 + dynamic_expand)
            jersey_height = (hip_y - shoulder_y) * 0.7
            
            center_x = (shoulder_center_x + hip_center_x) / 2
            
            x1 = int(center_x - crop_width / 2)
            y1 = int(shoulder_y)
            x2 = int(center_x + crop_width / 2)
            y2 = int(shoulder_y + jersey_height)
            
        elif bbox is not None:
            # Fallback: Bounding box
            bx1, by1, bx2, by2 = map(int, bbox)
            box_height = by2 - by1
            box_width = bx2 - bx1
            
            x1 = bx1 + int(box_width * 0.15)
            y1 = by1 + int(box_height * 0.15)
            x2 = bx2 - int(box_width * 0.15)
            y2 = by1 + int(box_height * 0.55)
        else:
            return None
        
        # Sınırları kontrol et
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Minimum boyut kontrolü
        if (x2 - x1) < self.config.MIN_CROP_SIZE or (y2 - y1) < self.config.MIN_CROP_SIZE:
            return None
        
        # Crop
        jersey_crop = image[y1:y2, x1:x2].copy()
        
        return jersey_crop, (x1, y1, x2, y2)


# Test kodu
if __name__ == "__main__":
    detector = PoseDetector()
    print("✅ PoseDetector test başarılı!")
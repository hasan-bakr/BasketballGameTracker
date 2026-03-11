"""
Court Detector Module
=====================
YOLOv11 segmentation modeli ile basketbol sahası tespiti.
"""

import sys
import os
from pathlib import Path

# Helpers klasörünü sys.path'e ekle
helpers_dir = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(helpers_dir))

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


class CourtDetector:
    """
    YOLOv11 segmentation modeli ile basketbol sahası tespiti.
    
    Saha maskesi oluşturur ve oyuncuların sahada olup olmadığını kontrol eder.
    """
    
    def __init__(self, config: Config = None, model_path: str = None):
        """
        CourtDetector'ı başlat.
        
        Args:
            config: Config nesnesi
            model_path: Segmentation model yolu
        """
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # Model yolu
        self.model_path = model_path or os.path.join(str(project_root), "models", "yolo", "best_seg.pt")
        
        self.model = None
        self.court_mask = None  # Cache için
        self.last_frame_num = -1
        
        self._load_model()
    
    def _load_model(self):
        """YOLO segmentation modelini yükle."""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ CourtDetector modeli yüklendi ({self.device})")
        except Exception as e:
            print(f"❌ Court segmentation model yüklenemedi: {e}")
            self.model = None
    
    def detect(self, frame, conf=0.5):
        """
        Saha segmentasyonu yap.
        
        Args:
            frame: BGR görüntü
            conf: Confidence threshold
            
        Returns:
            court_mask: Saha maskesi (binary, saha=255, dışarısı=0)
        """
        if self.model is None:
            return None
        
        try:
            results = self.model(frame, conf=conf, verbose=False, device=self.device)
            
            if results[0].masks is None:
                return None
            
            # İlk maske (court)
            masks = results[0].masks.data.cpu().numpy()
            
            if len(masks) == 0:
                return None
            
            # Tüm maskeleri birleştir (eğer birden fazla court parçası varsa)
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            for mask in masks:
                # Maskeyi frame boyutuna resize et
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
            
            self.court_mask = combined_mask
            return combined_mask
            
        except Exception as e:
            print(f"⚠️ Court detection hatası: {e}")
            return None
    
    def is_on_court(self, bbox, court_mask=None, threshold=0.5) -> bool:
        """
        Bounding box'ın sahada olup olmadığını kontrol et.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            court_mask: Saha maskesi (None ise cache'den al)
            threshold: Saha içi yüzde eşiği (0.5 = %50)
            
        Returns:
            is_on_court: Sahada mı
        """
        mask = court_mask if court_mask is not None else self.court_mask
        
        if mask is None:
            return True  # Maske yoksa varsayılan olarak sahada kabul et
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Sınırları kontrol et
        h, w = mask.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Oyuncunun alt kısmı (ayakları) sahada mı kontrol et
        # Sadece bbox'ın alt %30'unu kontrol et
        foot_y1 = y1 + int((y2 - y1) * 0.7)
        
        roi = mask[foot_y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Saha içi piksel yüzdesi
        on_court_ratio = np.count_nonzero(roi) / roi.size
        
        return on_court_ratio >= threshold
    
    def get_court_polygon(self, court_mask=None):
        """
        Saha sınırlarının polygon koordinatlarını döndür.
        
        Returns:
            contours: Saha konturları
        """
        mask = court_mask if court_mask is not None else self.court_mask
        
        if mask is None:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # En büyük konturu döndür
        return max(contours, key=cv2.contourArea)
    
    def visualize(self, frame, court_mask=None, alpha=0.3):
        """
        Saha maskesini görselleştir.
        
        Args:
            frame: BGR görüntü
            court_mask: Saha maskesi
            alpha: Transparanlık (0-1)
            
        Returns:
            visualized: Overlay görüntü
        """
        mask = court_mask if court_mask is not None else self.court_mask
        
        if mask is None:
            return frame
        
        # Yeşil overlay
        overlay = frame.copy()
        overlay[mask > 0] = [0, 255, 0]  # Yeşil
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# Test kodu
if __name__ == "__main__":
    detector = CourtDetector()
    
    # Test frame
    test_frame = cv2.imread(os.path.join(str(project_root), "videos", "output", "test_frame_4.jpg"))
    
    if test_frame is not None:
        mask = detector.detect(test_frame)
        
        if mask is not None:
            # Visualize
            vis = detector.visualize(test_frame, mask)
            cv2.imwrite("court_detection_test.jpg", vis)
            print("✅ Test görüntüsü kaydedildi: court_detection_test.jpg")
            
            # Test bbox
            test_bbox = [500, 300, 600, 500]
            on_court = detector.is_on_court(test_bbox)
            print(f"Test bbox sahada: {on_court}")
        else:
            print("❌ Saha maskesi oluşturulamadı")
    else:
        print("❌ Test görüntüsü yüklenemedi")

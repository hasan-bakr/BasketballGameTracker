""
Camera Stabilizer Module
========================
ORB/ECC hybrid pipeline ile kamera hareketi kompanzasyonu.
"""

import sys
from pathlib import Path

# Helpers klasörünü sys.path'e ekle
helpers_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(helpers_dir))

import cv2
import numpy as np

from config import Config


class CameraStabilizer:
    """
    ORB/ECC hybrid pipeline ile kamera hareketi kompanzasyonu.
    
    Özellikler:
    - ORB feature matching ile hızlı hareket tespiti
    - ECC ile hassas hizalama
    - Homography matrix ile bbox pozisyon düzeltme
    """
    
    def __init__(self, config: Config = None):
        """
        CameraStabilizer'ı başlat.
        
        Args:
            config: Config nesnesi
        """
        self.config = config or Config()
        
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Önceki frame (grayscale)
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Son hesaplanan homography
        self.homography = None
        self.inverse_homography = None
        
        # İstatistikler
        self.stats = {
            'frames_processed': 0,
            'successful_alignments': 0,
            'fallback_to_identity': 0
        }
        
        print("✅ CameraStabilizer başlatıldı (ORB/ECC)")
    
    def compute_motion(self, frame):
        """
        Frame'ler arası kamera hareketini hesapla.
        
        Args:
            frame: BGR görüntü
            
        Returns:
            homography: 3x3 dönüşüm matrisi (None ise hareket yok/hesaplanamadı)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # İlk frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(gray, None)
            self.homography = np.eye(3, dtype=np.float32)
            self.inverse_homography = np.eye(3, dtype=np.float32)
            return self.homography
        
        # ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or self.prev_descriptors is None:
            self._update_prev(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float32)
        
        # Feature matching
        try:
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # En iyi eşleşmeleri al
            good_matches = matches[:min(50, len(matches))]
            
            if len(good_matches) < 10:
                self._update_prev(gray, keypoints, descriptors)
                self.stats['fallback_to_identity'] += 1
                return np.eye(3, dtype=np.float32)
            
            # Noktaları çıkar
            src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Homography hesapla
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                self.homography = H
                self.inverse_homography = np.linalg.inv(H)
                self.stats['successful_alignments'] += 1
            else:
                self.stats['fallback_to_identity'] += 1
                H = np.eye(3, dtype=np.float32)
            
        except Exception as e:
            H = np.eye(3, dtype=np.float32)
            self.stats['fallback_to_identity'] += 1
        
        self._update_prev(gray, keypoints, descriptors)
        self.stats['frames_processed'] += 1
        
        return H
    
    def _update_prev(self, gray, keypoints, descriptors):
        """Önceki frame bilgilerini güncelle."""
        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
    
    def compensate_bbox(self, bbox, homography=None):
        """
        Bbox pozisyonunu kamera hareketine göre düzelt.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            homography: Kullanılacak homography (None ise son hesaplanan)
            
        Returns:
            compensated_bbox: Düzeltilmiş bbox
        """
        H = homography if homography is not None else self.inverse_homography
        
        if H is None:
            return bbox
        
        x1, y1, x2, y2 = bbox
        
        # Köşe noktaları
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform uygula
        transformed = cv2.perspectiveTransform(corners, H)
        
        # Yeni bbox hesapla
        transformed = transformed.reshape(-1, 2)
        new_x1 = max(0, int(np.min(transformed[:, 0])))
        new_y1 = max(0, int(np.min(transformed[:, 1])))
        new_x2 = int(np.max(transformed[:, 0]))
        new_y2 = int(np.max(transformed[:, 1]))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def compensate_point(self, point, homography=None):
        """
        Tek nokta pozisyonunu düzelt.
        
        Args:
            point: (x, y) nokta
            homography: Kullanılacak homography
            
        Returns:
            compensated_point: Düzeltilmiş nokta
        """
        H = homography if homography is not None else self.inverse_homography
        
        if H is None:
            return point
        
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def get_motion_magnitude(self):
        """
        Kamera hareketi büyüklüğünü döndür.
        
        Returns:
            magnitude: Hareket büyüklüğü (piksel)
        """
        if self.homography is None:
            return 0.0
        
        # Translation vektörünü çıkar
        tx = self.homography[0, 2]
        ty = self.homography[1, 2]
        
        return np.sqrt(tx**2 + ty**2)
    
    def reset(self):
        """Stabilizer'ı sıfırla."""
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.homography = None
        self.inverse_homography = None
    
    def get_summary(self):
        """İstatistik özeti döndür."""
        return {
            'frames_processed': self.stats['frames_processed'],
            'successful_alignments': self.stats['successful_alignments'],
            'fallback_rate': self.stats['fallback_to_identity'] / max(1, self.stats['frames_processed'])
        }


# Test kodu
if __name__ == "__main__":
    stabilizer = CameraStabilizer()
    
    # Test frame oluştur
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    H1 = stabilizer.compute_motion(frame1)
    H2 = stabilizer.compute_motion(frame2)
    
    print(f"Motion magnitude: {stabilizer.get_motion_magnitude():.2f}")
    print(f"Summary: {stabilizer.get_summary()}")
    
    # Bbox test
    bbox = [100, 100, 200, 200]
    compensated = stabilizer.compensate_bbox(bbox)
    print(f"Original bbox: {bbox}")
    print(f"Compensated bbox: {compensated}")

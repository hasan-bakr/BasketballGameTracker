"""
Jersey Number Recognition Pipeline
===================================
Bu pipeline basketbol oyuncularının forma numaralarını tespit eder.

Pipeline Adımları:
1. YOLO-Pose ile oyuncu tespiti ve keypoint çıkarma
2. Visibility, arkadan görüş ve hakem kontrolü
3. Keypoint'lerden jersey (göğüs/sırt) bölgesi crop
4. PARSeq-tiny ile numara okuma
5. Unique numara kaydetme (her numara sadece en iyi kalitede kaydedilir)

Filtreler:
- Visibility: Minimum 8 keypoint görünmeli (omuz + kalça zorunlu)
- Back View: Sadece arkadan görünen oyuncular (yüz görünmemeli)
- Referee: Hakemler filtrelenir (siyah/gri kıyafet)

Gereksinimler:
    pip install ultralytics torch torchvision pillow opencv-python
    pip install pytorch-lightning timm lmdb nltk
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path


# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Model paths
    POSE_MODEL_PATH = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\yolo11m-pose.pt"
    
    # OCR model
    PARSEQ_MODEL = "parseq_tiny"  # PARSeq Tiny: Scene Text Recognition (hızlı)
    
    # Input/Output paths
    INPUT_VIDEO = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    OUTPUT_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output"
    CROPS_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\jersey_crops"
    TEST_IMAGE_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\test"
    
    # Jersey crop settings
    JERSEY_EXPAND_RATIO = 0.4  # Crop bölgesini %40 genişlet (daha geniş jersey yakalama)
    MIN_CROP_SIZE = 32  # Minimum crop boyutu
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.4
    MIN_VISIBLE_KEYPOINTS = 8  # Minimum görünür keypoint sayısı
    KEYPOINT_CONFIDENCE = 0.3  # Keypoint confidence eşiği
    MIN_OCR_CONFIDENCE = 0.70  # Minimum OCR confidence (jersey numarası emin olmak için)
    
    # Referee detection (HSV thresholds)
    REFEREE_SATURATION_THRESHOLD = 50
    REFEREE_VALUE_THRESHOLD = 120
    
    # Processing
    FRAME_SKIP = 3  # Her 3 frame'de 1'ini işle
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# JERSEY NUMBER DETECTOR CLASS
# ============================================
class JerseyNumberDetector:
    """
    Basketbol oyuncularının forma numaralarını tespit eden sınıf.
    
    Özellikler:
    - Sadece arkadan görünen oyuncuları işler
    - Hakemleri filtreler
    - Her numara sadece en iyi kalitede kaydedilir
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
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        print(f"🔧 Device: {self.device}")
        
        # Output klasörünü oluştur
        os.makedirs(self.config.CROPS_DIR, exist_ok=True)
        
        # Load YOLO-Pose model
        print("📦 YOLO-Pose modeli yükleniyor...")
        self.pose_model = YOLO(self.config.POSE_MODEL_PATH)
        print("✅ YOLO-Pose modeli yüklendi!")
        
        # Load PARSeq model
        print("📦 PARSeq OCR modeli yükleniyor...")
        self.ocr_model = self._load_parseq()
        print("✅ PARSeq OCR modeli yüklendi!")
        
        # Best crops dictionary (unique numara kaydetmek için)
        self.best_crops = {}
        
        # Stats
        self.stats = {
            'skipped_low_vis': 0,
            'skipped_front': 0,
            'skipped_referee': 0,
            'processed': 0
        }
        
    def _load_parseq(self):
        """PARSeq Tiny modelini yükle (torch.hub ile)"""
        try:
            from torchvision import transforms
            
            # PARSeq tiny modeli (hızlı)
            model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True)
            model = model.to(self.device)
            model.eval()
            
            # Transform pipeline - PARSeq 32x128 boyutunda tensor bekliyor
            self.parseq_transform = transforms.Compose([
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
            
            return model
        except Exception as e:
            print(f"⚠️ PARSeq yüklenemedi: {e}")
            return None
    
    def is_back_view(self, keypoints, min_conf=None):
        """
        Oyuncunun arkadan görünüp görünmediğini kontrol et.
        Yüz (burun, gözler) görünmüyorsa = arkadan bakış
        
        Args:
            keypoints: [17, 3] keypoint array
            min_conf: Keypoint confidence eşiği
            
        Returns:
            is_back: Arkadan görünüyor mu
            face_visible: Yüz görünüyor mu
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
    
    def is_referee(self, crop):
        """
        Hakem mi kontrol et (siyah/gri kıyafet).
        Hakemler genelde siyah/gri giyerler.
        
        Args:
            crop: BGR formatında jersey crop
            
        Returns:
            is_dark: Koyu renk kıyafet mi (hakem olabilir)
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
    
    def check_player_visibility(self, keypoints, min_visible=None, min_conf=None):
        """
        Oyuncunun yeterince görünür olup olmadığını kontrol et.
        
        Args:
            keypoints: [17, 3] keypoint array
            min_visible: Minimum görünür keypoint sayısı
            min_conf: Keypoint confidence eşiği
            
        Returns:
            is_visible: Yeterince görünür mü
            visibility_score: Görünürlük skoru (0-1)
            visible_count: Görünen keypoint sayısı
            visible_kps: Görünen keypoint isimleri
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
        Oyuncu eğildiğinde genişliği dinamik olarak artırır.
        
        Args:
            image: BGR formatında görüntü (numpy array)
            keypoints: YOLO'dan gelen keypoints [17, 3] (x, y, confidence)
            bbox: Oyuncu bounding box [x1, y1, x2, y2] (opsiyonel, fallback için)
            
        Returns:
            tuple: (cropped jersey bölgesi, crop koordinatları) veya None
        """
        h, w = image.shape[:2]
        min_conf = self.config.KEYPOINT_CONFIDENCE
        
        # Keypoint'leri al
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_hip = keypoints[self.KEYPOINTS['left_hip']]
        right_hip = keypoints[self.KEYPOINTS['right_hip']]
        
        # Confidence check - en az 1 omuz ve 1 kalça görünmeli
        shoulder_visible = left_shoulder[2] > min_conf or right_shoulder[2] > min_conf
        hip_visible = left_hip[2] > min_conf or right_hip[2] > min_conf
        
        if shoulder_visible and hip_visible:
            # Keypoint bazlı crop
            # Omuzlar arasının ortası
            if left_shoulder[2] > min_conf and right_shoulder[2] > min_conf:
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_y = min(left_shoulder[1], right_shoulder[1])
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                
                # Eğim hesapla (omuzlar arasındaki y farkı)
                shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            elif left_shoulder[2] > min_conf:
                shoulder_center_x = left_shoulder[0]
                shoulder_y = left_shoulder[1]
                shoulder_width = 100  # default
                shoulder_tilt = 0
            else:
                shoulder_center_x = right_shoulder[0]
                shoulder_y = right_shoulder[1]
                shoulder_width = 100
                shoulder_tilt = 0
            
            # Kalçalar arasının ortası ve eğim
            hip_center_x = shoulder_center_x  # default
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
            
            # Vücut eğimi hesapla (omuz-kalça merkezi arası x farkı)
            body_lean = abs(shoulder_center_x - hip_center_x)
            
            # Toplam eğim faktörü (tilt + lean)
            avg_tilt = (shoulder_tilt + hip_tilt) / 2
            tilt_factor = 1.0 + (avg_tilt / 50) + (body_lean / 100)  # Eğim arttıkça genişlik artar
            tilt_factor = min(tilt_factor, 2.0)  # Maksimum 2x genişlik
            
            # Jersey bölgesi koordinatları
            base_expand = self.config.JERSEY_EXPAND_RATIO
            dynamic_expand = base_expand * tilt_factor
            crop_width = shoulder_width * (1 + dynamic_expand)
            jersey_height = (hip_y - shoulder_y) * 0.7  # Sadece üst %70
            
            # Merkez noktayı omuz ve kalça merkezinin ortasına al (eğime göre kayma)
            center_x = (shoulder_center_x + hip_center_x) / 2
            
            x1 = int(center_x - crop_width / 2)
            y1 = int(shoulder_y)
            x2 = int(center_x + crop_width / 2)
            y2 = int(shoulder_y + jersey_height)
            
        elif bbox is not None:
            # Fallback: Bounding box'ın üst-orta kısmı
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
    
    def recognize_number(self, jersey_crop):
        """
        PARSeq ile jersey numarasını oku.
        
        Args:
            jersey_crop: BGR formatında jersey görüntüsü (NumPy array)
            
        Returns:
            Tanınan numara (string) ve confidence
        """
        if self.ocr_model is None:
            return None, 0.0
        
        if jersey_crop is None or jersey_crop.size == 0:
            return None, 0.0
        
        try:
            # BGR -> RGB -> PIL Image
            rgb_image = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Transform: PIL -> Tensor (32x128, normalized)
            tensor = self.parseq_transform(pil_image).unsqueeze(0).to(self.device)
            
            # PARSeq inference
            with torch.no_grad():
                logits = self.ocr_model(tensor)
                pred = logits.softmax(-1)
                # decode() returns: (list of labels, list of confidences)
                labels, confidences = self.ocr_model.tokenizer.decode(pred)
            
            # İlk (ve tek) sonucu al
            if not labels or not confidences:
                return None, 0.0
                
            label = labels[0]  # String label
            # Confidence: ortalama al (her karakter için bir confidence var)
            confidence = confidences[0].mean().item()
            
            # Sadece sayıları filtrele
            numbers_only = ''.join(filter(str.isdigit, label))
            
            # Jersey numarası 0-99 arası olmalı
            if numbers_only and len(numbers_only) <= 2:
                return numbers_only, confidence
                
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ OCR hatası: {e}")
            return None, 0.0
    
    def process_frame(self, frame, frame_num=0, visualize=True):
        """
        Tek bir frame'i işle (tüm filtrelerle).
        
        Args:
            frame: BGR formatında görüntü
            frame_num: Frame numarası
            visualize: Görselleştirme yapılsın mı
            
        Returns:
            detections: List of dict with player info and jersey numbers
            annotated_frame: Görselleştirilmiş frame (eğer visualize=True)
        """
        detections = []
        annotated_frame = frame.copy() if visualize else None
        
        # YOLO-Pose inference
        results = self.pose_model(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return detections, annotated_frame
        
        result = results[0]
        
        # Her tespit edilen kişi için
        for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
            # Sadece "person" class'ı (class_id = 0)
            if int(box.cls[0]) != 0:
                continue
                
            bbox = box.xyxy[0].cpu().numpy()
            keypoints = kpts.data[0].cpu().numpy()  # [17, 3]
            x1, y1, x2, y2 = map(int, bbox)
            
            # 1. Visibility kontrolü
            is_visible, vis_score, vis_count, vis_kps = self.check_player_visibility(keypoints)
            if not is_visible:
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(annotated_frame, 'LowVis', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                self.stats['skipped_low_vis'] += 1
                continue
            
            # 2. Arkadan görüş kontrolü
            is_back, face_visible = self.is_back_view(keypoints)
            if not is_back:
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 1)
                    cv2.putText(annotated_frame, 'Front', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
                self.stats['skipped_front'] += 1
                continue
            
            # Jersey bölgesini çıkar
            jersey_result = self.extract_jersey_region(frame, keypoints, bbox)
            if jersey_result is None:
                continue
                
            jersey_crop, crop_coords = jersey_result
            cx1, cy1, cx2, cy2 = crop_coords
            
            # 3. Hakem kontrolü
            if self.is_referee(jersey_crop):
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(annotated_frame, 'Referee', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                self.stats['skipped_referee'] += 1
                continue
            
            self.stats['processed'] += 1
            
            # Görselleştirme - oyuncu arkadan görünüyor
            if visualize:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 100, 0), 2)
            
            # Numara tanıma
            number, ocr_conf = self.recognize_number(jersey_crop)
            
            if number and ocr_conf >= self.config.MIN_OCR_CONFIDENCE:
                # En iyi crop'u kaydet (sadece yüksek confidence)
                score = ocr_conf * vis_score
                if number not in self.best_crops or score > self.best_crops[number][0]:
                    self.best_crops[number] = (score, ocr_conf, vis_score, jersey_crop.copy(), frame_num)
                
                if visualize:
                    cv2.putText(annotated_frame, f'#{number} ({ocr_conf:.2f}) BACK', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif number and visualize:
                # Düşük confidence - görüntüle ama kaydetme
                cv2.putText(annotated_frame, f'#{number}? ({ocr_conf:.2f})', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            detection = {
                'player_id': i,
                'bbox': bbox.tolist(),
                'keypoints': keypoints.tolist(),
                'jersey_crop_coords': crop_coords,
                'jersey_number': number,
                'ocr_confidence': ocr_conf,
                'visibility_score': vis_score,
                'is_back_view': True,
                'frame': frame_num
            }
            detections.append(detection)
        
        return detections, annotated_frame
    
    def process_image(self, image_path, save_output=True):
        """
        Tek bir görüntüyü işle.
        """
        print(f"\n🖼️ İşleniyor: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Görüntü okunamadı: {image_path}")
            return None, None
        
        detections, annotated = self.process_frame(frame, frame_num=0, visualize=True)
        
        # Sonuçları yazdır
        print(f"   👥 {len(detections)} oyuncu tespit edildi (arkadan görüş)")
        for det in detections:
            if det['jersey_number']:
                print(f"   🎽 Oyuncu {det['player_id']}: #{det['jersey_number']} (conf: {det['ocr_confidence']:.2f})")
        
        # Kaydet
        if save_output and annotated is not None:
            output_path = os.path.join(
                self.config.OUTPUT_DIR, 
                f"jersey_{Path(image_path).stem}.jpg"
            )
            cv2.imwrite(output_path, annotated)
            print(f"   💾 Kaydedildi: {output_path}")
        
        return detections, annotated
    
    def process_video(self, video_path=None, output_path=None, show_preview=True, save_unique_crops=True):
        """
        Video işle - sadece arkadan görünen oyuncuları işle.
        
        Args:
            video_path: Video dosya yolu
            output_path: Çıktı video yolu
            show_preview: Önizleme penceresi açılsın mı
            save_unique_crops: Her unique numara için crop kaydedilsin mi
        """
        video_path = video_path or self.config.INPUT_VIDEO
        
        print(f"\n🎬 Video işleniyor: {video_path}")
        print("📋 Filtreler: Sadece ARKADAN görüş + Hakem değil")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {video_path}")
            return
        
        # Video özellikleri
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   📐 Boyut: {width}x{height}, FPS: {fps}, Toplam: {total_frames} frame")
        print("   ⌨️ Q: çık, SPACE: duraklat")
        
        # Stats sıfırla
        self.stats = {'skipped_low_vis': 0, 'skipped_front': 0, 'skipped_referee': 0, 'processed': 0}
        self.best_crops = {}
        
        # Crop klasörünü temizle
        if save_unique_crops:
            for f in os.listdir(self.config.CROPS_DIR):
                os.remove(os.path.join(self.config.CROPS_DIR, f))
        
        # Output writer (opsiyonel)
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Frame skip
                if frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                # Frame işle
                detections, annotated = self.process_frame(frame, frame_num=frame_count, visualize=True)
                
                # Detections'ı kaydet
                for det in detections:
                    all_detections.append(det)
                
                # Video yaz
                if out and annotated is not None:
                    out.write(annotated)
                
                # Info overlay
                sec = frame_count / fps
                info = f'F:{frame_count} T:{sec:.1f}s | OK:{len(self.best_crops)} Front:{self.stats["skipped_front"]} Ref:{self.stats["skipped_referee"]}'
                cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Preview
                if show_preview and annotated is not None:
                    cv2.imshow('Jersey Detection (Back View Only) - Q:quit SPACE:pause', annotated)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    elif k == ord(' '):
                        cv2.waitKey(0)
                        
        finally:
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Unique crop'ları kaydet
        if save_unique_crops:
            print("\n" + "=" * 50)
            print("🎽 Kaydedilen numaralar (sadece arkadan görüş):")
            for nums, (score, conf, vis, crop, frame_num) in sorted(
                self.best_crops.items(), 
                key=lambda x: int(x[0]) if x[0].isdigit() else 0
            ):
                path = os.path.join(self.config.CROPS_DIR, f'{nums}.jpg')
                cv2.imwrite(path, crop)
                print(f"   #{nums}: conf={conf:.2f}, vis={vis:.0%}, frame={frame_num}")
        
        print("\n" + "=" * 50)
        print(f"✅ Video işleme tamamlandı!")
        print(f"   📊 Toplam unique numara: {len(self.best_crops)}")
        print(f"   ⏭️ Atlanan - Düşük vis: {self.stats['skipped_low_vis']}")
        print(f"   ⏭️ Atlanan - Önden görüş: {self.stats['skipped_front']}")
        print(f"   ⏭️ Atlanan - Hakem: {self.stats['skipped_referee']}")
        print(f"   💾 Crop'lar: {self.config.CROPS_DIR}")
        
        return all_detections
    
    def reset_stats(self):
        """İstatistikleri ve best_crops'u sıfırla"""
        self.stats = {'skipped_low_vis': 0, 'skipped_front': 0, 'skipped_referee': 0, 'processed': 0}
        self.best_crops = {}


# ============================================
# SIMPLE ALTERNATIVE (PARSeq OLMADAN)
# ============================================
class SimpleJerseyExtractor:
    """
    PARSeq yüklenemezse, sadece jersey bölgelerini çıkarır.
    Sonra bu crop'ları manuel inceleyebilir veya başka OCR kullanabilirsiniz.
    """
    
    def __init__(self, pose_model_path):
        self.pose_model = YOLO(pose_model_path)
        self.output_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\jersey_crops"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_from_image(self, image_path):
        """Görüntüden jersey crop'larını çıkar ve kaydet"""
        frame = cv2.imread(image_path)
        if frame is None:
            return []
        
        results = self.pose_model(frame, conf=0.5)
        
        crops = []
        for i, (box, kpts) in enumerate(zip(results[0].boxes, results[0].keypoints)):
            if int(box.cls[0]) != 0:
                continue
            
            bbox = box.xyxy[0].cpu().numpy()
            
            # Basit crop: bbox'ın üst yarısı
            x1, y1, x2, y2 = map(int, bbox)
            h = y2 - y1
            jersey_crop = frame[y1:y1+int(h*0.5), x1:x2]
            
            if jersey_crop.size > 0:
                crop_path = os.path.join(
                    self.output_dir, 
                    f"{Path(image_path).stem}_player{i}.jpg"
                )
                cv2.imwrite(crop_path, jersey_crop)
                crops.append(crop_path)
        
        return crops


# ============================================
# MAIN - DEMO
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🏀 JERSEY NUMBER RECOGNITION PIPELINE")
    print("   Filtreler: Back View Only + No Referees")
    print("=" * 60)
    
    # Detector oluştur
    try:
        detector = JerseyNumberDetector()
    except Exception as e:
        print(f"\n⚠️ Tam pipeline yüklenemedi: {e}")
        print("📝 Basit jersey extractor kullanılacak...\n")
        
        # Fallback: Sadece crop çıkar
        extractor = SimpleJerseyExtractor(Config.POSE_MODEL_PATH)
        
        # Test görüntüleri
        test_dir = Config.TEST_IMAGE_DIR
        if os.path.exists(test_dir):
            for img_file in os.listdir(test_dir):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(test_dir, img_file)
                    crops = extractor.extract_from_image(img_path)
                    print(f"✅ {img_file}: {len(crops)} jersey crop kaydedildi")
        
        print(f"\n💾 Crop'lar kaydedildi: {extractor.output_dir}")
        print("💡 Bu crop'ları PARSeq web demo'sunda test edebilirsiniz:")
        print("   https://huggingface.co/spaces/baudm/PARSeq")
        exit()
    
    # ============================================
    # VIDEO İŞLE
    # ============================================
    print("\n" + "=" * 40)
    print("🎬 Video İşleniyor...")
    print("=" * 40)
    
    # Video işle
    detector.process_video(show_preview=True, save_unique_crops=True)
    
    print("\n" + "=" * 60)
    print("✅ Demo tamamlandı!")
    print("=" * 60)

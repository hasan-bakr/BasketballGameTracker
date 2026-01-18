"""
Jersey Number Pipeline
======================
Basketbol oyuncularının forma numaralarını tespit eden ana pipeline.

PoseDetector ve JerseyDetector sınıflarını birleştirir.
"""

import sys
from pathlib import Path

# Proje kök dizinini ve helpers klasörünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent
helpers_dir = Path(__file__).resolve().parent / "helpers"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(helpers_dir))

import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor

from config import Config
from pose_detector import PoseDetector
from jersey_detector import JerseyDetector
from player_tracker import PlayerTracker


class JerseyNumberPipeline:
    """
    Ana pipeline sınıfı.
    
    PoseDetector ve JerseyDetector'ı bir araya getirir.
    Frame/image/video işleme fonksiyonlarını sağlar.
    """
    
    def __init__(self, config: Config = None):
        """
        Pipeline'ı başlat.
        
        Args:
            config: Config nesnesi
        """
        self.config = config or Config()
        
        # Output klasörünü oluştur
        os.makedirs(self.config.CROPS_DIR, exist_ok=True)
        
        # Detectors
        print("📦 Modeller yükleniyor...")
        self.pose_detector = PoseDetector(self.config)
        self.jersey_detector = JerseyDetector(self.config)
        
        # Player Tracker (jersey doğrulama)
        self.player_tracker = PlayerTracker(
            confirmation_count=3,
            min_confidence=self.config.MIN_OCR_CONFIDENCE
        )
        
        # Tracking
        self.best_crops = {}
        self.stats = {
            'skipped_low_vis': 0,
            'skipped_front': 0,
            'skipped_referee': 0,
            'skipped_confirmed': 0,
            'processed': 0
        }
    
    def reset_stats(self):
        """İstatistikleri sıfırla."""
        self.stats = {
            'skipped_low_vis': 0,
            'skipped_front': 0,
            'skipped_referee': 0,
            'skipped_confirmed': 0,
            'processed': 0
        }
        self.best_crops = {}
        self.player_tracker.reset()
    
    def process_frame(self, frame, frame_num=0, visualize=True, use_tracking=True):
        """
        Tek bir frame'i işle.
        
        Args:
            frame: BGR formatında görüntü
            frame_num: Frame numarası
            visualize: Görselleştirme yapılsın mı
            use_tracking: Tracking kullanılsın mı
            
        Returns:
            detections: List of detection dictionaries
            annotated_frame: Görselleştirilmiş frame
        """
        detections = []
        annotated_frame = frame.copy() if visualize else None
        
        # YOLO-Pose inference (tracking veya normal)
        if use_tracking:
            results = self.pose_detector.detect_with_tracking(frame)
        else:
            results = self.pose_detector.detect(frame)
        
        if results is None or len(results) == 0 or results[0].keypoints is None:
            return detections, annotated_frame
        
        result = results[0]
        
        # Track ID'leri al (tracking modunda)
        track_ids = None
        if use_tracking and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Her tespit edilen kişi için
        for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
            # Sadece person class
            if int(box.cls[0]) != 0:
                continue
            
            # Track ID al
            track_id = track_ids[i] if track_ids is not None else i
            
            bbox = box.xyxy[0].cpu().numpy()
            keypoints = kpts.data[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Onaylı oyuncu kontrolü
            if self.player_tracker.is_confirmed(track_id):
                confirmed_number = self.player_tracker.get_jersey(track_id)
                self.stats['skipped_confirmed'] += 1
                
                if visualize:
                    # Onaylı oyuncu - yeşil çerçeve
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'#{confirmed_number} [OK]', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detection = {
                    'track_id': track_id,
                    'bbox': bbox.tolist(),
                    'jersey_number': confirmed_number,
                    'confirmed': True,
                    'frame': frame_num
                }
                detections.append(detection)
                continue
            
            # 1. Visibility kontrolü
            is_visible, vis_score, vis_count, vis_kps = self.pose_detector.check_player_visibility(keypoints)
            if not is_visible:
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(annotated_frame, 'LowVis', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                self.stats['skipped_low_vis'] += 1
                continue
            
            # 2. Arkadan görüş kontrolü
            is_back, face_visible = self.pose_detector.is_back_view(keypoints)
            if not is_back:
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 1)
                    cv2.putText(annotated_frame, 'Front', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
                self.stats['skipped_front'] += 1
                continue
            
            # Jersey bölgesini çıkar
            jersey_result = self.pose_detector.extract_jersey_region(frame, keypoints, bbox)
            if jersey_result is None:
                continue
            
            jersey_crop, crop_coords = jersey_result
            cx1, cy1, cx2, cy2 = crop_coords
            
            # 3. Hakem kontrolü
            if self.pose_detector.is_referee(jersey_crop):
                if visualize:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(annotated_frame, 'Referee', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                self.stats['skipped_referee'] += 1
                continue
            
            self.stats['processed'] += 1
            
            # Görselleştirme
            if visualize:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 100, 0), 2)
            
            # Numara tanıma
            number, ocr_conf = self.jersey_detector.recognize_number(jersey_crop)
            
            if number and ocr_conf >= self.config.MIN_OCR_CONFIDENCE:
                # PlayerTracker'a ekle
                newly_confirmed = self.player_tracker.add_detection(track_id, number, ocr_conf)
                
                # En iyi crop'u kaydet
                score = ocr_conf * vis_score
                if number not in self.best_crops or score > self.best_crops[number][0]:
                    self.best_crops[number] = (score, ocr_conf, vis_score, jersey_crop.copy(), frame_num)
                
                if visualize:
                    status = "✓" if newly_confirmed else "..."
                    cv2.putText(annotated_frame, f'#{number} ({ocr_conf:.2f}) {status}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif number and visualize:
                cv2.putText(annotated_frame, f'#{number}? ({ocr_conf:.2f})', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            detection = {
                'track_id': track_id,
                'bbox': bbox.tolist(),
                'keypoints': keypoints.tolist(),
                'jersey_crop_coords': crop_coords,
                'jersey_number': number,
                'ocr_confidence': ocr_conf,
                'visibility_score': vis_score,
                'is_back_view': True,
                'confirmed': False,
                'frame': frame_num
            }
            detections.append(detection)
        
        return detections, annotated_frame
    
    def process_image(self, image_path, save_output=True):
        """Tek bir görüntüyü işle."""
        print(f"\n🖼️ İşleniyor: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Görüntü okunamadı: {image_path}")
            return None, None
        
        detections, annotated = self.process_frame(frame, frame_num=0, visualize=True)
        
        print(f"   👥 {len(detections)} oyuncu tespit edildi (arkadan görüş)")
        for det in detections:
            if det['jersey_number']:
                print(f"   🎽 Oyuncu {det['player_id']}: #{det['jersey_number']} (conf: {det['ocr_confidence']:.2f})")
        
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
        Video işle.
        
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
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   📐 Boyut: {width}x{height}, FPS: {fps}, Toplam: {total_frames} frame")
        print("   ⌨️ Q: çık, SPACE: duraklat")
        
        # Stats sıfırla
        self.reset_stats()
        
        # Crop klasörünü temizle
        if save_unique_crops:
            for f in os.listdir(self.config.CROPS_DIR):
                os.remove(os.path.join(self.config.CROPS_DIR, f))
        
        # Output writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        all_detections = []
        last_tracking_result = None  # Tracking interval için
        start_time = time.time()
        
        # OCR için ThreadPoolExecutor
        ocr_executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        pending_ocr_futures = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Frame skip
                if frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                processed_count += 1
                
                # Tracking interval - her N frame'de tracking yap
                use_tracking = (processed_count % self.config.TRACKING_INTERVAL == 0)
                
                detections, annotated = self.process_frame(
                    frame, 
                    frame_num=frame_count, 
                    visualize=True,
                    use_tracking=use_tracking
                )
                
                for det in detections:
                    all_detections.append(det)
                
                if out and annotated is not None:
                    out.write(annotated)
                
                # Info overlay
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                confirmed = len(self.player_tracker.confirmed_players)
                info = f'F:{frame_count} | FPS:{fps_actual:.1f} | OK:{confirmed} | Skip:{self.stats["skipped_confirmed"]}'
                cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if show_preview and annotated is not None:
                    cv2.imshow('Jersey Detection (Optimized) - Q:quit SPACE:pause', annotated)
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


# Ana çalıştırma kodu
if __name__ == "__main__":
    print("=" * 60)
    print("🏀 JERSEY NUMBER RECOGNITION PIPELINE")
    print("   Filtreler: Back View Only + No Referees")
    print("=" * 60)
    
    # Pipeline oluştur
    pipeline = JerseyNumberPipeline()
    
    # Video işle
    print("\n" + "=" * 40)
    print("🎬 Video İşleniyor...")
    print("=" * 40)
    
    pipeline.process_video(video_path=r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4", show_preview=True, save_unique_crops=True)
    
    #pipeline.process_image(r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\test_frame_4.jpg")
    print("\n" + "=" * 60)
    print("✅ Demo tamamlandı!")
    print("=" * 60)

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
from court_detector import CourtDetector
from camera_stabilizer import CameraStabilizer
from sam_segmenter import SAMSegmenter


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
        self.court_detector = CourtDetector(self.config)
        
        # Player Tracker (jersey doğrulama + timeout)
        self.player_tracker = PlayerTracker(
            confirmation_count=3,
            min_confidence=self.config.MIN_OCR_CONFIDENCE,
            timeout_frames=30  # 30 frame (1 saniye) sonra unut
        )
        
        # Court mask cache
        self.court_mask = None
        
        # Camera Stabilizer (öpsiyonel)
        self.camera_stabilizer = None
        if self.config.USE_CAMERA_COMPENSATION:
            self.camera_stabilizer = CameraStabilizer(self.config)
        
        # SAM Segmenter (öpsiyonel)
        self.sam_segmenter = None
        if self.config.USE_SAM_SEGMENTATION:
            self.sam_segmenter = SAMSegmenter(self.config, self.config.SAM_MODEL)
        
        # Tracking
        self.best_crops = {}
        self.stats = {
            'skipped_low_vis': 0,
            'skipped_front': 0,
            'skipped_referee': 0,
            'skipped_off_court': 0,
            'skipped_confirmed': 0,
            'processed': 0
        }
    
    def reset_stats(self):
        """İstatistikleri sıfırla."""
        self.stats = {
            'skipped_low_vis': 0,
            'skipped_front': 0,
            'skipped_referee': 0,
            'skipped_off_court': 0,
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
        
        # Eski track'leri temizle (timeout kontrolü)
        self.player_tracker.cleanup_old_tracks(frame_num)
        
        # SAM2 Frame Prepare (Optimize encode once)
        if self.sam_segmenter is not None:
            self.sam_segmenter.prepare_frame(frame, frame_num)
        
        # Camera motion compensation
        if self.camera_stabilizer is not None:
            self.camera_stabilizer.compute_motion(frame)
        
        # Court detection (her 10 frame'de bir güncelle - performans için)
        if frame_num % 10 == 0 or self.court_mask is None:
            self.court_mask = self.court_detector.detect(frame)
        
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
            
            # Pose skeleton çiz - tüm oyunculara (hakem kontrolü sonra)
            # Önce jersey bölgesini al ve hakem mi kontrol et
            is_referee = False
            jersey_result = self.pose_detector.extract_jersey_region(frame, keypoints, bbox)
            if jersey_result is not None:
                jersey_crop, _ = jersey_result
                is_referee = self.pose_detector.is_referee(jersey_crop)
            
            # Hakem değilse skeleton çiz
            if visualize and self.config.SHOW_POSE_SKELETON and not is_referee:
                self.pose_detector.draw_skeleton(annotated_frame, keypoints)
            
            # 0. Saha kontrolü - sahada olmayan kişileri atla (opsiyonel)
            if self.config.USE_COURT_DETECTION and self.court_mask is not None:
                if not self.court_detector.is_on_court(bbox, self.court_mask):
                    if visualize:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        cv2.putText(annotated_frame, 'OffCourt', (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    self.stats['skipped_off_court'] += 1
                    continue
            
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
            
            # SAM mask segmentasyonu (opsiyonel)
            player_mask = None
            if self.sam_segmenter is not None:
                player_mask = self.sam_segmenter.segment_from_bbox(frame, bbox.tolist())
            
            # Görselleştirme
            if visualize:
                # SAM mask overlay
                if self.config.SHOW_SAM_MASK and player_mask is not None:
                    # Track ID'ye göre renk
                    color = ((track_id * 50) % 255, (track_id * 80 + 100) % 255, (track_id * 120 + 50) % 255)
                    annotated_frame = self.sam_segmenter.visualize_mask(annotated_frame, player_mask, color, alpha=0.3)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (255, 100, 0), 2)
            
            # Numara tanıma
            number, ocr_conf = self.jersey_detector.recognize_number(jersey_crop)
            
            if number and ocr_conf >= self.config.MIN_OCR_CONFIDENCE:
                # PlayerTracker'a ekle (frame_num ile)
                newly_confirmed = self.player_tracker.add_detection(
                    track_id, number, ocr_conf, frame_num=frame_num
                )
                
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
        target_width = self.config.INPUT_WIDTH
        target_height = self.config.INPUT_HEIGHT
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
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
                
                # Frame resize (1280x720)
                frame = cv2.resize(frame, (self.config.INPUT_WIDTH, self.config.INPUT_HEIGHT))
                
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
                
                # Court mask overlay (eğer aktifse)
                if self.config.SHOW_COURT_MASK and self.court_mask is not None:
                    annotated = self.court_detector.visualize(annotated, self.court_mask, alpha=0.2)
                
                if out and annotated is not None:
                    out.write(annotated)
                
                # Info overlay
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                confirmed = len(self.player_tracker.confirmed_players)
                off_court = self.stats.get('skipped_off_court', 0)
                info = f'F:{frame_count} | FPS:{fps_actual:.1f} | OK:{confirmed} | OffCourt:{off_court}'
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
    
    import os
    _ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(_ROOT_DIR, "videos", "input", "basketball_game.mp4")
    pipeline.process_video(video_path=video_path, show_preview=True, save_unique_crops=True)
    
    #test_image_path = os.path.join(_ROOT_DIR, "videos", "output", "test_frame_4.jpg")
    #pipeline.process_image(test_image_path)
    print("\n" + "=" * 60)
    print("✅ Demo tamamlandı!")
    print("=" * 60)


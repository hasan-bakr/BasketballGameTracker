"""
RFDETR + SAM2 Pipeline
Combines RFDETR (Object Detection) and SAM2 (Segmentation).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import cv2
import os
from .sam_helper import SAM2Helper
from .rfdetr_detector import RFDETRDetector
from .yolo_detector import YoloDetector

class GroundedSAM:
    """
    Segmentation Pipeline:
    1. Detection (RFDETR) -> BBoxes
    2. Segmentation (SAM2 ONNX) -> Masks
    Combines a Detector (RFDETR or YOLO) and SAM2 for segmentation.
    """
    def __init__(self, device: str = None, detector_model: str = "yolo", yolo_model_path: str = "models/yolo/best_detection.pt"):
        """
        Initialize Pipeline.
        Args:
            device: 'cuda' or 'cpu'
            detector_model: 'rfdetr' or 'yolo'
            yolo_model_path: Path to YOLO model (if using YOLO)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 GroundedSAM Initializing (Device: {self.device})...")
        
        # 1. Initialize Detector
        self.detector_type = detector_model
        
        if detector_model == "rfdetr":
            print("📦 Loading RFDETR...")
            self.detector = RFDETRDetector(
                model_path="models/rfdetr-medium.onnx",
                use_tensorrt=True
            )
        elif detector_model == "yolo":
            print(f"📦 Loading YOLO ({yolo_model_path})...")
            # Force CPU to avoid CUDA conflict with ONNX Runtime for now
            self.detector = YoloDetector(
                model_path=yolo_model_path,
                device="cpu" 
            )
        else:
            print(f"❌ Unknown detector model: {detector_model}")
            self.detector = None

        # 2. Initialize SAM2
        print("📦 Loading SAM2...")
        self.sam_onnx = SAM2Helper(device=self.device, use_tensorrt=False)

    def segment(self, image_input: Union[str, np.ndarray], conf_threshold: float = 0.50, rfdetr_vis_path: str = None, min_area: int = 1000) -> List[Dict]:
        """Detect and Segment (using selected detector)"""
        import time
        
        # Handle input type
        if isinstance(image_input, str):
            print(f"\n🔍 Processing: {image_input}")
            img = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _debug_path = image_input
        else:
            # Assume numpy array (BGR from cv2)
            img = image_input
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _debug_path = "Video Frame"

        if not self.detector:
            print("❌ Detector not active!")
            return []
            
        # 1. Detect
        t0 = time.time()
        
        if self.detector_type == "yolo":
            # Custom YOLO (best_detection.pt) Logic
            # Classes:
            # 0: ball, 3: player, 4: player-in-possession, 5: player-jump-shot, 
            # 6: player-layup-dunk, 7: player-shot-block
            # Exclude: 8 (referee), 1 (ball-in-basket?), 2 (number), 9 (rim)
            
            target_classes = [0, 3, 4, 5, 6, 7]
            detections = self.detector.detect(img, confidence_threshold=conf_threshold, classes=target_classes)
            
            # Label Mapping: Collapse all player variants to "player"
            for det in detections:
                cid = det['class_id']
                cname = det['class_name']
                if cid in [3, 4, 5, 6, 7]:
                    det['class_name'] = 'player'
                    # det['label'] logic handles usage later
        else:
            detections = self.detector.detect(img, confidence_threshold=conf_threshold)
            
        t_det = time.time() - t0
        
        # Filter by Area (for players/refs if needed, though class filtering handles most)
        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            area = w * h
            if area > min_area:
                filtered_detections.append(det)
            
        detections = filtered_detections
        
        # Save Detector-only visualization if requested
        if rfdetr_vis_path and detections and isinstance(image_input, str):
             self.detector.detect_and_draw(image_input, conf_threshold, show=False, save_path=rfdetr_vis_path)
        
        if not detections:
            return []

        # 2. Segment (SAM2)
        t0 = time.time()
        enc_outs, info = self.sam_onnx.encode(img_rgb)
        t_enc = time.time() - t0
        
        results = []
        t_dec_total = 0
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            # Use the updated class_name ('player')
            label = det.get('class_name', 'obj')
            confidence = det.get('confidence', 0.0)
            
            # Prepare prompts: 3 points (TL, BR, Center)
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            points = np.array([[[x1, y1], [x2, y2], [cx, cy]]], dtype=np.float32)
            labels = np.array([[2, 3, 1]], dtype=np.float32)
            
            # Decode
            t0 = time.time()
            mask, mask_score = self.sam_onnx.decode(enc_outs, points, labels, info)
            t_dec_total += (time.time() - t0)
            
            # Update detection object
            det['mask'] = mask
            det['mask_score'] = mask_score
            # Normalize keys for consistency
            det['label'] = label
            det['score'] = confidence
            
            results.append(det)
            
        if len(detections) > 0:
            print(f"⏱️ SAM2 Decoder (avg): {(t_dec_total/len(detections))*1000:.2f} ms (Total: {t_dec_total*1000:.2f} ms)")

        return results
    
    def visualize_results(
        self,
        image_path: str,
        results: List[Dict],
        save_path: str = "output.jpg",
        show: bool = False
    ):
        """Sonuçları görselleştir."""
        image = cv2.imread(image_path)
        
        for det in results:
            mask = det['mask']
            bbox = det['bbox']
            score = det['score']
            label = det['label']
            
            # Random color
            color = np.random.randint(0, 255, 3).tolist()
            
            # Maske
            if mask is not None:
                image[mask > 0] = np.array(color) * 0.5 + image[mask > 0] * 0.5
                
            # BBox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            text = f"{label} {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"💾 Kaydedildi: {save_path}")
            
        if show:
            import matplotlib.pyplot as plt
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    def process_video(self, video_path: str, output_path: str, conf_threshold: float = 0.50, min_area: int = 1000):
        """Video üzerinde processing yapar ve FPS ölçer."""
        import time
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video: {video_path}")
        print(f"   Size: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Output Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                t0 = time.time()
                
                # 1. Pipeline Run
                results = self.segment(frame, conf_threshold=conf_threshold, min_area=min_area)
                
                # 2. Visualize (Draw on frame)
                # We do this manually here to avoid re-reading form disk
                for det in results:
                    mask = det.get('mask')
                    bbox = det.get('bbox')
                    label = det.get('label', 'obj')
                    # score = det.get('score', 0.0)
                    
                    # Random color
                    color = np.random.randint(0, 255, 3).tolist()
                    
                    if mask is not None:
                         frame[mask > 0] = np.array(color) * 0.5 + frame[mask > 0] * 0.5
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                t1 = time.time()
                dt = t1 - t0
                total_time += dt
                current_fps = 1.0 / dt if dt > 0 else 0
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                out.write(frame)
                
                # Show video in real-time (User Request)
                try:
                    cv2.imshow("GroundedSAM Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Kullanıcı 'q' tuşu ile durdurdu.")
                        break
                except cv2.error:
                    # Likely headless environment
                    pass
                
                if frame_count % 10 == 0:
                    avg_fps = frame_count / total_time
                    print(f"   Frame {frame_count}/{total_frames}: FPS={current_fps:.1f} (Avg: {avg_fps:.1f})")
                
                # Limit test to 100 frames for speed (can be removed for full video)
                if frame_count >= 100:
                    print("🛑 Test limiti (100 kare) aşıldı.")
                    break
                    
        except KeyboardInterrupt:
            print("🛑 İşlem kullanıcı tarafından durduruldu.")
        finally:
            cap.release()
            out.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"✅ Video tamamlandı: {output_path}")
            print(f"   Ortalama FPS: {avg_fps:.2f}")


# ============ KULLANIM ÖRNEĞİ ============
if __name__ == "__main__":
    import os
    # Initialize Pipeline (RFDETR or YOLO) + SAM2
    # Force CPU for stable verification (avoid VRAM OOM)
    device = "cpu" 
    print(f"🚀 Başlatılıyor... Device: {device}")
    
    # Switch here: 'rfdetr' or 'yolo'
    pipeline = GroundedSAM(device=device, detector_model="yolo") 
    
    # Video Path
    video_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    output_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\benchmark_yolo_result.mp4"
    
    if os.path.exists(video_path):
        pipeline.process_video(video_path, output_path, conf_threshold=0.50, min_area=2000)
    else:
        print(f"❌ Video bulunamadı: {video_path}")

"""
YOLO Detector Class
===================
Ultralytics YOLO based object detection.
Mirrors the interface of RFDETRDetector for easy swapping.
"""
import cv2
import numpy as np
from typing import List, Dict, Union, Optional
from ultralytics import YOLO
import torch

class YoloDetector:
    """YOLO model wrapper for object detection."""
    
    def __init__(self, model_path: str = "yolo11m.pt", device: str = None):
        """
        Initialize YOLO Detector.
        
        Args:
            model_path: Path to model file (e.g. 'yolo11m.pt'). 
                        Ultralytics will download it automatically if not found.
            device: 'cuda' or 'cpu'. If None, detects automatically.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"📦 Loading YOLO Model: {model_path} ({device})...")
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)
        print(f"✅ YOLO Model loaded: {model_path}")

    def detect(self, image: Union[str, np.ndarray], confidence_threshold: float = 0.5, classes: List[int] = None) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: Image path (str) or numpy array (BGR).
            confidence_threshold: Confidence threshold.
            classes: List of class IDs to filter (optional). 
                     Note: YOLO class IDs might differ from RFDETR's COCO mapping if using custom models,
                     but standard YOLO uses COCO.
                     Person is 0 in YOLO (COCO), usually.
        
        Returns:
            List of dicts: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_name': str, 'class_id': int}
        """
        # Run inference
        # stream=False -> return Results list
        # verbose=False -> reduce print noise
        results = self.model.predict(
            source=image, 
            conf=confidence_threshold, 
            classes=classes, 
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Class
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_name': cls_name,
                    'class_id': cls_id
                })
                
        return detections

    def detect_and_draw(self, image_input: Union[str, np.ndarray], conf_threshold: float = 0.5, show: bool = False, save_path: str = None):
        """Helper to visualize detections."""
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()
            
        detections = self.detect(image, conf_threshold=conf_threshold)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Draw
            color = (0, 255, 0) # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"💾 YOLO Visualization saved: {save_path}")
            
        if show:
            cv2.imshow("YOLO Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.5):
        """
        Process video and visualizes detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Video can't be opened: {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"🎥 Processing YOLO Video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect
            # Using track for video consistency if available, or just detect
            # For pure detection inspection, simple detect is fine.
            results = self.model.predict(frame, conf=conf_threshold, verbose=False, device=self.device)
            
            # Annotated Frame (Ultralytics built-in is good, but let's draw manually for control if needed)
            # Actually, results[0].plot() gives a nice annotated image directly
            annotated_frame = results[0].plot() 
            
            # Additional custom text if needed
            cv2.putText(annotated_frame, "YOLO Inspection", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show
            try:
                cv2.imshow("YOLO Inspection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
                
            if output_path:
                out.write(annotated_frame)
                
        cap.release()
        if output_path:
            out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("✅ YOLO Video processing done.")

if __name__ == "__main__":
    # Test block
    # detector = YoloDetector(model_path="yolo11n.pt") 
    
    # Custom model test
    model_path = "models/yolo/best_detection.pt"
    video_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    output_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\yolo_only_inspection.mp4"
    
    import os
    if os.path.exists(model_path) and os.path.exists(video_path):
        print("▶️ Running YOLO Inspection...")
        detector = YoloDetector(model_path=model_path)
        detector.detect_video(video_path, output_path=output_path, conf_threshold=0.3)
    else:
        print("⚠️ Model or Video not found for test.")


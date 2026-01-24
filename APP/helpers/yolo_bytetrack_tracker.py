"""
YOLO + ByteTrack Basic Tracker
==============================
ID karışması/occlusion/swap testi için basit tracker.
SAM2 ile karşılaştırma amaçlı.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
import os

# Oyuncu sınıfları
PLAYER_CLASSES = [3, 4, 5, 6, 7]


def run_yolo_bytetrack(
    video_path: str,
    output_path: str,
    model_path: str = "models/yolo/best_detection.pt",
    max_frames: int = 400,
    confidence: float = 0.3
):
    """
    YOLO + ByteTrack ile oyuncu takibi.
    
    Args:
        video_path: Input video
        output_path: Output video
        model_path: YOLO model path
        max_frames: Max frames to process
        confidence: Detection confidence
    """
    print(f"📦 Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎬 Processing: {video_path}")
    print(f"   Resolution: {width}x{height}, FPS: {fps}")
    
    # Track history for visualization
    track_history: Dict[int, List] = {}
    
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO tracking with ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            classes=PLAYER_CLASSES,
            verbose=False
        )
        
        # Annotated frame
        annotated = frame.copy()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                # Rastgele renk (ID'ye göre sabit)
                np.random.seed(track_id)
                color = tuple(np.random.randint(50, 255, 3).tolist())
                
                # Bounding box çiz
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # ID yazdır
                label = f"ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Track history
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                
                # Son 30 nokta trail çiz
                points = track_history[track_id][-30:]
                for i in range(1, len(points)):
                    cv2.line(annotated, points[i-1], points[i], color, 2)
        
        # Frame info
        cv2.putText(annotated, f"Frame: {frame_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Tracks: {len(track_history)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, "YOLO + ByteTrack", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        writer.write(annotated)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"   Processed: {frame_count}/{max_frames} frames")
    
    cap.release()
    writer.release()
    
    print(f"\n✅ Done! Output: {output_path}")
    print(f"📊 Total tracks: {len(track_history)}")
    print(f"📊 Track IDs: {sorted(track_history.keys())}")


if __name__ == "__main__":
    video_in = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    video_out = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\yolo_bytetrack_output.mp4"
    
    run_yolo_bytetrack(video_in, video_out, max_frames=400)

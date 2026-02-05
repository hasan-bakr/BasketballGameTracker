"""
Test Custom Court Keypoint Model
================================
Yeni eğitilen YOLO-Pose modeli ile saha keypoint tespiti test et.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import deque

# Model path
MODEL_PATH = "models/keypoints/test_keypoint.pt"
VIDEO_PATH = "videos/input/court.mp4"
OUTPUT_PATH = "videos/output/basketball_game_keypoint_test.mp4"


class KeypointSmoother:
    """Temporal smoothing for keypoint stability."""
    
    def __init__(self, num_keypoints=18, alpha=0.7, buffer_size=5):
        """
        Args:
            num_keypoints: Number of keypoints
            alpha: EMA weight for current frame (0-1, higher = less smoothing)
            buffer_size: Number of frames to average
        """
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)
        self.smoothed = None
    
    def update(self, keypoints):
        """
        Apply temporal smoothing to keypoints.
        
        Args:
            keypoints: (N, 3) tensor [x, y, conf]
            
        Returns:
            smoothed keypoints
        """
        if keypoints is None:
            return self.smoothed
        
        kpts = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
        
        if self.smoothed is None:
            self.smoothed = kpts.copy()
        else:
            # Exponential Moving Average
            for i in range(len(kpts)):
                if kpts[i, 2] > 0.2:  # Only update if detection is confident
                    self.smoothed[i, :2] = (
                        self.alpha * kpts[i, :2] + 
                        (1 - self.alpha) * self.smoothed[i, :2]
                    )
                    self.smoothed[i, 2] = kpts[i, 2]
        
        return self.smoothed

# Model inference resolution - modelin eğitildiği çözünürlük ile aynı olmalı
MODEL_IMGSZ = 768  # Eğitimde kullandığın değer (640 veya 832)

# Confidence threshold - sadece bu eşiğin üzerindeki keypointler gösterilir
CONF_THRESHOLD = 0.5  # 0.3-0.7 arası dene, yüksek = daha az ama doğru keypoint

print("=" * 60)
print("Testing Custom Court Keypoint Model")
print("=" * 60)

# Load model
print(f"\n📦 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Model info
print(f"   Task: {model.task}")
print(f"   Names: {model.names}")
print(f"   Inference size: {MODEL_IMGSZ}")

# Test on single frame first
print(f"\n🖼️ Testing on first frame...")
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("❌ Could not read video!")
    exit(1)

print(f"   Frame size: {frame.shape}")

# Run inference with correct resolution
results = model(frame, imgsz=MODEL_IMGSZ, verbose=False)
result = results[0]

print(f"\n📊 Results:")
print(f"   Boxes: {result.boxes.shape if result.boxes is not None else 'None'}")
print(f"   Keypoints: {result.keypoints.shape if result.keypoints is not None else 'None'}")

if result.keypoints is not None:
    kpts = result.keypoints.data[0]  # First detection
    print(f"   Keypoint tensor shape: {kpts.shape}")
    print(f"   (num_keypoints, 3) -> x, y, confidence")
    
    # Count valid keypoints
    valid_count = (kpts[:, 2] > 0.3).sum().item()
    print(f"   Valid keypoints (conf > 0.3): {valid_count}/{len(kpts)}")
    
    # Draw keypoints
    output_frame = frame.copy()
    
    for i, kpt in enumerate(kpts):
        x, y, conf = int(kpt[0]), int(kpt[1]), float(kpt[2])
        
        if conf > 0.3:
            # Green circle for high confidence
            cv2.circle(output_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(output_frame, str(i), (x + 10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        elif conf > 0.1:
            # Yellow circle for low confidence
            cv2.circle(output_frame, (x, y), 6, (0, 255, 255), -1)
    
    # Draw connections (if keypoints form court lines)
    # Connect corner points
    corner_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Adjust based on keypoint order
    for i, j in corner_pairs:
        if i < len(kpts) and j < len(kpts):
            if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                cv2.line(output_frame, pt1, pt2, (255, 0, 255), 2)
    
    # Save single frame result
    cv2.imwrite("court_keypoint_custom_test.jpg", output_frame)
    print(f"\n✅ Saved: court_keypoint_custom_test.jpg")
else:
    print("   ❌ No keypoints detected!")

# Process video
print(f"\n🎬 Processing video: {VIDEO_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Initialize smoother for temporal stability
smoother = KeypointSmoother(num_keypoints=18, alpha=0.6)  # Lower alpha = more smoothing

frame_count = 0
max_frames = min(900, total_frames)  # Process first 900 frames

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference with correct resolution
    results = model(frame, imgsz=MODEL_IMGSZ, verbose=False)
    result = results[0]
    
    # Draw keypoints with temporal smoothing
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        raw_kpts = result.keypoints.data[0]
        
        # Apply temporal smoothing
        kpts = smoother.update(raw_kpts)
        
        for i, kpt in enumerate(kpts):
            x, y, conf = int(kpt[0]), int(kpt[1]), float(kpt[2])
            
            if conf > CONF_THRESHOLD:  # Sadece yüsek confidence keypointler
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"{i}:{conf:.2f}", (x + 8, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    
    # Add frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 50 == 0:
        print(f"   Processed {frame_count}/{max_frames} frames...")

cap.release()
out.release()

print(f"\n✅ Video saved: {OUTPUT_PATH}")
print(f"   Processed {frame_count} frames")

print("\n" + "=" * 60)

"""
Court Keypoint Detection Test
=============================
Roboflow'dan basketball court keypoint detection modelini test eder.

Model: basketball-court-detection-2 (YOLO26-pose)
Keypoints: Court lines, corners, arcs
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Roboflow model'i indirmek için
# pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace().project("basketball-court-detection-2")
# model = project.version(1).model

print("=" * 60)
print("Court Keypoint Detection Test")
print("=" * 60)

# Test video
VIDEO_PATH = "videos/input/court.mp4"
OUTPUT_PATH = "videos/output/court_keypoints_test.mp4"

# Check if we have a court keypoint model
COURT_MODEL_PATH = "models/court_keypoint.pt"

if not os.path.exists(COURT_MODEL_PATH):
    print(f"⚠️ Court keypoint model not found at {COURT_MODEL_PATH}")
    print("\nOptions:")
    print("1. Download from Roboflow:")
    print("   pip install roboflow")
    print("   from roboflow import Roboflow")
    print("   rf = Roboflow(api_key='YOUR_API_KEY')")
    print("   project = rf.workspace().project('basketball-court-detection-2')")
    print("   model = project.version(1).model")
    print("")
    print("2. Or use YOLO pose model directly:")
    print("   We can try with yolo26n-pose.pt for testing")
    
    # Try with existing pose model for testing structure
    POSE_MODEL_PATH = "yolo26n-pose.pt"
    if os.path.exists(POSE_MODEL_PATH):
        print(f"\n📦 Testing with existing pose model: {POSE_MODEL_PATH}")
        model = YOLO(POSE_MODEL_PATH)
        
        # Test on single frame
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"   Frame size: {frame.shape}")
            
            # Run inference
            results = model(frame, verbose=False)
            
            print("\n📊 Model Output Structure:")
            print(f"   Results type: {type(results)}")
            if results:
                r = results[0]
                print(f"   Boxes: {r.boxes.shape if r.boxes is not None else 'None'}")
                print(f"   Keypoints: {r.keypoints.shape if r.keypoints is not None else 'None'}")
                
                if r.keypoints is not None:
                    kpts = r.keypoints.data
                    print(f"   Keypoint tensor shape: {kpts.shape}")
                    print(f"   (batch, num_keypoints, 3) -> x, y, confidence")
    else:
        print(f"   No pose model found at {POSE_MODEL_PATH}")
else:
    print(f"✅ Loading court keypoint model: {COURT_MODEL_PATH}")
    model = YOLO(COURT_MODEL_PATH)
    
    # Process video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print(f"📽️ Processing video: {VIDEO_PATH}")
    print(f"   Resolution: {width}x{height}, FPS: {fps}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Draw keypoints
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data[0]  # First detection
            
            for i, kpt in enumerate(keypoints):
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"   Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"\n✅ Done! Output saved to: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

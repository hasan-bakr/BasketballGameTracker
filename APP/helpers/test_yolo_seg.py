"""
YOLO-Seg Test Script
====================
Tests YOLO11L-Seg model on basketball video to evaluate mask quality.
"""
import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_seg(video_path: str, output_path: str = None, model_name: str = "yolo11l-seg.pt"):
    """
    Test YOLO-Seg model on video.
    
    Args:
        video_path: Input video path
        output_path: Output video path (optional)
        model_name: YOLO-Seg model to use
    """
    print(f"📦 Loading YOLO-Seg Model: {model_name}...")
    model = YOLO(model_name)
    print(f"✅ Model loaded!")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video can't be opened: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"🎥 Processing: {video_path}")
    print(f"   Resolution: {width}x{height}, FPS: {fps}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run YOLO-Seg
        results = model.predict(frame, conf=0.5, verbose=False, classes=[0])  # 0 = person
        
        # Get annotated frame (built-in visualization)
        annotated = results[0].plot()
        
        # Additional info
        masks = results[0].masks
        num_masks = len(masks) if masks is not None else 0
        cv2.putText(annotated, f"Frame: {frame_count} | Masks: {num_masks}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show
        try:
            cv2.imshow("YOLO-Seg Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 User pressed 'q' to stop.")
                break
        except:
            pass
            
        if out:
            out.write(annotated)
            
        if frame_count % 30 == 0:
            print(f"   Frame {frame_count}: {num_masks} masks detected")
            
        # Limit for quick test
        if frame_count >= 300:
            print("🛑 Test limit (300 frames) reached.")
            break
            
    cap.release()
    if out:
        out.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
        
    print(f"✅ Test completed! Processed {frame_count} frames.")


if __name__ == "__main__":
    video_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    output_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\yolo_seg_test.mp4"
    
    # Test with yolo11l-seg (will auto-download if not present)
    test_yolo_seg(video_path, output_path, model_name="yolo11l-seg.pt")

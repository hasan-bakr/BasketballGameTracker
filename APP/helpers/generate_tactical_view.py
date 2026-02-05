"""
Generate Tactical View
======================
Full pipeline to generate tactical bird's eye view from basketball video.
Combines:
1. Court Keypoint Detection (Custom YOLO)
2. Homography Transformation
3. Player Detection (YOLO) [Optional for now, visualizing court first]
4. Tactical Map Rendering
"""

import cv2
import numpy as np
from court_keypoint_detector import CourtKeypointDetector
from homography_transformer import HomographyTransformer
from yolo_detector import YoloDetector

def render_tactical_map(background, player_positions):
    """Draw players on the tactical map."""
    frame = background.copy()
    for pid, (x, y) in player_positions.items():
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
    return frame

def main():
    VIDEO_PATH = "videos/input/nba_clip_trimmed.mp4"
    OUTPUT_PATH = "videos/output/tactical_view_test.mp4"
    COURT_TEMPLATE_PATH = "court_analysis_ref/tactical_view/court_images/basketball_court.png" # Need to check if exists or create simple one
    
    # Initialize Detectors
    keypoint_detector = CourtKeypointDetector(
        conf_threshold=0.5,
        imgsz=768
    )
    homography_transformer = HomographyTransformer()
    # player_detector = YoloDetector() # Uncomment to add players later
    
    # Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create Tactical Map Background (940x500)
    tactical_bg = np.zeros((500, 940, 3), dtype=np.uint8)
    # Draw simple court lines for visualization if template missing
    cv2.rectangle(tactical_bg, (0,0), (940,500), (200,150,100), -1) # Floor color
    cv2.line(tactical_bg, (470,0), (470,500), (255,255,255), 2) # Center line
    cv2.circle(tactical_bg, (470,250), 60, (255,255,255), 2) # Center circle
    
    # Setup Video Writer (Side-by-side view)
    # Output width = Video Width + Tactical Map Width
    out_width = width + 940
    out_height = max(height, 500)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, out_height))
    
    print(f"🚀 Processing video: {VIDEO_PATH}...")
    
    frame_idx = 0
    max_frames = 400
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect Keypoints
        keypoints = keypoint_detector.detect(frame)
        
        # 2. Update Homography
        H = homography_transformer.update(keypoints)
        
        # 3. Simulate Players (just detection centers for testing)
        # TODO: Replace with real player detection
        # Center of frame as dummy player to test transformation
        dummy_player_pos = np.array([[width/2, height/2]]) 
        
        # 4. Transform Points
        if H is not None:
            tactical_pos = homography_transformer.transform_points(dummy_player_pos, H)
            t_x, t_y = tactical_pos[0]
        else:
            t_x, t_y = -1, -1
            
        # 5. Render Views
        # Draw keypoints on main frame
        frame_vis = keypoint_detector.draw_keypoints(frame, keypoints)
        
        # Draw court lines: use keypoints when visible, homography prediction when not
        # Blue = both points visible, Cyan = at least one predicted
        frame_vis = homography_transformer.draw_court_lines_hybrid(frame_vis, keypoints, H)
        
        # Draw on tactical map
        tactical_vis = tactical_bg.copy()
        if t_x > 0:
            cv2.circle(tactical_vis, (int(t_x), int(t_y)), 15, (0, 255, 255), -1)
            
        # Combine Views
        combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        combined[:height, :width] = frame_vis
        combined[:500, width:] = tactical_vis
        
        out.write(combined)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"   Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    print(f"✅ Saved output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

"""
Manual Court Keypoint Selector
==============================
Saha köşelerini elle seçerek homography hesaplamak için interaktif araç.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import json
import os


class ManualCourtKeypointSelector:
    """
    Saha keypoint'lerini elle seçmek için interaktif araç.
    Seçilen noktalar kaydedilir ve homography hesaplanır.
    """
    
    # NBA Court dimensions in pixels (tactical view)
    COURT_WIDTH = 940   # 94 feet * 10
    COURT_HEIGHT = 500  # 50 feet * 10
    
    # Predefined court keypoint locations (tactical view coordinates)
    # These are the reference points on a top-down court view
    COURT_TEMPLATE_POINTS = {
        # Corners (clockwise from top-left)
        0: (0, 0),           # Top-left corner
        1: (COURT_WIDTH, 0), # Top-right corner
        2: (COURT_WIDTH, COURT_HEIGHT),  # Bottom-right
        3: (0, COURT_HEIGHT), # Bottom-left
        
        # Center line
        4: (COURT_WIDTH // 2, 0),  # Center-top
        5: (COURT_WIDTH // 2, COURT_HEIGHT),  # Center-bottom
        
        # Free throw line (left side)  
        6: (190, 190),   # Left FT line top
        7: (190, 310),   # Left FT line bottom
        
        # Free throw line (right side)
        8: (750, 190),   # Right FT line top
        9: (750, 310),   # Right FT line bottom
        
        # Three-point arc top
        10: (0, 140),    # Left 3pt corner top
        11: (940, 140),  # Right 3pt corner top
        
        # Three-point arc bottom
        12: (0, 360),    # Left 3pt corner bottom
        13: (940, 360),  # Right 3pt corner bottom
    }
    
    KEYPOINT_NAMES = {
        0: "Top-Left Corner",
        1: "Top-Right Corner", 
        2: "Bottom-Right Corner",
        3: "Bottom-Left Corner",
        4: "Center Line Top",
        5: "Center Line Bottom",
        6: "Left FT Top",
        7: "Left FT Bottom",
        8: "Right FT Top",
        9: "Right FT Bottom",
        10: "Left 3PT Top",
        11: "Right 3PT Top",
        12: "Left 3PT Bottom",
        13: "Right 3PT Bottom",
    }
    
    def __init__(self, cache_path: str = "court_keypoints_cache.json"):
        self.cache_path = cache_path
        self.selected_points = {}  # {keypoint_id: (x, y)}
        self.current_frame = None
        self.window_name = "Court Keypoint Selection"
        self.selecting_id = 0
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse click handler."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points[self.selecting_id] = (x, y)
            print(f"   Selected point {self.selecting_id} ({self.KEYPOINT_NAMES.get(self.selecting_id, 'Unknown')}): ({x}, {y})")
            self.selecting_id += 1
            self._draw_points()
    
    def _draw_points(self):
        """Draw selected points on frame."""
        if self.current_frame is None:
            return
        
        display = self.current_frame.copy()
        
        # Draw selected points
        for pid, (x, y) in self.selected_points.items():
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, str(pid), (x + 10, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Instructions
        next_name = self.KEYPOINT_NAMES.get(self.selecting_id, "Done")
        cv2.putText(display, f"Click: {next_name} (ID: {self.selecting_id})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'Q' to finish, 'R' to reset, 'U' to undo", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, display)
    
    def select_keypoints(self, frame: np.ndarray, min_points: int = 4) -> dict:
        """
        Interaktif olarak keypoint seç.
        
        Args:
            frame: Video frame
            min_points: Minimum gerekli nokta sayısı (homography için en az 4)
            
        Returns:
            selected_points: {keypoint_id: (x, y)}
        """
        self.current_frame = frame.copy()
        self.selected_points = {}
        self.selecting_id = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("Manuel Keypoint Seçimi")
        print("=" * 60)
        print("Saha üzerindeki noktaları sırayla tıklayın:")
        for pid, name in list(self.KEYPOINT_NAMES.items())[:8]:
            print(f"  {pid}: {name}")
        print("\nKontroller:")
        print("  Sol tık: Nokta seç")
        print("  U: Son noktayı geri al")
        print("  R: Tümünü sıfırla")  
        print("  Q: Bitir ve kaydet")
        print("=" * 60)
        
        self._draw_points()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if len(self.selected_points) >= min_points:
                    print(f"\n✅ {len(self.selected_points)} nokta seçildi")
                    break
                else:
                    print(f"⚠️ En az {min_points} nokta gerekli!")
            
            elif key == ord('u'):
                if self.selecting_id > 0:
                    self.selecting_id -= 1
                    if self.selecting_id in self.selected_points:
                        del self.selected_points[self.selecting_id]
                    print(f"   Geri alındı: {self.selecting_id}")
                    self._draw_points()
            
            elif key == ord('r'):
                self.selected_points = {}
                self.selecting_id = 0
                print("   Sıfırlandı")
                self._draw_points()
        
        cv2.destroyAllWindows()
        
        # Save to cache
        self._save_cache()
        
        return self.selected_points
    
    def _save_cache(self):
        """Save selected points to cache file."""
        with open(self.cache_path, 'w') as f:
            json.dump(self.selected_points, f)
        print(f"📁 Saved to: {self.cache_path}")
    
    def load_cache(self) -> Optional[dict]:
        """Load previously selected points."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.selected_points = {int(k): tuple(v) for k, v in json.load(f).items()}
            print(f"📁 Loaded {len(self.selected_points)} points from cache")
            return self.selected_points
        return None
    
    def get_homography_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matched point pairs for homography calculation.
        
        Returns:
            image_points: (N, 2) array of selected points in image
            court_points: (N, 2) array of corresponding tactical view points
        """
        image_points = []
        court_points = []
        
        for pid, (x, y) in self.selected_points.items():
            if pid in self.COURT_TEMPLATE_POINTS:
                image_points.append([x, y])
                court_points.append(list(self.COURT_TEMPLATE_POINTS[pid]))
        
        return np.array(image_points, dtype=np.float32), np.array(court_points, dtype=np.float32)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Court Keypoint Selector Test")
    print("=" * 60)
    
    selector = ManualCourtKeypointSelector()
    
    # Check for cached points
    cached = selector.load_cache()
    
    if cached and len(cached) >= 4:
        print(f"Using cached keypoints: {len(cached)} points")
    else:
        # Load video frame
        cap = cv2.VideoCapture("videos/input/court.mp4")
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Interactive selection
            points = selector.select_keypoints(frame, min_points=4)
            print(f"\nSelected points: {points}")
    
    # Get homography pairs
    img_pts, court_pts = selector.get_homography_points()
    print(f"\nHomography point pairs: {len(img_pts)}")
    
    print("\n" + "=" * 60)

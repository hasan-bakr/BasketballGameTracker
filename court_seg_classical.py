"""
Classical CV + YOLO Hybrid Court Detection
==========================================
Klasik bilgisayar görüşü (Hough Lines, edge detection) ile 
YOLO keypoint fusionlayarak saha tespiti.
"""

import cv2
import numpy as np
import argparse
import os
import sys

# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class RobustCourtDetector:
    def __init__(self, yolo_model_path=None):
        # YOLO keypoint model (varsa)
        if yolo_model_path and os.path.exists(yolo_model_path):
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
        else:
            self.yolo_model = None
        
        # Saha template (real-world coordinates)
        self.court_template = self.create_court_template()
        
        # Temporal smoothing
        self.last_H = None
        self.H_history = []
        
    def create_court_template(self):
        """NBA/FIBA standart saha (cm cinsinden)"""
        return {
            # Ana dikdörtgen köşeleri
            'corners': np.float32([
                [0, 0],           # Sol üst
                [2800, 0],        # Sağ üst  
                [2800, 1500],     # Sağ alt
                [0, 1500]         # Sol alt
            ]),
            
            # 3-point çizgisi (köşe noktaları)
            '3pt_corners': np.float32([
                [89, 141],
                [2711, 141],
                [2711, 1359],
                [89, 1359]
            ]),
            
            # Free throw çizgisi
            'free_throw': np.float32([
                [575, 0],
                [2225, 0],
                [2225, 190],
                [575, 190]
            ]),
            
            # Orta saha çizgisi
            'midcourt': np.float32([
                [0, 750],
                [2800, 750]
            ])
        }
    
    def detect_lines_classical(self, frame):
        """Klasik CV ile çizgi tespiti"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sadece beyaz çizgileri al
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Morfolojik operasyonlar
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        
        # Hough Lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=100,
            maxLineGap=20
        )
        
        return lines, white_mask
    
    def filter_court_lines(self, lines, frame):
        """Saha çizgilerine benzer olanları filtrele"""
        if lines is None:
            return []
        
        court_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 80:
                continue
            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            is_horizontal = (angle < 15) or (angle > 165)
            is_vertical = (75 < angle < 105)
            
            if is_horizontal or is_vertical:
                court_lines.append(line[0])
        
        return court_lines
    
    def find_court_keypoints(self, lines, frame):
        """Çizgi kesişim noktalarını bul"""
        if len(lines) < 4:
            return None
        
        keypoints = []
        
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1:]:
                pt = self.line_intersection(line1, line2)
                if pt is not None:
                    x, y = pt
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        keypoints.append(pt)
        
        if len(keypoints) < 4:
            return None
        
        keypoints = self.remove_duplicates(keypoints, threshold=20)
        
        return np.array(keypoints, dtype=np.float32)
    
    def line_intersection(self, line1, line2):
        """İki çizginin kesişim noktası"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return [int(x), int(y)]
    
    def remove_duplicates(self, points, threshold=20):
        """Birbirine çok yakın noktaları birleştir"""
        if len(points) == 0:
            return []
        
        unique = [points[0]]
        
        for pt in points[1:]:
            is_duplicate = False
            for upt in unique:
                dist = np.linalg.norm(np.array(pt) - np.array(upt))
                if dist < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(pt)
        
        return unique
    
    def match_keypoints_to_template(self, detected_kps, frame):
        """Tespit edilen keypoint'leri template ile eşleştir"""
        if detected_kps is None or len(detected_kps) < 4:
            return None
        
        hull = cv2.convexHull(detected_kps)
        
        if len(hull) < 4:
            return None
        
        corners = self.get_corner_points(hull, frame)
        
        if corners is None or len(corners) < 4:
            return None
        
        src_points = corners
        dst_points = self.court_template['corners']
        
        return src_points, dst_points
    
    def get_corner_points(self, hull, frame):
        """Convex hull'dan 4 köşe noktasını seç"""
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            corners = approx[:4].reshape(-1, 2).astype(np.float32)
            corners = self.order_points(corners)
            return corners
        
        return None
    
    def order_points(self, pts):
        """Noktaları saat yönü tersine sırala"""
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return pts[sorted_indices]
    
    def compute_homography(self, src_points, dst_points):
        """Homography matrisi hesapla"""
        if src_points is None or dst_points is None:
            return None
        
        if len(src_points) < 4 or len(dst_points) < 4:
            return None
        
        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        # Temporal smoothing
        if H is not None:
            if self.last_H is not None:
                alpha = 0.7
                H = alpha * H + (1 - alpha) * self.last_H
            
            self.last_H = H
            self.H_history.append(H)
        
        return H
    
    def detect(self, frame):
        """Ana detection fonksiyonu"""
        # 1. Klasik CV ile çizgi tespiti
        lines, white_mask = self.detect_lines_classical(frame)
        court_lines = self.filter_court_lines(lines, frame)
        
        # 2. Keypoint'leri bul
        keypoints = self.find_court_keypoints(court_lines, frame)
        
        # 3. Template ile eşleştir
        match_result = self.match_keypoints_to_template(keypoints, frame)
        
        if match_result is None:
            return None, keypoints, court_lines, white_mask
        
        src_points, dst_points = match_result
        
        # 4. Homography hesapla
        H = self.compute_homography(src_points, dst_points)
        
        return H, keypoints, court_lines, white_mask


def run_pipeline(video_path, save_output=False, output_path=None, max_frames=0):
    """Ana pipeline."""

    print("📦 Classical Court Detector başlatılıyor...")
    detector = RobustCourtDetector()
    print("✅ Hazır!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Video: {video_path}")
    print(f"   Çözünürlük: {frame_w}x{frame_h} | FPS: {fps:.1f} | Toplam Frame: {total_frames}")

    # Bird's eye view boyutu
    BIRD_W, BIRD_H = 420, 225

    out_width = frame_w + BIRD_W + 20
    out_height = max(frame_h, BIRD_H + 240)

    out_writer = None
    if save_output:
        if output_path is None:
            os.makedirs(os.path.join(ROOT_DIR, "videos", "output"), exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(ROOT_DIR, "videos", "output", f"{base_name}_classical.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        print(f"💾 Çıktı kaydedilecek: {output_path}")

    print(f"\n🚀 İşlem başlıyor... (Çıkmak için 'q' tuşuna basın)\n")

    frame_idx = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        H, keypoints, court_lines, white_mask = detector.detect(frame)

        # --- Görselleştirme ---
        vis_frame = frame.copy()

        # Çizgileri çiz
        n_lines = len(court_lines) if court_lines else 0
        if court_lines:
            for line in court_lines:
                x1, y1, x2, y2 = line
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Keypoint'leri çiz
        n_kps = 0
        if keypoints is not None:
            n_kps = len(keypoints)
            for pt in keypoints:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 6, (255, 0, 0), -1)
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 2)

        # Durum bilgisi
        if H is not None:
            success_count += 1
            cv2.putText(vis_frame, "Homography: OK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Homography: FAIL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(vis_frame, f"Lines: {n_lines} | KPs: {n_kps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bird's eye view
        bird_view = np.zeros((BIRD_H, BIRD_W, 3), dtype=np.uint8)
        bird_view[:] = (40, 80, 40)

        if H is not None:
            # Küçük bird's eye view
            warped = cv2.warpPerspective(frame, H, (2800, 1500))
            bird_view = cv2.resize(warped, (BIRD_W, BIRD_H))

        # White mask görselleştirme
        mask_display = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        mask_display = cv2.resize(mask_display, (BIRD_W, BIRD_H))

        # Birleşik panel
        combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        combined[:frame_h, :frame_w] = vis_frame

        # Sağ panel
        y_off = 10
        combined[y_off:y_off + BIRD_H, frame_w + 10:frame_w + 10 + BIRD_W] = bird_view
        cv2.putText(combined, "Bird's Eye View", (frame_w + 15, y_off + BIRD_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_off2 = y_off + BIRD_H + 35
        if y_off2 + BIRD_H <= out_height:
            combined[y_off2:y_off2 + BIRD_H, frame_w + 10:frame_w + 10 + BIRD_W] = mask_display
            cv2.putText(combined, "White Mask", (frame_w + 15, y_off2 + BIRD_H + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Göster
        display = combined
        max_display_w = 1600
        if combined.shape[1] > max_display_w:
            scale = max_display_w / combined.shape[1]
            display = cv2.resize(combined, (max_display_w, int(combined.shape[0] * scale)))

        cv2.imshow("Classical Court Detection", display)

        if out_writer is not None:
            out_writer.write(combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹ Kullanıcı tarafından durduruldu.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"   ✅ {frame_idx} frame | Homography: {success_count}/{frame_idx}")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"\n💾 Çıktı kaydedildi: {output_path}")
    cv2.destroyAllWindows()

    rate = (success_count / frame_idx * 100) if frame_idx > 0 else 0
    print(f"\n🏁 Tamamlandı! {frame_idx} frame | Homography başarı: {success_count}/{frame_idx} ({rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Classical CV Court Detection")
    parser.add_argument("--video", type=str,
                        default=os.path.join(ROOT_DIR, "videos", "input", "a.mp4"),
                        help="Giriş video dosyası")
    parser.add_argument("--save", action="store_true",
                        help="Çıktı videosunu kaydet")
    parser.add_argument("--output", type=str, default=None,
                        help="Çıktı video yolu")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maksimum frame sayısı (0 = sınırsız)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video bulunamadı: {args.video}")
        input_dir = os.path.join(ROOT_DIR, "videos", "input")
        if os.path.exists(input_dir):
            print("Mevcut videolar:")
            for f in os.listdir(input_dir):
                print(f"   - {f}")
        return

    run_pipeline(
        video_path=args.video,
        save_output=args.save,
        output_path=args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

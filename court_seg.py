"""
Court Segmentation with basketball_analysis Court Keypoint Model
================================================================
abdullahtarek/basketball_analysis projesinin YOLO keypoint modelini
kullanarak basketbol sahası tespiti, homography hesabı ve tactical view.

Pipeline:
1. YOLO court_keypoint_detector.pt ile 18 keypoint tespiti
2. Keypoint doğrulama (proportional distance check)
3. Homography hesabı (cv2.findHomography)
4. Tactical view dönüşümü + görselleştirme

Kullanım:
    python court_seg.py
    python court_seg.py --video videos/input/nba_clip_trimmed.mp4
    python court_seg.py --video videos/input/nba_clip_trimmed.mp4 --save
    python court_seg.py --max-frames 100
"""

import argparse
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# APP helpers path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'APP'))
from helpers.grounded_sam import GroundedSAM

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Model Path ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT_DIR, "models", "keypoints", "test_keypoint.pt")
COURT_IMAGE_PATH = os.path.join(ROOT_DIR, "basketball_analysis", "images", "basketball_court.png")

# ─── Basketball Court Keypoints (tactical view coordinates) ──────────────────
# 18 keypoints: saha boyutu 28m x 15m → 300x161 px tactical view
TACTICAL_WIDTH = 300
TACTICAL_HEIGHT = 161
ACTUAL_WIDTH_M = 28.0
ACTUAL_HEIGHT_M = 15.0

TACTICAL_KEYPOINTS = [
    # 0-5: Sol kenar (yukarıdan aşağıya)
    (0, 0),
    (0, int((0.91 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((14.1 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int(TACTICAL_HEIGHT)),

    # 6-7: Orta çizgi
    (int(TACTICAL_WIDTH / 2), TACTICAL_HEIGHT),
    (int(TACTICAL_WIDTH / 2), 0),

    # 8-9: Sol serbest atış çizgisi
    (int((5.79 / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (int((5.79 / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),

    # 10-15: Sağ kenar (aşağıdan yukarıya)
    (TACTICAL_WIDTH, int(TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((14.1 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((0.91 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, 0),

    # 16-17: Sağ serbest atış çizgisi
    (int(((ACTUAL_WIDTH_M - 5.79) / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (int(((ACTUAL_WIDTH_M - 5.79) / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
]

# Keypoint renkleri (görselleştirme)
KP_COLORS = [
    (0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 100, 0), (0, 200, 0), (0, 255, 0),   # Sol kenar
    (255, 255, 0), (255, 255, 0),                                                       # Orta çizgi
    (0, 0, 255), (0, 0, 255),                                                           # Sol FT
    (255, 0, 0), (255, 50, 0), (255, 100, 0), (255, 150, 0), (255, 50, 0), (255, 0, 0), # Sağ kenar
    (0, 165, 255), (0, 165, 255),                                                       # Sağ FT
]

KP_NAMES = [
    "L-TL", "L-BL1", "L-BL2", "L-BL3", "L-BL4", "L-BotL",
    "Mid-Bot", "Mid-Top",
    "FT-L1", "FT-L2",
    "R-BotR", "R-BR4", "R-BR3", "R-BR2", "R-BR1", "R-TR",
    "FT-R1", "FT-R2",
]


def measure_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def validate_keypoints(keypoints_xy):
    """
    Keypoint'leri orantısal uzaklık kontrolü ile doğrula.
    Hatalı keypoint'leri sıfırla.
    
    Args:
        keypoints_xy: (N, 2) numpy array, her satır (x, y) keypoint koordinatı.
                      (0, 0) = tespit edilememiş keypoint.
    Returns:
        (N, 2) numpy array, doğrulanmış keypoint'ler.
    """
    kp = keypoints_xy.copy()
    detected_indices = [i for i in range(len(kp)) if kp[i][0] > 0 and kp[i][1] > 0]

    if len(detected_indices) < 3:
        return kp

    invalid = []
    for i in detected_indices:
        if kp[i][0] == 0 and kp[i][1] == 0:
            continue

        other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid]
        if len(other_indices) < 2:
            continue

        j, k = other_indices[0], other_indices[1]

        d_ij = measure_distance(kp[i], kp[j])
        d_ik = measure_distance(kp[i], kp[k])

        t_ij = measure_distance(TACTICAL_KEYPOINTS[i], TACTICAL_KEYPOINTS[j])
        t_ik = measure_distance(TACTICAL_KEYPOINTS[i], TACTICAL_KEYPOINTS[k])

        if t_ij > 0 and t_ik > 0:
            prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
            prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

            error = abs((prop_detected - prop_tactical) / prop_tactical)

            if error > 0.8:
                kp[i] = [0.0, 0.0]
                invalid.append(i)

    return kp


def compute_homography(keypoints_xy):
    """
    Tespit edilen keypoint'lerden homography matrisi hesapla.
    
    Args:
        keypoints_xy: (N, 2) numpy array
    Returns:
        H: 3x3 homography matrisi veya None
    """
    valid_indices = [i for i in range(len(keypoints_xy))
                     if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0]

    if len(valid_indices) < 4:
        return None

    src = np.array([keypoints_xy[i] for i in valid_indices], dtype=np.float32)
    dst = np.array([TACTICAL_KEYPOINTS[i] for i in valid_indices], dtype=np.float32)

    try:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None
        return H
    except cv2.error:
        return None


def draw_keypoints_on_frame(frame, keypoints_xy, confidence=None):
    """Tespit edilen keypoint'leri frame üzerine çiz."""
    for i in range(len(keypoints_xy)):
        x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
        if x <= 0 and y <= 0:
            continue

        color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
        name = KP_NAMES[i] if i < len(KP_NAMES) else str(i)

        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

        # Confidence varsa göster
        if confidence is not None and i < len(confidence):
            label = f"{name} ({confidence[i]:.2f})"
        else:
            label = name
        cv2.putText(frame, label, (x + 12, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def draw_tactical_view(court_img, keypoints_xy, H, player_feet=None, player_jerseys=None):
    """
    Tactical view oluştur: saha imgesi + keypoint'ler + oyuncu pozisyonları.
    Keypoint'ler beklenen pozisyonlarına yakınsa snap edilir.
    """
    tactical = court_img.copy()
    SNAP_THRESHOLD = 60

    if H is not None:
        # Keypoint'leri tactical view'a dönüştür
        valid_indices = [i for i in range(len(keypoints_xy))
                         if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0]

        for i in valid_indices:
            src_pt = np.array([[keypoints_xy[i]]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, H)
            tx, ty = dst_pt[0][0][0], dst_pt[0][0][1]

            expected_x, expected_y = TACTICAL_KEYPOINTS[i]
            tact_dist = np.sqrt((tx - expected_x) ** 2 + (ty - expected_y) ** 2)
            if tact_dist <= SNAP_THRESHOLD:
                tx, ty = int(expected_x), int(expected_y)
            else:
                tx, ty = int(tx), int(ty)

            if 0 <= tx <= TACTICAL_WIDTH and 0 <= ty <= TACTICAL_HEIGHT:
                draw_x = min(tx, TACTICAL_WIDTH - 1)
                draw_y = min(ty, TACTICAL_HEIGHT - 1)
                color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
                cv2.circle(tactical, (draw_x, draw_y), 5, color, -1)
                cv2.circle(tactical, (draw_x, draw_y), 6, (255, 255, 255), 1)

        # Oyuncu pozisyonlarını tactical view'a çiz
        if player_feet is not None and len(player_feet) > 0:
            for idx, foot_pt in enumerate(player_feet):
                src_pt = np.array([[foot_pt]], dtype=np.float32)
                try:
                    dst_pt = cv2.perspectiveTransform(src_pt, H)
                    px, py = int(dst_pt[0][0][0]), int(dst_pt[0][0][1])
                    if 0 <= px < TACTICAL_WIDTH and 0 <= py < TACTICAL_HEIGHT:
                        cv2.circle(tactical, (px, py), 6, (255, 255, 255), -1)
                        cv2.circle(tactical, (px, py), 7, (0, 0, 0), 2)
                        # Jersey numarası varsa göster
                        if player_jerseys and idx < len(player_jerseys) and player_jerseys[idx]:
                            cv2.putText(tactical, player_jerseys[idx],
                                        (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35, (255, 255, 255), 1)
                except:
                    pass

    return tactical


def run_pipeline(video_path, save_output=False, output_path=None, max_frames=0):
    """Ana pipeline."""

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        print("   Lütfen modeli indirin:")
        print("   gdown '1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0' -O basketball_analysis/models/court_keypoint_detector.pt")
        return

    # --- Model Yükle ---
    print("📦 Court Keypoint modeli yükleniyor...")
    model = YOLO(MODEL_PATH)
    print("✅ Keypoint modeli yüklendi!")

    # GroundedSAM Pipeline (YOLO + SAM2 + Jersey)
    print("📦 GroundedSAM Pipeline yükleniyor...")
    pipeline = GroundedSAM(device="cuda")

    # Court image (tactical view arka planı)
    if os.path.exists(COURT_IMAGE_PATH):
        court_img_orig = cv2.imread(COURT_IMAGE_PATH)
        court_img_orig = cv2.resize(court_img_orig, (TACTICAL_WIDTH, TACTICAL_HEIGHT))
    else:
        court_img_orig = np.ones((TACTICAL_HEIGHT, TACTICAL_WIDTH, 3), dtype=np.uint8) * 40
        # Basit saha çizgileri
        cv2.rectangle(court_img_orig, (0, 0), (TACTICAL_WIDTH - 1, TACTICAL_HEIGHT - 1), (255, 255, 255), 1)
        cv2.line(court_img_orig, (TACTICAL_WIDTH // 2, 0), (TACTICAL_WIDTH // 2, TACTICAL_HEIGHT), (255, 255, 255), 1)

    # --- Video Aç ---
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

    # --- Video Writer ---
    out_width = frame_w
    out_height = frame_h

    out_writer = None
    if save_output:
        if output_path is None:
            os.makedirs(os.path.join(ROOT_DIR, "videos", "output"), exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(ROOT_DIR, "videos", "output", f"{base_name}_court_kp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        print(f"💾 Çıktı kaydedilecek: {output_path}")

    print(f"\n🚀 İşlem başlıyor... (Çıkmak için 'q' tuşuna basın)\n")

    frame_idx = 0
    success_count = 0
    last_H = None
    last_good_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # 1. YOLO ile keypoint tespiti
        results = model.predict(frame, conf=0.5, verbose=False)

        keypoints_xy = np.zeros((18, 2), dtype=np.float32)
        confidences = np.zeros(18, dtype=np.float32)

        if results and results[0].keypoints is not None:
            kp_data = results[0].keypoints
            if kp_data.xy is not None and len(kp_data.xy) > 0:
                xy = kp_data.xy[0].cpu().numpy()  # (N, 2)
                if len(xy) <= 18:
                    keypoints_xy[:len(xy)] = xy
                else:
                    keypoints_xy = xy[:18]

                # Confidence (varsa)
                if kp_data.conf is not None and len(kp_data.conf) > 0:
                    conf = kp_data.conf[0].cpu().numpy()
                    n = min(len(conf), 18)
                    confidences[:n] = conf[:n]

        # 2. Akıllı keypoint düzeltme
        SMOOTH_ALPHA = 0.6
        JUMP_THRESHOLD = 150

        # Kamera geçişi algılama
        is_camera_transition = False
        if last_good_keypoints is not None:
            jump_count = 0
            valid_count = 0
            for i in range(18):
                if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0:
                    if last_good_keypoints[i][0] > 0 and last_good_keypoints[i][1] > 0:
                        valid_count += 1
                        dist = np.sqrt((keypoints_xy[i][0] - last_good_keypoints[i][0]) ** 2 +
                                       (keypoints_xy[i][1] - last_good_keypoints[i][1]) ** 2)
                        if dist > JUMP_THRESHOLD:
                            jump_count += 1
            is_camera_transition = (valid_count > 0 and jump_count / valid_count > 0.5)

        if is_camera_transition:
            # Kamera geçişi → geçmiş verileri sıfırla, ham tespitleri kullan
            last_good_keypoints = None

        # İlk aşama: yüksek güvenli keypoint'lerden ön homography hesapla
        high_conf_kp = np.zeros((18, 2), dtype=np.float32)
        high_conf_count = 0
        for i in range(18):
            if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0 and confidences[i] >= 0.5:
                high_conf_kp[i] = keypoints_xy[i]
                high_conf_count += 1

        pre_H = compute_homography(high_conf_kp) if high_conf_count >= 4 else None

        # İkinci aşama: her keypoint'i doğrula
        # Eğer ön homography varsa, keypoint'i tactical view'a project et
        # ve beklenen taktik pozisyonundan çok uzaksa yanlış tespit → sıfırla
        for i in range(18):
            if keypoints_xy[i][0] <= 0 or keypoints_xy[i][1] <= 0:
                continue

            if pre_H is not None:
                # Bu keypoint tactical view'da nereye düşüyor?
                src_pt = np.array([[keypoints_xy[i]]], dtype=np.float32)
                dst_pt = cv2.perspectiveTransform(src_pt, pre_H)
                tx, ty = dst_pt[0][0][0], dst_pt[0][0][1]

                # Beklenen taktik pozisyon
                expected_x, expected_y = TACTICAL_KEYPOINTS[i]

                # Beklenen pozisyondan çok uzak mı?
                tact_dist = np.sqrt((tx - expected_x) ** 2 + (ty - expected_y) ** 2)
                if tact_dist > 50:  # Tactical view'da 50px'ten fazla sapma → yanlış tespit
                    keypoints_xy[i] = [0.0, 0.0]
                    continue

        # Üçüncü aşama: temporal smoothing (sadece valid keypoint'ler için)
        if last_good_keypoints is not None:
            for i in range(18):
                has_detection = keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0
                has_history = last_good_keypoints[i][0] > 0 and last_good_keypoints[i][1] > 0

                if has_detection and has_history:
                    # EMA smoothing
                    keypoints_xy[i] = (SMOOTH_ALPHA * keypoints_xy[i] +
                                       (1 - SMOOTH_ALPHA) * last_good_keypoints[i])
                elif not has_detection and has_history:
                    # Tespit yok → son iyi pozisyonu tut (ama sadece birkaç frame)
                    keypoints_xy[i] = last_good_keypoints[i].copy()

        last_good_keypoints = keypoints_xy.copy()

        # 3. Homography hesabı
        H = compute_homography(keypoints_xy)

        if H is not None:
            success_count += 1
            last_H = H
        else:
            H = last_H  # Son başarılı homography'yi kullan

        # 4. GroundedSAM Pipeline (detection + tracking + segmentation + jersey)
        results = pipeline.segment(frame)
        player_feet = pipeline.get_player_feet(results)
        player_jerseys = pipeline.get_player_jerseys(results)

        # 5. Görselleştirme
        frame_vis = pipeline.draw_results(frame, results)

        # Keypoint'leri çiz
        n_detected = sum(1 for i in range(18) if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0)
        frame_vis = draw_keypoints_on_frame(frame_vis, keypoints_xy, confidences)

        # Durum bilgisi
        status_color = (0, 255, 0) if H is not None else (0, 0, 255)
        cv2.putText(frame_vis, f"Keypoints: {n_detected}/18 | Players: {len(player_feet)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_vis, f"Homography: {'OK' if H is not None else 'FAIL'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame_vis, f"Frame: {frame_idx}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Tactical view (keypoint'ler + oyuncu pozisyonları)
        tactical = draw_tactical_view(court_img_orig, keypoints_xy, H, player_feet, player_jerseys)
        tactical_display = cv2.resize(tactical, (600, int(TACTICAL_HEIGHT * (600 / TACTICAL_WIDTH))))

        # Göster — ayrı pencereler
        display = frame_vis
        max_display_w = 1280
        if frame_vis.shape[1] > max_display_w:
            scale = max_display_w / frame_vis.shape[1]
            display = cv2.resize(frame_vis, (max_display_w, int(frame_vis.shape[0] * scale)))

        cv2.imshow("Court Keypoint Detection", display)
        cv2.imshow("Tactical View", tactical_display)

        if out_writer is not None:
            out_writer.write(frame_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹ Kullanıcı tarafından durduruldu.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"   ✅ {frame_idx} frame | Homography: {success_count}/{frame_idx} başarılı")

    # --- Temizlik ---
    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"\n💾 Çıktı kaydedildi: {output_path}")
    cv2.destroyAllWindows()

    rate = (success_count / frame_idx * 100) if frame_idx > 0 else 0
    print(f"\n🏁 Tamamlandı! {frame_idx} frame | Homography başarı: {success_count}/{frame_idx} ({rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Court Keypoint Detection (basketball_analysis)")
    parser.add_argument("--video", type=str,
                        default=os.path.join(ROOT_DIR, "videos", "input", "court2.mp4"),
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
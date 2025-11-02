from ultralytics import YOLO
import cv2
import os
import time
# -------------------------------
# 1️⃣ MODELİ YÜKLE
# -------------------------------
model_path = r'C:\Users\524ha\Desktop\Resources\BasketballGameTracker\models\yolo\best_seg.pt'
model = YOLO(model_path)

# -------------------------------
# 2️⃣ VİDEOYU YÜKLE
# -------------------------------
video_path = r'C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\court2.mp4'  # KENDİ VİDEONUZUN ADINI YAZIN

cap = cv2.VideoCapture(video_path)

# -------------------------------
# 3️⃣ FPS HESAPLAMA
# -------------------------------
prev_time = 0
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Boyut küçült (hız için)
    frame_resized = cv2.resize(frame, (1280, 720))

    # Zaman ölç
    start = time.time()

    # Tahmin (tek kare)
    results = model(frame_resized, conf=0.5, verbose=False, device=0)
    annotated_frame = results[0].plot()

    # FPS hesapla
    fps = 1 / (time.time() - start)
    fps_display = 0.9 * fps_display + 0.1 * fps  # smooth fps

    # FPS ekranda göster
    cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("YOLOv8 Segmentation (Optimized)", annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

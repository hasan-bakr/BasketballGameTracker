from ultralytics import YOLO
import cv2
import os
import time

# -------------------------------
# 1️⃣ MODELİ YÜKLE
# -------------------------------
model_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\models\yolo\best_seg.pt"
model = YOLO(model_path)

# -------------------------------
# 2️⃣ VİDEOYU YÜKLE VE HAZIRLA
# -------------------------------
video_path = r'C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\court2.mp4'  # KENDİ VİDEONUZUN ADINI YAZIN
output_video_path = r'C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\court2_segmentasyonlu.mp4' # KAYDEDİLECEK YOL VE İSİM

cap = cv2.VideoCapture(video_path)

# Orijinal video özelliklerini al
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Yeniden boyutlandırılmış kare boyutları
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FOURCC = cv2.VideoWriter_fourcc(*'MP4V') # Codec: MP4V yaygın olarak kullanılır

# Video Kaydediciyi (VideoWriter) Oluştur
out = cv2.VideoWriter(output_video_path, FOURCC, original_fps, (FRAME_WIDTH, FRAME_HEIGHT))


# -------------------------------
# 3️⃣ İŞLEM VE KAYIT DÖNGÜSÜ
# -------------------------------
prev_time = 0
fps_display = 0

print(f"İşlem başlıyor. Çıkış dosyası: {output_video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video sonu veya okuma hatası.")
        break

    # Boyut küçült (hız için)
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Zaman ölç
    start = time.time()

    # Tahmin (tek kare)
    results = model(frame_resized, conf=0.5, verbose=False, device=0)
    annotated_frame = results[0].plot() # YOLO sonuçlarının üzerine çizilmiş kare

    # FPS hesapla
    current_time = time.time()
    fps = 1 / (current_time - start)
    fps_display = 0.9 * fps_display + 0.1 * fps  # smooth fps

    # FPS ekranda göster
    cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 💾 KAREYİ KAYDET
    out.write(annotated_frame)

    # Görüntüyü göster
    cv2.imshow("YOLOv8 Segmentation (Optimized)", annotated_frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 4️⃣ KAYNAKLARI SERBEST BIRAK
# -------------------------------
cap.release()    # Giriş videosunu serbest bırak
out.release()    # Çıkış videosunu serbest bırak ve kaydetme işlemini bitir
cv2.destroyAllWindows()
print("İşlem tamamlandı. Video başarıyla kaydedildi.")
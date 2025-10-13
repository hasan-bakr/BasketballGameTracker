import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- AYARLAR ---
# İşlemek istediğiniz videonun ve çıktı videosunun yollarını belirtin
VIDEO_INPUT_PATH = "court.mp4"
VIDEO_OUTPUT_PATH = "output_video.mp4"
MODEL_PATH = "best_model.pth"

# Model ve transform ayarları eğitimdekiyle aynı olmalı
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- AYARLAR SONU ---

def process_video(model, device, video_path, output_path, transforms):
    """
    Bir videoyu kare kare işler, segmentasyon tahmini yapar ve sonuçları
    yeni bir videoya kaydeder.
    """
    # 1. Video okuyucuyu (capture) başlat
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: {video_path} açılamadı.")
        return

    # 2. Video yazıcıyı (writer) ayarla
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # MP4 formatı için 'mp4v' codec'ini kullanıyoruz
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Video işleniyor... Toplam Kare: {total_frames}, FPS: {fps}")

    model.eval() # Modeli değerlendirme moduna al

    # İlerleme çubuğu için tqdm kullan
    for _ in tqdm(range(total_frames), desc="Video İşleniyor"):
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Kareyi modelin anlayacağı formata çevir
        # OpenCV BGR okur, biz RGB'ye çeviriyoruz
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transformları uygula
        augmented = transforms(image=rgb_frame)
        input_tensor = augmented['image']
        
        # Batch boyutu ekle ve doğru cihaza gönder
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # 4. Tahmin yap
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            # Olasılıkları 0 veya 1'e çevir
            pred_mask = (pred_tensor > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        # 5. Maskeyi görselleştirmek için renklendir
        # Maskeyi orijinal boyutuna geri getir
        pred_mask_resized = cv2.resize(pred_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        
        # Renkli bir overlay oluştur (örneğin yeşil)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        # Sadece basketbol sahası olan pikselleri yeşil yap
        overlay[pred_mask_resized == 1] = [0, 255, 0] # Yeşil renk (BGR formatında)

        # 6. Orijinal kare ile maskeyi birleştir (blend)
        # 0.6 şeffaflık değeri ile maskeyi ekle
        output_frame = cv2.addWeighted(frame, 1, overlay, 0.6, 0)
        
        # 7. Sonucu ekranda göster ve dosyaya yaz
        cv2.imshow('Basketbol Sahası Segmentasyonu', output_frame)
        out.write(output_frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Her şeyi serbest bırak
    print(f"İşlem tamamlandı. Sonuç '{output_path}' dosyasına kaydedildi.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# --- ANA ÇALIŞTIRMA KISMI ---

if __name__ == "__main__":
    # Test için kullandığımız transformların aynısı
    video_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Modeli oluştur ve eğitilmiş ağırlıkları yükle
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    # Ana fonksiyonu çağır
    process_video(model, DEVICE, VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, video_transforms)
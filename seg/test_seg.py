import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
from collections import deque

# --- YENİ AYARLAR: DÜZELTME VE STABİLİZASYON İÇİN ---
# Zamansal Yumuşatma Ayarları
SMOOTHING_FRAMES = 5  # Ortalama için kullanılacak geçmiş kare sayısı. (Daha yüksek değer = daha stabil ama daha gecikmeli)
ALPHA = 0.6           # Mevcut karenin ağırlığı. (1'e yaklaştıkça mevcut kareye daha çok güvenir)

# Mekansal Düzeltme Ayarları
MIN_CONTOUR_AREA = 1000   # Bu piksel alanından küçük gürültüler temizlenecek.
APPROX_EPSILON = 0.001   # Sınırları ne kadar düzleştireceğimizi belirler. (Daha yüksek değer = daha fazla düzleştirme)
# --- YENİ AYARLAR SONU ---


# --- AYARLAR ---
VIDEO_INPUT_PATH = "./videos/input/court.mp4"
VIDEO_OUTPUT_PATH = "./videos/output/output_video_stabilized.mp4"
MODEL_PATH = "./models/seg/best_model (1).pth"

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
NUM_CLASSES = 4 # 1 Arka Plan + 3 Saha Sınıfı
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_COLORS = [
    [0, 0, 0],       # 0: Arka Plan
    [255, 0, 0],     # 1: Üç Sayı Bölgesi - Mavi
    [0, 255, 0],     # 2: İki Sayı Bölgesi - Yeşil
    [0, 0, 255],     # 3: Üç Saniye Alanı - Kırmızı
]
# --- AYARLAR SONU ---

# ⭐️ --- YENİ: Zamansal Yumuşatma için Yardımcı Sınıf --- ⭐️
class TemporalSmoother:
    def __init__(self, buffer_size, alpha):
        self.buffer = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.last_smoothed_logits = None

    def smooth(self, current_logits):
        if self.last_smoothed_logits is None:
            # İlk karede, sadece mevcut logitleri kullan
            smoothed = current_logits
        else:
            # Üstel hareketli ortalama (Exponential Moving Average)
            smoothed = self.alpha * current_logits + (1 - self.alpha) * self.last_smoothed_logits
        
        self.last_smoothed_logits = smoothed
        return smoothed

# ⭐️ --- YENİ: Mekansal Düzeltme için Yardımcı Fonksiyon --- ⭐️
def post_process_mask(mask):
    """
    Maskedeki gürültüyü temizler ve sınırları düzleştirir.
    """
    processed_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Her bir sınıf için ayrı ayrı işlem yap (arka plan hariç)
    for class_idx in range(1, NUM_CLASSES):
        # Sadece mevcut sınıfa ait pikselleri içeren bir binary maske oluştur
        class_mask = np.where(mask == class_idx, 255, 0).astype(np.uint8)
        
        # Bu maskedeki konturları (sınırları) bul
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Çok küçük alanları (gürültüyü) filtrele
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                # Konturun çevresini hesapla
                perimeter = cv2.arcLength(contour, True)
                # Konturu daha az noktadan oluşan bir poligona yaklaştırarak düzleştir
                approximated_poly = cv2.approxPolyDP(contour, APPROX_EPSILON * perimeter, True)
                
                # Düzleştirilmiş bu poligonu, ilgili sınıf rengiyle ana maskeye çiz
                cv2.drawContours(processed_mask, [approximated_poly], -1, (class_idx), thickness=cv2.FILLED)
                
    return processed_mask

def process_video(model, device, video_path, output_path, transforms, colors):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: {video_path} açılamadı.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Yumuşatıcıyı başlat
    smoother = TemporalSmoother(buffer_size=SMOOTHING_FRAMES, alpha=ALPHA)

    print(f"Video işleniyor... Çıkmak için 'q' tuşuna basın.")
    model.eval()

    for _ in tqdm(range(total_frames), desc="Video İşleniyor"):
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = transforms(image=rgb_frame)
        input_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 1. Ham Logit'leri al (argmax'tan önceki olasılık benzeri çıktılar)
            raw_logits = model(input_tensor)
            
            # 2. ZAMANSAL YUMUŞATMA
            smoothed_logits = smoother.smooth(raw_logits)
            
            # 3. Yumuşatılmış logit'lerden nihai kararı (maskeyi) oluştur
            pred_mask = torch.argmax(smoothed_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # 4. MEKANSAL DÜZELTME (Post-processing)
        clean_mask = post_process_mask(pred_mask)
        
        # Oluşturulan temiz maskeyi orijinal video boyutuna getir
        clean_mask_resized = cv2.resize(clean_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        
        # Renkli katmanı oluştur
        rgb_mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        for class_idx in range(1, NUM_CLASSES):
            rgb_mask_overlay[clean_mask_resized == class_idx] = colors[class_idx]

        # Sonuçları birleştir ve kaydet
        output_frame = cv2.addWeighted(frame, 1, rgb_mask_overlay, 0.7, 0)
        
        cv2.imshow('Stabil Basketbol Sahası Segmentasyonu', output_frame)
        out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"İşlem tamamlandı. Sonuç '{output_path}' dosyasına kaydedildi.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
# --- ANA ÇALIŞTIRMA KISMI ---
if __name__ == "__main__":
    video_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)

    process_video(model, DEVICE, VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, video_transforms, CLASS_COLORS)
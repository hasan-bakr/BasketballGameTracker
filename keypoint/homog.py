import cv2
import numpy as np

# --- 1. Adım: Görüntüyü Yükle ve Kullanıcıdan 4 Köşe Noktasını Al ---

# Video dosyasının yolunu kendi dosya yolunuzla güncelleyin
cap = cv2.VideoCapture(r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\court.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Video karesi okunamadı!")
    exit()

frame_display = frame.copy()
points = []

def click_event(event, x, y, flags, param):
    """Mouse tıklamalarını yakalayan ve noktaları saklayan fonksiyon."""
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(frame_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_display, str(len(points)), (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Saha Köşe Seçimi", frame_display)
        if len(points) == 4:
            print("Seçilen 4 köşe noktası:", points)

print("Lütfen sahanın 4 köşesini SAAT YÖNÜNDE veya TERSİNDE tıklayın.")
print("Önerilen Sıra: Sol Alt, Sol Üst, Sağ Üst, Sağ Alt")
cv2.imshow("Saha Köşe Seçimi", frame_display)
cv2.setMouseCallback("Saha Köşe Seçimi", click_event)

while len(points) < 4:
    if cv2.waitKey(1) & 0xFF == 27:
        print("İşlem kullanıcı tarafından iptal edildi.")
        cv2.destroyAllWindows()
        exit()
cv2.destroyAllWindows()

# --- 2. Adım: Standart NBA Sahası Boyutlarını ve Köşelerini Tanımla (FEET cinsinden) ---

# NBA saha boyutları: 94 feet x 50 feet
# Kolay çalışmak için ölçek: 1 foot = 10 piksel
STD_WIDTH = 940  # 94ft * 10
STD_HEIGHT = 500 # 50ft * 10

# Standart (tepeden görünüm) sahanın köşe noktaları
pts_std = np.array([
    [0, STD_HEIGHT],          # Sol Alt
    [0, 0],                   # Sol Üst
    [STD_WIDTH, 0],           # Sağ Üst
    [STD_WIDTH, STD_HEIGHT]   # Sağ Alt
], dtype=np.float32)

pts_src = np.array(points, dtype=np.float32)

# --- 3. Adım: Homografi Matrisini Hesapla ---
H, _ = cv2.findHomography(pts_std, pts_src)

# --- 4. Adım: Standart NBA Sahası Üzerindeki Kilit Noktaları Tanımla ---
# Tüm ölçüler feet cinsinden alınıp 10 ile çarpılarak piksele dönüştürülüyor
keypoints_std = {}
SCALE = 10

# Saha sınırları
keypoints_std['sidelines_and_baselines'] = np.array([
    [0, 0], [STD_WIDTH, 0], [STD_WIDTH, STD_HEIGHT], [0, STD_HEIGHT]
])

# Orta Çizgi
mid_x = STD_WIDTH / 2
keypoints_std['center_line'] = np.array([[mid_x, 0], [mid_x, STD_HEIGHT]])

# Sol Pota ve Çizgileri
hoop_from_baseline = 4 * SCALE         # Pota dip çizgiden 4 feet içeride
ft_from_baseline = 19 * SCALE          # Serbest atış çizgisi 19 feet
three_pt_radius = 23.75 * SCALE        # 3 sayı çizgisi yarıçapı 23 feet 9 inches (23.75)
key_width = 16 * SCALE                 # Boyalı alan genişliği 16 feet

# Sol potanın merkezi (yere izdüşümü)
left_hoop_center_x = hoop_from_baseline
mid_y = STD_HEIGHT / 2

# Sol serbest atış çizgisi ve boyalı alan (key)
keypoints_std['left_key_box'] = np.array([
    [0, mid_y - key_width/2],
    [ft_from_baseline, mid_y - key_width/2],
    [ft_from_baseline, mid_y + key_width/2],
    [0, mid_y + key_width/2]
])

# Sol 3 sayı çizgisi için noktalar oluştur (yarım daire)
arc_points = []
# NBA'de 3 sayı çizgisi yanlarda düz başlar. Pota merkezinden 22 feet (220 piksel) uzaklıkta.
three_pt_sideline_dist = 22 * SCALE
# y=0'dan hangi x değerinde düz çizginin bittiğini ve yayın başladığını bulalım
# (x-h)^2 + (y-k)^2 = r^2 -> (x-40)^2 + (y-250)^2 = 237.5^2
# y=0 için (x-40)^2 = 237.5^2 - 250^2 -> bu gerçek bir çözüm vermez, çünkü yay yan çizgiye ulaşmaz.
# Yanlardan 3 feet düz çizgi vardır.
corner_three_start_x = 3 * SCALE
keypoints_std['left_3pt_corner_bottom'] = np.array([[corner_three_start_x, STD_HEIGHT], [0, STD_HEIGHT]])
keypoints_std['left_3pt_corner_top'] = np.array([[corner_three_start_x, 0], [0, 0]])


# Daha basit bir yaklaşımla, sadece yayı çizelim.
# Not: NBA 3 sayı çizgisi yanlarda düzdür. Bu kod şimdilik sadece yayı çiziyor.
for angle in range(-90, 91, 5): 
    rad = np.deg2rad(angle)
    x = left_hoop_center_x + three_pt_radius * np.cos(rad)
    y = mid_y + three_pt_radius * np.sin(rad)
    if 0 <= x <= STD_WIDTH and 0 <= y <= STD_HEIGHT:
        arc_points.append([x, y])
keypoints_std['left_3pt_arc'] = np.array(arc_points)

# --- 5. Adım: Kilit Noktaları Dönüştür ve Orijinal Kare Üzerine Çiz ---

output_frame = frame.copy()

def draw_points(image, points_set, H, color=(255, 255, 0), thickness=2, is_closed=False):
    if len(points_set) == 0: return
    points_reshaped = points_set.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(points_reshaped, H)
    transformed_points = transformed_points.reshape(-1, 2).astype(np.int32)
    cv2.polylines(image, [transformed_points], isClosed=is_closed, color=color, thickness=thickness)

# Ana Sınırları Sarı Çiz
draw_points(output_frame, keypoints_std['sidelines_and_baselines'], H, color=(0, 255, 255), thickness=3, is_closed=True)
# Orta Çizgiyi Beyaz Çiz
draw_points(output_frame, keypoints_std['center_line'], H, color=(255, 255, 255), thickness=2)
# Boyalı Alanı Kırmızı Çiz
draw_points(output_frame, keypoints_std['left_key_box'], H, color=(0, 0, 255), thickness=2)
# 3 Sayı Yayını Kırmızı Çiz
draw_points(output_frame, keypoints_std['left_3pt_arc'], H, color=(0, 0, 255), thickness=3)

# Sonucu göster
cv2.imshow("NBA Saha Çizgileri", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
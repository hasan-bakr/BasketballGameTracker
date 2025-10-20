import os
import shutil

# Orijinal verilerin bulunduğu klasör
source_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\data\key_unready\train\labels"
output_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\data\key_ready"

# Alt klasörleri oluştur
images_dir = os.path.join(output_dir, "train", "images")
labels_dir = os.path.join(output_dir, "train", "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Tüm txt dosyalarını tara
for file in os.listdir(source_dir):
    if not file.endswith(".txt"):
        continue

    txt_path = os.path.join(source_dir, file)

    # Aynı adlı resim var mı kontrol et (.jpg veya .png)
    base = os.path.splitext(file)[0]
    for ext in [".jpg", ".png", ".jpeg"]:
        img_path = os.path.join(source_dir, base + ext)
        if os.path.exists(img_path):
            shutil.copy(img_path, images_dir)
            break
    else:
        print(f"⚠️ Görsel bulunamadı: {base}")
        continue

    # Etiket dosyasını kontrol et ve yeniden biçimlendir
    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.read().strip()

    # Parçala ve temizle
    parts = line.split()
    if len(parts) < 6:
        print(f"⚠️ Eksik veri: {file}")
        continue

    cls = parts[0]
    bbox_kpts = [float(x) for x in parts[1:]]
    formatted = [cls] + [f"{x:.6f}" for x in bbox_kpts]

    # Yeni dosyayı kaydet
    new_label_path = os.path.join(labels_dir, file)
    with open(new_label_path, "w") as f:
        f.write(" ".join(formatted) + "\n")

print("✅ YOLOv8-Pose veri seti yapısı oluşturuldu!")
print(f"📁 Klasör: {output_dir}")

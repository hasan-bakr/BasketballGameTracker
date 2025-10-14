import os
import random
import shutil
from tqdm import tqdm

# --- AYARLAR ---

# Girdi: Birleştirilecek ana veri seti klasörlerinin listesi
SOURCE_DIRS = ['courtv2', 'courtv2-2', 'courtv2-3']

SOURCE_DIRS = [os.path.join("data", d) for d in SOURCE_DIRS]
# Çıktı: Birleştirilmiş veri setinin oluşturulacağı ana klasör
OUTPUT_DIR = './data/courtv2-Final'

# Bölme oranları (toplamları 1.0 olmalı)
TRAIN_RATIO = 0.89
VALID_RATIO = 0.10
TEST_RATIO = 0.01

# Karıştırma işleminin her seferinde aynı sonucu vermesi için (opsiyonel ama önerilir)
RANDOM_SEED = 42

# --- AYARLAR SONU ---

def find_all_file_pairs(source_dirs):
    """
    Tüm kaynak klasörlerini tarar ve eşleşen (resim, maske) çiftlerini bulur.
    """
    all_pairs = []
    print("Dosya çiftleri aranıyor...")
    
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Uyarı: '{source_dir}' klasörü bulunamadı, atlanıyor.")
            continue
            
        for split_type in ['train', 'valid', 'test']:
            image_folder = os.path.join(source_dir, split_type, 'images')
            mask_folder = os.path.join(source_dir, split_type, 'masks')

            if not os.path.isdir(image_folder):
                continue

            for image_name in os.listdir(image_folder):
                # Resim adından uzantıyı ayırarak maske adını tahmin et
                base_name, _ = os.path.splitext(image_name)
                mask_name = base_name + '.png' # Maskelerin .png olduğunu varsayıyoruz
                
                image_path = os.path.join(image_folder, image_name)
                mask_path = os.path.join(mask_folder, mask_name)

                # Maske dosyasının var olup olmadığını kontrol et
                if os.path.exists(mask_path):
                    all_pairs.append((image_path, mask_path))
                else:
                    print(f"Uyarı: '{image_path}' için eşleşen maske bulunamadı, atlanıyor.")
                    
    return all_pairs

def create_output_dirs(base_dir):
    """
    Gerekli çıktı klasör yapısını oluşturur.
    """
    print(f"'{base_dir}' içinde çıktı klasörleri oluşturuluyor...")
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'masks'), exist_ok=True)

def copy_files(pairs, split_name, base_dir):
    """
    Belirtilen çiftleri hedef klasörlere kopyalar.
    """
    image_dest_folder = os.path.join(base_dir, split_name, 'images')
    mask_dest_folder = os.path.join(base_dir, split_name, 'masks')
    
    for img_path, mask_path in tqdm(pairs, desc=f"'{split_name}' dosyaları kopyalanıyor"):
        shutil.copy(img_path, image_dest_folder)
        shutil.copy(mask_path, mask_dest_folder)


if __name__ == "__main__":
    # 1. Adım: Tüm (resim, maske) çiftlerini bul
    file_pairs = find_all_file_pairs(SOURCE_DIRS)
    
    if not file_pairs:
        print("İşlenecek hiçbir dosya çifti bulunamadı. Lütfen klasör yapınızı kontrol edin.")
        exit()
        
    total_count = len(file_pairs)
    print(f"Toplam {total_count} adet eşleşen (resim, maske) çifti bulundu.")

    # 2. Adım: Çiftleri karıştır
    print(f"Dosya çiftleri RANDOM_SEED={RANDOM_SEED} ile karıştırılıyor...")
    random.seed(RANDOM_SEED)
    random.shuffle(file_pairs)

    # 3. Adım: Bölme sayılarını hesapla
    train_count = int(total_count * TRAIN_RATIO)
    valid_count = int(total_count * VALID_RATIO)
    # Geriye kalan her şey test setine gider (yuvarlama hatalarını önler)
    test_count = total_count - train_count - valid_count

    # 4. Adım: Karıştırılmış listeyi böl
    train_pairs = file_pairs[:train_count]
    valid_pairs = file_pairs[train_count : train_count + valid_count]
    test_pairs = file_pairs[train_count + valid_count :]
    
    print("\n--- Veri Seti Bölme Planı ---")
    print(f"Train Seti: {len(train_pairs)} dosya")
    print(f"Valid Seti: {len(valid_pairs)} dosya")
    print(f"Test Seti : {len(test_pairs)} dosya")
    print("-----------------------------\n")

    # 5. Adım: Çıktı klasörlerini oluştur
    if os.path.exists(OUTPUT_DIR):
        print(f"Uyarı: '{OUTPUT_DIR}' klasörü zaten mevcut. İçeriği üzerine yazılabilir.")
    create_output_dirs(OUTPUT_DIR)

    # 6. Adım: Dosyaları yeni yerlerine kopyala
    copy_files(train_pairs, 'train', OUTPUT_DIR)
    copy_files(valid_pairs, 'valid', OUTPUT_DIR)
    copy_files(test_pairs, 'test', OUTPUT_DIR)

    print("\nİşlem başarıyla tamamlandı!")
    print(f"Tüm dosyalar birleştirildi, karıştırıldı ve '{OUTPUT_DIR}' klasörüne bölündü.")
# Basketball Game Tracker 🏀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Basketbol maçlarını analiz etmek, oyuncu/top takibi yapmak ve saha içi istatistikleri çıkarmak amacıyla geliştirilen yapay zeka tabanlı bir bilgisayarlı görü projesidir.

> 🚧 **Not:** Bu proje şu anda aktif geliştirme aşamasındadır. Özellikler ve kod yapısı değişiklik gösterebilir.

## 🎯 Proje Hakkında

Bu proje, ham maç görüntülerini işleyerek anlamlı veriler çıkarmayı hedefler. Derin öğrenme modelleri ve görüntü işleme teknikleri kullanılarak sahadaki nesneler algılanır ve konumlandırılır.

### Öne Çıkan Özellik: Saha Segmentasyonu (Court Segmentation)
Projenin şu anki en güçlü yeteneklerinden biri, oyun alanını videodan ayrıştırıp maskeleyebilmesidir. Bu özellik, oyuncuların saha içindeki gerçek konumlarını (homography kullanarak) 2D bir haritaya yansıtmak için temel oluşturur.



![Saha Segmentasyonu Demo](CourtSegmentation.gif)


## ✨ Özellikler (Mevcut ve Planlanan)

* [x] **Saha Segmentasyonu:** Oyun alanının tespiti ve maskelenmesi.
* [ ] **Nesne Tespiti (Object Detection):** Oyuncuların, hakemlerin ve topun tespiti (YOLO).
* [ ] **Takım Ayrıştırma:** Oyuncuların forma renklerine göre takımlara ayrılması.
* [ ] **Perspektif Dönüşümü:** Kamera görüntüsünün 2D taktik tahtasına dönüştürülmesi.
* [ ] **Hareket Takibi:** Oyuncuların hız ve kat ettikleri mesafenin hesaplanması.

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/hasan-bakr/BasketballGameTracker.git](https://github.com/hasan-bakr/BasketballGameTracker.git)
    cd BasketballGameTracker
    ```

2.  **Sanal ortam oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    # Mac/Linux için:
    source venv/bin/activate
    ```

3.  **Gereksinimleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Kullanım

Modeli kendi videonuz üzerinde test etmek için:

```bash
python main.py --source "video_dosyasi.mp4"

BasketballGameTracker/
├── data/               # Test videoları ve görseller
├── models/             # Eğitilmiş YOLO ve segmentasyon modelleri
├── src/                # Kaynak kodlar
│   ├── tracker.py      # Takip algoritmaları
│   ├── segmentation.py # Saha segmentasyonu modülü
│   └── utils.py        # Yardımcı fonksiyonlar
├── main.py             # Ana çalıştırma dosyası
└── README.md
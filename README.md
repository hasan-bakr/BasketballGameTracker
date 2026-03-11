# Basketball Game Tracker 🏀

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![SAM2](https://img.shields.io/badge/Model-SAM2-purple)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-orange)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

Basketbol maçlarını analiz etmek, oyuncu/top takibi yapmak, forma numaralarını okumak ve saha içi pozisyonları kuşbakışı (tactical view) haritaya yansıtmak amacıyla geliştirilen yapay zeka tabanlı gelişmiş bir bilgisayarlı görü projesidir.

## 🎯 Proje Hakkında

Bu proje, ham maç görüntülerini işleyerek **Segment Anything Model 2 (SAM 2)** ve **YOLO** mimarilerinin gücüyle oyuncuları piksel hassasiyetinde takip eder. Gelişmiş Re-ID (Yeniden Tanımlama) algoritmaları ve Court Keypoint tespitiyle, sahadaki olayları 2 boyutlu taktiksel bir düzleme aktarır.

### Yeni ve Öne Çıkan Özellikler

* 🎯 **Robust SAM2 Tracking:** Bounding box yerine piksel bazlı oyuncu segmentasyonu ve hafıza (memory bank) destekli video yayılımı (propagation).
* 🔢 **Jersey OCR & IoU Re-ID:** YOLO ile forma numaralarının tespiti, **PARSeq OCR** ile metne dönüştürülmesi ve geçici olarak kaybedilen oyuncuların aynı ID ile tekrar tanınması.
* 🏟️ **Court Keypoint Detection:** Saha üzerindeki 18 stratejik referans noktasının özel bir model ile anlık kameradan tespiti.
* 🗺️ **Tactical View (Bird's-Eye Projection):** Tespit edilen noktalar üzerinden **Homografi (Homography)** hesaplanarak kamera açısındaki oyuncuların 2D taktik haritasına (Mini-Map) gerçek zamanlı yansıtılması.
* ⚡ **Performance:** Automatic Mixed Precision (AMP) ve TensorRT/ONNX optimizasyon yetenekleriyle hızlandırılmış işlem hattı.

---

### Eskiden Gelen Özellik: Saha Segmentasyonu (Court Segmentation)
Oyun alanının videodan ayrıştırıp maskelenmesi.

![Saha Segmentasyonu Demo](CourtSegmentation.gif)

---

## ✨ Özellikler (Mevcut Durum)

* [x] **SAM2 Segmentation:** Oyun alanı ve oyuncuların piksel bazlı maskelenmesi.
* [x] **Nesne Tespiti (YOLO):** Oyuncuların ve formaların tespiti.
* [x] **Takım / ID Ayrıştırma:** Forma numarası bazlı (Re-ID) oyuncu takibi.
* [x] **Perspektif Dönüşümü (Homography):** Kamera görüntüsünün 2D taktik tahtasına dönüştürülmesi.
* [ ] **Hareket Takibi & İstatistik:** Oyuncuların hız ve kat ettikleri mesafenin hesaplanması (Geliştirilecek).

## 🛠️ Kurulum

Proje ana olarak `dsci` adlı bir Conda ortamı (Python 3.12, PyTorch 2.10, CUDA 13) hedeflenerek geliştirilmiştir.

1.  **Repoyu klonlayın:**
    ```bash
    git clone https://github.com/hasan-bakr/BasketballGameTracker.git
    cd BasketballGameTracker
    ```

2.  **Conda Ortamını Kurun ve Aktifleştirin:**
    ```bash
    conda create -n dsci python=3.12
    conda activate dsci
    ```

3.  **Gereksinimleri yükleyin:**
    Gereken ana kütüphaneler: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `sam2` (Facebook Research), `scipy`, `tensorrt`.

## 🚀 Kullanım

Yeni nesil takip ve taktik haritası sistemini ana script üzerinden test edebilirsiniz:

```bash
python APP/helpers/robust_sam2_tracker.py
```
> Çıktılar (hem ana video maskelemesi hem de taktiksel harita) `videos/output/` dizinine `.mp4` olarak kaydedilecektir.

## 📂 Proje Yapısı

\`\`\`text
BasketballGameTracker/
├── APP/
│   ├── helpers/                 # Ana Tracker, YOLO, Jersey OCR ve SAM2 scriptleri
│   │   ├── robust_sam2_tracker.py   # 🌟 ANA ÇALIŞTIRMA DOSYASI
│   │   ├── yolo_detector.py
│   │   ├── jersey_detector.py
│   │   └── sam_helper.py
├── models/                      # Eğitilmiş YOLO, SAM2 ve Keypoint modelleri
├── videos/                      # Girdi (input) ve Çıktı (output) videoları
├── basketball_analysis/         # Eski/Referans yapılar ve Notebook'lar
├── .vscode/                     # Düzenleyici ayarları (Jedi/Pylance config)
├── .gitignore                   # Git yapılandırması
└── README.md
\`\`\`
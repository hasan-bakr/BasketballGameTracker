"""
Jersey Detector Module
======================
PARSeq OCR modeli ile jersey numarası tanıma.
"""

import sys
from pathlib import Path

# Proje kök dizinini ve parseq klasörünü sys.path'e ekle
project_root = Path(__file__).resolve().parent.parent.parent
parseq_root = project_root / "parseq"
helpers_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parseq_root))
sys.path.insert(0, str(helpers_dir))

import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

from config import Config


class JerseyDetector:
    """
    PARSeq modeli ile jersey numarası tanıma sınıfı.
    
    Sadece OCR işlevi görür - pose detection ayrı sınıfta.
    """
    
    def __init__(self, config: Config = None):
        """
        JerseyDetector'ı başlat.
        
        Args:
            config: Config nesnesi
        """
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = None
        self.img_transform = None
        self._load_model()
    
    def _load_model(self):
        """PARSeq modelini yükle."""
        try:
            self.model = torch.hub.load(
                'baudm/parseq', 'parseq', pretrained=True
            ).eval().to(self.device)
            
            # FP16 optimizasyonu
            if self.config.USE_HALF and self.device == 'cuda':
                self.model = self.model.half()
                print(f"✅ JerseyDetector modeli yüklendi ({self.device}, FP16)")
            else:
                print(f"✅ JerseyDetector modeli yüklendi ({self.device})")
            
            self.img_transform = SceneTextDataModule.get_transform(
                self.model.hparams.img_size
            )
        except Exception as e:
            print(f"❌ PARSeq model yüklenemedi: {e}")
            self.model = None
    
    def recognize_number(self, jersey_crop) -> tuple:
        """
        PARSeq ile jersey numarasını oku.
        
        Args:
            jersey_crop: PIL Image veya BGR NumPy array
            
        Returns:
            tuple: (number: str veya None, confidence: float)
        """
        if self.model is None:
            return None, 0.0
        
        if jersey_crop is None:
            return None, 0.0
        
        # NumPy array ise size kontrolü
        if hasattr(jersey_crop, 'size') and isinstance(jersey_crop.size, int) and jersey_crop.size == 0:
            return None, 0.0
        
        try:
            # PIL Image'e çevir
            if not isinstance(jersey_crop, Image.Image):
                # BGR -> RGB
                import cv2
                rgb_image = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2RGB)
                jersey_crop = Image.fromarray(rgb_image)
            
            # Transform: PIL -> Tensor
            tensor = self.img_transform(jersey_crop).unsqueeze(0).to(self.device)
            
            # PARSeq inference
            with torch.no_grad():
                logits = self.model(tensor)
                pred = logits.softmax(-1)
                labels, confidences = self.model.tokenizer.decode(pred)
            
            if not labels or not confidences:
                return None, 0.0
            
            label = labels[0]
            confidence = confidences[0].mean().item()
            
            # Sadece sayıları filtrele
            numbers_only = ''.join(filter(str.isdigit, label))
            
            # Jersey numarası 0-99 arası
            if numbers_only and len(numbers_only) <= 2:
                return numbers_only, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ OCR hatası: {e}")
            return None, 0.0
    
    def recognize_from_directory(self, directory_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> dict:
        """
        Klasördeki tüm görselleri tanı.
        
        Args:
            directory_path: Klasör yolu
            extensions: Dosya uzantıları
            
        Returns:
            dict: {dosya_adı: (numara, confidence)}
        """
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ Klasör bulunamadı: {directory_path}")
            return {}
        
        results = {}
        image_files = [f for f in directory.iterdir() if f.suffix.lower() in extensions]
        
        print(f"📁 {len(image_files)} görsel bulundu: {directory_path}")
        
        for img_path in sorted(image_files):
            try:
                img = Image.open(img_path).convert('RGB')
                number, confidence = self.recognize_number(img)
                results[img_path.name] = (number, confidence)
                
                status = f"✅ {number}" if number else "❌ Tanınamadı"
                print(f"  {img_path.name}: {status} (conf: {confidence:.4f})")
                
            except Exception as e:
                print(f"  {img_path.name}: ⚠️ Hata - {e}")
                results[img_path.name] = (None, 0.0)
        
        detected = sum(1 for v in results.values() if v[0] is not None)
        print(f"\n📊 Sonuç: {detected}/{len(results)} görsel tanındı")
        
        return results


# Test kodu
if __name__ == "__main__":
    import os
    detector = JerseyDetector()
    
    # Test klasörü
    results = detector.recognize_from_directory(
        os.path.join(str(project_root), "videos", "output", "jersey_crops")
    )
    print(results)
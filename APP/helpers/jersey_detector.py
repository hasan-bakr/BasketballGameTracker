"""
Jersey Number Detector with PARSeq OCR
=======================================
Oyuncu formalarındaki numaraları algılar ve Re-ID için kullanır.
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Dict

# PARSeq imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "parseq"))
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


class JerseyDetector:
    """
    PARSeq tabanlı forma numarası algılayıcı.
    
    Özellikler:
    - Oyuncu bbox'ından jersey bölgesini kırp
    - PARSeq ile OCR
    - Karakter → Sayı dönüşümü (O→0, I→1, vb.)
    """
    
    # Görsel benzerliği yüksek, güvenilir karakter → rakam dönüşümleri
    # Sadece OCR'ın neredeyse kesinlikle yanlış tanıyacağı karakterler
    CHAR_TO_DIGIT = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1', '|': '1',
        'Z': '2',
        'S': '5', 's': '5',
        'B': '8',
    }
    
    def __init__(
        self,
        model_id: str = "pretrained=parseq",
        device: str = "cuda",
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_id: PARSeq model identifier (e.g., "pretrained=parseq")
            device: cuda or cpu
            confidence_threshold: Minimum confidence for accepting OCR result
        """
        print(f"📦 Loading JerseyDetector with PARSeq...")
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load PARSeq model
        self.model = load_from_checkpoint(model_id).eval().to(device)
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
        
        print(f"✅ JerseyDetector Ready!")
    
    def detect_number(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Oyuncu bbox'ından forma numarasını algıla.
        
        Args:
            frame: BGR frame (numpy array)
            bbox: Player bounding box [x1, y1, x2, y2]
            mask: Optional player mask for better cropping
            
        Returns:
            Jersey number as string (e.g., "23") or None if not detected
        """
        # Jersey bölgesini kırp (üst gövde - bbox'ın üst yarısı)
        jersey_crop = self._extract_jersey_region(frame, bbox, mask)
        
        if jersey_crop is None or jersey_crop.size == 0:
            return None
        
        # OCR yap
        text, confidence = self._ocr(jersey_crop)
        
        if confidence < self.confidence_threshold:
            return None
        
        # Sayıya çevir
        number = self._text_to_number(text)
        
        return number
    
    def _extract_jersey_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Oyuncu bbox'ından jersey bölgesini kırp.
        Jersey genelde üst gövdede = bbox'ın üst %60'ı
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        
        # Sınırları kontrol et
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Jersey bölgesi: üst %35 - %70 arası (gövde)
        bbox_height = y2 - y1
        jersey_top = y1 + int(bbox_height * 0.20)  # Başın altı
        jersey_bottom = y1 + int(bbox_height * 0.65)  # Belin üstü
        
        # Crop
        jersey_crop = frame[jersey_top:jersey_bottom, x1:x2]
        
        return jersey_crop
    
    @torch.inference_mode()
    def _ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        PARSeq ile OCR yap.
        
        Returns:
            (text, confidence)
        """
        # BGR → RGB → PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Transform ve model input
        img_tensor = self.img_transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(img_tensor)
        probs = logits.softmax(-1)
        
        # Decode
        pred, prob = self.model.tokenizer.decode(probs)
        
        # Confidence = ortalama olasılık
        confidence = prob[0].mean().item() if len(prob) > 0 else 0.0
        
        return pred[0], confidence
    
    def _text_to_number(self, text: str) -> Optional[str]:
        """
        OCR metnini sayıya çevir.
        - Sadece rakamları al
        - Benzerliklere göre karakter → rakam dönüşümü
        """
        if not text:
            return None
        
        result = []
        
        for char in text:
            if char.isdigit():
                result.append(char)
            elif char in self.CHAR_TO_DIGIT:
                result.append(self.CHAR_TO_DIGIT[char])
            # Diğer karakterleri yoksay
        
        if not result:
            return None
        
        # Forma numaraları genelde 1-2 haneli
        number = ''.join(result)
        
        # 3+ haneli sonuçlar muhtemelen gürültü
        if len(number) > 2:
            # En olası 2 haneyi al (ortadaki karakterler genelde daha güvenilir)
            number = number[:2]
        
        return number if number else None
    
    def detect_batch(
        self,
        frame: np.ndarray,
        bboxes: list,
        masks: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, str]:
        """
        Birden fazla oyuncunun jersey numaralarını algıla.
        
        Args:
            frame: BGR frame
            bboxes: List of (obj_id, [x1, y1, x2, y2])
            masks: Optional dict of {obj_id: mask}
            
        Returns:
            {obj_id: jersey_number}
        """
        results = {}
        
        for obj_id, bbox in bboxes:
            mask = masks.get(obj_id) if masks else None
            number = self.detect_number(frame, bbox, mask)
            
            if number:
                results[obj_id] = number
        
        return results


class JerseyReIDBank:
    """
    Jersey numaralarına göre oyuncu Re-ID bankası.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # {obj_id: jersey_number}
        self.obj_to_jersey: Dict[int, str] = {}
        
        # {jersey_number: obj_id} - Ters mapping
        self.jersey_to_obj: Dict[str, int] = {}
        
        # Her jersey için tespit sayısı (güvenilirlik için)
        self.detection_counts: Dict[int, Dict[str, int]] = {}  # {obj_id: {number: count}}
    
    def register(self, obj_id: int, jersey_number: str):
        """
        Bir obj_id için jersey numarası kaydet.
        Çoklu tespitlerde en sık görülen numarayı kullan.
        """
        if obj_id not in self.detection_counts:
            self.detection_counts[obj_id] = {}
        
        counts = self.detection_counts[obj_id]
        counts[jersey_number] = counts.get(jersey_number, 0) + 1
        
        # En sık tespit edilen numarayı al
        best_number = max(counts, key=counts.get)
        
        # Eğer yeterli tespit varsa (en az 3 kez)
        if counts[best_number] >= 3:
            old_jersey = self.obj_to_jersey.get(obj_id)

            if old_jersey != best_number:
                # Bu numara zaten başka bir ID'ye kayıtlıysa, çakışmayı engelle
                existing_owner = self.jersey_to_obj.get(best_number)
                if existing_owner is not None and existing_owner != obj_id:
                    return

                # Eski eşleştirmeyi kaldır
                if old_jersey and self.jersey_to_obj.get(old_jersey) == obj_id:
                    del self.jersey_to_obj[old_jersey]

                # Yeni eşleştirmeyi kaydet
                self.obj_to_jersey[obj_id] = best_number
                self.jersey_to_obj[best_number] = obj_id

                if self.verbose:
                    print(f"   Jersey registered: ID {obj_id} → #{best_number}")
    
    def find_by_jersey(self, jersey_number: str) -> Optional[int]:
        """
        Jersey numarasına göre obj_id bul.
        """
        return self.jersey_to_obj.get(jersey_number)
    
    def get_jersey(self, obj_id: int) -> Optional[str]:
        """
        obj_id'nin jersey numarasını al.
        """
        return self.obj_to_jersey.get(obj_id)
    
    def get_all_mappings(self) -> Dict[int, str]:
        """
        Tüm eşleştirmeleri döndür.
        """
        return self.obj_to_jersey.copy()


if __name__ == "__main__":
    # Test
    print("Testing JerseyDetector...")
    
    detector = JerseyDetector()
    
    # Test image
    test_image = cv2.imread("demo_images/test_jersey.jpg")
    if test_image is not None:
        bbox = (100, 100, 300, 500)  # Example bbox
        number = detector.detect_number(test_image, bbox)
        print(f"Detected jersey number: {number}")
    else:
        print("No test image found. JerseyDetector initialized successfully.")

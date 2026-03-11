"""
RF-DETR ONNX Detector Class
TensorRT/CUDA destekli yüksek performanslı object detection
"""
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import List, Tuple, Optional, Union
import time


class RFDETRDetector:
    """RF-DETR modeli için ONNX Runtime tabanlı detector sınıfı."""
    
    # COCO sınıf isimleri
    COCO_CLASSES = {
        0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
        5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
        32: "sports ball", 37: "skateboard", 38: "surfboard", 39: "tennis racket"
    }
    
    def __init__(
        self,
        model_path: str,
        resolution: int = 576,
        use_tensorrt: bool = True,
    ):
        """
        RF-DETR Detector'ı başlatır.
        
        Args:
            model_path: ONNX model dosyasının yolu
            resolution: Model input çözünürlüğü (default: 576)
            use_tensorrt: TensorRT kullanılsın mı (default: True)
        """
        self.model_path = model_path
        self.resolution = resolution
        self.use_tensorrt = use_tensorrt
        
        # Provider'ları ayarla
        self.providers = self._setup_providers(use_tensorrt)
        
        # Session oluştur
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Input/Output bilgileri
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"✅ Model yüklendi: {model_path}")
        print(f"   Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"   Providers: {self.session.get_providers()}")
    
    def _setup_providers(
        self,
        use_tensorrt: bool,
    ) -> list:
        """Execution provider'ları ayarlar."""
        providers = []
        
        if use_tensorrt:
            providers.append(("TensorrtExecutionProvider"))
        
        providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Görseli model için hazırlar.
        
        Args:
            image: Görsel yolu (str) veya numpy array (BGR)
            
        Returns:
            Preprocessed numpy array [1, 3, H, W]
        """
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            # BGR -> RGB
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize
        img = img.resize((self.resolution, self.resolution))
        
        # Normalize ve transpose
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Batch ekle
        
        return img_array
    
    def postprocess(
        self,
        outputs: list,
        orig_size: Tuple[int, int],
        confidence_threshold: float = 0.5
    ) -> List[dict]:
        """
        Model çıktısını işler ve detection sonuçlarını döner.
        
        Args:
            outputs: session.run çıktısı [pred_boxes, pred_logits]
            orig_size: Orijinal görsel boyutu (width, height)
            confidence_threshold: Minimum güven skoru
            
        Returns:
            Detection listesi [{"bbox": [x1,y1,x2,y2], "class_id": int, "confidence": float, "class_name": str}]
        """
        pred_boxes = outputs[0][0]  # (300, 4) - [cx, cy, w, h] normalized
        pred_logits = outputs[1][0]  # (300, 91)
        
        # Softmax
        pred_probs = np.exp(pred_logits) / np.exp(pred_logits).sum(axis=1, keepdims=True)
        
        class_ids = np.argmax(pred_probs, axis=1)
        confidences = np.max(pred_probs, axis=1)
        
        orig_w, orig_h = orig_size
        detections = []
        
        for i in range(len(pred_boxes)):
            if confidences[i] > confidence_threshold and class_ids[i] != 0:
                cx, cy, w, h = pred_boxes[i]
                
                # Normalize -> Pixel koordinatları
                x1 = int((cx - w/2) * orig_w)
                y1 = int((cy - h/2) * orig_h)
                x2 = int((cx + w/2) * orig_w)
                y2 = int((cy + h/2) * orig_h)
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                    "class_name": self.COCO_CLASSES.get(class_ids[i], f"class_{class_ids[i]}")
                })
        
        return detections
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: float = 0.5
    ) -> List[dict]:
        """
        Görsel üzerinde object detection yapar.
        
        Args:
            image: Görsel yolu (str) veya numpy array (BGR)
            confidence_threshold: Minimum güven skoru
            
        Returns:
            Detection listesi
        """
        # Orijinal boyutu al
        if isinstance(image, str):
            img = Image.open(image)
            orig_size = img.size  # (width, height)
        else:
            orig_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Postprocess
        detections = self.postprocess(outputs, orig_size, confidence_threshold)
        
        return detections
    
    def detect_and_draw(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: float = 0.5,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Detection yapar ve sonuçları görsel üzerine çizer.
        
        Args:
            image: Görsel yolu (str) veya numpy array (BGR)
            confidence_threshold: Minimum güven skoru
            show: Görseli göster mi
            save_path: Kaydetme yolu (optional)
            
        Returns:
            Çizilmiş görsel (BGR numpy array)
        """
        # Görseli yükle
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Detection
        detections = self.detect(image, confidence_threshold)
        
        # Çiz
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Renk (class_id'ye göre)
            color = self._get_color(det["class_id"])
            
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label arka planı
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
            cv2.putText(img, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"💾 Kaydedildi: {save_path}")
        
        if show:
            import matplotlib.pyplot as plt
            # BGR -> RGB dönüşümü
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"RF-DETR Detections ({len(detections)} objects)")
            plt.tight_layout()
            plt.show()
        
        return img
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Class ID'ye göre renk döner (BGR)."""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def benchmark(self, image: Union[str, np.ndarray], num_runs: int = 50) -> dict:
        """
        Inference performansını ölçer.
        
        Args:
            image: Test görseli
            num_runs: Çalıştırma sayısı
            
        Returns:
            {"avg_time_ms": float, "fps": float}
        """
        input_data = self.preprocess(image)
        
        # Warm-up
        for _ in range(5):
            self.session.run(None, {self.input_name: input_data})
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: input_data})
        end = time.time()
        
        avg_time = (end - start) / num_runs * 1000
        fps = 1000 / avg_time
        
        return {"avg_time_ms": avg_time, "fps": fps}


# ============ KULLANIM ÖRNEĞİ ============
if __name__ == "__main__":
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Model yolu
    model_path = os.path.join(ROOT_DIR, "models", "rfdetr-medium.onnx")
    
    # Detector oluştur
    detector = RFDETRDetector(
        model_path=model_path,
        resolution=576,
        use_tensorrt=True  # CUDA EP kullan (PATH düzeltilince True yap)
    )
    
    # Test görseli
    image_path = os.path.join(ROOT_DIR, "videos", "output", "test_frame_4.jpg")
    
    # Benchmark
    print("\n📊 Benchmark yapılıyor...")
    results = detector.benchmark(image_path, num_runs=50)
    print(f"   Ortalama süre: {results['avg_time_ms']:.2f} ms")
    print(f"   FPS: {results['fps']:.1f}")
    
    # Detection
    print("\n🔍 Detection yapılıyor...")
    detections = detector.detect(image_path, confidence_threshold=0.5)
    print(f"   {len(detections)} nesne tespit edildi")
    
    for det in detections:
        print(f"   - {det['class_name']}: {det['confidence']:.2f}")
    
    # Görselleştir
    detector.detect_and_draw(image_path, confidence_threshold=0.65, show=True)
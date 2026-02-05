"""
Test Multiple Court Keypoint Models
====================================
Farklı modelleri karşılaştır, en iyi sonucu bul.
"""

from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import os

API_KEY = "lYJFYI6Qh2f8QOFW1i6e"

print("=" * 60)
print("Testing Multiple Court Keypoint Models")
print("=" * 60)

# Initialize client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Load test frame
VIDEO_PATH = "videos/input/court.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read video!")
    exit(1)

print(f"✅ Loaded frame: {frame.shape}")

# Models to test
MODELS = [
    "basketball-court-detection-2/4",
    "basketball-court-detection-2/3",
    "basketball-court-detection-2/2",
    "basketball-court-detection-2/1",
]

os.makedirs("model_comparison", exist_ok=True)

for model_id in MODELS:
    print(f"\n🔍 Testing: {model_id}")
    
    try:
        result = client.infer(frame, model_id=model_id)
        
        if 'predictions' in result and len(result['predictions']) > 0:
            pred = result['predictions'][0]
            
            if 'keypoints' in pred:
                kpts = pred['keypoints']
                print(f"   ✅ {len(kpts)} keypoints detected")
                
                # Draw keypoints
                output = frame.copy()
                valid_count = 0
                for i, kpt in enumerate(kpts):
                    x, y = int(kpt.get('x', 0)), int(kpt.get('y', 0))
                    conf = kpt.get('confidence', 0)
                    
                    if conf > 0.3:
                        valid_count += 1
                        cv2.circle(output, (x, y), 6, (0, 255, 0), -1)
                        cv2.putText(output, str(i), (x+8, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                print(f"   Valid (conf>0.3): {valid_count}")
                
                # Save
                filename = f"model_comparison/{model_id.replace('/', '_')}.jpg"
                cv2.imwrite(filename, output)
                print(f"   Saved: {filename}")
            else:
                print(f"   ❌ No keypoints in prediction")
        else:
            print(f"   ❌ No predictions")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")

print("\n" + "=" * 60)
print("Manuel Keypoint Seçimi Alternartifi:")
print("=" * 60)
print("""
Eğer hiçbir model iyi sonuç vermezse:
1. İlk frame'de saha köşelerini elle seç (4+ nokta)
2. Homography hesapla
3. Tüm frame'lere uygula

Bu yaklaşım daha güvenilir çünkü:
- Kamera açısına bağımlı değil
- Her video için customize edilebilir
""")

print("\n" + "=" * 60)

from ultralytics import YOLO
import cv2
import os

model = YOLO(r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\models\keypoints\best.pt")

input_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\data\keypoints\court_Key\test\images"
output_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Resim okunamadı: {img_path}")
        continue

    results = model(img, conf=0.8)
    annotated_img = results[0].plot()

    cv2.imshow("YOLO Inference", annotated_img)
    cv2.imwrite(os.path.join(output_dir, img_name), annotated_img)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
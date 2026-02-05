# =============================================================================
# Court Keypoint Detection Model Training - Google Colab
# =============================================================================

# Step 1: Install dependencies
!pip install ultralytics roboflow -q

# Step 2: Download dataset from Roboflow
from roboflow import Roboflow

API_KEY = "lYJFYI6Qh2f8QOFW1i6e"

rf = Roboflow(api_key=API_KEY)

# reloc2-den7l dataset (HanaFEKI'nin kullandığı - 18 keypoint)
project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
version = project.version(1)
dataset = version.download("yolov8", location="/content/court_keypoint_dataset")

print(f"Dataset downloaded to: {dataset.location}")

# Step 3: Check data.yaml
!cat /content/court_keypoint_dataset/data.yaml

# =============================================================================
# Step 4: Train YOLOv8-Pose Model
# =============================================================================

from ultralytics import YOLO

# IMPORTANT: Use POSE model for keypoint detection!
# yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt, yolo11l-pose.pt
model = YOLO("yolo11s-pose.pt")  # Small-pose model (good balance)

 

# =============================================================================
# Step 5: Validate and Export
# =============================================================================

# Validate on test set
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")

# Export to different formats
model.export(format="onnx")  # For ONNX Runtime
# model.export(format="engine")  # For TensorRT (if needed)

# Copy best model to Drive
!cp /content/gdrive/MyDrive/BASKETBALL_COURT/court_keypoint_v1/weights/best.pt \
    /content/gdrive/MyDrive/BASKETBALL_COURT/court_keypoint_best.pt

print("\n✅ Training complete!")
print("Best model saved to: /content/gdrive/MyDrive/BASKETBALL_COURT/court_keypoint_best.pt")

"""
Court Keypoint Model Training
==============================
Roboflow datasetini indir ve YOLOv8-Pose modeli eğit.
"""

from roboflow import Roboflow
from ultralytics import YOLO
import os

API_KEY = "lYJFYI6Qh2f8QOFW1i6e"

print("=" * 60)
print("Court Keypoint Model Training Setup")
print("=" * 60)

# Step 1: Download dataset
print("\n📦 Step 1: Downloading dataset from Roboflow...")

rf = Roboflow(api_key=API_KEY)

# Try the FYP project first (used by HanaFEKI)
try:
    project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
    print(f"   Project: {project.name}")
    print(f"   Type: {project.type}")
    
    version = project.version(1)
    dataset = version.download("yolov8", location="./datasets/court_keypoint")
    print(f"   ✅ Downloaded to: {dataset.location}")
    data_yaml = os.path.join(dataset.location, "data.yaml")
    
except Exception as e:
    print(f"   ❌ Error with reloc2-den7l: {e}")
    print("\n   Trying basketball-court-detection-2...")
    
    try:
        project = rf.workspace("basketball").project("basketball-court-detection-2")
        version = project.version(2)
        dataset = version.download("yolov8", location="./datasets/court_keypoint")
        print(f"   ✅ Downloaded to: {dataset.location}")
        data_yaml = os.path.join(dataset.location, "data.yaml")
    except Exception as e2:
        print(f"   ❌ Error: {e2}")
        exit(1)

# Step 2: Check dataset structure
print("\n📁 Step 2: Checking dataset structure...")
if os.path.exists(data_yaml):
    with open(data_yaml, 'r') as f:
        print(f.read())
else:
    print("   ❌ data.yaml not found!")
    exit(1)

# Step 3: Training setup
print("\n🚀 Step 3: Training setup...")
print("""
Training command (run separately or uncomment below):

yolo task=pose mode=train \\
    model=yolo11n-pose.pt \\
    data={data_yaml} \\
    imgsz=640 \\
    batch=16 \\
    epochs=100 \\
    project=runs/court_keypoint \\
    name=train_v1

Or with Python:
""".format(data_yaml=data_yaml))

# Uncomment to start training
# print("\n⏳ Starting training...")
# model = YOLO("yolo11n-pose.pt")
# model.train(
#     data=data_yaml,
#     imgsz=640,
#     batch=16,
#     epochs=100,
#     project="runs/court_keypoint",
#     name="train_v1"
# )
# print("✅ Training complete!")

print("\n" + "=" * 60)
print("Dataset ready for training!")
print("=" * 60)

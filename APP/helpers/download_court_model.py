"""
Download Basketball Court Keypoint Model from Roboflow
======================================================
"""

from roboflow import Roboflow
import os

API_KEY = "lYJFYI6Qh2f8QOFW1i6e"

print("=" * 60)
print("Downloading Basketball Court Keypoint Model")
print("=" * 60)

rf = Roboflow(api_key=API_KEY)

# Try the model used by HanaFEKI first
print("\n📦 Trying reloc2-den7l (FYP project)...")
try:
    project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
    print(f"   Project found: {project.name}")
    print(f"   Type: {project.type}")
    
    # Get latest version
    version = project.version(1)
    print(f"   Version: {version.version}")
    
    # Download for YOLOv8
    dataset = version.download("yolov8", location="./datasets/court_keypoint")
    print(f"   ✅ Downloaded to: {dataset.location}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n📦 Trying basketball-court-detection-2...")
    
    try:
        # Try alternative model
        project = rf.workspace("basketball").project("basketball-court-detection-2")
        print(f"   Project found: {project.name}")
        
        version = project.version(4)
        dataset = version.download("yolov8", location="./datasets/court_keypoint")
        print(f"   ✅ Downloaded to: {dataset.location}")
        
    except Exception as e2:
        print(f"   ❌ Error: {e2}")
        
        # List available workspaces
        print("\n🔍 Listing your workspaces...")
        try:
            workspaces = rf.workspaces()
            for ws in workspaces:
                print(f"   - {ws}")
        except Exception as e3:
            print(f"   Error listing workspaces: {e3}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)

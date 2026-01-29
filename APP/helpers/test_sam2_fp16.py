"""
SAM2 FP16 Test Script
=====================
SAM2 modelini FP16 modunda test eder ve performans karşılaştırması yapar.
"""

import os
import sys
import time
import torch
import numpy as np
import cv2

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("SAM2 FP16 Performance Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("=" * 60)

# ============================================================================
# Load SAM2 Model
# ============================================================================
print("\n[1/4] Loading SAM2 Model...")

from sam2.build_sam import build_sam2_video_predictor

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # Tiny
SAM2_CHECKPOINT = "models/sam2.1_hiera_tiny.pt"

# Load in FP32 first
predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
print("✅ Model loaded (FP32)")

# ============================================================================
# Test FP32 Performance
# ============================================================================
print("\n[2/4] Testing FP32 Performance...")

# Create dummy input
dummy_image = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.float32)

# Warmup
print("   Warming up...")
with torch.no_grad():
    for _ in range(3):
        _ = predictor.image_encoder(dummy_image)

torch.cuda.synchronize()

# Benchmark FP32
print("   Running FP32 benchmark (20 iterations)...")
times_fp32 = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        _ = predictor.image_encoder(dummy_image)
        torch.cuda.synchronize()
        times_fp32.append(time.time() - start)

avg_fp32 = np.mean(times_fp32) * 1000
print(f"   ✅ FP32: {avg_fp32:.2f}ms per frame ({1000/avg_fp32:.1f} FPS)")

# ============================================================================
# Convert to FP16 (Mixed Precision)
# ============================================================================
print("\n[3/4] Testing FP16 (Mixed Precision)...")

try:
    # Method 1: torch.cuda.amp (recommended)
    print("   Method 1: Using torch.cuda.amp autocast...")
    
    dummy_image_fp32 = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.float32)
    
    # Warmup with autocast
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for _ in range(3):
                _ = predictor.image_encoder(dummy_image_fp32)
    
    torch.cuda.synchronize()
    
    # Benchmark with autocast
    times_amp = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.time()
                _ = predictor.image_encoder(dummy_image_fp32)
                torch.cuda.synchronize()
                times_amp.append(time.time() - start)
    
    avg_amp = np.mean(times_amp) * 1000
    print(f"   ✅ AMP (autocast): {avg_amp:.2f}ms per frame ({1000/avg_amp:.1f} FPS)")
    print(f"   📈 Speedup: {avg_fp32/avg_amp:.2f}x")

except Exception as e:
    print(f"   ❌ AMP failed: {e}")
    avg_amp = None

# Method 2: Direct FP16 conversion
print("\n   Method 2: Direct model.half() conversion...")

try:
    # Try to convert model to FP16
    predictor_fp16 = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    
    # Convert to half precision
    predictor_fp16.image_encoder = predictor_fp16.image_encoder.half()
    
    dummy_image_fp16 = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.float16)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = predictor_fp16.image_encoder(dummy_image_fp16)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times_fp16 = []
    with torch.no_grad():
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.time()
            _ = predictor_fp16.image_encoder(dummy_image_fp16)
            torch.cuda.synchronize()
            times_fp16.append(time.time() - start)
    
    avg_fp16 = np.mean(times_fp16) * 1000
    print(f"   ✅ Direct FP16: {avg_fp16:.2f}ms per frame ({1000/avg_fp16:.1f} FPS)")
    print(f"   📈 Speedup: {avg_fp32/avg_fp16:.2f}x")

except Exception as e:
    print(f"   ❌ Direct FP16 failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    avg_fp16 = None

# ============================================================================
# Test Full Video Predictor with AMP
# ============================================================================
print("\n[4/4] Testing Full Video Predictor with AMP...")

try:
    # Test with a real video scenario
    VIDEO_PATH = "videos/court.mp4"
    
    if os.path.exists(VIDEO_PATH):
        # Extract first few frames
        cap = cv2.VideoCapture(VIDEO_PATH)
        frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        
        if frames:
            print(f"   Loaded {len(frames)} frames from video")
            
            # Save frames temporarily
            temp_dir = "temp_frames_fp16_test"
            os.makedirs(temp_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                cv2.imwrite(f"{temp_dir}/{i:05d}.jpg", frame)
            
            # Test FP32 video inference
            print("   Testing FP32 video inference...")
            predictor_test = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
            
            with torch.inference_mode():
                start = time.time()
                state = predictor_test.init_state(video_path=temp_dir)
                fp32_init_time = time.time() - start
            
            print(f"   FP32 init_state: {fp32_init_time*1000:.2f}ms")
            
            # Test with AMP
            print("   Testing AMP video inference...")
            predictor_amp = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    start = time.time()
                    state_amp = predictor_amp.init_state(video_path=temp_dir)
                    amp_init_time = time.time() - start
            
            print(f"   AMP init_state: {amp_init_time*1000:.2f}ms")
            print(f"   📈 Speedup: {fp32_init_time/amp_init_time:.2f}x")
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
    else:
        print(f"   ⚠️ Video not found: {VIDEO_PATH}")
        print("   Skipping full video test...")

except Exception as e:
    print(f"   ❌ Full video test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"FP32 Encoder:     {avg_fp32:.2f}ms ({1000/avg_fp32:.1f} FPS)")
if avg_amp:
    print(f"AMP (autocast):   {avg_amp:.2f}ms ({1000/avg_amp:.1f} FPS) - {avg_fp32/avg_amp:.2f}x speedup")
if avg_fp16:
    print(f"Direct FP16:      {avg_fp16:.2f}ms ({1000/avg_fp16:.1f} FPS) - {avg_fp32/avg_fp16:.2f}x speedup")
print("=" * 60)
print("\n💡 Recommendation: Use torch.cuda.amp.autocast() wrapper in your tracker!")
print("   It automatically handles mixed precision without manual conversion.")

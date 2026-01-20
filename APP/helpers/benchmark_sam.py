"""
SAM2 Model Benchmark - Speed Comparison
"""
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

def benchmark_sam_model(model_id: str, image, input_boxes, runs: int = 20):
    """Benchmark a SAM2 model."""
    device = 'cuda'
    
    print(f"\n📦 Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors='pt').to(device)
    
    # Warmup
    print("  Warming up...")
    for _ in range(3):
        with torch.no_grad():
            outputs = model(**inputs)
    
    # Benchmark
    print(f"  Running {runs} iterations...")
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(runs):
        with torch.no_grad():
            outputs = model(**inputs)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_ms = (elapsed / runs) * 1000
    fps = 1000 / avg_ms
    
    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    
    return avg_ms, fps


if __name__ == "__main__":
    print("=" * 50)
    print("SAM2 Model Speed Benchmark")
    print("=" * 50)
    
    # Setup
    image_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\test_frame_4.jpg"
    image = Image.open(image_path).convert("RGB")
    
    # Single bbox test
    input_boxes = [[[100, 100, 300, 400]]]
    
    models = [
        "facebook/sam2-hiera-small",
        "facebook/sam2-hiera-large",
    ]
    
    results = {}
    
    for model_id in models:
        avg_ms, fps = benchmark_sam_model(model_id, image, input_boxes, runs=20)
        results[model_id] = (avg_ms, fps)
        print(f"\n✅ {model_id.split('/')[-1]}:")
        print(f"   Time: {avg_ms:.1f} ms/mask")
        print(f"   FPS:  {fps:.1f} (single object)")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY - Real-time viability")
    print("=" * 50)
    
    for model_id, (avg_ms, fps) in results.items():
        name = model_id.split('/')[-1]
        # For 10 objects
        total_time = avg_ms * 10
        effective_fps = 1000 / total_time
        
        viability = "✅ OK" if effective_fps > 10 else "⚠️ Slow" if effective_fps > 5 else "❌ Too slow"
        
        print(f"\n{name}:")
        print(f"  Single mask: {avg_ms:.1f}ms ({fps:.1f} FPS)")
        print(f"  10 objects:  {total_time:.1f}ms ({effective_fps:.1f} FPS) {viability}")

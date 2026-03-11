"""
SAM2 ONNX Helper
================
TensorRT/CUDA supported high-performance segmentation using SAM2.
Mirroring style of rfdetr_detector.py
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from typing import Tuple, List, Union, Optional, Dict
from PIL import Image
import time
import matplotlib.pyplot as plt

class SAM2Helper:
    """SAM2 model for ONNX Runtime based segmentation."""
    
    def __init__(self, model_size: str = "sam2_hiera_small", device: str = "cuda", use_tensorrt: bool = True):
        """
        Initialize SAM2 Helper.
        
        Args:
            model_size: Model size typically 'sam2_hiera_small'
            device: 'cuda' or 'cpu'
            use_tensorrt: Enable TensorRT provider if available (default: True)
        """
        self.model_size = model_size
        self.device = device
        
        self.providers = []
        if device == 'cuda':
            if use_tensorrt:
                self.providers.append("TensorrtExecutionProvider")
            self.providers.append("CUDAExecutionProvider")
        self.providers.append("CPUExecutionProvider")
        
        # Session Options for memory optimization
        self.sess_options = ort.SessionOptions()
        self.sess_options.enable_cpu_mem_arena = False
        
        # Model paths
        self.model_dir = "models/sam2_onnx"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Download and load models
        self.encoder_path, self.decoder_path = self._ensure_models()
        
        print(f"📦 Loading SAM2 ONNX ({device})...")
        try:
            self.encoder_session = ort.InferenceSession(self.encoder_path, sess_options=self.sess_options, providers=self.providers)
            self.decoder_session = ort.InferenceSession(self.decoder_path, sess_options=self.sess_options, providers=self.providers)
            print("✅ SAM2 Models loaded successfully")
            print(f"   Providers (Encoder): {self.encoder_session.get_providers()}")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            raise

        # Image stats
        self.img_size = 1024 # Fixed input size for this ONNX model
        self.pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1).astype(np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1).astype(np.float32)

        self.sess_options = ort.SessionOptions()
        self.sess_options.enable_cpu_mem_arena = False # reduce memory usage

    def _ensure_models(self) -> Tuple[str, str]:
        """Download correct ONNX models if not present"""
        repo_id = "SharpAI/sam2-hiera-small-onnx"
        
        enc_path = os.path.join(self.model_dir, "encoder.onnx")
        dec_path = os.path.join(self.model_dir, "decoder.onnx")
        
        if not os.path.exists(enc_path):
            print(f"⬇️ Downloading encoder from {repo_id}...")
            hf_hub_download(repo_id=repo_id, filename="encoder.onnx", local_dir=self.model_dir)
            
        if not os.path.exists(dec_path):
            print(f"⬇️ Downloading decoder from {repo_id}...")
            hf_hub_download(repo_id=repo_id, filename="decoder.onnx", local_dir=self.model_dir)
            
        return enc_path, dec_path

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, dict]:
        """Preprocess image to 1024x1024"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]
        scale = self.img_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        
        img_padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img_padded[:new_h, :new_w, :] = img_resized
        
        img_tensor = img_padded.astype(np.float32)
        img_tensor = img_tensor.transpose(2, 0, 1)[None, :, :, :]
        img_tensor = (img_tensor - self.pixel_mean) / self.pixel_std
        
        return img_tensor, {"scale": scale, "orig_size": (orig_h, orig_w)}

    def encode(self, image: Union[np.ndarray, Image.Image, str]) -> Tuple[Dict[str, np.ndarray], dict]:
        """Run image encoder"""
        img_tensor, info = self.preprocess(image)
        
        try:
            output_names = [o.name for o in self.encoder_session.get_outputs()]
            outs = self.encoder_session.run(None, {"image": img_tensor})
            res = {name: val for name, val in zip(output_names, outs)}
            return res, info
        except Exception as e:
            print(f"❌ Encoder failed: {e}")
            raise

    def decode(self, encoder_outs: Dict[str, np.ndarray], prompt_points: np.ndarray, prompt_labels: np.ndarray, info: dict) -> Tuple[np.ndarray, float]:
        """Run mask decoder"""
        image_embed = encoder_outs.get('image_embed')
        high_res_0 = encoder_outs.get('high_res_feats_0')
        high_res_1 = encoder_outs.get('high_res_feats_1')
        
        if image_embed is None:
            raise ValueError(f"Missing required keys. Available: {encoder_outs.keys()}")

        scale = info["scale"]
        pts = (prompt_points * scale).astype(np.float32)
        
        if len(pts.shape) == 2:
            pts = pts[None, :, :]
        if len(prompt_labels.shape) == 1:
            lbls = prompt_labels[None, :]
        else:
            lbls = prompt_labels

        # Pad to exactly 4 points (Broadcasting requirement)
        current_points = pts.shape[1]
        target_points = 4 
        if current_points < target_points:
            n_pad = target_points - current_points
            pad_pts = np.zeros((pts.shape[0], n_pad, 2), dtype=np.float32)
            pad_lbls = -1 * np.ones((lbls.shape[0], n_pad), dtype=np.float32)
            pts = np.concatenate([pts, pad_pts], axis=1)
            lbls = np.concatenate([lbls, pad_lbls], axis=1)

        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)

        try:
            outs = self.decoder_session.run(None, {
                "image_embed": image_embed,
                "high_res_feats_0": high_res_0,
                "high_res_feats_1": high_res_1,
                "point_coords": pts,
                "point_labels": lbls,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input
            })
            
            masks = outs[0][0]
            scores = outs[1][0]
            
            # Debug shape (remove later)
            # print(f"DEBUG: masks shape: {masks.shape}") 
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx] # (256, 256)
            best_score = scores[best_idx]
            
            # SAM2 decoder yields 256x256 masks for 1024x1024 input
            # So we typically need to scale down the slicing coordinates by 4
            # OR resize the 256x256 mask to 1024x1024 first.
            # Efficient way: Slice then resize.
            
            orig_h, orig_w = info['orig_size']
            
            # Helper's internal image size
            model_input_size = self.img_size # 1024
            mask_output_size = best_mask.shape[0] # 256 typically
            
            # Scale factor between input image and mask output
            mask_scale_factor = mask_output_size / model_input_size # 0.25
            
            # Active area in the model input (before padding)
            active_h = int(orig_h * info["scale"])
            active_w = int(orig_w * info["scale"])
            
            # Active area in the mask output
            valid_h = int(active_h * mask_scale_factor)
            valid_w = int(active_w * mask_scale_factor)
            
            # Clamp to shape to be safe
            valid_h = min(valid_h, mask_output_size)
            valid_w = min(valid_w, mask_output_size)
            
            valid_mask = best_mask[:valid_h, :valid_w]
            
            # Resize back to original image size
            final_mask = cv2.resize(valid_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            return (final_mask > 0.0).astype(bool), float(best_score)
            
        except Exception as e:
            print(f"❌ Decoder failed: {e}")
            raise

    def segment_bbox(self, image: Union[np.ndarray, str], bbox: List[int]):
        """
        Segment using bbox [x1, y1, x2, y2].
        Adds a center point prompt to encourage full object segmentation.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 3 points: 
        # 1. Top-Left Box Corner (Label 2)
        # 2. Bottom-Right Box Corner (Label 3)
        # 3. Center Point (Label 1 - Positive Click) -> Forces selecting the object body
        points = np.array([[[x1, y1], [x2, y2], [cx, cy]]], dtype=np.float32)
        labels = np.array([[2, 3, 1]], dtype=np.float32)
        
        enc_outs, info = self.encode(image)
        mask, score = self.decode(enc_outs, points, labels, info)
        return mask, score

    def benchmark(self, image_input: Union[str, np.ndarray], num_runs: int = 10):
        """Benchmark performance"""
        print(f"🏎️ Benchmarking SAM2 ONNX ({num_runs} runs)...")
        
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_input

        # Warmup
        print("   Warmup...")
        enc_outs, info = self.encode(image)
        dummy_box = [100, 100, 200, 200]
        self.segment_bbox(image, dummy_box)
        
        # Encoder Benchmark
        start = time.time()
        for _ in range(num_runs):
            self.encode(image)
        enc_time = (time.time() - start) / num_runs * 1000
        
        # Decoder Benchmark
        # Prepare inputs once
        x1, y1, x2, y2 = dummy_box
        points = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
        labels = np.array([[2, 3]], dtype=np.float32)
        
        start = time.time()
        for _ in range(num_runs):
             self.decode(enc_outs, points, labels, info)
        dec_time = (time.time() - start) / num_runs * 1000
        
        print(f"   Encoder: {enc_time:.2f} ms")
        print(f"   Decoder: {dec_time:.2f} ms")
        return {"encoder_ms": enc_time, "decoder_ms": dec_time}

    def visualize(self, image, mask, bbox=None, alpha=0.5, save_path=None, show=True):
        """Visualize binary mask on image"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            if img.shape[2] == 3: # RGB/BGR check
                 # Assuming Input is RGB for consistency with other tools, convert to BGR for OpenCV
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        overlay = img.copy()
        color = (0, 255, 0) # Green
        
        # Draw Mask
        overlay[mask] = np.array(color) * alpha + overlay[mask] * (1 - alpha)
        
        # Draw BBox
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"💾 Saved: {save_path}")
            
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        return overlay

# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    helper = SAM2Helper()
    
    # Test image
    img_path = os.path.join(ROOT_DIR, "videos", "output", "test_frame_4.jpg")
    if os.path.exists(img_path):
        bbox = [450, 200, 600, 500] # Example bbox
        
        print("\n🔍 Segmentation...")
        mask, score = helper.segment_bbox(img_path, bbox)
        print(f"   Score: {score:.3f}")
        
        helper.visualize(img_path, mask, bbox, save_path="sam_test_vis.jpg")
        
        helper.benchmark(img_path)


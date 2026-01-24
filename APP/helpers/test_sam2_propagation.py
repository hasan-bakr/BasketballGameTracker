import os
import cv2
import numpy as np
import torch
import gc
import shutil
import sys

# SAM2 Video imports
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from APP.helpers.yolo_detector import YoloDetector

class SAM2BatchProcessor:
    def __init__(self, video_path, output_path, device="cuda"):
        self.video_path = video_path
        self.output_path = output_path
        self.device = device
        
        # Load Models
        print("📦 Loading Models...")
        self.yolo = YoloDetector(
            model_path="models/yolo/best_detection.pt", 
            device="cpu"  # Keep YOLO on CPU to save VRAM for SAM2
        )
        self.predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "models/sam2.1_hiera_small.pt",
            device=device
        )
        
        # Video Props
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output Video Writer (will be init on first frame)
        self.writer = None
        self.temp_dir = "temp_batch_frames"
        
        # State
        self.last_masks = {} # {obj_id: mask_array} from the end of previous batch
        self.next_obj_id = 1
        
    def _setup_writer(self, frame_size):
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, frame_size
            )

    def process_video(self, batch_size=50):
        print(f"🚀 Starting Batch Processing: {self.total_frames} frames in batches of {batch_size}")
        
        frame_idx = 0
        while True:
            # 1. Extract Batch
            frames, original_frames = self._extract_batch(batch_size)
            if not frames:
                break
                
            current_batch_size = len(frames)
            print(f"\n🔄 Processing Batch: Frames {frame_idx} - {frame_idx + current_batch_size}")
            
            # 2. Initialize SAM2 State for this batch
            inference_state = self.predictor.init_state(video_path=self.temp_dir)
            
            # 3. Provide Prompts
            if frame_idx == 0:
                # First batch: Use YOLO
                self._init_with_yolo(inference_state, original_frames[0])
            else:
                # Next batches: Use masks from previous batch
                self._init_with_previous_masks(inference_state)
            
            # 4. Propagate
            self._propagate_and_save(inference_state, original_frames)
            
            # 5. Cleanup
            self.predictor.reset_state(inference_state)
            frame_idx += current_batch_size
            
            # Free memory
            del frames
            del original_frames
            gc.collect()
            torch.cuda.empty_cache()
            
        if self.writer:
            self.writer.release()
        
        # Remove temp dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
        print(f"✅ Processing Complete! Output: {self.output_path}")

    def _extract_batch(self, batch_size):
        """Buffer frames to disk for SAM2"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        frames = []
        original_frames = [] # To draw on
        
        for i in range(batch_size):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Keep original for output
            original_frames.append(frame)
            
            # Resize for SAM2 (Max 1024)
            h, w = frame.shape[:2]
            scale = 1.0
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
            
            # Save for SAM2
            cv2.imwrite(f"{self.temp_dir}/{i:05d}.jpg", frame_resized)
            frames.append(frame_resized)
            
            # Initialize Writer if needed
            if self.writer is None:
               self._setup_writer((w, h)) 

        return frames, original_frames

    def _init_with_yolo(self, inference_state, first_frame):
        print("🔍 First Batch: detecting with YOLO...")
        detections = self.yolo.detect(first_frame, classes=[0]) # 0 is person in COCO (check your model classes)
        # Note: User's previous code used classes=[3,4,5...] check that!
        # Assuming YOLO model returns standard classes or specific ones. 
        # Updating to match user's previous code: classes=[3, 4, 5, 6, 7]
        
        detections = self.yolo.detect(first_frame, confidence_threshold=0.5) 
        # Filter for players if needed, or assume YOLO returns players.
        
        print(f"   Found {len(detections)} objects")
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            # Resize bbox if frame was resized? No, SAM2 reads the file from disk which IS resized.
            # BUT we passed 'original_frames[0]' to YOLO.
            # So we must scale bbox to match 'temp_batch_frames/00000.jpg' size.
            
            # Get scale
            h, w = first_frame.shape[:2]
            scale = 1.0
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
            
            box = np.array([x1*scale, y1*scale, x2*scale, y2*scale], dtype=np.float32)
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=self.next_obj_id,
                box=box
            )
            self.next_obj_id += 1

    def _init_with_previous_masks(self, inference_state):
        """Use masks from end of last batch as prompt for start of this batch"""
        print(f"🔗 Continuing tracking for {len(self.last_masks)} objects...")
        
        # Note: frame_idx=0 here refers to the 0th frame of THIS BATCH (which is the continuation)
        for obj_id, mask in self.last_masks.items():
            # mask is [H, W] boolean or float
            # SAM2 add_new_mask expects logits or binary mask? 
            # documentation says 'mask' argument.
            
            # We need to resize mask to match current batch frame size (already resized)
            # stored mask should be in resized coordinates from previous batch output?
            
            if mask is not None:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0, 
                    obj_id=obj_id,
                    mask=mask
                )

    def _propagate_and_save(self, inference_state, original_frames):
        # Propagate
        video_segments = {} # frame_idx (in batch) -> {obj_id: mask}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
             video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Save output and store last masks
        h_orig, w_orig = original_frames[0].shape[:2]
        h_resized, w_resized = cv2.imread(f"{self.temp_dir}/00000.jpg").shape[:2]
        
        for i, frame in enumerate(original_frames):
            if i in video_segments:
                for obj_id, mask in video_segments[i].items():
                    # Handle mask dim
                    if mask.ndim == 3: mask = mask[0]
                    
                    # Store last mask (for next batch)
                    if i == len(original_frames) - 1:
                        self.last_masks[obj_id] = mask # Keep in resized scale
                    
                    # Resize mask back to original for drawing
                    mask_orig = cv2.resize(mask.astype(np.float32), (w_orig, h_orig))
                    mask_bool = mask_orig > 0.5
                    
                    # Draw
                    color = self._get_color(obj_id)
                    frame[mask_bool] = frame[mask_bool] * 0.6 + np.array(color) * 0.4
                    
            self.writer.write(frame)
            
    def _get_color(self, obj_id):
        np.random.seed(obj_id)
        return np.random.randint(0, 255, 3).tolist()

if __name__ == "__main__":
    mp4_in = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    mp4_out = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\sam2_batch_output.mp4"
    
    processor = SAM2BatchProcessor(mp4_in, mp4_out)
    processor.process_video(batch_size=50) # Safe batch size

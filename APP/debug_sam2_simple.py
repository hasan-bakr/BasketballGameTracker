"""
debug_sam2_simple.py
====================
Orijinal (Ocak 2026) SAM2 tracking yaklaşımı — sade versiyon.

- Court filter YOK
- Memory bank YOK
- Chest-level prompt YOK — düz YOLO bbox kullanır
- Sadece: YOLO init → SAM2 propagate → batch sınırında add_new_mask

Kullanım:
    python -m APP.debug_sam2_simple \
        --input  videos/input/game.mp4 \
        --output videos/output/simple_out.mp4 \
        --max-frames 300 --batch-size 50 --start 0
"""

import argparse
import gc
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from sam2.build_sam import build_sam2_video_predictor
from APP.helpers.yolo_detector import YoloDetector

PLAYER_CLASSES = [3, 4, 5, 6, 7]
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT       = os.path.join(ROOT_DIR, "models", "sam2.1_hiera_base_plus.pt")
YOLO_PATH       = os.path.join(ROOT_DIR, "models", "yolo", "best_detection.pt")
TEMP_DIR        = "/dev/shm/dbg_sam2_simple"

np.random.seed(42)
COLORS = [tuple(int(c) for c in np.random.randint(60, 255, 3)) for _ in range(200)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_batch(cap, n_frames: int, frame_skip: int = 1):
    """cap'ten n_frames*frame_skip frame oku, frame_skip'e göre filtrele, diske yaz."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    frames, idx = [], 0
    for i in range(n_frames * frame_skip):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip != 0:
            continue
        frames.append(frame)
        h, w = frame.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            out = cv2.resize(frame, (int(w * s), int(h * s)))
        else:
            out = frame
        cv2.imwrite(f"{TEMP_DIR}/{idx:05d}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    return frames


def draw_masks(frame, masks, jersey_ids=None):
    out = frame.copy()
    for obj_id, mask in masks.items():
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32),
                              (frame.shape[1], frame.shape[0])) > 0.5
        color = COLORS[obj_id % len(COLORS)]
        out[mask] = (out[mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
        m8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
        if cnts:
            M = cv2.moments(cnts[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = str(obj_id)
                cv2.putText(out, label, (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(out, label, (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return out


# ── Core ──────────────────────────────────────────────────────────────────────

def run(args):
    device = args.device
    yolo = YoloDetector(model_path=YOLO_PATH, device="cpu")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device=device)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # start offset
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    tracked: dict = {}   # {obj_id: {"last_mask": ndarray|None}}
    next_id = [1]

    writer = None
    frame_idx = 0
    total = args.max_frames

    pbar = tqdm(total=total, desc="Processing")

    while frame_idx < total:
        batch_n = min(args.batch_size, total - frame_idx)
        frames = extract_batch(cap, batch_n, args.frame_skip)
        if not frames:
            break

        if writer is None:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps / args.frame_skip, (w, h))

        # SAM2 state
        inference_state = predictor.init_state(video_path=TEMP_DIR)

        if frame_idx == 0:
            # İlk batch: YOLO bbox ile init
            dets = [d for d in yolo.detect(frames[0], confidence_threshold=args.conf)
                    if d["class_id"] in PLAYER_CLASSES]
            h_o, w_o = frames[0].shape[:2]
            ref = cv2.imread(f"{TEMP_DIR}/00000.jpg")
            h_r, w_r = ref.shape[:2]
            sx, sy = w_r / w_o, h_r / h_o

            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                box = np.array([x1*sx, y1*sy, x2*sx, y2*sy], dtype=np.float32)
                oid = next_id[0]; next_id[0] += 1
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0, obj_id=oid, box=box)
                tracked[oid] = {"last_mask": None}
            print(f"  Init: {len(dets)} players")
        else:
            # Devam: önceki batch son maskesini prompt olarak ver
            for oid, obj in tracked.items():
                if obj["last_mask"] is not None:
                    m256 = cv2.resize(obj["last_mask"].astype(np.float32),
                                      (256, 256)) > 0.5
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0, obj_id=oid, mask=m256)

        # Propagate
        batch_masks: dict = {}
        h_o, w_o = frames[0].shape[:2]

        for out_fi, out_ids, out_logits in predictor.propagate_in_video(inference_state):
            fm = {}
            for i, oid in enumerate(out_ids):
                logit = out_logits[i]
                mask = (logit > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask = cv2.resize(mask.astype(np.float32), (w_o, h_o)) > 0.5
                if mask.sum() > 100:
                    fm[oid] = mask
                if oid in tracked:
                    tracked[oid]["last_mask"] = mask
            batch_masks[out_fi] = fm

        predictor.reset_state(inference_state)

        # Annotate & write
        for i, frame in enumerate(frames):
            masks = batch_masks.get(i, {})
            out = draw_masks(frame, masks)
            cv2.putText(out, f"Frame {frame_idx + i}  objs:{len(masks)}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            writer.write(out)

        frame_idx += len(frames)
        pbar.update(len(frames))
        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    print(f"\nDone → {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--max-frames",  type=int,   default=300)
    p.add_argument("--batch-size",  type=int,   default=50)
    p.add_argument("--start",       type=float, default=0.0, help="Start offset in seconds")
    p.add_argument("--frame-skip",  type=int,   default=1)
    p.add_argument("--conf",        type=float, default=0.5)
    p.add_argument("--device",      default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()

"""RF-DETR basketball detector via ONNX Runtime.

Uses the Roboflow-trained basketball model (cached ONNX) directly through
onnxruntime, no `inference-gpu` SDK / pycuda required.

Output format mirrors RFDETRDetector.detect():
    [{'bbox': [x1,y1,x2,y2], 'confidence': float,
      'class_name': str, 'class_id': int}]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort

# Same class IDs as the rest of the pipeline expects.
_NAME_TO_CID: Dict[str, int] = {
    "player":     3,
    "referee":    8,
    "ball":       0,
    "rim":        9,
}


def _resolve_class_id(name: str) -> int:
    lower = name.lower()
    for kw, cid in _NAME_TO_CID.items():
        if kw in lower:
            return cid
    return 99


class RFDETROnnx:
    DEFAULT_DIR = Path(__file__).resolve().parents[2] / "models" / "rfdetr_basketball"

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_DIR,
        device: str = "cuda",
        input_size: int = 1280,
    ):
        model_dir = Path(model_dir)
        weights   = model_dir / "weights.onnx"
        names_fp  = model_dir / "class_names.txt"
        cfg_fp    = model_dir / "inference_config.json"
        if not weights.exists():
            raise FileNotFoundError(f"Missing ONNX weights: {weights}")

        self.class_names: List[str] = [
            ln.strip() for ln in names_fp.read_text().splitlines() if ln.strip()
        ]
        cfg = json.loads(cfg_fp.read_text()) if cfg_fp.exists() else {}
        net_in = cfg.get("network_input", {})
        self.input_size  = int(net_in.get("training_input_size", {}).get("height", input_size))
        self.scale_div   = float(net_in.get("scaling_factor", 255.0))
        self.color_rgb   = net_in.get("color_mode", "rgb").lower() == "rgb"
        self.resize_mode = net_in.get("resize_mode", "stretch").lower()

        providers: Sequence[str]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(f"Loading RF-DETR ONNX: {weights} ({providers[0]})...")
        self.session = ort.InferenceSession(str(weights), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()[0]
        if active != providers[0]:
            print(f"  WARNING: requested {providers[0]} but using {active}")
        print(f"RF-DETR ready. classes={len(self.class_names)} input={self.input_size}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        h0, w0 = frame.shape[:2]
        s = self.input_size
        if self.resize_mode == "stretch":
            img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_LINEAR)
        else:  # letterbox-style fallback (not used by basketball model)
            img = cv2.resize(frame, (s, s), interpolation=cv2.INTER_LINEAR)
        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / self.scale_div
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x)

    @staticmethod
    def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        return np.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=-1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iw = np.maximum(0.0, xx2 - xx1)
            ih = np.maximum(0.0, yy2 - yy1)
            inter = iw * ih
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-9)
            order = order[1:][iou <= iou_thr]
        return np.asarray(keep, dtype=np.int64)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        h0, w0 = frame.shape[:2]
        x = self._preprocess(frame)
        out = self.session.run(None, {self.input_name: x})[0]  # [1, 4+C, A]
        out = out[0].T  # [A, 4+C]

        boxes_in = out[:, :4]
        cls_probs = out[:, 4:]
        if cls_probs.shape[1] != len(self.class_names):
            # Fallback if config mismatch: trust the channel count.
            self.class_names = [f"cls{i}" for i in range(cls_probs.shape[1])]

        cls_id  = cls_probs.argmax(axis=1)
        cls_conf = cls_probs.max(axis=1)
        keep = cls_conf >= confidence_threshold
        if not keep.any():
            return []

        boxes_in = boxes_in[keep]
        cls_id   = cls_id[keep]
        cls_conf = cls_conf[keep]

        # Boxes are in input-image pixel coords (cx,cy,w,h). Scale to original.
        sx = w0 / float(self.input_size)
        sy = h0 / float(self.input_size)
        boxes_xyxy = self._xywh_to_xyxy(boxes_in)
        boxes_xyxy[:, 0::2] *= sx
        boxes_xyxy[:, 1::2] *= sy
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, w0 - 1)
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, h0 - 1)

        # Class-agnostic NMS (matches Roboflow inference default).
        keep_idx = self._nms(boxes_xyxy, cls_conf, iou_threshold)
        boxes_xyxy = boxes_xyxy[keep_idx]
        cls_id     = cls_id[keep_idx]
        cls_conf   = cls_conf[keep_idx]

        target_set = set(classes) if classes is not None else None
        out_list: List[Dict] = []
        for box, cid_raw, conf in zip(boxes_xyxy, cls_id, cls_conf):
            x1, y1, x2, y2 = box.astype(int).tolist()
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            name = self.class_names[int(cid_raw)] if int(cid_raw) < len(self.class_names) else "obj"
            cid  = _resolve_class_id(name)
            if target_set is not None and cid not in target_set:
                continue
            out_list.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_name": name,
                "class_id":   int(cid),
            })
        return out_list

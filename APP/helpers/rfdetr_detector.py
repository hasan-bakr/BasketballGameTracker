"""RF-DETR detector via Roboflow Inference SDK.

Requires:
  pip install inference-gpu
  ROBOFLOW_API_KEY env var
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Maps RF-DETR class names → pipeline class IDs (must match MCByte tracker wrapper)
_NAME_TO_CID: Dict[str, int] = {
    "player":     3,
    "referee":    8,
    "ref":        8,
    "ball":       0,
    "basketball": 0,
    "rim":        9,
    "hoop":       9,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_result(result: Any) -> List[Any]:
    if isinstance(result, (list, tuple)) and result:
        result = result[0]
    preds = _get(result, "predictions", None)
    return list(preds) if preds is not None else []


def _to_xyxy(pred: Any) -> Optional[Tuple[int, int, int, int]]:
    x1 = _get(pred, "x1")
    y1 = _get(pred, "y1")
    x2 = _get(pred, "x2")
    y2 = _get(pred, "y2")
    if None not in (x1, y1, x2, y2):
        return int(x1), int(y1), int(x2), int(y2)
    x = _get(pred, "x")
    y = _get(pred, "y")
    w = _get(pred, "width")
    h = _get(pred, "height")
    if None in (x, y, w, h):
        return None
    return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)


def _resolve_class_id(name: str, raw_id: Any) -> int:
    lower = name.lower()
    for kw, cid in _NAME_TO_CID.items():
        if kw in lower:
            return cid
    return int(raw_id) if raw_id is not None else 99


class RFDETRDetector:
    """Roboflow Inference RF-DETR wrapper.

    Returns the normalized detection format used by the pipeline:
        [{'bbox': [x1,y1,x2,y2], 'confidence': float,
          'class_name': str, 'class_id': int}]
    """

    DEFAULT_MODEL_ID = "basketball-player-detection-3-ycjdo/4"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
        if not os.getenv("ROBOFLOW_API_KEY"):
            raise SystemExit("ROBOFLOW_API_KEY env var is not set.")

        # Suppress inference SDK optional-model warnings
        for var in (
            "PALIGEMMA_ENABLED", "FLORENCE2_ENABLED", "QWEN_2_5_ENABLED",
            "QWEN_3_ENABLED", "CORE_MODEL_SAM_ENABLED", "CORE_MODEL_SAM3_ENABLED",
            "CORE_MODEL_GAZE_ENABLED", "SMOLVLM2_ENABLED", "DEPTH_ESTIMATION_ENABLED",
            "MOONDREAM2_ENABLED", "CORE_MODEL_TROCR_ENABLED", "CORE_MODEL_GROUNDINGDINO_ENABLED",
        ):
            os.environ.setdefault(var, "False")

        try:
            from inference import get_model
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency. Run: pip uninstall inference -y && pip install inference-gpu"
            ) from exc

        print(f"Loading RF-DETR: {model_id} ({device})...")
        self.model    = get_model(model_id=model_id)
        self.device   = device
        self.model_id = model_id
        print("RF-DETR ready.")

    def detect(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.9,
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        result      = self.model.infer(image, confidence=confidence_threshold, iou_threshold=iou_threshold)
        predictions = _normalize_result(result)
        target_set  = set(classes) if classes is not None else None
        detections  = []

        for pred in predictions:
            xyxy = _to_xyxy(pred)
            if xyxy is None:
                continue
            raw_name = str(_get(pred, "class_name", _get(pred, "class", "player")))
            raw_cid  = _get(pred, "class_id", None)
            conf     = float(_get(pred, "confidence", 0.0))
            cid      = _resolve_class_id(raw_name, raw_cid)
            if target_set is not None and cid not in target_set:
                continue
            x1, y1, x2, y2 = xyxy
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": raw_name,
                "class_id":   cid,
            })

        return detections

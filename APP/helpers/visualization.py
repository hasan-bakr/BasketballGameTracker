"""
Visualization Helpers
=====================
Standalone drawing utilities for BotSort tracks and mask overlays.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Set


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_tracks_with_ids(
    frame: np.ndarray,
    tracks: list,
    colors: List[tuple],
    jersey_bank,
) -> np.ndarray:
    """Draw a dot at player feet + floating badge above. Jersey-locked tracks get a halo."""
    result = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        tid    = track["track_id"]
        is_ref = track["is_referee"]

        fx = (x1 + x2) // 2
        fy = y2
        bbox_h = max(1, y2 - y1)
        dot_r  = max(5, min(10, bbox_h // 12))

        if is_ref:
            color  = (0, 200, 255)
            jersey = None
            label  = "REF"
        else:
            color  = colors[tid % len(colors)]
            jersey = jersey_bank.get_jersey(tid)
            label  = f"#{jersey}" if jersey else str(tid)

        # Foot dot
        if jersey:
            cv2.circle(result, (fx, fy), dot_r + 4, color, 2, cv2.LINE_AA)
        cv2.circle(result, (fx, fy), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(result, (fx, fy), dot_r, (255, 255, 255), 1, cv2.LINE_AA)

        # Floating badge above head
        font_scale = 0.75 if jersey else 0.55
        thickness  = 2 if jersey else 1
        font       = cv2.FONT_HERSHEY_DUPLEX if jersey else FONT
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 5
        lx  = fx - tw // 2
        ly  = y1 - pad * 2
        bx1, by1 = lx - pad, max(ly - th - pad, 0)
        bx2, by2 = lx + tw + pad, max(ly + pad // 2, 0)

        cv2.rectangle(result, (bx1, by1), (bx2, by2), color, -1)
        if jersey:
            cv2.rectangle(result, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(result, label, (lx, ly), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return result


def _make_stripe_pattern(h: int, w: int, stripe_width: int = 8) -> np.ndarray:
    """Return a boolean array of diagonal stripe pixels (45-degree)."""
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    return ((x_idx + y_idx) // stripe_width) % 2 == 0


def draw_masks_with_ids(
    frame: np.ndarray,
    masks: Dict[int, np.ndarray],
    colors: List[tuple],
    jersey_bank,
    confidences: Optional[Dict[int, float]] = None,
    ball_obj_ids: Optional[Set[int]] = None,
) -> np.ndarray:
    """Draw segmentation masks with player ID / jersey labels.

    Rendering rules:
    - Ball IDs  → vivid orange filled mask + "Ball" label.
    - Jersey detected players → diagonal zebra-stripe mask + opaque jersey
      number label (LinkedIn-friendly style).
    - Regular players  → semi-transparent solid colour overlay + ID label.

    Args:
        frame: BGR source frame.
        masks: {obj_id: boolean mask} dict.
        colors: List of BGR color tuples indexed by obj_id.
        jersey_bank: JerseyReIDBank instance for looking up jersey numbers.
        confidences: Optional {obj_id: float} (kept for API compatibility).
        ball_obj_ids: Set of obj_ids that correspond to the basketball.

    Returns:
        Annotated copy of the frame.
    """
    if ball_obj_ids is None:
        ball_obj_ids = set()

    result = frame.copy()
    h, w = frame.shape[:2]
    stripe = _make_stripe_pattern(h, w)  # shared diagonal pattern

    # Split obj_ids into three render groups so jersey-detected always wins overlaps:
    #   Pass 1 — balls + regular players (solid overlay, renders first)
    #   Pass 2 — jersey-detected players (stripe, renders on top → wins any overlap)
    jersey_ids  = {oid for oid in masks if oid not in ball_obj_ids and jersey_bank.get_jersey(oid)}
    regular_ids = {oid for oid in masks if oid not in ball_obj_ids and oid not in jersey_ids}
    render_order = list(ball_obj_ids & masks.keys()) + list(regular_ids) + list(jersey_ids)

    # Pre-compute jersey bboxes for overlap suppression (bounding box IoU)
    def _mask_bbox(m: np.ndarray):
        """Returns (x1, y1, x2, y2) bounding box of a boolean mask, or None."""
        ys, xs = np.where(m)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _bbox_iou(a, b) -> float:
        """Intersection / min(area_a, area_b) for two (x1,y1,x2,y2) boxes."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        denom = min(area_a, area_b)
        return inter / denom if denom > 0 else 0.0

    jersey_bboxes: List[tuple] = []
    for oid in jersey_ids:
        jm = masks[oid]
        if jm.shape[:2] != (h, w):
            jm = cv2.resize(jm.astype(np.float32), (w, h)) > 0.5
        bb = _mask_bbox(jm)
        if bb is not None:
            jersey_bboxes.append(bb)

    for obj_id in render_order:
        mask = masks[obj_id]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

        color = colors[obj_id % len(colors)]
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── Ball ──────────────────────────────────────────────────────────────
        if obj_id in ball_obj_ids:
            BALL_COLOR = (0, 140, 255)  # vivid orange (BGR)
            result[mask] = (result[mask] * 0.35 + np.array(BALL_COLOR) * 0.65).astype(np.uint8)
            cv2.drawContours(result, contours, -1, BALL_COLOR, 3)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = "Ball"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(result,
                                  (cx - tw // 2 - 4, cy - th - 6),
                                  (cx + tw // 2 + 4, cy + 4),
                                  BALL_COLOR, -1)
                    cv2.putText(result, label, (cx - tw // 2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            continue

        jersey = jersey_bank.get_jersey(obj_id)

        # ── Jersey-detected player → zebra stripe mask ────────────────────────
        if jersey:
            # Stripe areas: vivid colour; gaps: near-transparent dark fill
            stripe_in_mask = mask & stripe
            gap_in_mask    = mask & ~stripe
            color_arr = np.array(color, dtype=np.float32)
            result[stripe_in_mask] = np.clip(
                result[stripe_in_mask] * 0.25 + color_arr * 0.75, 0, 255
            ).astype(np.uint8)
            result[gap_in_mask] = np.clip(
                result[gap_in_mask] * 0.55 + color_arr * 0.10, 0, 255
            ).astype(np.uint8)
            cv2.drawContours(result, contours, -1, color, 2)

            # Opaque badge with jersey number — LinkedIn style
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text = f"#{jersey}"
                    font_scale = 0.85
                    thickness  = 2
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                    )
                    pad = 6
                    bx1, by1 = cx - tw // 2 - pad, cy - th - pad - baseline
                    bx2, by2 = cx + tw // 2 + pad, cy + baseline + pad // 2
                    # Fully opaque background
                    cv2.rectangle(result, (bx1, by1), (bx2, by2), color, -1)
                    # Dark border
                    cv2.rectangle(result, (bx1, by1), (bx2, by2), (30, 30, 30), 1)
                    cv2.putText(result, text, (cx - tw // 2, cy),
                                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)

        # ── Regular player → semi-transparent solid ───────────────────────────
        else:
            # Jersey-detected bir oyuncunun bbox'ıyla >30% örtüşme varsa atla
            # (maske pikselleri ayrı bölgelerde olsa bile bbox çakışmasını yakalar)
            reg_bb = _mask_bbox(mask)
            if reg_bb is not None and any(
                _bbox_iou(reg_bb, jbb) > 0.30 for jbb in jersey_bboxes
            ):
                continue
            result[mask] = (result[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            cv2.drawContours(result, contours, -1, color, 2)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text = f"ID:{obj_id}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(result,
                                  (cx - tw // 2 - 3, cy - th - 5),
                                  (cx + tw // 2 + 3, cy + 5),
                                  color, -1)
                    cv2.putText(result, text, (cx - tw // 2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return result

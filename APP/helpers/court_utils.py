"""
Court Utilities
===============
Basketball court constants, homography computation, and drawing helpers
for tactical bird's-eye view projection.
"""

import os
import cv2
import colorsys
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Court dimensions ──────────────────────────────────────────────────────────
TACTICAL_WIDTH = 300
TACTICAL_HEIGHT = 161
ACTUAL_WIDTH_M = 28.0
ACTUAL_HEIGHT_M = 15.0

# ── Tactical keypoint positions (pixel coords on the 300x161 court image) ─────
TACTICAL_KEYPOINTS = [
    (0, 0),
    (0, int((0.91 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int((14.1 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (0, int(TACTICAL_HEIGHT)),
    (int(TACTICAL_WIDTH / 2), TACTICAL_HEIGHT),
    (int(TACTICAL_WIDTH / 2), 0),
    (int((5.79 / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (int((5.79 / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int(TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((14.1 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, int((0.91 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (TACTICAL_WIDTH, 0),
    (int(((ACTUAL_WIDTH_M - 5.79) / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((5.18 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
    (int(((ACTUAL_WIDTH_M - 5.79) / ACTUAL_WIDTH_M) * TACTICAL_WIDTH), int((10.0 / ACTUAL_HEIGHT_M) * TACTICAL_HEIGHT)),
]

# ── Keypoint display metadata ─────────────────────────────────────────────────
KP_COLORS = [
    (0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 100, 0), (0, 200, 0), (0, 255, 0),
    (255, 255, 0), (255, 255, 0),
    (0, 0, 255), (0, 0, 255),
    (255, 0, 0), (255, 50, 0), (255, 100, 0), (255, 150, 0), (255, 50, 0), (255, 0, 0),
    (0, 165, 255), (0, 165, 255),
]

KP_NAMES = [
    "L-TL", "L-BL1", "L-BL2", "L-BL3", "L-BL4", "L-BotL",
    "Mid-Bot", "Mid-Top",
    "FT-L1", "FT-L2",
    "R-BotR", "R-BR4", "R-BR3", "R-BR2", "R-BR1", "R-TR",
    "FT-R1", "FT-R2",
]

# Roboflow basketball-court-detection-2/19 uses the Roboflow Sports NBA
# basketball schema: 33 labels out of a 41-point court annotation layout.
ROBOFLOW_COURT_KEYPOINT_LABELS = [
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13",
    "14", "15", "16", "17", "19", "21", "23", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "34", "35", "37", "38", "40", "41",
]

# Native Roboflow keypoint index -> the older 18-point tactical index.
# Kept for debug logs and compatibility with the local 18-keypoint model path.
ROBOFLOW_TO_TACTICAL_INDEX = [
    0,   # 01 left top corner
    1,   # 02 left sideline, three-point corner
    2,   # 04 left sideline, paint top
    3,   # 05 left sideline, paint bottom
    4,   # 07 left sideline, three-point corner
    5,   # 08 left bottom corner
    -1,  # 09 left rim center
    -1,  # 10 left three-point straight top end
    -1,  # 11 left three-point straight bottom end
    8,   # 12 left paint/free-throw top
    -1,  # 13 left free-throw center
    9,   # 14 left paint/free-throw bottom
    -1,  # 15 left top throw-in mark
    -1,  # 16 left three-point arc center
    -1,  # 17 left bottom throw-in mark
    7,   # 19 half-court top
    -1,  # 21 court center
    6,   # 23 half-court bottom
    -1,  # 25 right top throw-in mark
    -1,  # 26 right three-point arc center
    -1,  # 27 right bottom throw-in mark
    16,  # 28 right paint/free-throw top
    -1,  # 29 right free-throw center
    17,  # 30 right paint/free-throw bottom
    -1,  # 31 right three-point straight top end
    -1,  # 32 right three-point straight bottom end
    -1,  # 33 right rim center
    15,  # 34 right top corner
    14,  # 35 right sideline, three-point corner
    13,  # 37 right sideline, paint top
    12,  # 38 right sideline, paint bottom
    11,  # 40 right sideline, three-point corner
    10,  # 41 right bottom corner
]

ROBOFLOW_LABEL_TO_TACTICAL_INDEX = {
    label: ROBOFLOW_TO_TACTICAL_INDEX[idx]
    for idx, label in enumerate(ROBOFLOW_COURT_KEYPOINT_LABELS)
}

# Roboflow-native NBA court vertices projected to the tactical 300x161 image.
# Source geometry is the Roboflow Sports basketball CourtConfiguration:
# length=2865cm, width=1524cm, paint_length=579cm, paint_width=488cm,
# sideline_to_three_point_line=91cm, baseline_to_rim_center=160cm,
# three_point_arc_radius=724cm, baseline_to_throw_line=835cm.
ROBOFLOW_TACTICAL_KEYPOINTS = [
    (0, 0),
    (0, 10),
    (0, 55),
    (0, 106),
    (0, 151),
    (0, 161),
    (17, 80),
    (44, 10),
    (44, 151),
    (61, 55),
    (61, 80),
    (61, 106),
    (87, 0),
    (93, 80),
    (87, 161),
    (150, 0),
    (150, 80),
    (150, 161),
    (213, 0),
    (207, 80),
    (213, 161),
    (239, 55),
    (239, 80),
    (239, 106),
    (256, 10),
    (256, 151),
    (283, 80),
    (300, 0),
    (300, 10),
    (300, 55),
    (300, 106),
    (300, 151),
    (300, 161),
]

# ── Default model / asset paths ───────────────────────────────────────────────
DEFAULT_KEYPOINT_MODEL = os.path.join(ROOT_DIR, "models", "keypoints", "yolo26l-fine-tuned.pt")
DEFAULT_COURT_IMAGE = os.path.join(ROOT_DIR, "APP", "assets", "basketball_court.png")


# ── Helper functions ──────────────────────────────────────────────────────────

def _track_color_bgr(track_id: int) -> tuple:
    h = ((int(track_id) % 60) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple:
    return (
        int(np.clip(round(x), 0, width - 1)),
        int(np.clip(round(y), 0, height - 1)),
    )

def compute_homography(
    keypoints_xy: np.ndarray,
    max_reproj_px: float = 60.0,
    dst_keypoints: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Compute a RANSAC homography matrix from detected court keypoints.

    Args:
        keypoints_xy: (N, 2) array of detected keypoint pixel positions.
                      Zero entries indicate undetected keypoints.
        max_reproj_px: Reject H if median src->dst reprojection error exceeds this.
        dst_keypoints: Optional tactical destination points. Defaults to TACTICAL_KEYPOINTS.

    Returns:
        3x3 homography matrix, or None if sanity checks fail.
    """
    valid_indices = [i for i in range(len(keypoints_xy))
                     if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0]
    if len(valid_indices) < 4:
        return None
    dst_template = TACTICAL_KEYPOINTS if dst_keypoints is None else dst_keypoints
    src = np.array([keypoints_xy[i] for i in valid_indices], dtype=np.float32)
    dst = np.array([dst_template[i] for i in valid_indices], dtype=np.float32)
    try:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    except cv2.error:
        return None
    if H is None or not np.isfinite(H).all():
        return None
    if abs(float(np.linalg.det(H))) < 1e-8:
        return None
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(projected - dst, axis=1)
    if float(np.median(residuals)) > max_reproj_px:
        return None
    return H


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
    draw_labels: bool = True,
) -> np.ndarray:
    """Overlay detected court keypoints on a frame.

    Args:
        frame: BGR image to draw on (modified in place).
        keypoints_xy: (N, 2) detected positions; zeros are skipped.
        confidence: Optional (N,) confidence scores.
        min_confidence: Skip points below this confidence when confidence is available.
        draw_labels: Draw keypoint names/confidence next to points.

    Returns:
        The annotated frame.
    """
    for i in range(len(keypoints_xy)):
        x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
        if x <= 0 and y <= 0:
            continue
        if confidence is not None and i < len(confidence):
            conf = float(confidence[i])
            if not np.isfinite(conf) or conf < min_confidence:
                continue
        color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
        if len(keypoints_xy) == len(ROBOFLOW_COURT_KEYPOINT_LABELS):
            name = f"RF-{ROBOFLOW_COURT_KEYPOINT_LABELS[i]}"
        else:
            name = KP_NAMES[i] if i < len(KP_NAMES) else str(i)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        if draw_labels:
            if confidence is not None and i < len(confidence):
                label = f"{name} ({float(confidence[i]):.2f})"
            else:
                label = name
            cv2.putText(frame, label, (x + 12, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def build_minimal_court(width: int = TACTICAL_WIDTH, height: int = TACTICAL_HEIGHT) -> np.ndarray:
    """Render a clean NBA-style top-down court for tactical output."""
    scale = 4
    w, h = width * scale, height * scale
    img = np.zeros((h, w, 3), dtype=np.uint8)

    wood = np.array((78, 126, 176), dtype=np.uint8)
    wood_dark = np.array((69, 112, 160), dtype=np.uint8)
    img[:, :] = wood
    for x in range(0, w, 28 * scale):
        img[:, x:x + 14 * scale] = wood_dark

    line = (245, 248, 250)
    muted_line = (224, 231, 236)
    paint = (42, 58, 174)
    paint_dark = (34, 47, 145)
    restricted_fill = (64, 76, 185)
    t = max(1, 2 * scale)

    def p(x: float, y: float) -> Tuple[int, int]:
        return int(round(x * scale)), int(round(y * scale))

    def r(v: float) -> int:
        return int(round(v * scale))

    y_top = int((518 / 1524) * height)
    y_bot = int((1006 / 1524) * height)
    left_paint_x = int((579 / 2865) * width)
    right_paint_x = width - left_paint_x
    ft_radius = int((183 / 2865) * width)
    restricted_radius = int((122 / 2865) * width)
    three_radius = int((724 / 2865) * width)

    cv2.rectangle(img, p(0, y_top), p(left_paint_x, y_bot), paint, -1, cv2.LINE_AA)
    cv2.rectangle(img, p(right_paint_x, y_top), p(width - 1, y_bot), paint, -1, cv2.LINE_AA)
    cv2.rectangle(img, p(0, y_top), p(left_paint_x, y_bot), paint_dark, t, cv2.LINE_AA)
    cv2.rectangle(img, p(right_paint_x, y_top), p(width - 1, y_bot), paint_dark, t, cv2.LINE_AA)

    cv2.rectangle(img, p(1, 1), p(width - 2, height - 2), line, t, cv2.LINE_AA)
    cv2.line(img, p(width / 2, 1), p(width / 2, height - 2), muted_line, t, cv2.LINE_AA)
    cv2.circle(img, p(width / 2, height / 2), r(ft_radius), muted_line, t, cv2.LINE_AA)

    left_rim = p((160 / 2865) * width, height / 2)
    right_rim = p(width - (160 / 2865) * width, height / 2)
    left_ft = p(left_paint_x, height / 2)
    right_ft = p(right_paint_x, height / 2)

    for center, start, end in ((left_rim, -90, 90), (right_rim, 90, 270)):
        cv2.ellipse(img, center, (r(restricted_radius), r(restricted_radius)),
                    0, start, end, restricted_fill, -1, cv2.LINE_AA)
        cv2.ellipse(img, center, (r(restricted_radius), r(restricted_radius)),
                    0, start, end, line, t, cv2.LINE_AA)

    cv2.circle(img, left_ft, r(ft_radius), muted_line, t, cv2.LINE_AA)
    cv2.circle(img, right_ft, r(ft_radius), muted_line, t, cv2.LINE_AA)

    corner_y_top = int((91 / 1524) * height)
    corner_y_bot = height - corner_y_top
    straight_x = int((424 / 2865) * width)
    cv2.line(img, p(1, corner_y_top), p(straight_x, corner_y_top), line, t, cv2.LINE_AA)
    cv2.line(img, p(1, corner_y_bot), p(straight_x, corner_y_bot), line, t, cv2.LINE_AA)
    cv2.line(img, p(width - 1, corner_y_top), p(width - straight_x, corner_y_top), line, t, cv2.LINE_AA)
    cv2.line(img, p(width - 1, corner_y_bot), p(width - straight_x, corner_y_bot), line, t, cv2.LINE_AA)
    cv2.ellipse(img, left_rim, (r(three_radius), r(three_radius)),
                0, -69, 69, line, t, cv2.LINE_AA)
    cv2.ellipse(img, right_rim, (r(three_radius), r(three_radius)),
                0, 111, 249, line, t, cv2.LINE_AA)

    rim_radius = max(2, r(23 / 2865 * width))
    cv2.circle(img, left_rim, rim_radius, (20, 35, 55), -1, cv2.LINE_AA)
    cv2.circle(img, right_rim, rim_radius, (20, 35, 55), -1, cv2.LINE_AA)
    cv2.circle(img, left_rim, rim_radius, line, 1 * scale, cv2.LINE_AA)
    cv2.circle(img, right_rim, rim_radius, line, 1 * scale, cv2.LINE_AA)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def draw_tactical_view(
    court_img: np.ndarray,
    keypoints_xy: np.ndarray,
    H: Optional[np.ndarray],
    player_ids: Optional[List[int]] = None,
    player_feet: Optional[List[List[float]]] = None,
    player_jerseys: Optional[List[Optional[str]]] = None,
    position_history: Optional[Dict[int, object]] = None,
    point_radius: int = 7,
    trail_thickness: int = 2,
    label_scale: float = 0.32,
    tactical_keypoints: Optional[np.ndarray] = None,
    tactical_keypoint_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Render a bird's-eye tactical view onto the court template image.

    Args:
        court_img: Base court image (300x161 by default).
        keypoints_xy: (N, 2) detected keypoint positions (unused for drawing, kept for API compat).
        H: 3x3 homography matrix (camera → court). None → return plain court.
        draw_players: If False, skip player circles and trails (used when H is stale/untrusted).
        label_scale: Font scale for the in-circle label.

    Returns:
        Annotated tactical view image.
    """
    tactical = court_img.copy()

    if H is None and not position_history:
        return tactical

    if tactical_keypoints is not None:
        for i, pt in enumerate(tactical_keypoints):
            x, y = float(pt[0]), float(pt[1])
            if x < 0 or y < 0:
                continue
            px, py = _clamp_point(x, y, TACTICAL_WIDTH, TACTICAL_HEIGHT)
            label = (
                tactical_keypoint_labels[i]
                if tactical_keypoint_labels and i < len(tactical_keypoint_labels)
                else str(i)
            )
            cv2.circle(tactical, (px, py), 3, (18, 24, 34), -1, cv2.LINE_AA)
            cv2.circle(tactical, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                tactical,
                label,
                (px + 4, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.24,
                (12, 18, 28),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                tactical,
                label,
                (px + 4, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.24,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Draw movement trails
    if position_history:
        for track_id, history in position_history.items():
            pts = [_clamp_point(pt[0], pt[1], TACTICAL_WIDTH, TACTICAL_HEIGHT) for pt in history]
            n = len(pts)
            if n >= 2:
                color = _track_color_bgr(int(track_id))
                for i in range(1, n):
                    alpha = i / n
                    blended = tuple(int(c * (0.25 + 0.75 * alpha)) for c in color)
                    cv2.line(tactical, pts[i - 1], pts[i], blended, trail_thickness, cv2.LINE_AA)

    # Draw player circles with in-circle label (jersey if available, else track ID)
    if player_feet:
        for idx, foot_pt in enumerate(player_feet):
            track_id = player_ids[idx] if player_ids and idx < len(player_ids) else idx
            history = position_history.get(int(track_id)) if position_history else None
            try:
                if history and len(history) > 0:
                    px, py = _clamp_point(history[-1][0], history[-1][1], TACTICAL_WIDTH, TACTICAL_HEIGHT)
                else:
                    continue  # no trusted history yet, skip until positioned correctly
                color = _track_color_bgr(int(track_id))
                cv2.circle(tactical, (px + 1, py + 2), point_radius + 2, (28, 34, 42), -1, cv2.LINE_AA)
                cv2.circle(tactical, (px, py), point_radius + 2, (248, 250, 252), -1, cv2.LINE_AA)
                cv2.circle(tactical, (px, py), point_radius, color, -1, cv2.LINE_AA)
                cv2.circle(tactical, (px, py), point_radius + 1, (20, 24, 30), 1, cv2.LINE_AA)
                jersey = player_jerseys[idx] if (player_jerseys and idx < len(player_jerseys)) else None
                label = jersey if jersey else str(int(track_id))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
                tx = px - tw // 2
                ty = py + th // 2
                cv2.putText(tactical, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            label_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(tactical, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            label_scale, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass

    return tactical

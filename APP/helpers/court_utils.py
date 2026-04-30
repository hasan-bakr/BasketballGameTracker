"""
Court Utilities
===============
Basketball court constants, homography computation, and drawing helpers
for tactical bird's-eye view projection.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional

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

# ── Default model / asset paths ───────────────────────────────────────────────
DEFAULT_KEYPOINT_MODEL = os.path.join(ROOT_DIR, "models", "keypoints", "yolo26l-fine-tuned.pt")
DEFAULT_COURT_IMAGE = os.path.join(ROOT_DIR, "APP", "assets", "basketball_court.png")


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_homography(keypoints_xy: np.ndarray) -> Optional[np.ndarray]:
    """Compute a RANSAC homography matrix from detected court keypoints.

    Args:
        keypoints_xy: (18, 2) array of detected keypoint pixel positions.
                      Zero entries indicate undetected keypoints.

    Returns:
        3x3 homography matrix, or None if fewer than 4 valid keypoints.
    """
    valid_indices = [i for i in range(len(keypoints_xy))
                     if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0]
    if len(valid_indices) < 4:
        return None
    src = np.array([keypoints_xy[i] for i in valid_indices], dtype=np.float32)
    dst = np.array([TACTICAL_KEYPOINTS[i] for i in valid_indices], dtype=np.float32)
    try:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H
    except cv2.error:
        return None


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """Overlay detected court keypoints on a frame.

    Args:
        frame: BGR image to draw on (modified in place).
        keypoints_xy: (18, 2) detected positions; zeros are skipped.
        confidence: Optional (18,) confidence scores shown as labels.

    Returns:
        The annotated frame.
    """
    for i in range(len(keypoints_xy)):
        x, y = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
        if x <= 0 and y <= 0:
            continue
        color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
        name = KP_NAMES[i] if i < len(KP_NAMES) else str(i)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        if confidence is not None and i < len(confidence):
            label = f"{name} ({confidence[i]:.2f})"
        else:
            label = name
        cv2.putText(frame, label, (x + 12, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def draw_tactical_view(
    court_img: np.ndarray,
    keypoints_xy: np.ndarray,
    H: Optional[np.ndarray],
    player_feet: Optional[List[List[float]]] = None,
    player_jerseys: Optional[List[Optional[str]]] = None,
    position_history: Optional[Dict[int, object]] = None,
) -> np.ndarray:
    """Render a bird's-eye tactical view by projecting keypoints and player
    positions through the homography matrix onto the court template image.

    Args:
        court_img: Base court image (300x161 by default).
        keypoints_xy: (18, 2) detected keypoint positions.
        H: 3x3 homography matrix (camera → court). None → return plain court.
        player_feet: List of [x, y] foot positions in camera frame.
        player_jerseys: Jersey number strings aligned with player_feet.

    Returns:
        Annotated tactical view image.
    """
    tactical = court_img.copy()
    SNAP_THRESHOLD = 60

    if H is None:
        return tactical

    # Draw projected keypoints
    valid_indices = [i for i in range(len(keypoints_xy))
                     if keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0]
    for i in valid_indices:
        src_pt = np.array([[keypoints_xy[i]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt, H)
        tx, ty = dst_pt[0][0][0], dst_pt[0][0][1]
        expected_x, expected_y = TACTICAL_KEYPOINTS[i]
        if np.sqrt((tx - expected_x) ** 2 + (ty - expected_y) ** 2) <= SNAP_THRESHOLD:
            tx, ty = int(expected_x), int(expected_y)
        else:
            tx, ty = int(tx), int(ty)
        if 0 <= tx <= TACTICAL_WIDTH and 0 <= ty <= TACTICAL_HEIGHT:
            draw_x = min(tx, TACTICAL_WIDTH - 1)
            draw_y = min(ty, TACTICAL_HEIGHT - 1)
            color = KP_COLORS[i] if i < len(KP_COLORS) else (0, 255, 0)
            cv2.circle(tactical, (draw_x, draw_y), 5, color, -1)
            cv2.circle(tactical, (draw_x, draw_y), 6, (255, 255, 255), 1)

    # Draw movement trails
    if position_history and H is not None:
        for history in position_history.values():
            pts = []
            for foot in history:
                src = np.array([[foot]], dtype=np.float32)
                try:
                    dst = cv2.perspectiveTransform(src, H)
                    px, py = int(dst[0][0][0]), int(dst[0][0][1])
                    if 0 <= px < TACTICAL_WIDTH and 0 <= py < TACTICAL_HEIGHT:
                        pts.append((px, py))
                except Exception:
                    continue
            n = len(pts)
            if n >= 2:
                for i in range(1, n):
                    alpha = i / n
                    intensity = int(60 + 195 * alpha)
                    cv2.line(tactical, pts[i - 1], pts[i], (0, intensity, intensity), 1, cv2.LINE_AA)

    # Draw player positions
    if player_feet:
        for idx, foot_pt in enumerate(player_feet):
            src_pt = np.array([[foot_pt]], dtype=np.float32)
            try:
                dst_pt = cv2.perspectiveTransform(src_pt, H)
                px, py = int(dst_pt[0][0][0]), int(dst_pt[0][0][1])
                if 0 <= px < TACTICAL_WIDTH and 0 <= py < TACTICAL_HEIGHT:
                    cv2.circle(tactical, (px, py), 6, (255, 255, 255), -1)
                    cv2.circle(tactical, (px, py), 7, (0, 0, 0), 2)
                    if player_jerseys and idx < len(player_jerseys) and player_jerseys[idx]:
                        cv2.putText(tactical, player_jerseys[idx],
                                    (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35, (255, 255, 255), 1)
            except Exception:
                pass

    return tactical

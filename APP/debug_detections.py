"""
debug_detections.py
===================
YOLO detection debug aracı — SAM2 olmadan hızlı çalışır.

Her frame üzerine:
  • Yeşil  bbox → saha içi (court filter PASS)
  • Kırmızı bbox → saha dışı (court filter REJECT)
  • Sarı   bbox → player sınıfı DEĞİL ama YOLO'nun gördüğü her şey
  • Her bbox üzerinde: cls_id | conf | reject_reason

Kullanım:
    python -m APP.debug_detections \\
        --input  videos/input/nba_game_h264.mp4 \\
        --output videos/output/debug_det.mp4 \\
        --start  150  --max-frames 150  --frame-skip 2
"""

import argparse
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from APP.helpers.yolo_detector import YoloDetector
from APP.helpers.court_filter  import CourtFilterMixin
from APP.helpers.court_utils   import (
    DEFAULT_KEYPOINT_MODEL, DEFAULT_COURT_IMAGE,
    draw_keypoints_on_frame,
)

# Renk haritası: sınıf id → BGR
CLASS_COLORS = {
    0: (200, 200, 200),  # ball
    1: (255, 165,   0),  # rim
    2: (255, 255,   0),  # number
    3: (  0, 200, 255),  # player-A
    4: (  0, 200, 255),
    5: (  0, 200, 255),
    6: (  0, 200, 255),
    7: (  0, 200, 255),
    8: (255,  80, 200),  # referee
    9: (200,  80, 255),  # rim-det
}
CLASS_NAMES = {
    0: "ball", 1: "rim", 2: "number",
    3: "plA", 4: "plB", 5: "plC", 6: "plD", 7: "plE",
    8: "ref", 9: "rim2",
}
PLAYER_CLASSES = {3, 4, 5, 6, 7}

# ── Interp helper (standalone; mixin'den bağımsız) ────────────────────────────
from APP.helpers.court_filter import interp_x_at_y


def _vis_kp(keypoints_xy, indices):
    return sorted(
        [(float(keypoints_xy[i][0]), float(keypoints_xy[i][1]))
         for i in indices
         if i < len(keypoints_xy) and keypoints_xy[i][0] > 0 and keypoints_xy[i][1] > 0],
        key=lambda p: p[1],
    )


LEFT_KP   = [0, 1, 2, 3, 4, 5]
RIGHT_KP  = [10, 11, 12, 13, 14, 15]
TOP_KP    = [0, 7, 15]
BOT_KP    = [5, 6, 10]
INNER_KP  = {8, 9, 16, 17}


def court_reject_reason(bbox, keypoints_xy) -> str:
    """Hangi sınırda reddedildi? '' → geçti."""
    if not any(kp[0] > 0 and kp[1] > 0 for kp in keypoints_xy):
        return ""

    x1, y1, x2, y2 = [float(c) for c in bbox]
    cy = (y1 + y2) * 0.5

    left_pts = _vis_kp(keypoints_xy, LEFT_KP)
    if left_pts and x2 < interp_x_at_y(left_pts, cy):
        return "LEFT"

    right_pts = _vis_kp(keypoints_xy, RIGHT_KP)
    if right_pts and x1 > interp_x_at_y(right_pts, cy):
        return "RIGHT"

    top_pts  = _vis_kp(keypoints_xy, TOP_KP)
    edge_pts = _vis_kp(keypoints_xy, [i for i in range(18) if i not in INNER_KP])
    ref_top  = top_pts if top_pts else edge_pts
    if ref_top and y2 < min(p[1] for p in ref_top):
        return "TOP"

    bot_pts = _vis_kp(keypoints_xy, BOT_KP)
    if bot_pts and y2 > max(p[1] for p in bot_pts):
        return "BOTTOM"

    return ""


def size_reject_reason(bbox, min_height: int, min_area: int) -> str:
    """Boyut filtresine takılıyor mu? '' → geçti."""
    x1, y1, x2, y2 = [float(c) for c in bbox]
    h = y2 - y1
    a = (x2 - x1) * h
    if h < min_height:
        return f"H<{min_height}(h={h:.0f})"
    if a < min_area:
        return f"A<{min_area}(a={a:.0f})"
    return ""


def draw_detection(frame, det, court_reason: str, size_reason: str, show_all: bool):
    x1, y1, x2, y2 = [int(c) for c in det['bbox']]
    cls  = det['class_id']
    conf = det['confidence']
    is_player = cls in PLAYER_CLASSES

    if not show_all and not is_player:
        return frame

    if is_player:
        if size_reason:
            color = (0, 140, 255)   # turuncu — boyut filtresi
        elif court_reason:
            color = (0, 60, 220)    # kırmızı — saha dışı
        else:
            color = (0, 220, 60)    # yeşil — geçti
    else:
        color = CLASS_COLORS.get(cls, (180, 180, 180))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    h_px = y2 - y1
    a_px = (x2 - x1) * h_px
    size_tag = f" h={h_px} a={a_px}" if is_player else ""
    label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}{size_tag}"
    reason = size_reason or court_reason
    if reason:
        label += f" [{reason}]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    lx, ly = x1, max(y1 - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 3), (lx + tw + 4, ly + 1), color, -1)
    cv2.putText(frame, label, (lx + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return frame




def draw_boundary_lines(frame, keypoints_xy):
    """Sol '/' ve sağ '\' sınır çizgilerini frame'e çiz."""
    h, w = frame.shape[:2]

    def _pts(indices):
        return _vis_kp(keypoints_xy, indices)

    # Sol sınır — sarı noktalı çizgi
    lpts = _pts(LEFT_KP)
    if len(lpts) >= 2:
        for i in range(len(lpts) - 1):
            cv2.line(frame,
                     (int(lpts[i][0]), int(lpts[i][1])),
                     (int(lpts[i+1][0]), int(lpts[i+1][1])),
                     (0, 220, 255), 2, cv2.LINE_AA)

    # Sağ sınır — sarı noktalı çizgi
    rpts = _pts(RIGHT_KP)
    if len(rpts) >= 2:
        for i in range(len(rpts) - 1):
            cv2.line(frame,
                     (int(rpts[i][0]), int(rpts[i][1])),
                     (int(rpts[i+1][0]), int(rpts[i+1][1])),
                     (0, 220, 255), 2, cv2.LINE_AA)

    # Üst sınır — mavi yatay çizgi (min-y keypoint'te)
    top_pts  = _pts(TOP_KP)
    edge_pts = _pts([i for i in range(18) if i not in INNER_KP])
    ref_top  = top_pts if top_pts else edge_pts
    if ref_top:
        top_y = int(min(p[1] for p in ref_top))
        cv2.line(frame, (0, top_y), (w, top_y), (255, 80, 0), 2, cv2.LINE_AA)

    # Alt sınır — mavi yatay çizgi
    bot_pts = _pts(BOT_KP)
    if bot_pts:
        bot_y = int(max(p[1] for p in bot_pts))
        cv2.line(frame, (0, bot_y), (w, bot_y), (255, 80, 0), 2, cv2.LINE_AA)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO detection debug visualizer")
    parser.add_argument("--input",      "-i", required=True)
    parser.add_argument("--output",     "-o", required=True)
    parser.add_argument("--max-frames", type=int,   default=300)
    parser.add_argument("--start",      type=float, default=0.0,
                        help="Start time in seconds — same as main app (default: 0)")
    parser.add_argument("--frame-skip", type=int,   default=1)
    parser.add_argument("--conf",       type=float, default=0.10,
                        help="YOLO confidence threshold (default: 0.10 — show everything)")
    parser.add_argument("--min-height", type=int,   default=20,
                        help="PLAYER_MIN_HEIGHT_PX — bu değerin altındaki bbox'lar turuncu (default: 20)")
    parser.add_argument("--min-area",   type=int,   default=500,
                        help="PLAYER_MIN_AREA_PX   — bu değerin altındaki bbox alanları turuncu (default: 500)")
    parser.add_argument("--show-all",   action="store_true",
                        help="Show non-player detections too (ball, rim, etc.)")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--yolo-model", default="models/yolo/best_detection.pt")

    args = parser.parse_args()

    yolo    = YoloDetector(model_path=args.yolo_model, device=args.device)
    kp_model = None
    if os.path.exists(DEFAULT_KEYPOINT_MODEL):
        kp_model = YOLO(DEFAULT_KEYPOINT_MODEL)
        print(f"Keypoint model loaded: {DEFAULT_KEYPOINT_MODEL}")
    else:
        print("WARNING: keypoint model not found — boundary lines won't be drawn")

    cap     = cv2.VideoCapture(args.input)
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_raw / args.frame_skip

    # Ana app ile birebir aynı formula: fps = fps_raw / frame_skip
    fps_out     = fps_raw / args.frame_skip
    start_frame = int(args.start * fps_out)   # start_sec * (fps_raw / frame_skip)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_remaining = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    total = min(raw_remaining // args.frame_skip, args.max_frames)

    writer   = None
    fi       = 0          # output frame index
    raw_read = 0          # toplam okunan raw frame

    KEYPOINT_BORDER_MARGIN = 30

    while fi < total:
        # frame_skip: ilk frame'i al, kalanını atla
        ret, frame = cap.read()
        if not ret:
            break
        raw_read += 1
        for _ in range(args.frame_skip - 1):
            ret2, _ = cap.read()
            if not ret2:
                break
            raw_read += 1

        # ── Keypoint tespiti ───────────────────────────────────────────────
        keypoints_xy = np.zeros((18, 2), dtype=np.float32)
        confidences  = np.zeros(18, dtype=np.float32)
        if kp_model is not None:
            h_f, w_f = frame.shape[:2]
            kp_res = kp_model.predict(frame, conf=0.3, verbose=False,
                                      half=(args.device == 'cuda'), device=args.device)
            if kp_res and kp_res[0].keypoints is not None:
                kp_data = kp_res[0].keypoints
                if kp_data.xy is not None and len(kp_data.xy) > 0:
                    xy = kp_data.xy[0].cpu().numpy()
                    keypoints_xy[:min(len(xy), 18)] = xy[:18]
                    if kp_data.conf is not None and len(kp_data.conf) > 0:
                        c = kp_data.conf[0].cpu().numpy()
                        confidences[:min(len(c), 18)] = c[:18]
            m = KEYPOINT_BORDER_MARGIN
            for i in range(18):
                x, y = float(keypoints_xy[i][0]), float(keypoints_xy[i][1])
                if not (m <= x < w_f - m and m <= y < h_f - m):
                    keypoints_xy[i] = 0.0

        # ── YOLO tespiti ───────────────────────────────────────────────────
        dets = yolo.detect(frame, confidence_threshold=args.conf)

        # ── Çizim ──────────────────────────────────────────────────────────
        result = frame.copy()
        result = draw_boundary_lines(result, keypoints_xy)
        result = draw_keypoints_on_frame(result, keypoints_xy, confidences)

        n_in = n_out_court = n_out_size = 0
        for det in dets:
            is_player = det['class_id'] in PLAYER_CLASSES
            c_reason  = court_reject_reason(det['bbox'], keypoints_xy) if is_player else "non-player"
            s_reason  = size_reject_reason(det['bbox'], args.min_height, args.min_area) if (is_player and c_reason == "") else ""
            if is_player:
                if c_reason:
                    n_out_court += 1
                elif s_reason:
                    n_out_size += 1
                else:
                    n_in += 1
            result = draw_detection(result, det, c_reason, s_reason, args.show_all)

        # HUD
        total_players = sum(1 for d in dets if d['class_id'] in PLAYER_CLASSES)
        abs_frame = start_frame + raw_read - 1
        cv2.putText(result, f"Frame {abs_frame}  conf>={args.conf:.2f}  minH={args.min_height} minA={args.min_area}",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(result,
                    f"Players: {total_players}  GREEN={n_in}  RED={n_out_court}(court)  ORANGE={n_out_size}(size)",
                    (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 180), 2)

        # Legend
        cv2.rectangle(result, (12, 82), (240, 148), (30, 30, 30), -1)
        cv2.putText(result, "GREEN  = in-court, size OK",   (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 60),  1)
        cv2.putText(result, "RED    = court filter reject", (16, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 60, 220),  1)
        cv2.putText(result, "ORANGE = size filter reject",  (16, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 1)

        if writer is None:
            h, w = result.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, fps_out, (w, h))

        writer.write(result)
        fi += 1
        if fi % 20 == 0:
            print(f"  frame {fi}/{total}  players={total_players}  green={n_in}  red={n_out_court}  orange={n_out_size}")

    cap.release()
    if writer:
        writer.release()
    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()

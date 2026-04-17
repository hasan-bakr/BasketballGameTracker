"""
debug_prompt.py
===============
SAM2 seed-frame prompt debug görselleştirici — SAM2 olmadan çalışır.

Seed frame seçim sürecini ve prompt noktalarını görselleştirir:
  • Cyan   bbox + daire  → SAM2'ye seçilen prompt (in-court, boyut OK, NMS geçti)
  • Yeşil  bbox          → in-court, boyut OK ama NMS ile çıkarıldı (IoU > 0.5 çakışma)
  • Turuncu bbox         → boyut filtresi (court geçti, boyut küçük)
  • Kırmızı bbox         → saha dışı (court filter REJECT)
  • Pembe  bbox          → hakem detection (referee)
  • ★ SEED FRAME banner  → SAM2 init frame olarak seçilen kare

Kullanım:
    python -m APP.debug_prompt \\
        --input  videos/input/game.mp4 \\
        --output videos/output/debug_prompt.mp4 \\
        --start  0
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from APP.helpers.yolo_detector import YoloDetector
from APP.helpers.court_filter  import interp_x_at_y
from APP.helpers.court_utils   import (
    DEFAULT_KEYPOINT_MODEL, draw_keypoints_on_frame,
)

# ── Tracker sabitleri (robust_sam2_tracker.py ile senkron) ────────────────────
PLAYER_CLASSES       = {3, 4, 5, 6, 7}
REFEREE_CLASS        = 8
PLAYER_MIN_CONF      = 0.8
REFEREE_MIN_CONF     = 0.85
PLAYER_MIN_HEIGHT_PX = 40
PLAYER_MIN_AREA_PX   = 900
KEYPOINT_BORDER_MARGIN = 30
CLUSTER_TRIGGER_COUNT   = 3
CLUSTER_MIN_SPREAD_PX   = 100
CLUSTER_MAX_SCAN_FRAMES = 60

LEFT_KP   = [0, 1, 2, 3, 4, 5]
RIGHT_KP  = [10, 11, 12, 13, 14, 15]
TOP_KP    = [0, 7, 15]
BOT_KP    = [5, 6, 10]
INNER_KP  = {8, 9, 16, 17}

# ── Renkler ───────────────────────────────────────────────────────────────────
COL_SELECTED  = (255, 220,   0)   # cyan-ish → seçildi, prompt var
COL_NMS_DUP   = ( 60, 220,  60)   # yeşil    → in-court ama NMS çıkardı
COL_SIZE_REJ  = (  0, 140, 255)   # turuncu  → boyut küçük
COL_COURT_REJ = ( 50,  50, 220)   # kırmızı  → saha dışı
COL_REFEREE   = (255,  80, 200)   # pembe    → hakem
COL_PROMPT_PT = (255, 255,   0)   # sarı     → prompt noktası


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _vis_kp(kp_xy, indices):
    return sorted(
        [(float(kp_xy[i][0]), float(kp_xy[i][1]))
         for i in indices
         if i < len(kp_xy) and kp_xy[i][0] > 0 and kp_xy[i][1] > 0],
        key=lambda p: p[1],
    )


def court_reject_reason(bbox, kp_xy) -> str:
    """'' → geçti. Değilse hangi kenarda reddedildiğini döndür."""
    if not any(kp[0] > 0 and kp[1] > 0 for kp in kp_xy):
        return ""
    x1, y1, x2, y2 = [float(c) for c in bbox]
    cy = (y1 + y2) * 0.5

    lpts = _vis_kp(kp_xy, LEFT_KP)
    if lpts and x2 < interp_x_at_y(lpts, cy):
        return "LEFT"

    rpts = _vis_kp(kp_xy, RIGHT_KP)
    if rpts and x1 > interp_x_at_y(rpts, cy):
        return "RIGHT"

    top_pts  = _vis_kp(kp_xy, TOP_KP)
    edge_pts = _vis_kp(kp_xy, [i for i in range(18) if i not in INNER_KP])
    ref_top  = top_pts if top_pts else edge_pts
    if ref_top and y2 < min(p[1] for p in ref_top):
        return "TOP"

    bot_pts = _vis_kp(kp_xy, BOT_KP)
    if bot_pts and y2 > max(p[1] for p in bot_pts):
        return "BOTTOM"

    return ""


def size_reject_reason(bbox) -> str:
    """'' → geçti."""
    x1, y1, x2, y2 = [float(c) for c in bbox]
    h = y2 - y1
    a = (x2 - x1) * h
    if h < PLAYER_MIN_HEIGHT_PX:
        return f"H<{PLAYER_MIN_HEIGHT_PX}(h={h:.0f})"
    if a < PLAYER_MIN_AREA_PX:
        return f"A<{PLAYER_MIN_AREA_PX}(a={a:.0f})"
    return ""


def bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def compute_spread(dets: list) -> float:
    if len(dets) < 2:
        return float('inf')
    centers = [((d['bbox'][0] + d['bbox'][2]) / 2,
                (d['bbox'][1] + d['bbox'][3]) / 2)
               for d in dets]
    total, count = 0.0, 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            total += (dx * dx + dy * dy) ** 0.5
            count += 1
    return total / count


# ── Çizim ─────────────────────────────────────────────────────────────────────

def draw_boundary_lines(frame, kp_xy):
    h, w = frame.shape[:2]
    for indices, color in [
        (LEFT_KP,  (0, 220, 255)),
        (RIGHT_KP, (0, 220, 255)),
    ]:
        pts = _vis_kp(kp_xy, indices)
        for i in range(len(pts) - 1):
            cv2.line(frame,
                     (int(pts[i][0]), int(pts[i][1])),
                     (int(pts[i+1][0]), int(pts[i+1][1])),
                     color, 2, cv2.LINE_AA)

    top_pts  = _vis_kp(kp_xy, TOP_KP)
    edge_pts = _vis_kp(kp_xy, [i for i in range(18) if i not in INNER_KP])
    ref_top  = top_pts if top_pts else edge_pts
    if ref_top:
        cv2.line(frame, (0, int(min(p[1] for p in ref_top))),
                 (w, int(min(p[1] for p in ref_top))), (255, 80, 0), 2, cv2.LINE_AA)

    bot_pts = _vis_kp(kp_xy, BOT_KP)
    if bot_pts:
        cv2.line(frame, (0, int(max(p[1] for p in bot_pts))),
                 (w, int(max(p[1] for p in bot_pts))), (255, 80, 0), 2, cv2.LINE_AA)
    return frame


def _put_label(frame, text, x1, y1, color, font_scale=0.48):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    lx, ly = int(x1), max(int(y1) - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 3), (lx + tw + 4, ly + 1), color, -1)
    cv2.putText(frame, text, (lx + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_frame(frame, dets, kp_xy, selected_boxes, frame_idx, best_idx,
               spread, clustering, scan_limit):
    """Tek bir tarama frame'ini render et."""
    out = frame.copy()
    out = draw_boundary_lines(out, kp_xy)
    out = draw_keypoints_on_frame(out, kp_xy, np.zeros(18))

    for det in dets:
        cls = det['class_id']
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(c) for c in bbox]
        conf = det['confidence']

        if cls == REFEREE_CLASS:
            color = COL_REFEREE
            label = f"REF {conf:.2f}"
        elif cls in PLAYER_CLASSES:
            c_reason = court_reject_reason(bbox, kp_xy)
            s_reason = size_reject_reason(bbox) if not c_reason else ""
            is_selected = any(bbox_iou(bbox, sb) > 0.5 for sb in selected_boxes)

            if c_reason:
                color = COL_COURT_REJ
                label = f"pl {conf:.2f} [{c_reason}]"
            elif s_reason:
                color = COL_SIZE_REJ
                label = f"pl {conf:.2f} [{s_reason}]"
            elif is_selected:
                color = COL_SELECTED
                label = f"pl {conf:.2f} ✓PROMPT"
                # Prompt noktası (bbox merkezi)
                px = (x1 + x2) // 2
                py = (y1 + y2) // 2
                cv2.circle(out, (px, py), 8, COL_PROMPT_PT, -1)
                cv2.circle(out, (px, py), 8, (0, 0, 0), 2)
            else:
                color = COL_NMS_DUP
                label = f"pl {conf:.2f} [NMS-dup]"
        else:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, label, x1, y1, color)

    # ── Seed frame banner ──────────────────────────────────────────────────────
    h, w = out.shape[:2]
    if frame_idx == best_idx:
        cv2.rectangle(out, (0, 0), (w, h), COL_SELECTED, 6)
        banner = " SEED FRAME "
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        bx, by = (w - bw) // 2, h // 2 - bh // 2
        cv2.rectangle(out, (bx - 10, by - bh - 10), (bx + bw + 10, by + 10),
                      COL_SELECTED, -1)
        cv2.putText(out, banner, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cluster_str = f" CLUSTERING EXTEND={scan_limit}" if clustering else ""
    spread_str  = f"{spread:.0f}" if spread != float('inf') else "inf"
    hud1 = f"Scan frame {frame_idx}  spread={spread_str}px{cluster_str}"
    hud2 = f"Best so far: frame {best_idx}  |  selected={len(selected_boxes)}"

    cv2.putText(out, hud1, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(out, hud2, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 60),  2)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend = [
        (COL_SELECTED,  "Cyan   = PROMPT seçildi"),
        (COL_NMS_DUP,   "Yesil  = in-court, NMS dup"),
        (COL_SIZE_REJ,  "Turuncu= boyut kucuk"),
        (COL_COURT_REJ, "Kirmizi= saha disi"),
        (COL_REFEREE,   "Pembe  = hakem"),
    ]
    lx0, ly0 = 12, h - 12 - len(legend) * 22
    cv2.rectangle(out, (lx0 - 4, ly0 - 6), (lx0 + 260, ly0 + len(legend) * 22), (30, 30, 30), -1)
    for i, (col, txt) in enumerate(legend):
        cv2.putText(out, txt, (lx0, ly0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2 seed-frame prompt debug visualizer")
    parser.add_argument("--input",      "-i", required=True)
    parser.add_argument("--output",     "-o", required=True)
    parser.add_argument("--start",      type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--frame-skip", type=int,   default=1)
    parser.add_argument("--max-frames",  type=int, default=None,
                        help="Max scan window override — overrides clustering-based limit")
    parser.add_argument("--conf",       type=float, default=PLAYER_MIN_CONF,
                        help=f"YOLO confidence (default: {PLAYER_MIN_CONF})")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--yolo-model", default="models/yolo/best_detection.pt")
    args = parser.parse_args()

    # ── Model yükleme ──────────────────────────────────────────────────────────
    yolo     = YoloDetector(model_path=args.yolo_model, device=args.device)
    kp_model = None
    if os.path.exists(DEFAULT_KEYPOINT_MODEL):
        kp_model = YOLO(DEFAULT_KEYPOINT_MODEL)
        print(f"Keypoint model: {DEFAULT_KEYPOINT_MODEL}")
    else:
        print("WARNING: keypoint model bulunamadı — saha sınırları çizilmeyecek")

    # ── Video aç ──────────────────────────────────────────────────────────────
    cap       = cv2.VideoCapture(args.input)
    fps_raw   = cap.get(cv2.CAP_PROP_FPS)
    fps_out   = fps_raw / args.frame_skip
    start_frame = int(args.start * fps_out)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ── Scan ──────────────────────────────────────────────────────────────────
    scan_limit = args.max_frames if args.max_frames else 10
    clustering = False

    scan_frames  = []   # (frame, kp_xy, all_dets, in_court_dets, spread)
    best_idx     = 0
    best_dets    = []
    best_kp      = np.zeros((18, 2), dtype=np.float32)
    best_spread  = 0.0
    best_has_kp  = False

    fi = 0
    print(f"Scanning up to {CLUSTER_MAX_SCAN_FRAMES} frames for seed selection...")

    while fi < scan_limit:
        # frame_skip uygula
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(args.frame_skip - 1):
            ret2, _ = cap.read()
            if not ret2:
                break

        # ── Keypoints ─────────────────────────────────────────────────────────
        kp_xy = np.zeros((18, 2), dtype=np.float32)
        if kp_model is not None:
            h_f, w_f = frame.shape[:2]
            kp_res = kp_model.predict(frame, conf=0.3, verbose=False,
                                      half=(args.device == 'cuda'), device=args.device)
            if kp_res and kp_res[0].keypoints is not None:
                kp_data = kp_res[0].keypoints
                if kp_data.xy is not None and len(kp_data.xy) > 0:
                    xy = kp_data.xy[0].cpu().numpy()
                    kp_xy[:min(len(xy), 18)] = xy[:18]
            m = KEYPOINT_BORDER_MARGIN
            for i in range(18):
                x, y = float(kp_xy[i][0]), float(kp_xy[i][1])
                if not (m <= x < w_f - m and m <= y < h_f - m):
                    kp_xy[i] = 0.0

        # ── YOLO ──────────────────────────────────────────────────────────────
        all_dets = yolo.detect(frame, confidence_threshold=args.conf)
        player_dets = [d for d in all_dets if d['class_id'] in PLAYER_CLASSES]

        in_court = []
        for d in player_dets:
            if court_reject_reason(d['bbox'], kp_xy):
                continue
            if size_reject_reason(d['bbox']):
                continue
            in_court.append(d)

        spread = compute_spread(in_court)

        # ── Clustering tespiti ────────────────────────────────────────────────
        is_clustered = (
            len(in_court) >= CLUSTER_TRIGGER_COUNT
            and spread < CLUSTER_MIN_SPREAD_PX
        )
        if is_clustered and not clustering:
            clustering = True
            scan_limit = CLUSTER_MAX_SCAN_FRAMES
            print(f"  [init] Frame {fi}: clustering detected (spread={spread:.0f}px)"
                  f" — scan genişletildi → {scan_limit}")

        # ── En iyi frame seçim kriteri ─────────────────────────────────────────
        has_kp = any(kp[0] > 0 and kp[1] > 0 for kp in kp_xy)
        kp_upgrade    = has_kp and not best_has_kp
        better_count  = len(in_court) > len(best_dets)
        better_spread = spread > best_spread and len(in_court) >= max(len(best_dets) - 1, 1)

        is_better = (
            kp_upgrade
            or (has_kp == best_has_kp and better_spread)
            or (has_kp == best_has_kp and better_count and not is_clustered)
        )
        if is_better:
            best_idx    = fi
            best_dets   = in_court
            best_kp     = kp_xy.copy()
            best_spread = spread
            best_has_kp = has_kp

        scan_frames.append((frame.copy(), kp_xy.copy(), list(all_dets),
                            list(in_court), spread))
        fi += 1

    cap.release()

    if not scan_frames:
        print("ERROR: hiç frame okunamadı.")
        return

    # ── NMS uygula (seed frame'deki seçilenler üzerinde) ─────────────────────
    # Tracker _initialize_objects'teki mantığın aynısı
    sorted_dets = sorted(best_dets, key=lambda d: d['confidence'], reverse=True)
    selected = []
    for det in sorted_dets:
        if not any(bbox_iou(det['bbox'], s['bbox']) > 0.5 for s in selected):
            selected.append(det)

    selected_boxes = [d['bbox'] for d in selected]
    print(f"\nSeed frame: {best_idx}  |  "
          f"in-court={len(best_dets)}  →  after NMS={len(selected)}")
    for i, d in enumerate(selected):
        x1, y1, x2, y2 = d['bbox']
        px, py = (x1 + x2) / 2, (y1 + y2) / 2
        print(f"  Prompt {i}: obj→{d['class_id']} conf={d['confidence']:.2f}"
              f"  bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
              f"  point=({px:.0f},{py:.0f})")

    # ── Çıktı dizinleri ────────────────────────────────────────────────────────
    out_stem    = os.path.splitext(args.output)[0]
    frames_dir  = out_stem + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # ── JSON log ───────────────────────────────────────────────────────────────
    log = {
        "input":      args.input,
        "start_sec":  args.start,
        "conf":       args.conf,
        "seed_frame": best_idx,
        "clustering": clustering,
        "scan_limit": scan_limit,
        "prompts": [],
        "frames":  [],
    }

    for i, d in enumerate(selected):
        x1, y1, x2, y2 = [float(c) for c in d['bbox']]
        log["prompts"].append({
            "prompt_idx": i,
            "class_id":   d['class_id'],
            "confidence": round(float(d['confidence']), 3),
            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            "point":      [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
        })

    # ── Video + frame JPEG + log ───────────────────────────────────────────────
    writer = None
    for fi, (frame, kp_xy, all_dets, in_court_dets, spread) in enumerate(scan_frames):
        s_boxes = selected_boxes if fi == best_idx else [d['bbox'] for d in in_court_dets]
        vis = draw_frame(frame, all_dets, kp_xy, s_boxes,
                         fi, best_idx, spread, clustering, scan_limit)

        # Video yazıcı (ilk frame'de aç)
        if writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, max(fps_out, 2.0), (w, h))
        writer.write(vis)

        # Bireysel JPEG
        tag = "_SEED" if fi == best_idx else ""
        img_path = os.path.join(frames_dir, f"frame_{fi:03d}{tag}.jpg")
        cv2.imwrite(img_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # JSON frame kaydı
        frame_entry = {
            "frame_idx": fi,
            "is_seed":   fi == best_idx,
            "spread_px": round(spread, 1) if spread != float('inf') else None,
            "detections": [],
        }
        for det in all_dets:
            cls = det['class_id']
            bbox = [float(c) for c in det['bbox']]
            x1, y1, x2, y2 = bbox
            if cls in PLAYER_CLASSES:
                c_r = court_reject_reason(det['bbox'], kp_xy)
                s_r = size_reject_reason(det['bbox']) if not c_r else ""
                is_sel = any(bbox_iou(det['bbox'], sb) > 0.5 for sb in s_boxes)
                if c_r:
                    status = f"court_reject:{c_r}"
                elif s_r:
                    status = f"size_reject:{s_r}"
                elif is_sel:
                    status = "prompt_selected"
                else:
                    status = "nms_removed"
            elif cls == REFEREE_CLASS:
                status = "referee"
            else:
                continue
            frame_entry["detections"].append({
                "class_id":   cls,
                "confidence": round(float(det['confidence']), 3),
                "bbox":       [int(c) for c in bbox],
                "point":      [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                "status":     status,
            })
        log["frames"].append(frame_entry)

        print(f"  frame {fi}  spread={spread:.0f}  in_court={len(in_court_dets)}"
              f"{'  ← SEED' if fi == best_idx else ''}")

    if writer:
        writer.release()

    # Seed frame'i ayrıca yüksek kalite kaydet
    seed_frame, seed_kp, seed_dets, _, seed_spread = scan_frames[best_idx]
    seed_vis = draw_frame(seed_frame, seed_dets, seed_kp, selected_boxes,
                          best_idx, best_idx, seed_spread, clustering, scan_limit)
    seed_path = out_stem + "_seed.jpg"
    cv2.imwrite(seed_path, seed_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    json_path = out_stem + "_prompts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\nDone:")
    print(f"  video      → {args.output}")
    print(f"  seed image → {seed_path}")
    print(f"  frames/    → {frames_dir}/  ({len(scan_frames)} JPEGs)")
    print(f"  log        → {json_path}")


if __name__ == "__main__":
    main()

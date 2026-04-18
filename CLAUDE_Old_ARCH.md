# CLAUDE_Old_ARCH.md

Orijinal (Ocak 2026) SAM2 tracking mimarisi — `APP/debug_sam2_simple.py` ile yeniden üretilmiş hali.
Karşılaştırma ve regresyon testi için referans döküman.

## Run Command

```bash
python -m APP.debug_sam2_simple \
    --input  videos/input/game.mp4 \
    --output videos/output/simple_out.mp4 \
    --max-frames 300 --batch-size 50 --start 70
```

**All options:**
```
--max-frames N      Frames to process (default: 300)
--batch-size N      SAM2 propagation batch size (default: 50)
--start S           Start offset in seconds (default: 0)
--frame-skip N      Process 1 in N frames (default: 1)
--conf F            YOLO confidence threshold (default: 0.5)
--device            cuda or cpu (default: cuda)
```

## Architecture Overview

Tek dosya, sade pipeline — mixin yok, court filter yok, memory bank yok.

```
debug_sam2_simple.py
  ├── extract_batch()          — frame → /dev/shm JPEG
  ├── run() → frame_idx == 0   — YOLO bbox ile SAM2 init
  ├── run() → frame_idx > 0    — add_new_mask ile batch devam
  ├── propagate_in_video()     — forward pass, logit > 0.0
  └── draw_masks()             — overlay + kontur + ID label
```

**Single-pass processing:** Propagation ve annotation aynı döngüde — double buffer yok, background thread yok.

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `PLAYER_CLASSES` | `[3,4,5,6,7]` | Seed edilecek YOLO sınıfları |
| `confidence_threshold` | `0.5` (CLI `--conf`) | YOLO init detection eşiği |
| `logit_threshold` | `0.0` | SAM2 mask: logit > 0 → True |
| `min_mask_pixels` | `100` | Maskede en az 100 piksel olmalı |
| `batch_size` | `50` (CLI) | SAM2'ye verilen JPEG sayısı |
| `mask_resize` | `256×256` | `add_new_mask` için maske boyutu |
| `frame_resize` | `1024px` (long edge) | JPEG yazılırken SAM2 input boyutu |
| `temp_dir` | `/dev/shm/dbg_sam2_simple` | Geçici JPEG klasörü (RAM disk) |

## What Does NOT Exist

Şu anki ana mimaride olan ama bu eski mimaride **olmayan** her şey:

| Özellik | Durum |
|---|---|
| Court filter (saha içi/dışı bbox filtresi) | YOK |
| Keypoint detection | YOK |
| Homography / tactical view | YOK |
| Chest-level prompt point | YOK — düz YOLO **bbox** kullanır |
| New-prompt gating (`NEW_PROMPT_MIN_CONF`, `OVERLAP_EXISTING`) | YOK |
| Edge margin filter (`PLAYER_EDGE_MARGIN_PX`) | YOK |
| Cross-batch memory (maskmem token injection) | YOK |
| Mask conflict resolution (logit argmax) | YOK — logit > 0.0, maskeler çakışabilir |
| Aspect ratio / area ratio mask filter | YOK |
| Jersey OCR / Re-ID | YOK |
| Init frame search (clustering / spread) | YOK — her zaman frame 0 |
| Double-buffer extraction | YOK |
| AMP / FP16 | YOK |
| `propagate_in_video(reverse=True)` | YOK |

## Init Logic

```
İlk batch (frame_idx == 0):
  YOLO.detect(frame[0], conf=0.5)
  → class_id in [3,4,5,6,7]
  → add_new_points_or_box(box=scaled_bbox)   # bbox ölçeği: orijinal → resized JPEG boyutuna

Devam eden batch (frame_idx > 0):
  for each tracked object:
    mask_256 = resize(last_mask, 256×256) > 0.5
    add_new_mask(frame_idx=0, mask=mask_256)
```

Hiçbir ek filtreleme yok — YOLO'nun gördüğü her oyuncu doğrudan SAM2'ye eklenir.

## Propagation Logic

```python
for out_fi, out_ids, out_logits in predictor.propagate_in_video(inference_state):
    for i, obj_id in enumerate(out_ids):
        mask = (out_logits[i] > 0.0).cpu().numpy()   # logit eşiği: 0.0
        mask = resize(mask, original_frame_size) > 0.5
        if mask.sum() > 100:
            frame_masks[obj_id] = mask
        tracked[obj_id]["last_mask"] = mask           # sonraki batch için sakla
```

Maskeler arasında **çakışma çözümü yok** — SAM2 doğal olarak hangi pixeli hangi objeye atadıysa o kalır.

## Differences vs Current Architecture

| Konu | Eski (bu dosya) | Şu anki (`python -m APP`) |
|---|---|---|
| Prompt tipi | `add_new_points_or_box` (bbox) | `add_new_points_or_box` (chest-level **point**) |
| Init frame | Her zaman frame 0 | En iyi frame aranır (clustering scan) |
| Yeni oyuncu ekleme | Sadece frame 0'da | Her batch sonunda yeni oyuncular eklenir |
| Frame edge gating | Yok | `PLAYER_EDGE_MARGIN_PX=60` |
| Gating threshold | Yok | `NEW_PROMPT_MIN_CONF=0.70`, `OVERLAP=0.30` |
| Mask logit threshold | `0.0` | `0.15` |
| Çakışan maskeler | SAM2'ye bırakılır | Logit argmax ile çözülür |
| Cross-batch memory | Yok (sadece `add_new_mask`) | 3 batch maskmem token inject |
| Court filter | Yok | `CourtFilterMixin` |
| Jersey OCR | Yok | `PlayerDetectionMixin` + PARSeq |
| Tactical view | Yok | Homography → 300×161 kuş bakışı |
| AMP | Yok | FP16 (disable: `--no-amp`) |

## Source Reference

Orijinal commit: `0e74209` (25 Ocak 2026) — `APP/helpers/robust_sam2_tracker.py` ilk hali.
Yeniden üretilmiş dosya: `APP/debug_sam2_simple.py`

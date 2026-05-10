"""Basketball Game Tracker — CLI entry point (MCByte pipeline).

Usage:
    python -m APP --input videos/input/game.mp4 --output videos/output/result.mp4
"""

import argparse
import contextlib
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv yoksa .env yüklenmez, env var elle set edilmeli


_SUPPRESSED = "Not enough SMs to use max_autotune_gemm mode"


class _SuppressTorchWarning(logging.Filter):
    def filter(self, record):
        return _SUPPRESSED not in record.getMessage()


def _configure_warnings():
    warnings.filterwarnings("ignore", message=f".*{_SUPPRESSED}.*")
    flt = _SuppressTorchWarning()
    for name in ("torch", "torch._inductor", "torch._inductor.utils"):
        logging.getLogger(name).addFilter(flt)


class _StdoutMirror:
    """Tee stdout to log file; mirror key progress lines to terminal."""

    _MIRROR = (
        "Run metadata:",
        "Output files:",
        "Done! Output:",
        "Tactical:",
        "Homography success:",
    )

    def __init__(self, log_fp, terminal_fp, mirror_all: bool = False):
        self.log_fp      = log_fp
        self.terminal_fp = terminal_fp
        self.mirror_all  = mirror_all
        self._buf        = ""

    def _emit(self, line):
        if self.mirror_all or any(line.strip().startswith(p) for p in self._MIRROR):
            self.terminal_fp.write(line)
            self.terminal_fp.flush()

    def write(self, text):
        self.log_fp.write(text)
        self.log_fp.flush()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""
        self.log_fp.flush()
        self.terminal_fp.flush()

    def isatty(self):
        return self.terminal_fp.isatty()


def main():
    from APP.helpers.config import PipelineConfig

    default_config = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Basketball Game Tracker: player tracking, jersey OCR, tactical view."
    )
    parser.add_argument("--input",      "-i", required=True,  help="Input video path")
    parser.add_argument("--output",     "-o", required=True,  help="Annotated output video path")
    parser.add_argument("--max-frames", type=int,   default=300,  help="Max frames to process (default: 300)")
    parser.add_argument("--start",      type=float, default=0.8,  help="Start time in seconds (default: 0.8)")
    parser.add_argument("--confidence",      type=float, default=default_config.detector.confidence,  help=f"Detection confidence threshold (default: {default_config.detector.confidence})")
    parser.add_argument("--track-thresh", type=float, default=default_config.tracker.track_thresh,
                        help=f"MCByte high-score tracking threshold (default: {default_config.tracker.track_thresh})")
    parser.add_argument("--new-track-thresh", type=float, default=default_config.tracker.new_track_thresh,
                        help=f"MCByte threshold for initializing new tracks (default: {default_config.tracker.new_track_thresh})")
    parser.add_argument("--track-buffer", type=int, default=default_config.tracker.track_buffer,
                        help=f"Frames to keep lost MCByte tracks (default: {default_config.tracker.track_buffer})")
    parser.add_argument("--cmc-method", default=default_config.tracker.cmc_method,
                        choices=["orb", "sift", "ecc", "sparseOptFlow", "none"],
                        help=f"Camera motion compensation method (default: {default_config.tracker.cmc_method})")
    parser.add_argument("--assoc1-thresh", type=float, default=default_config.tracker.assoc1_thresh,
                        help=f"MCByte first association max cost (default: {default_config.tracker.assoc1_thresh})")
    parser.add_argument("--assoc2-thresh", type=float, default=default_config.tracker.assoc2_thresh,
                        help=f"MCByte second association max cost (default: {default_config.tracker.assoc2_thresh})")
    parser.add_argument("--unconfirmed-assoc-thresh", type=float, default=default_config.tracker.unconfirmed_assoc_thresh,
                        help=f"MCByte unconfirmed-track association max cost (default: {default_config.tracker.unconfirmed_assoc_thresh})")
    parser.add_argument("--mask-duplicate-min-fill", type=float, default=default_config.tracker.mask_duplicate_min_fill,
                        help=f"Minimum dominant-mask fill ratio to suppress duplicate tracks (default: {default_config.tracker.mask_duplicate_min_fill})")
    parser.add_argument("--ref-player-conflict-iou", type=float, default=default_config.tracker.ref_player_conflict_iou,
                        help=f"Minimum player/referee IoU to suppress a player mask conflict (default: {default_config.tracker.ref_player_conflict_iou})")
    parser.add_argument("--ref-player-conflict-mask-fill", type=float, default=default_config.tracker.ref_player_conflict_mask_fill,
                        help=f"Minimum player bbox mask fill to suppress a player/referee conflict (default: {default_config.tracker.ref_player_conflict_mask_fill})")
    parser.add_argument("--ref-player-conflict-conf-margin", type=float, default=default_config.tracker.ref_player_conflict_conf_margin,
                        help=f"Required referee confidence advantage over player confidence (default: {default_config.tracker.ref_player_conflict_conf_margin})")
    parser.add_argument("--keypoint-confidence", type=float, default=default_config.keypoints.confidence,
                        help=f"Court keypoint detector confidence threshold (default: {default_config.keypoints.confidence})")
    parser.add_argument("--court-mask-activation", action=argparse.BooleanOptionalAction,
                        default=default_config.detector.activate_tracks_by_court_mask,
                        help=f"Gate new player tracks until their MCByte mask enters the court (default: {default_config.detector.activate_tracks_by_court_mask})")
    parser.add_argument("--court-mask-min-overlap", type=float, default=default_config.detector.court_mask_min_overlap,
                        help=f"Minimum mask ratio projected inside court before activating a player track (default: {default_config.detector.court_mask_min_overlap})")
    parser.add_argument("--court-mask-min-area", type=int, default=default_config.detector.court_mask_min_area_px,
                        help=f"Minimum estimated mask pixels inside court before activating a player track (default: {default_config.detector.court_mask_min_area_px})")
    parser.add_argument("--device",          default=default_config.device,            help=f"Compute device (default: {default_config.device})")
    parser.add_argument("--rfdetr-model-id", default=default_config.detector.rfdetr_model_id,
                        help="Roboflow RF-DETR model ID for player/referee detection")
    parser.add_argument("--frame-skip", type=int,   default=1,    help="Process 1 of every N frames (default: 1)")
    parser.add_argument("--log-file",   default=None,              help="Verbose log path (default: output_dir/log.txt)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed tracking, mask, and keypoint diagnostics in the log")
    args = parser.parse_args()

    _configure_warnings()

    from APP.helpers.pipeline import BasketballTrackingPipeline

    output_dir = os.path.dirname(os.path.abspath(args.output)) or os.getcwd()
    log_file   = os.path.abspath(args.log_file) if args.log_file else os.path.join(output_dir, "log.txt")
    base, ext  = os.path.splitext(args.output)
    tactical   = f"{base}_tactical{ext}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"Running tracker. Logs: {log_file}", flush=True)

    with open(log_file, "w", encoding="utf-8") as log_fp:
        for line in (
            "Basketball Game Tracker verbose log\n",
            f"input={args.input}\n",
            f"output={args.output}\n",
            f"tactical={tactical}\n",
            f"device={args.device}\n",
            f"debug={int(args.debug)}\n",
            f"detection_confidence={args.confidence}\n",
            f"track_thresh={args.track_thresh}\n",
            f"new_track_thresh={args.new_track_thresh}\n",
            f"track_buffer={args.track_buffer}\n",
            f"cmc_method={args.cmc_method}\n",
            f"assoc1_thresh={args.assoc1_thresh}\n",
            f"assoc2_thresh={args.assoc2_thresh}\n",
            f"unconfirmed_assoc_thresh={args.unconfirmed_assoc_thresh}\n",
            f"mask_duplicate_min_fill={args.mask_duplicate_min_fill}\n",
            f"ref_player_conflict_iou={args.ref_player_conflict_iou}\n",
            f"ref_player_conflict_mask_fill={args.ref_player_conflict_mask_fill}\n",
            f"ref_player_conflict_conf_margin={args.ref_player_conflict_conf_margin}\n",
            f"keypoint_confidence={args.keypoint_confidence}\n",
            f"court_mask_activation={int(args.court_mask_activation)}\n",
            f"court_mask_min_overlap={args.court_mask_min_overlap}\n",
            f"court_mask_min_area={args.court_mask_min_area}\n",
            f"frame_skip={args.frame_skip}\n\n",
        ):
            log_fp.write(line)
        log_fp.flush()

        stdout_mirror = _StdoutMirror(log_fp, sys.__stdout__)
        stderr_mirror = _StdoutMirror(log_fp, sys.__stderr__, mirror_all=True)

        try:
            with contextlib.redirect_stdout(stdout_mirror), contextlib.redirect_stderr(stderr_mirror):
                config = PipelineConfig()
                config.device = args.device
                config.detector.rfdetr_model_id = args.rfdetr_model_id
                config.detector.confidence = args.confidence
                config.tracker.track_thresh = args.track_thresh
                config.tracker.new_track_thresh = args.new_track_thresh
                config.tracker.track_buffer = args.track_buffer
                config.tracker.cmc_method = args.cmc_method
                config.tracker.assoc1_thresh = args.assoc1_thresh
                config.tracker.assoc2_thresh = args.assoc2_thresh
                config.tracker.unconfirmed_assoc_thresh = args.unconfirmed_assoc_thresh
                config.tracker.mask_duplicate_min_fill = args.mask_duplicate_min_fill
                config.tracker.ref_player_conflict_iou = args.ref_player_conflict_iou
                config.tracker.ref_player_conflict_mask_fill = args.ref_player_conflict_mask_fill
                config.tracker.ref_player_conflict_conf_margin = args.ref_player_conflict_conf_margin
                config.keypoints.confidence = args.keypoint_confidence
                config.detector.activate_tracks_by_court_mask = args.court_mask_activation
                config.detector.court_mask_min_overlap = args.court_mask_min_overlap
                config.detector.court_mask_min_area_px = args.court_mask_min_area
                if args.debug:
                    config.debug.enabled = True
                    config.debug.tracking = True
                    config.debug.masks = True
                    config.debug.keypoints = True
                tracker = BasketballTrackingPipeline(config=config)
                tracker.process_video(
                    args.input,
                    args.output,
                    max_frames=args.max_frames,
                    start_sec=args.start,
                    frame_skip=args.frame_skip,
                )
        except Exception:
            traceback.print_exc(file=log_fp)
            log_fp.flush()
            print(f"Run failed. Check log: {log_file}", file=sys.stderr, flush=True)
            raise

    print(f"Run complete. Log: {log_file}", flush=True)


if __name__ == "__main__":
    main()

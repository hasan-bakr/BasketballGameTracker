"""Basketball Game Tracker — CLI entry point (BotSort pipeline).

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

    def __init__(self, log_fp, terminal_fp):
        self.log_fp      = log_fp
        self.terminal_fp = terminal_fp
        self._buf        = ""

    def _emit(self, line):
        if any(line.strip().startswith(p) for p in self._MIRROR):
            self.terminal_fp.write(line)
            self.terminal_fp.flush()

    def write(self, text):
        self.log_fp.write(text)
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
    parser = argparse.ArgumentParser(
        description="Basketball Game Tracker: player tracking, jersey OCR, tactical view."
    )
    parser.add_argument("--input",      "-i", required=True,  help="Input video path")
    parser.add_argument("--output",     "-o", required=True,  help="Annotated output video path")
    parser.add_argument("--max-frames", type=int,   default=300,  help="Max frames to process (default: 300)")
    parser.add_argument("--start",      type=float, default=0.8,  help="Start time in seconds (default: 0)")
    parser.add_argument("--confidence",      type=float, default=0.4,  help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--device",          default="cuda",            help="Compute device (default: cuda)")
    parser.add_argument("--rfdetr-model-id", default="basketball-player-detection-3-ycjdo/4",
                        help="Roboflow RF-DETR model ID for player/referee detection")
    parser.add_argument("--frame-skip", type=int,   default=1,    help="Process 1 of every N frames (default: 1)")
    parser.add_argument("--log-file",   default=None,              help="Verbose log path (default: output_dir/log.txt)")
    args = parser.parse_args()

    _configure_warnings()

    from APP.helpers.botsort_pipeline import BotSortPipeline

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
            f"frame_skip={args.frame_skip}\n\n",
        ):
            log_fp.write(line)
        log_fp.flush()

        mirror = _StdoutMirror(log_fp, sys.__stdout__)

        try:
            with contextlib.redirect_stdout(mirror):
                tracker = BotSortPipeline(
                    rfdetr_model_id=args.rfdetr_model_id,
                    device=args.device,
                    confidence_threshold=args.confidence,
                )
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

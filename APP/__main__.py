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

    parser = argparse.ArgumentParser(
        description="Basketball Game Tracker: player tracking, jersey OCR, tactical view."
    )
    parser.add_argument("--input",      "-i", required=True,  help="Input video path")
    parser.add_argument("--output",     "-o", required=True,  help="Annotated output video path")
    parser.add_argument("--max-frames", type=int,   default=300,   help="Max frames to process (default: 300)")
    parser.add_argument("--start",      type=float, default=0.8,   help="Start time in seconds (default: 0.8)")
    parser.add_argument("--frame-skip", type=int,   default=1,     help="Process 1 of every N frames (default: 1)")
    parser.add_argument("--device",     default=None,               help="Compute device override (default: from config)")
    parser.add_argument("--log-file",   default=None,               help="Verbose log path (default: output_dir/LOG.log)")
    parser.add_argument("--confidence", type=float, default=None,   help="Detector confidence override (default: from config)")
    parser.add_argument("--debug",      action="store_true",        help="Enable tracking/mask/keypoint diagnostics")
    parser.add_argument("--tracking-report-json", default=None,     help="Write lifecycle summary JSON to this path")
    args = parser.parse_args()

    _configure_warnings()

    from APP.helpers.pipeline import BasketballTrackingPipeline

    output_dir = os.path.dirname(os.path.abspath(args.output)) or os.getcwd()
    log_file   = os.path.abspath(args.log_file) if args.log_file else os.path.join(output_dir, "LOG.log")
    base, ext  = os.path.splitext(args.output)
    tactical   = f"{base}_tactical{ext}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"Running tracker. Logs: {log_file}", flush=True)

    with open(log_file, "w", encoding="utf-8") as log_fp:
        log_fp.write(f"input={args.input}\noutput={args.output}\ntactical={tactical}\nframe_skip={args.frame_skip}\n\n")
        log_fp.flush()

        stdout_mirror = _StdoutMirror(log_fp, sys.__stdout__)
        stderr_mirror = _StdoutMirror(log_fp, sys.__stderr__, mirror_all=True)

        try:
            with contextlib.redirect_stdout(stdout_mirror), contextlib.redirect_stderr(stderr_mirror):
                config = PipelineConfig()
                if args.device:
                    config.device = args.device
                if args.confidence is not None:
                    config.detector.confidence = args.confidence
                if args.debug:
                    config.debug.enabled = True
                    config.debug.tracking = True
                    config.debug.masks = True
                    config.debug.keypoints = True
                if args.tracking_report_json:
                    config.tracking_report_json = args.tracking_report_json
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

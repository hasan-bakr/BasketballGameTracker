"""Basketball Game Tracker — CLI entry point.

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


_SUPPRESSED_TORCH_WARNING = "Not enough SMs to use max_autotune_gemm mode"


class _SuppressTorchInductorWarning(logging.Filter):
    def filter(self, record):
        return _SUPPRESSED_TORCH_WARNING not in record.getMessage()


def _configure_torch_warning_suppression():
    warnings.filterwarnings("ignore", message=f".*{_SUPPRESSED_TORCH_WARNING}.*")
    warning_filter = _SuppressTorchInductorWarning()
    for logger_name in ("torch", "torch._inductor", "torch._inductor.utils"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(warning_filter)


class _ProgressBarStderr:
    """Write everything to log, but mirror only the SAM2 tqdm bar to the terminal."""

    def __init__(self, log_fp, terminal_fp):
        self.log_fp = log_fp
        self.terminal_fp = terminal_fp
        self._mirror_active = False
        self._suppress_terminal_tokens = (
            "Not enough SMs to use max_autotune_gemm mode",
        )

    def write(self, text):
        self.log_fp.write(text)

        is_progress_chunk = "propagate in video" in text
        is_progress_update = self._mirror_active and ("\r" in text or "%" in text)
        is_suppressed = any(token in text for token in self._suppress_terminal_tokens)

        if not is_suppressed and (is_progress_chunk or is_progress_update):
            self.terminal_fp.write(text)
            self.terminal_fp.flush()
            self._mirror_active = not text.endswith("\n")

        return len(text)

    def flush(self):
        self.log_fp.flush()
        self.terminal_fp.flush()

    def isatty(self):
        return self.terminal_fp.isatty()


class _StdoutMirror:
    """Write stdout to log and selectively mirror key run info to terminal."""

    def __init__(self, log_fp, terminal_fp):
        self.log_fp = log_fp
        self.terminal_fp = terminal_fp
        self._buffer = ""
        self._mirror_prefixes = (
            "Run metadata:",
            "Output files:",
            "Batch: processed frames",
            "Done! Output:",
            "Tactical:",
            "Objects tracked:",
            "Homography success:",
        )

    def _emit_line(self, line):
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in self._mirror_prefixes):
            self.terminal_fp.write(line)
            self.terminal_fp.flush()

    def write(self, text):
        self.log_fp.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        self.log_fp.flush()
        self.terminal_fp.flush()

    def isatty(self):
        return self.terminal_fp.isatty()


def main():
    parser = argparse.ArgumentParser(
        description="Basketball Game Tracker: player segmentation, jersey OCR, tactical view."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input video file")
    parser.add_argument("--output", "-o", required=True,
                        help="Path for annotated output video")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames to process (default: 300)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--batch-size", type=int, default=150,
                        help="Frames per SAM2 propagation batch (default: 150)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="YOLO detection confidence threshold (default: 0.5)")
    parser.add_argument("--device", default="cuda",
                        help="Compute device: cuda or cpu (default: cuda)")
    parser.add_argument("--sam2-checkpoint", default="models/sam2.1_hiera_base_plus.pt",
                        help="Path to SAM2 checkpoint .pt file")
    parser.add_argument("--yolo-model", default="models/yolo/best_detection.pt",
                        help="Path to YOLO detection model (.pt or .onnx)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision (FP16)")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process 1 out of every N frames (default: 1 = all frames)")
    parser.add_argument("--log-file", default=None,
                        help="Path to verbose run log (default: output_dir/log.txt)")
    parser.add_argument("--debug-prompts", action="store_true",
                        help="Overlay prompt events on output video and save _prompt_events.json")
    args = parser.parse_args()
    _configure_torch_warning_suppression()

    from APP.helpers.robust_sam2_tracker import RobustSAM2Tracker

    output_dir = os.path.dirname(os.path.abspath(args.output)) or os.getcwd()
    log_file = os.path.abspath(args.log_file) if args.log_file else os.path.join(output_dir, "log.txt")
    base, ext = os.path.splitext(args.output)
    tactical_output = f"{base}_tactical{ext}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"Running tracker. Detailed logs: {log_file}", flush=True)

    with open(log_file, "w", encoding="utf-8") as log_fp:
        log_fp.write("Basketball Game Tracker verbose log\n")
        log_fp.write("log_mode=truncate_on_start\n")
        log_fp.write(f"input={args.input}\n")
        log_fp.write(f"output={args.output}\n")
        log_fp.write(f"output_tactical={tactical_output}\n")
        log_fp.write(f"log_file={log_file}\n")
        log_fp.write(f"device={args.device}\n")
        log_fp.write(f"batch_size={args.batch_size}\n")
        log_fp.write(f"frame_skip={args.frame_skip}\n\n")
        log_fp.flush()
        stdout_mirror = _StdoutMirror(log_fp, sys.__stdout__)
        progress_stderr = _ProgressBarStderr(log_fp, sys.__stderr__)

        try:
            with contextlib.redirect_stdout(stdout_mirror), contextlib.redirect_stderr(progress_stderr):
                tracker = RobustSAM2Tracker(
                    sam2_checkpoint=args.sam2_checkpoint,
                    yolo_path=args.yolo_model,
                    device=args.device,
                    confidence_threshold=args.confidence,
                    use_amp=not args.no_amp,
                )
                tracker.process_video(
                    args.input,
                    args.output,
                    max_frames=args.max_frames,
                    batch_size=args.batch_size,
                    start_sec=args.start,
                    frame_skip=args.frame_skip,
                    debug_prompts=args.debug_prompts,
                )
        except Exception:
            traceback.print_exc(file=log_fp)
            log_fp.flush()
            print(f"Run failed. Check log: {log_file}", file=sys.stderr, flush=True)
            raise

    print(f"Run complete. Log saved to {log_file}", flush=True)


if __name__ == "__main__":
    main()

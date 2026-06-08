"""
Drum Remover — CLI entry point.

Usage:
    python drumremover.py --input <filepath> [options]
    python drumremover.py --list-models
    python drumremover.py --version

All core logic lives in core.py. This file is purely argument parsing,
logging setup, user-facing output, and exit codes.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

VERSION = "0.1.0"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "drumremover.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cli_progress(fraction: float, message: str) -> None:
    pct = int(fraction * 100)
    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:3d}%  {message:<45}", end="", flush=True)
    if fraction >= 1.0:
        print()  # newline after 100%


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drumremover",
        description="Remove drums from an audio file using Demucs stem separation.",
    )
    p.add_argument("--input",  "-i", metavar="FILE",  help="Input MP3 or WAV file")
    p.add_argument("--output", "-o", metavar="DIR",   help="Output directory (default: same as input)")
    p.add_argument("--model",  "-m", metavar="MODEL", default=None,
                   help="Demucs model to use (default from config, usually htdemucs)")
    p.add_argument("--device", "-d", choices=["gpu", "cpu"], default=None,
                   help="Processing device (default from config, usually gpu)")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    p.add_argument("--version",     action="store_true", help="Show version and exit")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"Drum Remover v{VERSION}")
        return 0

    if args.list_models:
        from core import list_models
        print("Available models:")
        for m in list_models():
            print(f"  {m}")
        return 0

    if not args.input:
        parser.print_help()
        return 1

    import config
    cfg = config.load()

    log_dir = Path(cfg.get("log_dir", "logs"))
    setup_logging(log_dir)

    model  = args.model  or cfg.get("model",  "htdemucs")
    device = args.device or cfg.get("device", "gpu")

    input_path  = Path(args.input)
    output_dir  = Path(args.output) if args.output else None

    from core import remove_drums, resolve_output_paths, check_overwrite, DrumRemoverError

    # Warn before overwrite
    no_drums_path, drums_path = resolve_output_paths(input_path, output_dir)
    existing = check_overwrite([no_drums_path, drums_path])
    if existing:
        print("Warning: the following output file(s) already exist:")
        for p in existing:
            print(f"  {p}")
        answer = input("Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 0

    print(f"\nDrum Remover v{VERSION}")
    print(f"  Input : {input_path}")
    print(f"  Output: {no_drums_path.parent}")
    print(f"  Model : {model}  |  Device: {device}")
    print()

    start = time.time()
    logging.info(
        "Starting removal — input=%s  model=%s  device=%s",
        input_path, model, device,
    )

    try:
        no_drums, drums = remove_drums(
            input_path=input_path,
            output_dir=output_dir,
            model=model,
            device_preference=device,
            progress_callback=cli_progress,
        )
    except DrumRemoverError as e:
        print(f"\nError: {e}", file=sys.stderr)
        logging.error("DrumRemoverError: %s", e)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error")
        return 1

    elapsed = time.time() - start
    logging.info("Completed in %.1f s — no_drums=%s  drums=%s", elapsed, no_drums, drums)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  No-drums: {no_drums}")
    print(f"  Drums   : {drums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

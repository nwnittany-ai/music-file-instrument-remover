"""
Music File Instrument Remover — CLI entry point.

Usage:
    python drumremover.py --input <filepath> [options]
    python drumremover.py --list-models
    python drumremover.py --list-stems
    python drumremover.py --version

All core logic lives in core.py. This file handles argument parsing,
logging setup, user-facing output, and exit codes only.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

VERSION = "0.2.0"


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
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drumremover",
        description=(
            "Separate stems from an audio file using Demucs AI stem separation.\n\n"
            "Stems:\n"
            "  drums   — isolate or remove drums (default)\n"
            "  vocals  — isolate or remove vocals (karaoke)\n"
            "  bass    — isolate or remove bass\n"
            "  other   — isolate or remove guitar/keys\n"
            "  all     — full 4-stem separation (drums, bass, vocals, other)\n\n"
            "Output Formats:\n"
            "  wav  — 24-bit WAV, best quality (default)\n"
            "  mp3  — 320kbps MP3, ~8x smaller\n\n"
            "Models:\n"
            "  htdemucs_ft  — best quality, recommended (default)\n"
            "  htdemucs     — fast, good quality\n"
            "  mdx_extra    — alternative architecture\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",       "-i", metavar="FILE",  help="Input MP3 or WAV file")
    p.add_argument("--output",      "-o", metavar="DIR",   help="Output directory (default: same as input)")
    p.add_argument("--stem",        "-s", metavar="STEM",  default=None,
                   help="Stem to separate: drums, vocals, bass, other, all (default: drums)")
    p.add_argument("--model",       "-m", metavar="MODEL", default=None,
                   help="Demucs model (default from config, usually htdemucs_ft)")
    p.add_argument("--format",      "-f", choices=["wav", "mp3"], default=None,
                   help="Output format: wav (24-bit, default) or mp3 (320kbps)")
    p.add_argument("--device",      "-d", choices=["gpu", "cpu"], default=None,
                   help="Processing device (default from config, usually gpu)")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    p.add_argument("--list-stems",  action="store_true", help="List available stems and exit")
    p.add_argument("--version",     action="store_true", help="Show version and exit")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"Music File Instrument Remover v{VERSION}")
        return 0

    if args.list_models:
        from core import list_models
        print("Available models:")
        for m in list_models():
            print(f"  {m}")
        return 0

    if args.list_stems:
        from core import STEM_LABELS
        print("Available stems:")
        for key, label in STEM_LABELS.items():
            print(f"  {key:<8} — {label}")
        return 0

    if not args.input:
        parser.print_help()
        return 1

    import config
    cfg = config.load()

    log_dir = Path(cfg.get("log_dir", "logs"))
    setup_logging(log_dir)

    model         = args.model  or cfg.get("model",         "htdemucs_ft")
    device        = args.device or cfg.get("device",        "gpu")
    stem          = args.stem   or cfg.get("stem",          "drums")
    output_format = args.format or cfg.get("output_format", "wav")

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    from core import separate_stems, resolve_output_paths, check_overwrite, SeparationError, STEM_LABELS

    # Overwrite check
    output_paths = resolve_output_paths(input_path, output_dir, stem, output_format)
    existing = check_overwrite(output_paths)
    if existing:
        print("Warning: the following output file(s) already exist:")
        for p in existing:
            print(f"  {p}")
        answer = input("Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return 0

    label = STEM_LABELS.get(stem, stem)
    print(f"\nMusic File Instrument Remover v{VERSION}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_paths[0].parent}")
    print(f"  Mode  : {label}")
    print(f"  Format: {output_format.upper()}")
    print(f"  Model : {model}  |  Device: {device}")
    print()

    start = time.time()
    logging.info("Starting — input=%s  stem=%s  model=%s  device=%s",
                 input_path, stem, model, device)

    try:
        results = separate_stems(
            input_path=input_path,
            stem=stem,
            output_dir=output_dir,
            model=model,
            device_preference=device,
            output_format=output_format,
            progress_callback=cli_progress,
        )
    except SeparationError as e:
        print(f"\nError: {e}", file=sys.stderr)
        logging.error("SeparationError: %s", e)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error")
        return 1

    elapsed = time.time() - start
    logging.info("Completed in %.1f s", elapsed)

    print(f"\nCompleted in {elapsed:.1f}s")
    print("Output files:")
    for p in results:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

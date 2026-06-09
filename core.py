"""
Music File Instrument Remover — core separation logic.

Shared by CLI and GUI. All Demucs interaction lives here.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

SUPPORTED_EXTENSIONS = {".mp3", ".wav"}

AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"]
DEFAULT_MODEL = "htdemucs_ft"

# Stems supported for isolation. "all" = full 4-stem separation.
AVAILABLE_STEMS = ["drums", "vocals", "bass", "other", "all"]
DEFAULT_STEM = "drums"

# Output formats
AVAILABLE_FORMATS = ["wav", "mp3"]
DEFAULT_FORMAT = "wav"
MP3_BITRATE = "320k"

# Human-readable labels for the GUI
STEM_LABELS = {
    "drums":  "Remove Drums",
    "vocals": "Remove Vocals",
    "bass":   "Remove Bass",
    "other":  "Remove Other (guitar/keys)",
    "all":    "Full Separation (4 stems)",
}


class SeparationError(Exception):
    """Raised for user-facing errors (no raw tracebacks shown to users)."""

# Keep old name as alias so any external code doesn't break
DrumRemoverError = SeparationError


def resolve_device(preference: str) -> str:
    """Return 'cuda' or 'cpu' based on preference and GPU availability."""
    if preference.lower() in ("gpu", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            logging.warning("GPU requested but CUDA is not available — falling back to CPU.")
        except ImportError:
            logging.warning("PyTorch not found — falling back to CPU.")
        return "cpu"
    return "cpu"


def validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise SeparationError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise SeparationError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SeparationError(
            f"Unsupported file format '{input_path.suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def resolve_output_paths(
    input_path: Path,
    output_dir: Optional[Path],
    stem: str = DEFAULT_STEM,
    output_format: str = DEFAULT_FORMAT,
) -> list[Path]:
    """
    Return list of output Paths for the given stem mode and format.

    For single-stem mode (e.g. 'drums'):
        [<name>_no_<stem>.<ext>, <name>_<stem>.<ext>]
    For 'all' mode:
        [<name>_drums.<ext>, <name>_bass.<ext>, <name>_vocals.<ext>, <name>_other.<ext>]
    """
    file_stem = input_path.stem
    out_dir = Path(output_dir) if output_dir else input_path.parent
    ext = f".{output_format.lower()}"

    if stem == "all":
        return [out_dir / f"{file_stem}_{s}{ext}" for s in ["drums", "bass", "vocals", "other"]]
    else:
        without  = out_dir / f"{file_stem}_no_{stem}{ext}"
        isolated = out_dir / f"{file_stem}_{stem}{ext}"
        return [without, isolated]


def check_overwrite(paths: list[Path]) -> list[Path]:
    """Return list of paths that already exist (caller decides how to handle)."""
    return [p for p in paths if p.exists()]


def _copy_metadata(input_path: Path, output_path: Path, stem_label: str) -> None:
    """
    Copy ID3/metadata tags from input_path to output_path.
    Adds a COMM (comment) tag identifying which stem the file contains.
    Silently skips if input has no tags or mutagen can't handle the format.
    """
    try:
        from mutagen import File
        from mutagen.id3 import ID3, COMM, TIT2
        from mutagen.mp3 import MP3
        from mutagen.wave import WAVE

        src = File(str(input_path))
        if src is None or src.tags is None:
            return

        dst = File(str(output_path), easy=False)
        if dst is None:
            return

        # Ensure destination has a tag container
        if dst.tags is None:
            dst.add_tags()

        # Copy all tags from source
        for key, value in src.tags.items():
            try:
                dst.tags[key] = value
            except Exception:
                pass  # skip any tag that doesn't apply to the output format

        # Add/overwrite a comment tag identifying the stem
        comment_tag = COMM(encoding=3, lang="eng", desc="stem", text=[stem_label])
        try:
            dst.tags.add(comment_tag)
        except Exception:
            try:
                dst.tags["COMM::eng"] = comment_tag
            except Exception:
                pass

        dst.save()
        logging.debug("Metadata copied to %s (stem: %s)", output_path.name, stem_label)

    except Exception as e:
        logging.warning("Could not copy metadata to %s: %s", output_path.name, e)


def _save_audio(tensor, sample_rate: int, out_path: Path, output_format: str) -> None:
    """Save a (channels, samples) tensor to WAV or MP3."""
    import numpy as np
    import soundfile as sf
    import subprocess
    import tempfile

    if output_format.lower() == "wav":
        sf.write(str(out_path), tensor.numpy().T, sample_rate, subtype="PCM_24")
    elif output_format.lower() == "mp3":
        # Write a temp WAV then encode to MP3 via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), tensor.numpy().T, sample_rate, subtype="PCM_24")
            cmd = [
                "ffmpeg", "-y", "-i", str(tmp_path),
                "-b:a", MP3_BITRATE,
                "-q:a", "0",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise SeparationError(f"Unknown output format '{output_format}'. Choose wav or mp3.")


def separate_stems(
    input_path: Path,
    stem: str = DEFAULT_STEM,
    output_dir: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    device_preference: str = "gpu",
    output_format: str = DEFAULT_FORMAT,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[Path]:
    """
    Run Demucs stem separation on input_path.

    Args:
        input_path:         Path to input MP3 or WAV file.
        stem:               Which stem to isolate: 'drums', 'vocals', 'bass', 'other', or 'all'.
        output_dir:         Output directory. Defaults to same directory as input file.
        model:              Demucs model name (default: htdemucs_ft).
        device_preference:  'gpu' or 'cpu'.
        output_format:      Output file format: 'wav' (default, 24-bit) or 'mp3' (320kbps).
        progress_callback:  Optional callable(fraction: float, message: str).

    Returns:
        List of Paths to output files.
        - Single stem: [no_<stem>.<ext>, <stem>.<ext>]
        - 'all':       [drums.<ext>, bass.<ext>, vocals.<ext>, other.<ext>]

    Raises:
        SeparationError on any user-facing problem.
    """
    if stem not in AVAILABLE_STEMS:
        raise SeparationError(
            f"Unknown stem '{stem}'. Choose from: {', '.join(AVAILABLE_STEMS)}"
        )

    input_path = Path(input_path).resolve()
    validate_input(input_path)

    out_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_preference)
    logging.info("Using device: %s | model: %s | stem: %s", device, model, stem)

    if progress_callback:
        progress_callback(0.0, f"Loading model '{model}'...")

    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch
        import torchaudio
    except ImportError as e:
        raise SeparationError(f"Required package not installed: {e}") from e

    try:
        logging.info("Loading model: %s", model)
        demucs_model = get_model(model)
        demucs_model.to(device)
        demucs_model.eval()
    except Exception as e:
        raise SeparationError(f"Failed to load model '{model}': {e}") from e

    model_sr   = demucs_model.samplerate
    stem_names = list(demucs_model.sources)  # e.g. ['drums', 'bass', 'vocals', 'other']

    # Validate the requested stem exists in this model
    if stem != "all" and stem not in stem_names:
        raise SeparationError(
            f"Model '{model}' does not support stem '{stem}'. "
            f"Available: {', '.join(stem_names)}"
        )

    if progress_callback:
        progress_callback(0.1, "Loading audio...")

    try:
        from demucs.audio import AudioFile
        waveform = AudioFile(input_path).read(streams=0, samplerate=model_sr, channels=2)
    except Exception as e:
        raise SeparationError(f"Could not read audio file: {e}") from e

    if progress_callback:
        progress_callback(0.2, "Separating stems (this may take a minute)...")

    waveform = waveform.unsqueeze(0).to(device)  # (1, channels, samples)

    try:
        with torch.no_grad():
            sources = apply_model(demucs_model, waveform, device=device, progress=False)
    except Exception as e:
        raise SeparationError(f"Stem separation failed: {e}") from e

    sources = sources.squeeze(0).cpu()  # (num_stems, channels, samples)

    fmt = output_format.lower()
    fmt_label = "MP3 (320kbps)" if fmt == "mp3" else "WAV (24-bit)"
    if progress_callback:
        progress_callback(0.85, f"Saving output files ({fmt_label})...")

    try:
        output_paths = []
        ext = f".{fmt}"

        if stem == "all":
            for i, name in enumerate(stem_names):
                out_path = out_dir / f"{input_path.stem}_{name}{ext}"
                _save_audio(sources[i], model_sr, out_path, fmt)
                _copy_metadata(input_path, out_path, name)
                output_paths.append(out_path)
                logging.info("Output: %s", out_path)
        else:
            stem_idx      = stem_names.index(stem)
            isolated      = sources[stem_idx]
            without       = sum(sources[i] for i in range(len(stem_names)) if i != stem_idx)
            without_path  = out_dir / f"{input_path.stem}_no_{stem}{ext}"
            isolated_path = out_dir / f"{input_path.stem}_{stem}{ext}"

            _save_audio(without,  model_sr, without_path,  fmt)
            _copy_metadata(input_path, without_path, f"no_{stem}")
            _save_audio(isolated, model_sr, isolated_path, fmt)
            _copy_metadata(input_path, isolated_path, stem)

            output_paths = [without_path, isolated_path]
            logging.info("Output: %s", without_path)
            logging.info("Output: %s", isolated_path)

    except SeparationError:
        raise
    except Exception as e:
        raise SeparationError(f"Failed to write output files: {e}") from e

    if progress_callback:
        progress_callback(1.0, "Done.")

    return output_paths


# ---------------------------------------------------------------------------
# Convenience wrapper — keeps old callers working unchanged
# ---------------------------------------------------------------------------

def remove_drums(
    input_path: Path,
    output_dir: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    device_preference: str = "gpu",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[Path, Path]:
    """Backward-compatible wrapper around separate_stems for drum removal."""
    paths = separate_stems(
        input_path=input_path,
        stem="drums",
        output_dir=output_dir,
        model=model,
        device_preference=device_preference,
        progress_callback=progress_callback,
    )
    return paths[0], paths[1]  # (no_drums, drums)


def list_models() -> list[str]:
    return list(AVAILABLE_MODELS)


def list_stems() -> list[str]:
    return list(AVAILABLE_STEMS)

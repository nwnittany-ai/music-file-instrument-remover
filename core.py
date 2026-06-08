"""
Core drum-removal logic — shared by CLI and GUI.

All Demucs interaction lives here. The CLI and GUI import from this module;
neither duplicates this logic.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

SUPPORTED_EXTENSIONS = {".mp3", ".wav"}
AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"]
DEFAULT_MODEL = "htdemucs"


class DrumRemoverError(Exception):
    """Raised for user-facing errors (no raw tracebacks shown to users)."""


def resolve_device(preference: str) -> str:
    """
    Return 'cuda' or 'cpu' based on preference and availability.
    Logs a warning and falls back to CPU if GPU requested but unavailable.
    """
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
        raise DrumRemoverError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise DrumRemoverError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DrumRemoverError(
            f"Unsupported file format '{input_path.suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def resolve_output_paths(input_path: Path, output_dir: Optional[Path]) -> tuple[Path, Path]:
    """
    Return (no_drums_path, drums_path) for the given input file.
    If output_dir is None, output goes alongside the input file.
    """
    stem = input_path.stem
    out_dir = output_dir if output_dir else input_path.parent
    no_drums = out_dir / f"{stem}_no_drums.wav"
    drums    = out_dir / f"{stem}_drums.wav"
    return no_drums, drums


def check_overwrite(paths: list[Path]) -> list[Path]:
    """Return list of paths that already exist (caller decides how to handle)."""
    return [p for p in paths if p.exists()]


def remove_drums(
    input_path: Path,
    output_dir: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    device_preference: str = "gpu",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> tuple[Path, Path]:
    """
    Run Demucs stem separation (drums vs. no_drums) on input_path.

    Args:
        input_path:         Path to input MP3 or WAV file.
        output_dir:         Directory for output files. Defaults to input file's directory.
        model:              Demucs model name (default: htdemucs).
        device_preference:  'gpu' or 'cpu'.
        progress_callback:  Optional callable(fraction: float, message: str) for progress updates.

    Returns:
        (no_drums_path, drums_path) — absolute Paths to the two output WAV files.

    Raises:
        DrumRemoverError on any user-facing problem.
    """
    input_path = Path(input_path).resolve()
    validate_input(input_path)

    out_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_preference)
    logging.info("Using device: %s", device)

    no_drums_path, drums_path = resolve_output_paths(input_path, out_dir)

    if progress_callback:
        progress_callback(0.0, f"Loading model '{model}'...")

    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch
        import torchaudio
    except ImportError as e:
        raise DrumRemoverError(f"Required package not installed: {e}") from e

    try:
        logging.info("Loading model: %s", model)
        demucs_model = get_model(model)
        demucs_model.to(device)
        demucs_model.eval()
    except Exception as e:
        raise DrumRemoverError(f"Failed to load model '{model}': {e}") from e

    if progress_callback:
        progress_callback(0.1, "Loading audio...")

    model_sr = demucs_model.samplerate

    try:
        from demucs.audio import AudioFile
        audio_file = AudioFile(input_path)
        # read() returns (channels, samples) tensor, resampled to model_sr, stereo
        waveform = audio_file.read(streams=0, samplerate=model_sr, channels=2)
        sample_rate = model_sr
    except Exception as e:
        raise DrumRemoverError(f"Could not read audio file: {e}") from e

    if progress_callback:
        progress_callback(0.2, "Separating stems (this may take a minute)...")

    # apply_model expects shape (batch, channels, samples)
    waveform = waveform.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            sources = apply_model(demucs_model, waveform, device=device, progress=False)
    except Exception as e:
        raise DrumRemoverError(f"Stem separation failed: {e}") from e

    # sources shape: (batch, num_stems, channels, samples)
    # Stem order depends on model; find drums index by name
    stem_names = demucs_model.sources
    if "drums" not in stem_names:
        raise DrumRemoverError(f"Model '{model}' does not have a 'drums' stem.")

    drums_idx = stem_names.index("drums")
    sources = sources.squeeze(0).cpu()  # (num_stems, channels, samples)

    drums_audio   = sources[drums_idx]
    no_drums_audio = sum(sources[i] for i in range(len(stem_names)) if i != drums_idx)

    if progress_callback:
        progress_callback(0.85, "Saving output files...")

    try:
        import soundfile as sf
        import numpy as np
        # soundfile expects (samples, channels); our tensors are (channels, samples)
        sf.write(str(no_drums_path), no_drums_audio.numpy().T, sample_rate, subtype="PCM_24")
        sf.write(str(drums_path),    drums_audio.numpy().T,    sample_rate, subtype="PCM_24")
    except Exception as e:
        raise DrumRemoverError(f"Failed to write output files: {e}") from e

    if progress_callback:
        progress_callback(1.0, "Done.")

    logging.info("Output: %s", no_drums_path)
    logging.info("Output: %s", drums_path)
    return no_drums_path, drums_path


def list_models() -> list[str]:
    return list(AVAILABLE_MODELS)

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
) -> list[Path]:
    """
    Return list of output Paths for the given stem mode.

    For single-stem mode (e.g. 'drums'):
        [<stem>_no_<stem>.wav, <stem>_<stem>.wav]
    For 'all' mode:
        [<stem>_drums.wav, <stem>_bass.wav, <stem>_vocals.wav, <stem>_other.wav]
    """
    file_stem = input_path.stem
    out_dir = Path(output_dir) if output_dir else input_path.parent

    if stem == "all":
        return [out_dir / f"{file_stem}_{s}.wav" for s in ["drums", "bass", "vocals", "other"]]
    else:
        isolated    = out_dir / f"{file_stem}_{stem}.wav"
        without     = out_dir / f"{file_stem}_no_{stem}.wav"
        return [without, isolated]


def check_overwrite(paths: list[Path]) -> list[Path]:
    """Return list of paths that already exist (caller decides how to handle)."""
    return [p for p in paths if p.exists()]


def separate_stems(
    input_path: Path,
    stem: str = DEFAULT_STEM,
    output_dir: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    device_preference: str = "gpu",
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
        progress_callback:  Optional callable(fraction: float, message: str).

    Returns:
        List of Paths to output WAV files.
        - Single stem: [no_<stem>.wav, <stem>.wav]
        - 'all':       [drums.wav, bass.wav, vocals.wav, other.wav]

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

    if progress_callback:
        progress_callback(0.85, "Saving output files...")

    try:
        import soundfile as sf

        output_paths = []

        if stem == "all":
            # Write all 4 stems individually
            for i, name in enumerate(stem_names):
                out_path = out_dir / f"{input_path.stem}_{name}.wav"
                sf.write(str(out_path), sources[i].numpy().T, model_sr, subtype="PCM_24")
                output_paths.append(out_path)
                logging.info("Output: %s", out_path)
        else:
            # Write isolated stem and "everything else" mix
            stem_idx    = stem_names.index(stem)
            isolated    = sources[stem_idx]
            without     = sum(sources[i] for i in range(len(stem_names)) if i != stem_idx)

            without_path   = out_dir / f"{input_path.stem}_no_{stem}.wav"
            isolated_path  = out_dir / f"{input_path.stem}_{stem}.wav"

            sf.write(str(without_path),  without.numpy().T,  model_sr, subtype="PCM_24")
            sf.write(str(isolated_path), isolated.numpy().T, model_sr, subtype="PCM_24")

            output_paths = [without_path, isolated_path]
            logging.info("Output: %s", without_path)
            logging.info("Output: %s", isolated_path)

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

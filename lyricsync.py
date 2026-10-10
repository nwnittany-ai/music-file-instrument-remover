"""
lyricsync - timestamp a lyric file against a song.

  python lyricsync.py song.mp3 lyrics.txt
  python lyricsync.py song.mp3 lyrics.txt --words --model large-v3
  python lyricsync.py song.mp3 lyrics.txt --device cpu          (leave the GPU free for other jobs)
  python lyricsync.py song.mp3 lyrics.txt --vocals "song_vocals.wav"   (reuse an existing vocal stem)

Writes next to the song (or to --output-dir):
  <song>_timed.txt    readable summary + every line with start-end times (paste into Claude chat/design)
  <song>.lrc          standard timed-lyrics file (add --words for per-word tags)
  <song>_timing.json  everything, including per-word times and confidence, for other tools

Device: --device auto (default) uses the GPU only when it has enough free memory (so it will not
fight a running animation job) and otherwise the CPU; gpu / cpu force a choice. If "auto" picks the
GPU and still runs out of memory, it retries on the CPU.

Vocal stems are cached in lyricsync_cache/ so re-running a song (e.g. after fixing a lyric) skips
the slow Demucs step. The GUI's Lyric Timing tab calls run() below.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import lyricsync_core as lc

CACHE_DIR = Path(__file__).parent / "lyricsync_cache"

# Rough peak GPU memory needs in GB (including headroom). The steps run one after another.
DEMUCS_GB = 4.0
WHISPER_GB = {"tiny": 1.0, "base": 1.0, "small": 2.0, "medium": 3.5, "large": 6.5, "large-v3": 6.5}
DEFAULT_WHISPER_GB = 4.0

# progress(fraction 0-1 or None, message)
Progress = Callable[[Optional[float], str], None]


# ---------------------------------------------------------------------------
# Device choice
# ---------------------------------------------------------------------------

def gpu_status() -> Optional[tuple[float, float]]:
    """(free GB, total GB) of the first NVIDIA GPU, or None if there is none / nvidia-smi is
    missing. Uses nvidia-smi rather than torch so that merely checking does not reserve any VRAM."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()[0]
        free_mb, total_mb = (float(x) for x in out.split(","))
        return free_mb / 1024, total_mb / 1024
    except Exception:
        return None


def needed_gb(model: str, need_demucs: bool) -> float:
    whisper = WHISPER_GB.get(model, DEFAULT_WHISPER_GB)
    return max(whisper, DEMUCS_GB) if need_demucs else whisper


def choose_device(pref: str, needed: float) -> tuple[str, str]:
    """Resolve 'auto' / 'gpu' / 'cpu' to ('gpu' or 'cpu', explanation)."""
    if pref == "cpu":
        return "cpu", "Using the CPU (as selected)."
    st = gpu_status()
    if st is None:
        return "cpu", "No NVIDIA GPU found; using the CPU."
    free, total = st
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return "cpu", "PyTorch in this environment cannot use the GPU; using the CPU."
    if pref == "gpu":
        return "gpu", f"Using the GPU (as selected); {free:.1f} of {total:.1f} GB free."
    if free >= needed:
        return "gpu", f"Using the GPU ({free:.1f} of {total:.1f} GB free; needs about {needed:.1f})."
    return "cpu", (f"GPU is busy (only {free:.1f} of {total:.1f} GB free; this needs about {needed:.1f}), "
                   "so using the CPU. This is slower but will not disturb other GPU jobs.")


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


# ---------------------------------------------------------------------------
# Vocal stem (cached)
# ---------------------------------------------------------------------------

def find_vocals(outputs: list[Path]) -> Path:
    for p in outputs:
        if p.stem.lower().endswith("vocals") and not p.stem.lower().endswith("no_vocals"):
            return p
    raise lc.LyricSyncError(f"Vocal separation produced no vocals file: {[p.name for p in outputs]}")


def _cache_folder(audio: Path, demucs_model: str) -> Path:
    st = audio.stat()
    return CACHE_DIR / f"{audio.stem}_{st.st_size}_{int(st.st_mtime)}_{demucs_model}"


def cached_vocals_path(audio: Path, demucs_model: str) -> Optional[Path]:
    """The vocal stem for this exact file (name + size + modified time + model), if already made."""
    folder = _cache_folder(audio, demucs_model)
    found = [p for p in sorted(folder.glob("*_vocals.wav")) if not p.stem.endswith("no_vocals")] if folder.is_dir() else []
    return found[0] if found else None


def make_vocals(audio: Path, demucs_model: str, device_pref: str, progress: Progress) -> Path:
    import core  # the existing Demucs wrapper in this project

    progress(0.0, "Separating vocals with Demucs (the slow part)...")
    outs = core.separate_stems(audio, stem="vocals", output_dir=_cache_folder(audio, demucs_model),
                               model=demucs_model, device_preference=device_pref, output_format="wav",
                               progress_callback=lambda frac, msg: progress(frac * 0.7, msg))
    return find_vocals(outs)


# ---------------------------------------------------------------------------
# Main entry point (CLI and GUI both call this)
# ---------------------------------------------------------------------------

def _pipeline(audio: Path, lyrics_text: str, output_dir: Path, model: str, device: str, language: str,
              words: bool, vocals: Optional[Path], separate: bool, demucs_model: str, say: Progress) -> dict:
    lines = lc.parse_lyrics(lyrics_text)
    title = lc.find_title(lyrics_text) or audio.stem
    say(0.0, f"{len(lines)} sung lines parsed ({sum(1 for l in lines if l.hint is not None)} with rough times).")

    if vocals:
        vocal_path = Path(vocals).resolve()
        say(0.7, f"Using existing vocal track: {vocal_path.name}")
    elif separate:
        vocal_path = cached_vocals_path(audio, demucs_model)
        if vocal_path:
            say(0.7, "Reusing the vocal track from last time.")
        else:
            vocal_path = make_vocals(audio, demucs_model, device, say)
    else:
        vocal_path = audio
        say(0.7, "Skipping vocal separation (aligning against the full mix).")

    lc.align_vocals(vocal_path, lines, model_name=model, device=("cuda" if device == "gpu" else "cpu"),
                    language=language, progress=lambda m: say(0.85, m))

    stem = audio.stem
    duration = lc.audio_duration(audio)
    meta = {"title": title, "audio": str(audio), "vocals": str(vocal_path), "model": model, "language": language,
            "device": device, "duration": duration, "line_count": len(lines)}
    files = {
        "timed": output_dir / f"{stem}_timed.txt",
        "lrc": output_dir / f"{stem}.lrc",
        "json": output_dir / f"{stem}_timing.json",
    }
    timed_text = lc.to_timed_text(lines, title, duration)
    files["timed"].write_text(timed_text, encoding="utf-8")
    files["lrc"].write_text(lc.to_lrc(lines, words=words), encoding="utf-8")
    files["json"].write_text(lc.to_json(lines, meta), encoding="utf-8")
    flagged = sum(1 for l in lines if l.flags)
    say(1.0, f"Done. {len(lines)} lines timed, {flagged} flagged.")
    return {"files": {k: str(v) for k, v in files.items()}, "lines": len(lines), "flagged": flagged,
            "timed_text": timed_text, "report": lc.report(lines), "output_dir": str(output_dir)}


def run(audio: Path, lyrics_text: str, output_dir: Path, model: str = "medium", device: str = "auto",
        language: str = "en", words: bool = False, vocals: Optional[Path] = None, separate: bool = True,
        demucs_model: str = "htdemucs_ft", progress: Optional[Progress] = None) -> dict:
    """device: 'auto' (GPU only if enough memory is free), 'gpu' or 'cpu'."""
    say: Progress = progress or (lambda frac, msg: print(msg))
    audio = Path(audio).resolve()
    if not audio.is_file():
        raise lc.LyricSyncError(f"Audio file not found: {audio}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    need_demucs = bool(separate and not vocals and cached_vocals_path(audio, demucs_model) is None)
    chosen, note = choose_device(device, needed_gb(model, need_demucs))
    say(0.0, note)
    args = (audio, lyrics_text, output_dir, model)
    rest = (language, words, vocals, separate, demucs_model, say)
    try:
        result = _pipeline(*args, chosen, *rest)
    except Exception as exc:
        if chosen == "gpu" and device == "auto" and _is_oom(exc):
            say(0.0, "The GPU ran out of memory (another job probably started); retrying on the CPU...")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            chosen, note = "cpu", "Retried on the CPU after the GPU ran out of memory."
            result = _pipeline(*args, chosen, *rest)
        else:
            raise
    result["device_used"], result["device_note"] = chosen, note
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Timestamp a lyric file against a song.")
    ap.add_argument("audio", type=Path, help="song (.mp3 or .wav)")
    ap.add_argument("lyrics", type=Path, help="lyrics text file, one sung line per line")
    ap.add_argument("--output-dir", type=Path, help="default: next to the audio file")
    ap.add_argument("--model", default="medium", help="speech model: tiny, base, small, medium, large-v3 (default medium)")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto = GPU only if enough memory is free (default); gpu / cpu force it")
    ap.add_argument("--language", default="en")
    ap.add_argument("--words", action="store_true", help="write per-word timestamps into the .lrc (always kept in the .json)")
    ap.add_argument("--vocals", type=Path, help="use this existing vocal-only file instead of separating")
    ap.add_argument("--no-separate", action="store_true", help="align against the full mix (less accurate)")
    ap.add_argument("--demucs-model", default="htdemucs_ft")
    a = ap.parse_args()
    try:
        if not a.lyrics.is_file():
            raise lc.LyricSyncError(f"Lyrics file not found: {a.lyrics}")
        res = run(a.audio, a.lyrics.read_text(encoding="utf-8-sig"), a.output_dir or a.audio.resolve().parent,
                  a.model, a.device, a.language, a.words, a.vocals, not a.no_separate, a.demucs_model)
        print()
        print(res["report"])
        print()
        for label, p in res["files"].items():
            print(f"{label:6s} {p}")
    except Exception as exc:  # one clear line instead of a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

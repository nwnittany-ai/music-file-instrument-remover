"""
lyricsync - timestamp a lyric file against a song.

  python lyricsync.py song.mp3 lyrics.txt
  python lyricsync.py song.mp3 lyrics.txt --words --model large-v3
  python lyricsync.py song.mp3 lyrics.txt --vocals "song_vocals.wav"   (reuse an existing vocal stem)

Writes next to the song (or to --output-dir):
  <song>_timed.txt    readable summary + every line with start-end times (paste into Claude chat/design)
  <song>.lrc          standard timed-lyrics file (add --words for per-word tags)
  <song>_timing.json  everything, including per-word times and confidence, for other tools

Vocal stems are cached in lyricsync_cache/ so re-running a song (e.g. after fixing a lyric) skips
the slow Demucs step. The GUI's Lyric Timing tab calls run() below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import lyricsync_core as lc

CACHE_DIR = Path(__file__).parent / "lyricsync_cache"

# progress(fraction 0-1 or None, message)
Progress = Callable[[Optional[float], str], None]


def find_vocals(outputs: list[Path]) -> Path:
    for p in outputs:
        if p.stem.lower().endswith("vocals") and not p.stem.lower().endswith("no_vocals"):
            return p
    raise lc.LyricSyncError(f"Vocal separation produced no vocals file: {[p.name for p in outputs]}")


def cached_vocals(audio: Path, demucs_model: str, device_pref: str, progress: Progress) -> Path:
    """Vocal stem for this exact file (name + size + modified time + model), made once."""
    st = audio.stat()
    folder = CACHE_DIR / f"{audio.stem}_{st.st_size}_{int(st.st_mtime)}_{demucs_model}"
    existing = [p for p in sorted(folder.glob("*_vocals.wav")) if not p.stem.endswith("no_vocals")] if folder.is_dir() else []
    if existing:
        progress(0.7, "Reusing the vocal track from last time.")
        return existing[0]
    import core  # the existing Demucs wrapper in this project

    progress(0.0, "Separating vocals with Demucs (the slow part)...")
    outs = core.separate_stems(audio, stem="vocals", output_dir=folder, model=demucs_model,
                               device_preference=device_pref, output_format="wav",
                               progress_callback=lambda frac, msg: progress(frac * 0.7, msg))
    return find_vocals(outs)


def run(audio: Path, lyrics_text: str, output_dir: Path, model: str = "medium", device: str = "gpu",
        language: str = "en", words: bool = False, vocals: Optional[Path] = None, separate: bool = True,
        demucs_model: str = "htdemucs_ft", progress: Optional[Progress] = None) -> dict:
    say: Progress = progress or (lambda frac, msg: print(msg))
    audio = Path(audio).resolve()
    if not audio.is_file():
        raise lc.LyricSyncError(f"Audio file not found: {audio}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = lc.parse_lyrics(lyrics_text)
    title = lc.find_title(lyrics_text) or audio.stem
    say(0.0, f"{len(lines)} sung lines parsed ({sum(1 for l in lines if l.hint is not None)} with rough times).")

    dev_pref = "cpu" if device == "cpu" else "gpu"
    if vocals:
        vocal_path = Path(vocals).resolve()
        say(0.7, f"Using existing vocal track: {vocal_path.name}")
    elif separate:
        vocal_path = cached_vocals(audio, demucs_model, dev_pref, say)
    else:
        vocal_path = audio
        say(0.7, "Skipping vocal separation (aligning against the full mix).")

    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        pass
    lc.align_vocals(vocal_path, lines, model_name=model, device=("cuda" if device != "cpu" and cuda_ok else "cpu"),
                    language=language, progress=lambda m: say(0.85, m))

    stem = audio.stem
    duration = lc.audio_duration(audio)
    meta = {"title": title, "audio": str(audio), "vocals": str(vocal_path), "model": model, "language": language,
            "duration": duration, "line_count": len(lines)}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Timestamp a lyric file against a song.")
    ap.add_argument("audio", type=Path, help="song (.mp3 or .wav)")
    ap.add_argument("lyrics", type=Path, help="lyrics text file, one sung line per line")
    ap.add_argument("--output-dir", type=Path, help="default: next to the audio file")
    ap.add_argument("--model", default="medium", help="speech model: tiny, base, small, medium, large-v3 (default medium)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
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

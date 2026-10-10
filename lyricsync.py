"""
lyricsync - timestamp a lyric file against a song.

  python lyricsync.py song.mp3 lyrics.txt
  python lyricsync.py song.mp3 lyrics.txt --words --model large-v3
  python lyricsync.py song.mp3 lyrics.txt --vocals "song_vocals.wav"   (reuse an existing vocal stem)

Writes next to the song (or to --output-dir):
  <song>_timed.txt    readable summary + every line with start-end times (paste into Claude chat/design)
  <song>.lrc          standard timed-lyrics file (add --words for per-word tags)
  <song>_timing.json  everything, including per-word times and confidence, for other tools
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lyricsync_core as lc


def find_vocals(outputs: list[Path]) -> Path:
    for p in outputs:
        if p.stem.lower().endswith("vocals") and not p.stem.lower().endswith("no_vocals"):
            return p
    raise lc.LyricSyncError(f"Vocal separation produced no vocals file: {[p.name for p in outputs]}")


def run(audio: Path, lyrics_file: Path, output_dir: Path, model: str, device: str, language: str,
        words: bool, vocals: Path | None, separate: bool, demucs_model: str) -> dict:
    audio, lyrics_file = Path(audio).resolve(), Path(lyrics_file).resolve()
    if not audio.is_file():
        raise lc.LyricSyncError(f"Audio file not found: {audio}")
    if not lyrics_file.is_file():
        raise lc.LyricSyncError(f"Lyrics file not found: {lyrics_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    lyric_text = lyrics_file.read_text(encoding="utf-8-sig")
    lines = lc.parse_lyrics(lyric_text)
    title = lc.find_title(lyric_text) or audio.stem
    print(f"{len(lines)} sung lines parsed ({sum(1 for l in lines if l.hint is not None)} with rough times).")

    dev_pref = "cpu" if device == "cpu" else "gpu"
    if vocals:
        vocal_path = Path(vocals).resolve()
        print(f"Using existing vocal track: {vocal_path.name}")
    elif separate:
        import core  # the existing Demucs wrapper in this project

        work = output_dir / "_lyricsync_work"
        print("Separating vocals with Demucs (this is the slow part)...")
        outs = core.separate_stems(audio, stem="vocals", output_dir=work, model=demucs_model,
                                   device_preference=dev_pref, output_format="wav")
        vocal_path = find_vocals(outs)
    else:
        vocal_path = audio
        print("Skipping vocal separation (aligning against the full mix).")

    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        pass
    lc.align_vocals(vocal_path, lines, model_name=model, device=("cuda" if device != "cpu" and cuda_ok else "cpu"),
                    language=language, progress=print)

    stem = audio.stem
    duration = lc.audio_duration(audio)
    meta = {"title": title, "audio": str(audio), "vocals": str(vocal_path), "model": model, "language": language,
            "duration": duration, "line_count": len(lines)}
    files = {
        "timed": output_dir / f"{stem}_timed.txt",
        "lrc": output_dir / f"{stem}.lrc",
        "json": output_dir / f"{stem}_timing.json",
    }
    files["timed"].write_text(lc.to_timed_text(lines, title, duration), encoding="utf-8")
    files["lrc"].write_text(lc.to_lrc(lines, words=words), encoding="utf-8")
    files["json"].write_text(lc.to_json(lines, meta), encoding="utf-8")
    print()
    print(lc.report(lines))
    print()
    for label, p in files.items():
        print(f"{label:6s} {p}")
    return {"files": {k: str(v) for k, v in files.items()}, "lines": len(lines),
            "flagged": sum(1 for l in lines if l.flags)}


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
        run(a.audio, a.lyrics, a.output_dir or a.audio.resolve().parent, a.model, a.device, a.language,
            a.words, a.vocals, not a.no_separate, a.demucs_model)
    except Exception as exc:  # one clear line instead of a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

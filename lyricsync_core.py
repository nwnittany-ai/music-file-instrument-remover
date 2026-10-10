"""
Lyric timing core: align a known lyric text to a song's vocals and produce timestamped output.

Pure functions (parsing, mapping, formatting) are separate from the one function that needs the
speech model (align_vocals), so everything except alignment can be tested without Whisper.
Word-level timings are always kept in the data when the aligner provides them; they are only
written to the .lrc file when requested, so adding word-level output later needs no redesign.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

SECTION_RE = re.compile(r"^\s*[\[\(].*[\]\)]\s*$")  # [Verse 1], (Chorus), etc. - labels, not sung text
TITLE_RE = re.compile(r"^\s*\[title\]\s*(.*)$", re.IGNORECASE)
HINT_RE = re.compile(r"^\s*(\d+):(\d{1,2}(?:\.\d+)?)\s+(?=\S)")  # optional leading "m:ss " rough time
HINT_TOLERANCE_SECONDS = 4.0  # flag a line whose aligned start is this far from the user's rough time
GAP_REPORT_SECONDS = 4.0      # instrumental gaps at least this long are listed in the summary
LOW_CONFIDENCE = 0.5          # mean word probability below this flags a line
SUSPICIOUS_MIN_SECONDS = 0.25  # a sung line shorter than this is probably mis-aligned
FAILED_CONFIDENCE = 0.1       # below this a line is treated as a failed alignment and retried
SUSPICIOUS_MAX_SECONDS = 25.0  # ...and so is one longer than this


class LyricSyncError(Exception):
    """A user-facing problem (bad input, alignment failed, etc.)."""


@dataclass
class Line:
    index: int
    text: str
    section: Optional[str] = None      # section label that begins at this line, if any
    hint: Optional[float] = None       # rough start time the user typed in the lyrics file, if any
    start: Optional[float] = None      # seconds from the start of the audio
    end: Optional[float] = None
    confidence: Optional[float] = None  # mean word probability, 0-1, when the aligner provides it
    words: list = field(default_factory=list)  # [{"word", "start", "end", "probability"}]
    flags: list = field(default_factory=list)


def find_title(text: str) -> Optional[str]:
    for raw in text.splitlines():
        m = TITLE_RE.match(raw)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def parse_lyrics(text: str) -> list[Line]:
    """Sung lines only. [Section] labels are attached to the line that follows them, a [title] line
    is skipped, lines with no letters (e.g. "...") are skipped, and a leading "m:ss " on a line is
    kept as a rough-time hint (not part of the sung text)."""
    lines: list[Line] = []
    pending: Optional[str] = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s or TITLE_RE.match(s):
            continue
        if SECTION_RE.match(s):
            pending = s.strip("[]() ").strip()
            continue
        hint = None
        m = HINT_RE.match(s)
        if m:
            hint = int(m.group(1)) * 60 + float(m.group(2))
            s = s[m.end():].strip()
        if not any(c.isalpha() for c in s):
            continue
        lines.append(Line(len(lines), s, pending, hint))
        pending = None
    if not lines:
        raise LyricSyncError("The lyrics file has no sung lines.")
    return lines


def map_words_to_lines(lines: list[Line], words: list) -> None:
    """Assign aligned words (objects with .word/.start/.end/.probability) to lyric lines by
    counting whitespace-separated tokens per line. Alignment keeps the text's order, so repeated
    choruses each get their own time. Raises if the counts disagree (tokenisation mismatch)."""
    expected = sum(len(l.text.split()) for l in lines)
    if len(words) != expected:
        raise LyricSyncError(
            f"Aligner returned {len(words)} words but the lyrics have {expected}; cannot map lines safely."
        )
    i = 0
    for line in lines:
        n = len(line.text.split())
        chunk = words[i:i + n]
        i += n
        line.words = [
            {"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
             "probability": None if getattr(w, "probability", None) is None else round(float(w.probability), 3)}
            for w in chunk
        ]
        line.start, line.end = line.words[0]["start"], line.words[-1]["end"]
        probs = [w["probability"] for w in line.words if w["probability"] is not None]
        line.confidence = round(sum(probs) / len(probs), 3) if probs else None


def flag_lines(lines: list[Line]) -> None:
    """Cheap sanity checks so a human knows which lines to double-check."""
    prev_end = None
    for l in lines:
        l.flags = []
        if l.start is None:
            l.flags.append("not aligned")
            continue
        dur = l.end - l.start
        if dur < SUSPICIOUS_MIN_SECONDS:
            l.flags.append(f"very short ({dur:.2f}s)")
        if dur > SUSPICIOUS_MAX_SECONDS:
            l.flags.append(f"very long ({dur:.1f}s)")
        if l.confidence is not None and l.confidence < LOW_CONFIDENCE:
            l.flags.append(f"low confidence ({l.confidence:.2f})")
        if l.hint is not None and abs(l.start - l.hint) > HINT_TOLERANCE_SECONDS:
            l.flags.append(f"starts at {_clock(l.start)} but your rough time was {_clock(l.hint)}")
        if prev_end is not None and l.start < prev_end - 0.05:
            l.flags.append("overlaps previous line")
        prev_end = l.end


def _clock(t: float, tenths: bool = True) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}" if tenths else f"{int(m)}:{int(s):02d}"


def _lrc_stamp(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def to_lrc(lines: list[Line], words: bool = False) -> str:
    """Standard .lrc (one timestamp per line). With words=True, adds enhanced-LRC <mm:ss.xx>
    per-word tags after the line stamp."""
    out = []
    for l in lines:
        if l.start is None:
            continue
        if words and l.words:
            body = " ".join(f"<{_lrc_stamp(w['start'])}>{w['word']}" for w in l.words)
        else:
            body = l.text
        out.append(f"[{_lrc_stamp(l.start)}]{body}")
    return "\n".join(out) + "\n"


def to_timed_text(lines: list[Line], title: str, duration: Optional[float]) -> str:
    """Readable summary meant to be pasted into Claude chat/design: where lyrics begin and end,
    instrumental gaps, then every line with its start-end time."""
    timed = [l for l in lines if l.start is not None]
    out = [f"Song: {title}" + (f"  (length {_clock(duration)})" if duration else "")]
    if timed:
        first, last = timed[0], timed[-1]
        out.append(f"First lyric starts at {_clock(first.start)} (intro of {first.start:.1f}s)")
        tail = f"; {duration - last.end:.1f}s of outro after it" if duration and duration > last.end else ""
        out.append(f"Last lyric ends at {_clock(last.end)}{tail}")
        gaps = [(a.end, b.start) for a, b in zip(timed, timed[1:]) if b.start - a.end >= GAP_REPORT_SECONDS]
        if gaps:
            out.append("Instrumental gaps (>= %.0fs): " % GAP_REPORT_SECONDS
                       + ", ".join(f"{_clock(a)}-{_clock(b)} ({b - a:.1f}s)" for a, b in gaps))
    flagged = [l for l in lines if l.flags]
    if flagged:
        out.append(f"{len(flagged)} line(s) flagged for a second look (marked with !)")
    out.append("")
    for l in lines:
        if l.section:
            out.append(f"[{l.section}]")
        if l.start is None:
            out.append(f"  (not aligned)  {l.text}")
            continue
        mark = " !" if l.flags else ""
        out.append(f"{_clock(l.start)} - {_clock(l.end)}  {l.text}{mark}")
    return "\n".join(out) + "\n"


def to_json(lines: list[Line], meta: dict) -> str:
    return json.dumps({"meta": meta, "lines": [asdict(l) for l in lines]}, indent=2, ensure_ascii=False)


def report(lines: list[Line]) -> str:
    flagged = [l for l in lines if l.flags]
    if not flagged:
        return "All lines aligned with no flags."
    return "\n".join(f"line {l.index + 1} ({_clock(l.start) if l.start is not None else '--'}): "
                     f"{l.text}  <- {'; '.join(l.flags)}" for l in flagged)


def audio_duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def _is_failed(l: Line) -> bool:
    """Whisper sometimes gives up on a stretch and squeezes its lines into a near-zero span."""
    return l.start is None or (l.end - l.start) < SUSPICIOUS_MIN_SECONDS or (l.confidence or 1.0) < FAILED_CONFIDENCE


def repair_failed(model, audio_path: str, lines: list[Line], language: str, say) -> list[int]:
    """Re-align each run of failed lines on only the audio between the neighbouring lines that did
    align. Returns the indexes of lines that were re-timed."""
    from types import SimpleNamespace

    from whisper.audio import SAMPLE_RATE, load_audio

    failed = [_is_failed(l) for l in lines]
    if not any(failed) or all(failed):
        return []
    audio = None
    fixed: list[int] = []
    i = 0
    while i < len(lines):
        if not failed[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and failed[j + 1]:
            j += 1
        before = next((lines[k] for k in range(i - 1, -1, -1) if not failed[k]), None)
        after = next((lines[k] for k in range(j + 1, len(lines)) if not failed[k]), None)
        t0 = before.end if before else 0.0
        if audio is None:
            audio = load_audio(audio_path)
        t1 = after.start if after else len(audio) / SAMPLE_RATE
        group = lines[i:j + 1]
        say(f"Retrying lines {i + 1}-{j + 1} on {t0:.1f}s-{t1:.1f}s...")
        try:
            res = model.align(audio[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)],
                              "\n".join(l.text for l in group), language=language, original_split=True)
            shifted = [SimpleNamespace(word=w.word, start=w.start + t0, end=w.end + t0,
                                       probability=getattr(w, "probability", None)) for w in res.all_words()]
            map_words_to_lines(group, shifted)
            if not any(_is_failed(l) for l in group):
                fixed.extend(range(i, j + 1))
        except Exception as exc:
            say(f"  retry failed: {exc}")
        i = j + 1
    return fixed


def align_vocals(vocals_path: Path, lines: list[Line], model_name: str = "medium", device: str = "cuda",
                 language: str = "en", progress: Optional[Callable[[str], None]] = None) -> None:
    """Align the lyric lines to the vocal track in place. Needs the stable-ts package."""
    try:
        import stable_whisper
    except ImportError as exc:
        raise LyricSyncError("The 'stable-ts' package is not installed in this environment "
                             "(pip install stable-ts).") from exc
    say = progress or (lambda m: None)
    say(f"Loading speech model '{model_name}' on {device}...")
    model = stable_whisper.load_model(model_name, device=device)
    say("Aligning lyrics to vocals...")
    text = "\n".join(l.text for l in lines)
    result = model.align(str(vocals_path), text, language=language, original_split=True)
    words = list(result.all_words())
    map_words_to_lines(lines, words)
    flag_lines(lines)
    repaired = repair_failed(model, str(vocals_path), lines, language, say)
    if repaired:
        flag_lines(lines)
        for i in repaired:
            lines[i].flags.append("re-aligned in a second pass")
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

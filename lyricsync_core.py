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
LOW_CONFIDENCE = 0.3          # mean word probability below this flags a line (sung words score low even when placed right)
SUSPICIOUS_MIN_SECONDS = 0.25  # a sung line shorter than this is probably mis-aligned
FAILED_CONFIDENCE = 0.1       # below this a line is treated as a failed alignment and retried
SUSPICIOUS_MAX_SECONDS = 25.0  # ...and so is one longer than this

# --- finding where the singing is (see find_sung_blocks / plan_blocks) ---
BLOCK_MIN_LENGTH_S = 1.5       # a "block" shorter than this is a blip (stray noise, a shout), not a verse
BLOCK_MIN_RUN_S = 0.25         # a stretch of sound shorter than this is noise, not singing, when measuring pauses
# What counts as a break is measured from the song itself, not fixed: ordinary pauses between sung lines are all
# about the same length (2-3.3 s in the songs tried so far), instrumental breaks are much longer. A fixed 3 s cut-off
# sat right at the size of an ordinary pause and chopped a verse into line-sized pieces (see choose_gap_threshold).
BLOCK_BREAK_MIN_S = 5.0        # a pause must last at least this long to be an instrumental break
BLOCK_BREAK_JUMP = 1.6         # ...and be at least this many times longer than the longest ordinary pause
BLOCK_PAD_S = 1.5              # extra audio on each side of a block when aligning, so edge words are not clipped
INSTRUMENTAL_WORDS = ("guitar", "solo", "fill", "instrumental", "interlude", "break", "intro", "drum", "riff")
PLAN_NON_SECTION_CUT_PENALTY = 6.0   # seconds-equivalent cost of splitting a block's lyrics in the middle of a section
PLAN_INSTRUMENTAL_CUT_BONUS = 4.0    # ...and the reward for cutting right where an instrumental label sits
PLAN_SKIP_BLOCK_PENALTY = 3.0        # extra cost for deciding a block of singing has no lyrics (ad-libs, backing vocals)
PLAN_HINT_MISS_PENALTY = 10.0        # the user's rough time falls outside the block a line was assigned to
PLAN_HINT_SLACK_S = 4.0
FIT_OK_RANGE = (0.5, 2.0)       # block length / expected length outside this is reported as worth a look
TRIM_PAD_S = 0.2               # a line's end is placed this far after the last singing under it...
TRIM_MAX_INNER_GAP_S = 3.0     # ...and never reaches across a silence longer than this (that is the next section's singing)
SECTION_WORDS = ("verse", "chorus", "bridge", "intro", "outro", "pre", "refrain", "hook", "solo", "interlude",
                 "instrumental", "break", "guitar", "drum", "fill", "riff", "coda", "tag", "vamp", "breakdown")
EVIDENCE_WEIGHT = 3.0          # seconds-equivalent value of one lyric word heard in a block (see plan_blocks)
EVIDENCE_LENGTH_WEIGHT = 0.1   # with that evidence, block length is only a tie-breaker
EVIDENCE_FUZZY = 0.75          # two words this similar (difflib ratio) count as the same when matching a transcript
TRIM_ECHO_GAP_S = 2.0          # for a line followed by echo lines: a silence this long ends the line (the echo is next)
SNAP_LEAD_MIN_S = 0.6          # a line whose first stretch is this long without singing is moved to where the sound starts
SNAP_PAD_S = 0.1               # ...a little before the first sustained sound
SNAP_WINDOW_S = 0.6            # "sustained sound" = most of this window is voiced...
SNAP_DENSITY = 0.6             # ...i.e. at least this fraction of it
HELD_BRIDGE_S = 0.3            # a held word's sound may have dips this short (breath, vibrato) and still be one note
ESTIMATE_FLAG ="estimated: spread over the singing block by word count (alignment failed here)"


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
    repaired: bool = False             # re-timed by the second pass (kept out of flags: it passed the checks)
    labels: list = field(default_factory=list)  # every [label] between the previous sung line and this one
    echoes: list = field(default_factory=list)  # "(echo)" lines written right after this line: sung, but not timed
    rest_before: bool = False          # a rest marker line ("...", "----") sits just before this line in the lyrics

    @property
    def sung_text(self) -> str:
        """What is actually sung for this line, echoes included. This is what the aligner is given, so the echo's
        singing is matched to the echo's words instead of being squeezed into the end of the line before it."""
        return " ".join([self.text] + list(self.echoes))


def find_title(text: str) -> Optional[str]:
    for raw in text.splitlines():
        m = TITLE_RE.match(raw)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def _is_section_name(label: str) -> bool:
    """"(Chorus)" is a section label; "(Who will he see next?)" is an echo line. Told apart by section words."""
    return any(w in SECTION_WORDS for w in re.findall(r"[a-z]+", label.lower()))


def parse_lyrics(text: str) -> list[Line]:
    """Sung lines only. [Section] labels are attached to the line that follows them, a [title] line
    is skipped, lines with no letters (e.g. "...") are skipped, and a leading "m:ss " on a line is
    kept as a rough-time hint (not part of the sung text). A line of only punctuation ("...", "----") is a rest
    marker: it is not sung, but the next sung line is remembered as starting after a rest (Line.rest_before)."""
    lines: list[Line] = []
    pending_labels: list[str] = []
    pending_rest = False
    for raw in text.splitlines():
        s = raw.strip()
        if not s or TITLE_RE.match(s):
            continue
        if SECTION_RE.match(s):
            label = s.strip("[]() ").strip()
            if s.lstrip().startswith("(") and not _is_section_name(label):
                if lines and not pending_labels:   # an echo / ad-lib: it belongs to the line it follows
                    lines[-1].echoes.append(label)
                continue
            pending_labels.append(label)
            continue
        hint = None
        m = HINT_RE.match(s)
        if m:
            hint = int(m.group(1)) * 60 + float(m.group(2))
            s = s[m.end():].strip()
        if not any(c.isalpha() for c in s):
            pending_rest = pending_rest or bool(lines)   # a rest before the first line is just the intro
            continue
        lines.append(Line(len(lines), s, section=pending_labels[-1] if pending_labels else None, hint=hint,
                          labels=list(pending_labels), rest_before=pending_rest))
        pending_labels = []
        pending_rest = False
    if not lines:
        raise LyricSyncError("The lyrics file has no sung lines.")
    return lines


def map_words_to_lines(lines: list[Line], words: list) -> None:
    """Assign aligned words (objects with .word/.start/.end/.probability) to lyric lines by
    counting whitespace-separated tokens per line. Alignment keeps the text's order, so repeated
    choruses each get their own time. Raises if the counts disagree (tokenisation mismatch)."""
    expected = sum(len(l.sung_text.split()) for l in lines)
    if len(words) != expected:
        raise LyricSyncError(
            f"Aligner returned {len(words)} words but the lyrics have {expected}; cannot map lines safely."
        )
    i = 0
    for line in lines:
        n = len(line.sung_text.split())
        chunk = words[i:i + len(line.text.split())]   # the echo words that follow are not part of the line
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
    if not tenths:
        m, s = divmod(max(t, 0.0), 60)
        return f"{int(m)}:{int(s):02d}"
    m, s = divmod(round(max(t, 0.0) * 10) / 10, 60)  # round first, so 59.97 s reads 1:00.0 and never 0:60.0
    return f"{int(m)}:{s:04.1f}"


def _lrc_stamp(t: float) -> str:
    m, s = divmod(round(max(t, 0.0) * 100) / 100, 60)
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


def to_timed_text(lines: list[Line], title: str, duration: Optional[float], blocks: Optional[list] = None,
                  mode: Optional[str] = None) -> str:
    """Readable summary meant to be pasted into Claude chat/design: where lyrics begin and end,
    instrumental gaps, then every line with its start-end time. `blocks` (from align_vocals) adds where
    the singing was found; `mode` says which alignment path was used ("singing blocks" or "whole song")."""
    timed = [l for l in lines if l.start is not None]
    out = [f"Song: {title}" + (f"  (length {_clock(duration)})" if duration else "")]
    if mode:
        out.append(f"Alignment: {mode}")
    if blocks:
        out.append("Singing found at: " + ", ".join(
            f"{_clock(b['start'])}-{_clock(b['end'])}"
            + (f" (lines {b['first_line']}-{b['last_line']})" if b.get("first_line") else " (no lyrics matched)")
            for b in blocks))
        odd = [b for b in blocks if b.get("fit") and not FIT_OK_RANGE[0] <= b["fit"] <= FIT_OK_RANGE[1]]
        for b in odd:
            out.append(f"Check lines {b['first_line']}-{b['last_line']}: the singing at {_clock(b['start'])}-{_clock(b['end'])} "
                       f"is {'much longer' if b['fit'] > 1 else 'much shorter'} than those lines should take - "
                       "the lyric sections may be matched to the wrong stretch of singing.")
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


VOICED_MIN_FRACTION = 0.4   # a line needs at least this much singing under it (vocal-track loudness)
HINT_PAD_BEFORE = 6.0       # when retrying, look this far before the user's rough time...
HINT_PAD_AFTER = 8.0        # ...and this far after the last line's rough time
REPAIR_ROUNDS = 4


def voiced_mask(audio, sample_rate: int, hop_s: float = 0.02):
    """Per-20ms 'is someone singing' flags from the vocal track's loudness. Threshold is relative
    to the track's own loud passages, so it works for quiet and loud songs alike."""
    import numpy as np

    hop = max(1, int(sample_rate * hop_s))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool), hop_s
    frames = audio[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype("float64") ** 2).mean(axis=1))
    return rms > max(1e-4, 0.1 * float(np.percentile(rms, 90))), hop_s


def voiced_fraction(mask, hop_s: float, t0: float, t1: float) -> float:
    a, b = int(t0 / hop_s), max(int(t1 / hop_s), int(t0 / hop_s) + 1)
    seg = mask[a:b]
    return float(seg.mean()) if len(seg) else 1.0


def _sound_runs(mask, hop_s: float) -> list[tuple[float, float]]:
    """(start_s, end_s) of each stretch of sound in the vocal track, ignoring specks shorter than BLOCK_MIN_RUN_S."""
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    padded = np.concatenate(([False], m, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(float(s), float(e)) for s, e in zip(edges[0::2] * hop_s, edges[1::2] * hop_s) if e - s >= BLOCK_MIN_RUN_S]


def choose_gap_threshold(gaps: list[float]) -> Optional[float]:
    """Where to draw the line between an ordinary pause and an instrumental break, from the song's own pauses.

    Pauses between sung lines are all about the same length; breaks (a fill, a solo, an intro) are much longer.
    The line goes at the shortest pause length where the next one up is both a real break (>= BLOCK_BREAK_MIN_S) and
    clearly longer (>= BLOCK_BREAK_JUMP x). A virtual 1 s pause is included so a song whose every pause is a long
    break is handled. When pauses grow gradually (no clear jump, e.g. 4.3, 6.3, 10.0, 10.4) it falls back to the plain
    minimum: every pause of at least BLOCK_BREAK_MIN_S is a break. Returns None only when no pause is that long (the
    whole song is one stretch of singing).
    """
    jump = _jump_threshold(gaps)
    if jump is not None:
        return jump
    return BLOCK_BREAK_MIN_S if max([1.0] + [g for g in gaps if g >= 1.0]) >= BLOCK_BREAK_MIN_S else None


def _jump_threshold(gaps: list[float]) -> Optional[float]:
    """The clear-jump rule on its own (see choose_gap_threshold); None when no pause stands out that way."""
    ordered = [1.0] + sorted(g for g in gaps if g >= 1.0)
    for lower, upper in zip(ordered, ordered[1:]):
        if upper >= BLOCK_BREAK_MIN_S and upper / lower >= BLOCK_BREAK_JUMP:
            return (lower * upper) ** 0.5
    return None


def trim_line_ends(lines: list[Line], mask, hop_s: float) -> int:
    """Pull each line's end back to where the singing under it actually stops (plus TRIM_PAD_S). Alignment can stretch
    a line's end across a long pause or an instrumental stretch; the vocal track shows where the voice really stopped.
    Only ever shortens. The singing counted is between the line's start and its end (or the next line's start, if
    that is earlier), and stops at the first silence longer than TRIM_MAX_INNER_GAP_S. Returns how many ends moved.

    A line followed by echo lines is different: the echo is sung right after it, so the aligner cannot tell where the
    line stops and the echo starts (a held last note is shared out between them). There the end is simply where the
    unbroken singing that starts with the line stops - a silence of TRIM_ECHO_GAP_S or more ends it - and it may be
    moved later as well as earlier, but never past the next line's start."""
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    moved = 0
    for i, l in enumerate(lines):
        if l.start is None or l.end is None:
            continue
        nxt = next((o.start for o in lines[i + 1:] if o.start is not None), None)
        echo = bool(l.echoes)
        limit = (nxt if nxt is not None and nxt > l.start else len(m) * hop_s) if echo else l.end
        if not echo and nxt is not None and l.start < nxt < limit:
            limit = nxt
        a, b = int(l.start / hop_s), int(limit / hop_s)
        voiced = np.flatnonzero(m[a:b])
        if voiced.size == 0:
            continue
        cut = np.flatnonzero(np.diff(voiced) * hop_s > (TRIM_ECHO_GAP_S if echo else TRIM_MAX_INNER_GAP_S))
        last = voiced[cut[0]] if cut.size else voiced[-1]
        new_end = round((a + int(last) + 1) * hop_s + TRIM_PAD_S, 3)
        if nxt is not None and echo:
            new_end = min(new_end, nxt)
        if new_end < l.end - 0.05 or (echo and new_end > l.end + 0.05):
            l.end = max(new_end, l.start)
            for w in l.words:
                w["end"] = min(w["end"], l.end)
                w["start"] = min(w["start"], w["end"])
            if l.words and l.end > l.words[-1]["end"]:
                l.words[-1]["end"] = l.end
            moved += 1
    return moved


def _first_onset(m, hop_s: float, t0: float, t1: float) -> Optional[float]:
    """Time of the first sustained start of singing in [t0, t1): a voiced frame followed by a window that is mostly
    voiced. A stray tail of the previous line's sound (a few frames) does not count. None when there is none."""
    import numpy as np

    w = max(1, int(SNAP_WINDOW_S / hop_s))
    cum = np.concatenate(([0], np.cumsum(np.asarray(m, dtype=np.int64))))
    a, b = max(0, int(t0 / hop_s)), min(len(m), int(t1 / hop_s))
    for i in range(a, b):
        if m[i] and (cum[min(i + w, len(m))] - cum[i]) >= SNAP_DENSITY * w:
            return i * hop_s
    return None


def snap_line_starts(lines: list[Line], mask, hop_s: float) -> int:
    """Move a line's start forward to where its sound begins. The aligner tends to stretch a line's first word back
    across the silence before it ("we" spanning 2 s of nothing), so the line looks like it starts long before anyone
    sings. If the first SNAP_LEAD_MIN_S of a line is silent in the vocal track and singing starts later inside the
    line, the start moves there (less SNAP_PAD_S). Only ever moves a start later. Returns how many were moved."""
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    moved = 0
    for l in lines:
        if l.start is None or l.end is None:
            continue
        if voiced_fraction(m, hop_s, l.start, l.start + SNAP_LEAD_MIN_S) >= SNAP_DENSITY:
            continue   # already on the sound
        onset = _first_onset(m, hop_s, l.start, l.end)
        if onset is None or onset - l.start < SNAP_LEAD_MIN_S:
            continue
        new_start = round(min(onset - SNAP_PAD_S, l.end), 3)
        if new_start <= l.start:
            continue
        l.start = new_start
        for w in l.words:
            if w["start"] < new_start:
                w["start"] = new_start
                w["end"] = max(w["end"], new_start)
        moved += 1
    return moved


def stretch_held_lines(lines: list[Line], mask, hop_s: float) -> list[int]:
    """A held word ("time....." sung for four seconds) often comes back from the aligner as a sliver of zero length
    somewhere in the middle of the note, with a confidence near zero. For such a line the voice is still going
    strongly, so its start and end are taken from the sound itself: back to where the singing under it began and
    forward to where it stops, but never into the previous line or past the next line's start. Returns the indexes
    of the lines changed."""
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    gap = max(1, int(HELD_BRIDGE_S / hop_s))
    changed: list[int] = []
    for i, l in enumerate(lines):
        if l.start is None or l.end is None:
            continue
        if not ((l.end - l.start) < SUSPICIOUS_MIN_SECONDS or (l.confidence is not None and l.confidence < FAILED_CONFIDENCE)):
            continue
        prev_end = max((o.end for o in lines[:i] if o.end is not None), default=0.0)
        nxt = next((o.start for o in lines[i + 1:] if o.start is not None), None)
        floor = int(prev_end / hop_s)
        ceil = min(len(m), int(nxt / hop_s) if nxt is not None else len(m))
        core = next((k for k in range(max(floor, int((l.start - HELD_BRIDGE_S) / hop_s)),
                                      min(ceil, int((l.end + HELD_BRIDGE_S) / hop_s) + 1)) if m[k]), None)
        if core is None:
            continue
        lo = hi = core   # grow over the sound, bridging silences shorter than HELD_BRIDGE_S
        for k in range(core - 1, floor - 1, -1):
            if m[k]:
                lo = k
            elif lo - k > gap:
                break
        for k in range(core + 1, ceil):
            if m[k]:
                hi = k
            elif k - hi > gap:
                break
        new_start, new_end = round(lo * hop_s, 3), round(min((hi + 1) * hop_s, nxt if nxt is not None else 1e9), 3)
        if new_end - new_start <= (l.end - l.start) + 0.05:
            continue
        l.start, l.end = new_start, new_end
        if l.words:
            l.words[0]["start"], l.words[-1]["end"] = new_start, new_end
            for w in l.words:
                w["start"] = min(max(w["start"], new_start), new_end)
                w["end"] = min(max(w["end"], w["start"]), new_end)
        changed.append(i)
    return changed


def find_sung_blocks(mask, hop_s: float, min_gap_s: Optional[float] = None,
                     min_length_s: float = BLOCK_MIN_LENGTH_S) -> list[tuple[float, float]]:
    """Where the singing is: stretches of sound in the vocal track, joined across pauses shorter than min_gap_s,
    dropping anything shorter than min_length_s (stray noise). min_gap_s defaults to the song's own break length
    (choose_gap_threshold); a song with no real break comes back as one block. Returns [(start_s, end_s)]."""
    runs = _sound_runs(mask, hop_s)
    if not runs:
        return []
    if min_gap_s is None:
        min_gap_s = choose_gap_threshold([b[0] - a[1] for a, b in zip(runs, runs[1:])])
        if min_gap_s is None:
            return [(runs[0][0], runs[-1][1])] if runs[-1][1] - runs[0][0] >= min_length_s else []
    blocks: list[tuple[float, float]] = []
    for s, e in runs:
        if blocks and s - blocks[-1][1] < min_gap_s:
            blocks[-1] = (blocks[-1][0], e)
        else:
            blocks.append((s, e))
    return [(s, e) for s, e in blocks if e - s >= min_length_s]


def _expects_break(line: Line) -> bool:
    """True when a label just before this line names an instrumental part ([Guitar Solo], [Intro], ...)."""
    return any(w in label.lower() for label in line.labels for w in INSTRUMENTAL_WORDS)


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= EVIDENCE_FUZZY


def match_evidence(lines: list[Line], transcripts: list[str]) -> Optional[list[list[int]]]:
    """Which lyric lines were actually heard in which block, from a plain transcript of each block.

    The transcripts (in block order) are lined up with the lyrics as one long in-order sequence match (longest common
    subsequence of words, similar spellings count as equal), so a chorus sung three times is matched to its three
    places in order. Echo lines are part of the lyric stream (they are sung) but count for the line before them.
    Returns counts[block][line] = words of that line heard in that block, or None when nothing matched at all."""
    lyr, owner = [], []
    for i, l in enumerate(lines):
        for w in _norm_words(l.text) + [w for e in l.echoes for w in _norm_words(e)]:
            lyr.append(w)
            owner.append(i)
    heard, block_of = [], []
    for k, t in enumerate(transcripts):
        for w in _norm_words(t):
            heard.append(w)
            block_of.append(k)
    if not lyr or not heard:
        return None
    n, m = len(lyr), len(heard)
    eq: dict = {}

    def same(a, b):
        key = (a, b)
        if key not in eq:
            eq[key] = _same_word(a, b)
        return eq[key]

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        row, prev = dp[a], dp[a - 1]
        for b in range(1, m + 1):
            best = prev[b] if prev[b] >= row[b - 1] else row[b - 1]
            if prev[b - 1] + 1 > best and same(lyr[a - 1], heard[b - 1]):
                best = prev[b - 1] + 1
            row[b] = best
    counts = [[0] * len(lines) for _ in transcripts]
    a, b = n, m
    while a > 0 and b > 0:
        if dp[a][b] == dp[a - 1][b]:
            a -= 1
        elif dp[a][b] == dp[a][b - 1]:
            b -= 1
        else:
            counts[block_of[b - 1]][owner[a - 1]] += 1
            a -= 1
            b -= 1
    return counts if any(any(r) for r in counts) else None


def plan_blocks(lines: list[Line], blocks: list[tuple[float, float]],
                evidence: Optional[list[list[int]]] = None) -> list[list[int]]:
    """Decide which lyric lines belong to which block of singing.

    Lines stay in order and each block gets a run of consecutive lines (possibly none: ad-libs, backing vocals).
    The best split is the one where each block's length best matches the time its words should take (the singing
    pace is worked out from the song itself: total singing time / total words), preferring to cut where a new
    [section] starts - especially after an instrumental label like [Guitar Solo] - and where the user's own rough
    times (if given) agree. Returns one list of line indexes per block.

    Block length is only a rough guide (a chant repeated at the end, a reprise that is not in the lyrics, or echo
    lines that are sung but not timed make a section sing much longer than its words suggest). When `evidence` is
    given (match_evidence: what was actually heard in each block) it decides instead, and length only breaks ties.
    """
    K, N = len(blocks), len(lines)
    if K == 0:
        return []
    if K == 1:
        return [list(range(N))]
    words = [max(1, len(l.text.split())) for l in lines]
    prefix = [0]
    for w in words:
        prefix.append(prefix[-1] + w)
    dur = [e - s for s, e in blocks]
    rate = sum(dur) / prefix[N]  # seconds of singing per word
    heard = None
    if evidence is not None:
        heard = []   # heard[k][i]: words heard in block k that belong to the lines before i (cumulative)
        for k in range(K):
            acc = [0]
            for i in range(N):
                acc.append(acc[-1] + evidence[k][i])
            heard.append(acc)

    # misses[k][i]: lines before index i whose rough time falls outside block k (cumulative)
    misses = []
    for s, e in blocks:
        acc = [0]
        for l in lines:
            out = l.hint is not None and not (s - PLAN_HINT_SLACK_S <= l.hint <= e + PLAN_HINT_SLACK_S)
            acc.append(acc[-1] + (1 if out else 0))
        misses.append(acc)

    def cut_cost(j: int) -> float:  # starting a new block's lyrics at line j
        if j <= 0 or j >= N:
            return 0.0
        if not lines[j].labels and not lines[j].rest_before:   # a rest the user wrote is as good a cut place as a new section
            return PLAN_NON_SECTION_CUT_PENALTY
        return -PLAN_INSTRUMENTAL_CUT_BONUS if _expects_break(lines[j]) else 0.0

    def block_cost(k: int, j: int, i: int) -> float:
        if i == j:
            return (0.0 if heard else dur[k]) + PLAN_SKIP_BLOCK_PENALTY
        length_gap = abs(dur[k] - rate * (prefix[i] - prefix[j]))
        if heard:
            value = EVIDENCE_LENGTH_WEIGHT * length_gap - EVIDENCE_WEIGHT * (heard[k][i] - heard[k][j])
        else:
            value = length_gap
        return value + cut_cost(j) + PLAN_HINT_MISS_PENALTY * (misses[k][i] - misses[k][j])

    INF = float("inf")
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    back = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0
    for k in range(1, K + 1):
        for i in range(N + 1):
            for j in range(i + 1):
                if dp[k - 1][j] == INF:
                    continue
                c = dp[k - 1][j] + block_cost(k - 1, j, i)
                if c < dp[k][i]:
                    dp[k][i], back[k][i] = c, j
    plan: list[list[int]] = [[] for _ in range(K)]
    i = N
    for k in range(K, 0, -1):
        j = back[k][i]
        plan[k - 1] = list(range(j, i))
        i = j
    return plan


def block_fit(lines: list[Line], blocks: list[tuple[float, float]], plan: list[list[int]]) -> list[Optional[float]]:
    """How well each block's length matches the lines assigned to it: block seconds / expected seconds, where the
    expected time uses the song's own average pace per word. About 1.0 is a good match; far from 1 means the
    lyric sections and the singing were probably paired up wrongly (None for a block with no lyrics)."""
    total_words = sum(max(1, len(l.text.split())) for l in lines)
    rate = sum(e - s for s, e in blocks) / total_words if total_words and blocks else 0.0
    out: list[Optional[float]] = []
    for (s, e), idxs in zip(blocks, plan):
        words = sum(max(1, len(lines[i].text.split())) for i in idxs)
        out.append(round((e - s) / (rate * words), 2) if idxs and rate else None)
    return out


def estimate_block(group: list[Line], start: float, end: float) -> None:
    """Last resort for a block the aligner could not place: spread its lines (and words) over the block by word
    count. Rough, but in order and inside the singing, which beats a pile-up on one instant."""
    total_words = sum(max(1, len(l.text.split())) for l in group)
    t = start
    for l in group:
        toks = l.text.split()
        span = (end - start) * max(1, len(toks)) / total_words
        step = span / max(1, len(toks))
        l.words = [{"word": w, "start": round(t + k * step, 3), "end": round(t + (k + 1) * step, 3),
                    "probability": None} for k, w in enumerate(toks)]
        l.start, l.end, l.confidence = round(t, 3), round(t + span, 3), None
        t += span


def _is_failed(l: Line) -> bool:
    """Whisper sometimes gives up on a stretch and squeezes its lines into a near-zero span."""
    return l.start is None or (l.end - l.start) < SUSPICIOUS_MIN_SECONDS or (l.confidence or 1.0) < FAILED_CONFIDENCE


def _is_bad(l: Line, mask, hop_s: float) -> bool:
    """Failed, or placed where nobody is singing, or far from the user's own rough time."""
    if _is_failed(l):
        return True
    if voiced_fraction(mask, hop_s, l.start, l.end) < VOICED_MIN_FRACTION:
        return True
    return l.hint is not None and abs(l.start - l.hint) > HINT_TOLERANCE_SECONDS


def repair_failed(model, audio_path: str, lines: list[Line], language: str, say, audio=None):
    """Re-align every run of bad lines on only the audio between the nearest good lines (narrowed
    by the user's rough times when given). Tries the run alone, then widened by one line each side.
    A retry is kept only if it leaves fewer bad lines. Repeats while it keeps making progress.
    Returns (indexes of lines that were re-timed, voiced mask, mask hop in seconds)."""
    from types import SimpleNamespace

    from whisper.audio import SAMPLE_RATE, load_audio

    if audio is None:
        audio = load_audio(audio_path)
    total = len(audio) / SAMPLE_RATE
    mask, hop_s = voiced_mask(audio, SAMPLE_RATE)
    touched: set[int] = set()

    def bad_flags() -> list[bool]:
        return [_is_bad(l, mask, hop_s) for l in lines]

    def snapshot(group):
        return [(l.start, l.end, l.confidence, l.words) for l in group]

    def restore(group, snap):
        for l, (s, e, c, w) in zip(group, snap):
            l.start, l.end, l.confidence, l.words = s, e, c, w

    for _ in range(REPAIR_ROUNDS):
        bad = bad_flags()
        if not any(bad) or all(bad):
            break
        progressed = False
        i = 0
        while i < len(lines):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(lines) and bad[j + 1]:
                j += 1
            fixed_group = False
            reached = j
            for widen in (0, 1):
                a, b = max(0, i - widen), min(len(lines) - 1, j + widen)
                group = lines[a:b + 1]
                before = next((lines[k] for k in range(a - 1, -1, -1) if not bad[k]), None)
                after = next((lines[k] for k in range(b + 1, len(lines)) if not bad[k]), None)
                lo = before.end if before else 0.0
                hi = after.start if after else total
                if group[0].hint is not None and group[0].hint - HINT_PAD_BEFORE > lo:
                    lo = group[0].hint - HINT_PAD_BEFORE
                if group[-1].hint is not None and group[-1].hint + HINT_PAD_AFTER < hi:
                    hi = group[-1].hint + HINT_PAD_AFTER
                n_words = sum(len(l.text.split()) for l in group)
                if hi - lo < 0.3 * n_words:
                    continue  # no room for these words; try a wider window or give up
                say(f"Retrying lines {a + 1}-{b + 1} on {lo:.1f}s-{hi:.1f}s...")
                snap = snapshot(group)
                before_bad = sum(_is_bad(l, mask, hop_s) for l in group if l.start is not None) + \
                    sum(1 for l in group if l.start is None)
                try:
                    res = model.align(audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)],
                                      "\n".join(l.sung_text for l in group), language=language, original_split=True)
                    shifted = [SimpleNamespace(word=w.word, start=w.start + lo, end=w.end + lo,
                                               probability=getattr(w, "probability", None)) for w in res.all_words()]
                    map_words_to_lines(group, shifted)
                    after_bad = sum(_is_bad(l, mask, hop_s) for l in group)
                    if after_bad < before_bad:
                        touched.update(range(a, b + 1))
                        progressed = fixed_group = True
                        reached = max(j, b)
                        break
                    restore(group, snap)
                except Exception as exc:
                    restore(group, snap)
                    say(f"  retry failed: {exc}")
            i = reached + 1
            if fixed_group:
                bad = bad_flags()  # neighbours' windows depend on what is now good
        if not progressed:
            break
    return sorted(touched), mask, hop_s


def _align_group(model, audio, group: list[Line], lo: float, hi: float, language: str) -> bool:
    """Align the lines of `group` inside audio[lo:hi] (seconds). True when it clearly worked."""
    from types import SimpleNamespace

    from whisper.audio import SAMPLE_RATE

    try:
        res = model.align(audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)], "\n".join(l.sung_text for l in group),
                          language=language, original_split=True)
        shifted = [SimpleNamespace(word=w.word, start=w.start + lo, end=w.end + lo,
                                   probability=getattr(w, "probability", None)) for w in res.all_words()]
        map_words_to_lines(group, shifted)
    except Exception:
        return False
    return sum(_is_failed(l) for l in group) <= len(group) // 2


def _transcribe_blocks(model, audio, blocks, language: str, say) -> Optional[list[str]]:
    """Plain speech-to-text of each block of singing: evidence of which lyrics are where. Best effort - None when the
    model cannot transcribe (the plan then falls back to block lengths)."""
    from whisper.audio import SAMPLE_RATE

    texts = []
    try:
        for k, (b_start, b_end) in enumerate(blocks):
            say(f"Listening to stretch {k + 1} of {len(blocks)} to see which lyrics it holds...")
            a, b = int(max(0.0, b_start - BLOCK_PAD_S) * SAMPLE_RATE), int((b_end + BLOCK_PAD_S) * SAMPLE_RATE)
            res = model.transcribe(audio[a:b], language=language, verbose=None, condition_on_previous_text=False,
                                   fp16=False)
            texts.append(getattr(res, "text", "") or "")
    except Exception as exc:
        say(f"  (could not transcribe the singing, using block lengths instead: {str(exc)[:80]})")
        return None
    return texts


def align_vocals(vocals_path: Path, lines: list[Line], model_name: str = "medium", device: str = "cuda",
                 language: str = "en", progress: Optional[Callable[[str], None]] = None) -> dict:
    """Align the lyric lines to the vocal track in place. Needs the stable-ts package.

    First finds where the singing actually is (blocks separated by instrumental stretches), matches the lyric
    sections to those blocks, and aligns each section inside its own block, so the speech model never has to
    cross a long guitar solo or a silent intro. Songs with no clear gaps are aligned in one pass. Blocks the
    aligner cannot place are spread over the singing by word count and flagged. Returns what it found:
    {"mode": "singing blocks" | "whole song", "blocks": [{"start", "end", "first_line", "last_line"}]}.
    """
    try:
        import stable_whisper
    except ImportError as exc:
        raise LyricSyncError("The 'stable-ts' package is not installed in this environment "
                             "(pip install stable-ts).") from exc
    from whisper.audio import SAMPLE_RATE, load_audio

    say = progress or (lambda m: None)
    say(f"Loading speech model '{model_name}' on {device}...")
    model = stable_whisper.load_model(model_name, device=device)
    audio = load_audio(str(vocals_path))
    total = len(audio) / SAMPLE_RATE
    mask, hop_s = voiced_mask(audio, SAMPLE_RATE)
    runs = _sound_runs(mask, hop_s)
    pauses = [b[0] - a[1] for a, b in zip(runs, runs[1:])]
    break_s = choose_gap_threshold(pauses)
    blocks = find_sung_blocks(mask, hop_s)
    info: dict = {"mode": "whole song", "blocks": [],
                  "break_length_s": None if break_s is None else round(break_s, 1)}  # pauses longer than this split the singing
    estimated: set[int] = set()

    if len(blocks) >= 2:
        info["mode"] = "singing blocks"
        say(f"Found {len(blocks)} stretches of singing; matching lyric sections to them...")
        # Block lengths are only a rough guide to which section is where (held notes and rests make the pace very
        # uneven, and a clear-looking set of breaks can still be paired wrongly), so always listen to each block and
        # see which lyrics it actually holds; lengths then only break ties.
        evidence = None
        heard_text = _transcribe_blocks(model, audio, blocks, language, say)
        if heard_text:
            evidence = match_evidence(lines, heard_text)
            info["evidence"] = "transcript" if evidence else "none"
        plan = plan_blocks(lines, blocks, evidence)
        fits = block_fit(lines, blocks, plan)
        for k, ((b_start, b_end), idxs) in enumerate(zip(blocks, plan)):
            info["blocks"].append({"start": round(b_start, 2), "end": round(b_end, 2),
                                   "first_line": idxs[0] + 1 if idxs else None,
                                   "last_line": idxs[-1] + 1 if idxs else None, "fit": fits[k]})
            if not idxs:
                continue
            prev_end = blocks[k - 1][1] if k else 0.0
            next_start = blocks[k + 1][0] if k + 1 < len(blocks) else total
            lo = max(b_start - BLOCK_PAD_S, (prev_end + b_start) / 2 if k else 0.0)
            hi = min(b_end + BLOCK_PAD_S, (b_end + next_start) / 2 if k + 1 < len(blocks) else total)
            group = [lines[i] for i in idxs]
            say(f"Aligning lines {idxs[0] + 1}-{idxs[-1] + 1} inside {_clock(lo)}-{_clock(hi)}...")
            if not _align_group(model, audio, group, lo, hi, language):
                say(f"  could not place lines {idxs[0] + 1}-{idxs[-1] + 1}; spreading them over the singing instead")
                estimate_block(group, b_start, b_end)
                estimated.update(idxs)
    else:
        say("Aligning lyrics to vocals...")
        result = model.align(str(vocals_path), "\n".join(l.sung_text for l in lines), language=language,
                             original_split=True)
        map_words_to_lines(lines, list(result.all_words()))

    flag_lines(lines)
    repaired, mask, hop_s = repair_failed(model, str(vocals_path), lines, language, say, audio=audio)
    info["lines_trimmed"] = trim_line_ends(lines, mask, hop_s)  # before the final flags, so "very long" means still long
    info["lines_snapped"] = snap_line_starts(lines, mask, hop_s)
    stretched = stretch_held_lines(lines, mask, hop_s)
    info["lines_stretched"] = len(stretched)
    flag_lines(lines)
    for i in stretched:
        lines[i].flags.append("held note: start and end taken from the sound under it")
    for i, l in enumerate(lines):
        if l.start is not None and voiced_fraction(mask, hop_s, l.start, l.end) < VOICED_MIN_FRACTION:
            l.flags.append("little singing detected at this time")
        if i in estimated:
            l.flags.append(ESTIMATE_FLAG)
    for i in repaired:
        lines[i].repaired = True
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return info

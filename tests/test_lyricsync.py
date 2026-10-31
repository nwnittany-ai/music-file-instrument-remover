"""Tests for lyric timing (lyricsync_core.py / lyricsync.py). No pytest, no model download, no real audio:
a stand-in speech model and a generated "song" (bursts of sound with silent gaps) are used.

  .venv\\Scripts\\python.exe tests\\test_lyricsync.py          runs every test_* function, prints PASS/FAIL
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import lyricsync  # noqa: E402
import lyricsync_core as lc  # noqa: E402

SR = 16000


# ----------------------------------------------------------------------------------------------- helpers

def _song(bursts, total, path: Path) -> Path:
    """A mono WAV: noise in each (start, end) burst, silence elsewhere."""
    rng = np.random.default_rng(1)
    y = np.zeros(int(total * SR), dtype="float32")
    for s, e in bursts:
        y[int(s * SR):int(e * SR)] = (0.3 * rng.standard_normal(int((e - s) * SR))).astype("float32")
    sf.write(str(path), y, SR)
    return path


class FakeModel:
    """Spreads the words evenly over the audio it is given (like a perfect aligner would inside a clean block)."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, []

    @staticmethod
    def _was_path(_audio) -> bool:
        return FakeModel._last_was_path

    def align(self, audio, text, language=None, original_split=True):
        if self.fail:
            raise RuntimeError("model could not align")
        FakeModel._last_was_path = isinstance(audio, str)
        if isinstance(audio, str):
            audio = sf.read(audio)[0]
        toks = text.split()
        self.calls.append(("path" if self._was_path(audio) else "slice", len(audio) / SR, len(toks)))
        loud = np.flatnonzero(np.abs(audio) > 0.02)        # a real aligner puts words where the sound is
        lo, hi = (loud[0] / SR, loud[-1] / SR) if loud.size else (0.0, len(audio) / SR)
        dur = hi - lo
        words = [SimpleNamespace(word=w, start=lo + dur * (i + 0.05) / len(toks), end=lo + dur * (i + 0.95) / len(toks),
                                 probability=0.9) for i, w in enumerate(toks)]
        return SimpleNamespace(all_words=lambda: words)


class patched_model:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        import stable_whisper
        self._saved = stable_whisper.load_model
        stable_whisper.load_model = lambda name, device=None, **kw: self.model
        return self.model

    def __exit__(self, *exc):
        import stable_whisper
        stable_whisper.load_model = self._saved


THREE_SECTIONS = """\
[title] Test Song
[Intro]

[Verse 1]
one two three
four five six
seven eight nine

[Guitar Solo]

[Verse 2]
ten eleven twelve
thirteen fourteen fifteen
sixteen seventeen eighteen

[Guitar Fill]
[Verse 3]
nineteen twenty twentyone
twentytwo twentythree twentyfour
twentyfive twentysix twentyseven
"""
BURSTS = [(4.0, 14.0), (22.0, 32.0), (42.0, 52.0)]


# ----------------------------------------------------------------------------------------------- parsing

def test_labels_between_lines_are_all_remembered():
    lines = lc.parse_lyrics(THREE_SECTIONS)
    assert len(lines) == 9
    assert lines[0].labels == ["Intro", "Verse 1"] and lines[0].section == "Verse 1"
    assert lines[3].labels == ["Guitar Solo", "Verse 2"]
    assert lines[6].labels == ["Guitar Fill", "Verse 3"]
    assert lines[1].labels == [] and lines[1].section is None
    assert lc.find_title(THREE_SECTIONS) == "Test Song"
    assert lc._expects_break(lines[3]) and lc._expects_break(lines[0]) and not lc._expects_break(lines[1])


def test_rough_times_and_non_lyric_lines_still_parse():
    lines = lc.parse_lyrics("[Verse]\n0:12 first line\n...\n1:05 second line\n")
    assert [(l.text, l.hint) for l in lines] == [("first line", 12.0), ("second line", 65.0)]


def test_a_rest_marker_line_is_remembered_by_the_line_after_it():
    lines = lc.parse_lyrics("[Chorus]\ntime.....\n....and space...\n....\ntime.....\n----\n....and space...\n")
    assert [l.rest_before for l in lines] == [False, False, True, True]
    assert lc.parse_lyrics("...\n[V]\nfirst\n\nsecond\n")[0].rest_before is False      # a rest before the first line is the intro
    assert [l.rest_before for l in lc.parse_lyrics("first\n\nsecond\n")] == [False, False]   # a blank line is not a marker


def test_echo_lines_in_parentheses_belong_to_the_line_before_and_are_not_timed():
    lines = lc.parse_lyrics("[Chorus]\nWho is next\n(Who is next?)\n(Who is next?)\nSweet face\n[Verse 2]\nA new day\n")
    assert [l.text for l in lines] == ["Who is next", "Sweet face", "A new day"]
    assert lines[0].echoes == ["Who is next?", "Who is next?"] and lines[1].echoes == []
    assert lines[1].section is None and lines[1].labels == []      # the echo is not mistaken for a section label
    assert lines[2].section == "Verse 2"


def test_echo_words_are_aligned_with_the_line_but_not_reported_as_part_of_it():
    lines = lc.parse_lyrics("[V]\nWho is next\n(Who is next?)\nSweet face\n")
    assert lines[0].sung_text == "Who is next Who is next?" and lines[1].sung_text == "Sweet face"
    toks = "Who is next Who is next? Sweet face".split()
    words = [SimpleNamespace(word=w, start=float(i), end=i + 0.5, probability=0.9) for i, w in enumerate(toks)]
    lc.map_words_to_lines(lines, words)
    assert [w["word"] for w in lines[0].words] == ["Who", "is", "next"]        # the echo's words are dropped...
    assert lines[0].end == 2.5                                                   # ...so the line ends where it is sung
    assert [w["word"] for w in lines[1].words] == ["Sweet", "face"] and lines[1].start == 6.0


def test_parenthesised_section_names_are_still_labels_and_a_leading_echo_is_dropped():
    lines = lc.parse_lyrics("(echo before anything)\n(Chorus)\nla la\n(Guitar Solo)\nda da\n")
    assert lines[0].labels == ["Chorus"] and lines[0].echoes == []
    assert lines[1].labels == ["Guitar Solo"] and lc._expects_break(lines[1])


# ----------------------------------------------------------------------------------------------- finding the singing

def _mask(runs, total_s, hop=0.02):
    m = np.zeros(int(total_s / hop), dtype=bool)
    for s, e in runs:
        m[int(s / hop):int(e / hop)] = True
    return m


def test_find_sung_blocks_joins_breaths_splits_on_long_gaps_and_ignores_blips():
    m = _mask([(2.0, 6.0), (7.0, 12.0),      # 1 s breath: one block
               (20.0, 25.0),                 # after a long gap: a new block
               (30.0, 30.4)], 40.0)          # a 0.4 s blip: ignored
    blocks = lc.find_sung_blocks(m, 0.02)
    assert [(round(s), round(e)) for s, e in blocks] == [(2, 12), (20, 25)]


def test_find_sung_blocks_edge_cases():
    assert lc.find_sung_blocks(np.zeros(0, dtype=bool), 0.02) == []
    assert lc.find_sung_blocks(np.zeros(500, dtype=bool), 0.02) == []
    (only,) = lc.find_sung_blocks(np.ones(1000, dtype=bool), 0.02)
    assert only[0] == 0.0 and abs(only[1] - 20.0) < 0.05


def test_break_length_comes_from_the_songs_own_pauses_not_a_fixed_number():
    # the pauses measured in the song that broke a fixed 3 s rule: line pauses 3.0-3.3 s, breaks of 9.6 s and 29.8 s
    gaps = [3.0, 3.0, 9.6, 3.1, 3.1, 29.8, 3.1, 3.0, 3.1, 3.3, 3.1, 4.4, 3.1, 3.1]
    thr = lc.choose_gap_threshold(gaps)
    assert 4.4 < thr < 9.6
    assert sum(g > thr for g in gaps) == 2                       # only the two real breaks split the song
    shorter = [2.1, 2.4, 2.8, 2.6, 12.0, 2.5, 21.9, 38.2]       # the first recording: pauses 2-3 s
    assert 2.8 < lc.choose_gap_threshold(shorter) < 12.0


def test_break_length_edge_cases():
    assert lc.choose_gap_threshold([]) is None
    assert lc.choose_gap_threshold([2.8, 3.1, 3.0, 2.9]) is None            # only ordinary pauses: no break at all
    # a smooth continuum has no jump, but pauses of 5 s or more are still breaks (Daylight Monster, below)
    assert lc.choose_gap_threshold([3.0, 3.6, 4.3, 5.2, 6.3, 7.6]) == lc.BLOCK_BREAK_MIN_S
    assert lc.choose_gap_threshold([3.0, 3.6, 4.3, 4.9]) is None             # ...but nothing reaches 5 s: no break
    assert lc.choose_gap_threshold([0.3, 0.5, 0.8]) is None                  # micro-pauses are ignored
    assert 1.0 < lc.choose_gap_threshold([8.0]) < 8.0                        # a single long break splits the song
    both = lc.choose_gap_threshold([8.0, 10.0])                              # a song where every pause is a break
    assert both is not None and both < 8.0


DAYLIGHT_PAUSES = [1.1, 1.4, 1.9, 2.2, 2.6, 2.6, 3.3, 4.3, 6.3, 10.0, 10.4]   # pauses >= 1 s from "Daylight Monster"


def test_pauses_that_grow_gradually_still_split_at_five_seconds():
    # 6.3/4.3 = 1.47 and 10.0/6.3 = 1.59 are both under the 1.6 jump rule, so the jump rule alone finds nothing
    ordered = [1.0] + sorted(DAYLIGHT_PAUSES)
    assert not any(u >= lc.BLOCK_BREAK_MIN_S and u / l >= lc.BLOCK_BREAK_JUMP for l, u in zip(ordered, ordered[1:]))
    assert lc.choose_gap_threshold(DAYLIGHT_PAUSES) == lc.BLOCK_BREAK_MIN_S
    runs, t = [], 0.0
    for g in [0.0] + DAYLIGHT_PAUSES:   # one 3 s stretch of singing after each pause
        t += g
        runs.append((t, t + 3.0))
        t += 3.0
    blocks = lc.find_sung_blocks(_mask(runs, t + 1), 0.02)
    assert len(blocks) == 4   # split at the 6.3, 10.0 and 10.4 s pauses; the 4.3 s one is an ordinary pause


def test_the_jump_rule_still_wins_when_it_finds_a_break():
    gaps = [2.0, 2.4, 2.6, 3.0, 9.0]
    assert 3.0 < lc.choose_gap_threshold(gaps) < 9.0 and lc.choose_gap_threshold(gaps) != lc.BLOCK_BREAK_MIN_S


def test_line_ends_are_pulled_back_to_where_the_singing_stops():
    mask = _mask([(10.0, 14.0), (14.8, 18.0)], 40.0)   # singing 10-18 s, silence after
    line = lc.Line(0, "a b", start=10.0, end=35.0, words=[
        {"word": "a", "start": 10.0, "end": 12.0, "probability": 0.9}, {"word": "b", "start": 12.0, "end": 35.0, "probability": 0.9}])
    assert lc.trim_line_ends([line], mask, 0.02) == 1
    assert abs(line.end - 18.2) < 0.05, line.end          # last voiced frame + 0.2 s pad
    assert abs(line.words[-1]["end"] - line.end) < 1e-6 and line.words[0]["end"] == 12.0


def test_a_line_followed_by_echoes_ends_where_its_unbroken_singing_ends():
    # singing 118-124.2, a 4 s silence, then the echo at 128-133 (as in "Daylight Monster" chorus 1)
    mask = _mask([(118.0, 124.2), (128.0, 133.0)], 140.0)
    nxt = lc.Line(1, "next verse", start=143.0, end=147.0)
    for aligned_end in (121.0, 133.0):   # the aligner may end it early (held note shared with the echo) or late
        line = lc.Line(0, "Who will he see next", start=118.0, end=aligned_end, echoes=["Who will he see next?"],
                       words=[{"word": "next", "start": 120.0, "end": aligned_end, "probability": 0.5}])
        lc.trim_line_ends([line, nxt], mask, 0.02)
        assert abs(line.end - 124.4) < 0.05, (aligned_end, line.end)
        assert line.words[-1]["end"] == line.end
    plain = lc.Line(0, "Who will he see next", start=118.0, end=121.0)   # no echoes: only ever shortened
    lc.trim_line_ends([plain, nxt], mask, 0.02)
    assert plain.end == 121.0


def test_an_echo_line_never_runs_into_the_next_line():
    mask = _mask([(10.0, 13.0), (13.5, 20.0)], 30.0)   # the echo-bearing line is followed straight away by the next one
    a = lc.Line(0, "x", start=10.0, end=12.0, echoes=["x"])
    b = lc.Line(1, "y", start=13.5, end=20.0)
    lc.trim_line_ends([a, b], mask, 0.02)
    assert a.end <= 13.5 and b.start == 13.5


def test_trimming_never_reaches_into_the_next_lines_singing_or_across_a_long_silence():
    mask = _mask([(10.0, 14.0), (30.0, 34.0)], 40.0)    # a 16 s silence, then the next section
    a = lc.Line(0, "x", start=10.0, end=33.0)
    b = lc.Line(1, "y", start=30.0, end=34.0)
    lc.trim_line_ends([a, b], mask, 0.02)
    assert abs(a.end - 14.2) < 0.05, a.end                # not 33: the singing at 30+ belongs to the next line
    assert b.start == 30.0 and b.end == 34.0              # a line that already ends with the singing is left alone
    c = lc.Line(0, "z", start=10.0, end=40.0)
    lc.trim_line_ends([c], mask, 0.02)
    assert abs(c.end - 14.2) < 0.05, c.end                # no next line: still stops at the silence


def test_trimming_only_shortens_and_leaves_odd_lines_alone():
    mask = _mask([(10.0, 14.0)], 40.0)
    short = lc.Line(0, "x", start=10.0, end=12.0)
    nothing = lc.Line(1, "y", start=20.0, end=25.0)       # no singing under it at all
    unaligned = lc.Line(2, "z")
    assert lc.trim_line_ends([short, nothing, unaligned], mask, 0.02) == 0
    assert short.end == 12.0 and nothing.end == 25.0 and unaligned.end is None


def test_a_trimmed_line_is_no_longer_flagged_very_long():
    mask = _mask([(10.0, 14.0)], 60.0)
    line = lc.Line(0, "x", start=10.0, end=50.0, confidence=0.9)
    lc.flag_lines([line])
    assert any("very long" in f for f in line.flags)
    lc.trim_line_ends([line], mask, 0.02)
    lc.flag_lines([line])
    assert not any("very long" in f for f in line.flags)


def test_a_line_that_starts_in_silence_is_moved_to_where_the_sound_begins():
    mask = _mask([(10.0, 14.0), (20.0, 24.0)], 40.0)
    # the aligner stretched "we" back across the 6 s of silence between the two phrases
    words = [{"word": "we", "start": 14.0, "end": 20.0, "probability": 0.5}, {"word": "go", "start": 20.0, "end": 24.0, "probability": 0.9}]
    line = lc.Line(0, "we go", start=14.0, end=24.0, words=words)
    assert lc.snap_line_starts([line], mask, 0.02) == 1
    assert abs(line.start - 19.9) < 0.05 and line.end == 24.0, line.start
    assert line.words[0]["start"] == line.start and line.words[0]["end"] == 20.0 and line.words[1]["start"] == 20.0


def test_starts_already_on_the_sound_or_on_a_stray_tail_are_left_alone():
    mask = _mask([(10.0, 14.0), (14.3, 14.45), (20.0, 24.0)], 40.0)    # a 0.15 s blip of tail just after the first phrase
    on_sound = lc.Line(0, "a", start=11.0, end=13.0)
    after_tail = lc.Line(1, "b", start=14.3, end=24.0)                  # starts in the tail: the sustained sound is at 20
    nothing = lc.Line(2, "c", start=30.0, end=35.0)                      # no sound anywhere under it
    unaligned = lc.Line(3, "d")
    assert lc.snap_line_starts([on_sound, nothing, unaligned], mask, 0.02) == 0
    assert on_sound.start == 11.0 and nothing.start == 30.0
    assert lc.snap_line_starts([after_tail], mask, 0.02) == 1 and abs(after_tail.start - 19.9) < 0.05


def test_a_line_that_starts_in_the_tail_of_the_previous_phrase_is_moved_to_its_own_sound():
    # "Every Day the Same": the aligner put "go to work" at 24.8, inside a 0.4 s tail of "get awake"; the phrase is at 26.4
    mask = _mask([(23.6, 25.24), (26.4, 27.7)], 40.0)
    line = lc.Line(0, "go to work", start=24.82, end=27.38)
    assert lc.snap_line_starts([line], mask, 0.02) == 1 and abs(line.start - 26.3) < 0.05, line.start


def test_a_short_first_word_followed_by_a_brief_pause_is_not_skipped():
    short_then_pause = _mask([(10.0, 10.4), (10.7, 13.0)], 20.0)     # "go" ... "to work": 0.3 s breath between them
    line = lc.Line(0, "go to work", start=10.0, end=13.0)
    assert lc.snap_line_starts([line], short_then_pause, 0.02) == 0 and line.start == 10.0
    in_tail = _mask([(9.0, 10.4), (10.7, 13.0)], 20.0)               # starts in the tail, but only a 0.3 s breath follows
    tail_line = lc.Line(1, "to work", start=10.0, end=13.0)
    assert lc.snap_line_starts([tail_line], in_tail, 0.02) == 0 and tail_line.start == 10.0


def test_lines_squeezed_into_one_moment_are_reported_and_flagged():
    lines = [lc.Line(i, f"l{i}", start=s, end=e) for i, (s, e) in enumerate(
        [(10.0, 12.0), (14.0, 16.0), (20.0, 20.0), (20.0, 20.0), (20.1, 20.1), (20.1, 20.2), (30.0, 32.0)])]
    mask = _mask([(10.0, 12.0), (14.0, 16.0), (20.0, 23.0), (30.0, 32.0)], 40.0)
    (warn,) = lc.sanity_warnings(lines, mask, 0.02)
    assert warn["lines"] == [3, 6] and "squeezed" in warn["text"] and "4 lines" in warn["text"]
    assert [bool(l.flags) for l in lines] == [False, False, True, True, True, True, False]
    two = [lc.Line(0, "a", start=5.0, end=5.1), lc.Line(1, "b", start=5.2, end=5.3)]
    assert lc.sanity_warnings(two, _mask([(5.0, 5.4)], 10.0), 0.02) == []        # two close lines are not a pile-up


def test_sound_with_no_lyric_on_it_is_reported_between_lines_and_after_the_last_line():
    mask = _mask([(10.0, 14.0), (14.5, 60.0)], 80.0)
    lines = [lc.Line(0, "a", start=10.0, end=14.0), lc.Line(1, "b", start=30.0, end=33.0)]
    warns = lc.sanity_warnings(lines, mask, 0.02, blocks=[{"start": 10.0, "end": 60.5}])
    assert [w["lines"] for w in warns] == [[1, 2], [2, 2]]
    assert "16 s of singing-like sound at 0:14.0-0:30.0" in warns[0]["text"] and "after the last lyric" in warns[1]["text"]
    # silence between lines, or a short held note, is normal and stays quiet
    quiet = lc.sanity_warnings([lc.Line(0, "a", start=10.0, end=14.0), lc.Line(1, "b", start=30.0, end=33.0)],
                               _mask([(10.0, 14.0), (30.0, 33.0)], 80.0), 0.02)
    assert quiet == []
    held = lc.sanity_warnings([lc.Line(0, "a", start=10.0, end=14.0), lc.Line(1, "b", start=22.0, end=25.0)],
                              _mask([(10.0, 25.0)], 80.0), 0.02)
    assert held == []                                                            # an 8 s held note is under the limit


def test_a_later_block_without_lyrics_is_not_reported_as_singing_after_the_last_lyric():
    mask = _mask([(10.0, 20.0), (60.0, 100.0)], 120.0)                           # a reprise block with no lyrics matched
    lines = [lc.Line(0, "a", start=10.0, end=19.5)]
    blocks = [{"start": 10.0, "end": 20.0, "first_line": 1}, {"start": 60.0, "end": 100.0, "first_line": None}]
    assert lc.sanity_warnings(lines, mask, 0.02, blocks) == []


def test_warnings_appear_in_the_summary_and_the_report():
    line = lc.Line(0, "x", start=1.0, end=2.0)
    w = [{"lines": [1, 1], "text": "something looks off"}]
    assert "WARNING: something looks off" in lc.to_timed_text([line], "t", 10.0, warnings=w)
    assert "WARNING:" not in lc.to_timed_text([line], "t", 10.0)
    assert lc.report([line], w).startswith("WARNING: something looks off") and "no flags" in lc.report([line], w)
    assert lc.report([line]) == "All lines aligned with no flags."


def test_a_held_word_that_came_back_as_a_sliver_is_stretched_over_its_note():
    mask = _mask([(100.0, 112.0)], 130.0)            # one long sustained note, 100-112 s
    prev = lc.Line(0, "before", start=90.0, end=96.0, confidence=0.9)
    held = lc.Line(1, "time.....", start=105.0, end=105.0, confidence=0.002, words=[
        {"word": "time.....", "start": 105.0, "end": 105.0, "probability": 0.002}])
    nxt = lc.Line(2, "and space", start=108.0, end=111.0, confidence=0.5)
    assert lc.stretch_held_lines([prev, held, nxt], mask, 0.02) == [1]
    assert abs(held.start - 100.0) < 0.05 and abs(held.end - 108.0) < 0.05          # note start .. next line's start
    assert held.words[0]["start"] == held.start and held.words[0]["end"] == held.end
    assert prev.end == 96.0 and nxt.start == 108.0                                   # neighbours and confident lines untouched


def test_a_held_note_never_reaches_back_into_the_previous_line():
    mask = _mask([(100.0, 130.0)], 140.0)             # unbroken sound under two lines
    prev = lc.Line(0, "every day", start=100.0, end=116.8, confidence=0.9)
    held = lc.Line(1, "time", start=123.0, end=123.0, confidence=0.01)
    nxt = lc.Line(2, "and space", start=125.0, end=129.0, confidence=0.5)
    lc.stretch_held_lines([prev, held, nxt], mask, 0.02)
    assert held.start >= 116.8 - 1e-6 and abs(held.end - 125.0) < 0.05, (held.start, held.end)
    nobody = lc.Line(0, "x", start=50.0, end=50.0, confidence=0.0)                   # held-looking, but silent: nothing to stretch to
    assert lc.stretch_held_lines([nobody], mask, 0.02) == [] and (nobody.start, nobody.end) == (50.0, 50.0)


def test_the_summary_header_says_which_path_was_used():
    line = lc.Line(0, "x", start=1.0, end=2.0)
    assert "Alignment: whole song" in lc.to_timed_text([line], "t", 10.0, mode="whole song")
    assert "Alignment: singing blocks" in lc.to_timed_text([line], "t", 10.0, mode="singing blocks")
    assert "Alignment:" not in lc.to_timed_text([line], "t", 10.0)


# ----------------------------------------------------------------------------------------------- what was heard where

EV_LYRICS = """[Verse]
he walks to the park
he sees a small cat
[Chorus]
who is next tonight
sweet face black breath
[Bridge]
a mom with a stroller
tiny hands folded sleep
"""
# Real shape of "Daylight Monster": the middle block sings the chorus twice as long as its words, and a last block
# repeats the first verse (not in the lyrics at all), so block lengths alone cannot pair sections with blocks.
EV_BLOCKS = [(10.0, 40.0), (50.0, 120.0), (130.0, 140.0), (150.0, 200.0)]
EV_HEARD = ["He walks to the park. He sees a small cat. Who is next tonight? Sweet face, black breath.",
            "Who is next tonight? Who is next tonight? Who is next tonight? Sweet face black breath.",
            "A mom with a stroller. Tiny hands, folded sleep.",
            "He walks to the park. He sees a small cat."]


def test_what_was_heard_in_each_block_decides_which_lyrics_it_holds():
    lines = lc.parse_lyrics(EV_LYRICS)
    ev = lc.match_evidence(lines, EV_HEARD)
    assert ev is not None and len(ev) == 4 and len(ev[0]) == 6
    plan = lc.plan_blocks(lines, EV_BLOCKS, ev)
    assert plan[2] == [4, 5] and plan[3] == []                    # the bridge block; the reprise block gets no lyrics
    assert plan[0] + plan[1] == [0, 1, 2, 3]                      # verse and chorus come first, in order


def test_without_evidence_the_plan_is_decided_by_block_lengths_as_before():
    lines = lc.parse_lyrics(EV_LYRICS)
    assert lc.plan_blocks(lines, EV_BLOCKS) == lc.plan_blocks(lines, EV_BLOCKS, None)
    assert lc.match_evidence(lines, ["", ""]) is None                            # nothing heard: no evidence
    assert lc.match_evidence(lines, ["completely unrelated words here"]) is None


REST_LYRICS = "[Chorus]\ntime\nand space\n...\ntime\nand space\n"


def test_a_rest_the_user_wrote_is_a_good_place_to_split_between_blocks():
    # "time" is held and never heard, so the evidence ties between cutting before line 3 or before line 4; block
    # lengths favour 3+1. Without the "..." the 3+1 split wins; with it, the cut goes where the singer rests.
    blocks = [(0.0, 9.0), (20.0, 23.0)]
    heard = ["time and space", "and space"]
    plain = lc.parse_lyrics("[Chorus]\ntime\nand space\ntime\nand space\n")
    assert lc.plan_blocks(plain, blocks, lc.match_evidence(plain, heard)) == [[0, 1, 2], [3]]
    rested = lc.parse_lyrics(REST_LYRICS)
    assert lc.plan_blocks(rested, blocks, lc.match_evidence(rested, heard)) == [[0, 1], [2, 3]]


def test_similar_spellings_count_as_the_same_word():
    lines = lc.parse_lyrics("[V]\nPressed shirt clean shoes\n")
    assert lc.match_evidence(lines, ["Press shirt, clean shoe."]) == [[4]]


def test_a_model_that_cannot_transcribe_falls_back_to_block_lengths():
    said = []
    assert lc._transcribe_blocks(object(), np.zeros(16000 * 5, dtype="float32"), [(1.0, 3.0)], "en", said.append) is None
    assert any("block lengths" in m for m in said)


def test_a_slow_song_whose_line_pauses_look_like_breaks_is_not_chopped_into_lines():
    # 12 sung lines, 1.5 s each, 3.1 s apart; then a 9.6 s break; 12 more; a 29.8 s break; 12 more
    runs, t = [], 0.0
    for part in range(3):
        for _ in range(12):
            runs.append((t, t + 1.5))
            t += 1.5 + 3.1
        t += (9.6 if part == 0 else 29.8) - 3.1
    blocks = lc.find_sung_blocks(_mask(runs, t + 5), 0.02)
    assert len(blocks) == 3
    assert abs(blocks[0][0] - 0.0) < 0.1 and abs(blocks[1][0] - runs[12][0]) < 0.1 and abs(blocks[2][0] - runs[24][0]) < 0.1


def test_a_song_with_only_ordinary_pauses_is_one_block():
    runs = [(i * 4.6, i * 4.6 + 1.5) for i in range(20)]
    assert len(lc.find_sung_blocks(_mask(runs, 100.0), 0.02)) == 1


# ----------------------------------------------------------------------------------------------- matching sections to blocks

def test_plan_matches_sections_to_blocks_of_the_right_size():
    lines = lc.parse_lyrics(THREE_SECTIONS)
    plan = lc.plan_blocks(lines, [(4, 14), (22, 32), (42, 52)])
    assert plan == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_plan_follows_block_lengths_not_just_section_count():
    # a short verse then a long one: the long block must get the longer section
    text = "[Verse 1]\n" + "\n".join("a b c" for _ in range(2)) + "\n[Solo]\n[Verse 2]\n" + "\n".join("d e f" for _ in range(6))
    lines = lc.parse_lyrics(text)
    plan = lc.plan_blocks(lines, [(0, 10), (30, 60)])
    assert plan == [[0, 1], [2, 3, 4, 5, 6, 7]]


def test_plan_lets_a_block_without_lyrics_be_skipped():
    lines = lc.parse_lyrics(THREE_SECTIONS)
    blocks = [(4, 14), (16, 18.5), (22, 32), (42, 52)]   # a 2.5 s ad-lib between verse 1 and 2
    plan = lc.plan_blocks(lines, blocks)
    assert plan[1] == [] and plan[0] == [0, 1, 2] and plan[2] == [3, 4, 5] and plan[3] == [6, 7, 8]


def test_plan_trivial_cases():
    lines = lc.parse_lyrics(THREE_SECTIONS)
    assert lc.plan_blocks(lines, []) == []
    assert lc.plan_blocks(lines, [(0, 30)]) == [list(range(9))]
    many = lc.plan_blocks(lines[:2], [(0, 3), (5, 8), (10, 13), (15, 18)])
    assert sorted(i for g in many for i in g) == [0, 1] and len(many) == 4  # every line placed exactly once


def test_plan_uses_the_users_rough_times_when_given():
    # two identical verses and identical blocks: only the rough times can tell the right order from a swap... they
    # must keep the lines in order and agree with the hints
    text = "[Verse 1]\n0:05 a b c\n0:08 d e f\n[Verse 2]\n0:25 g h i\n0:28 j k l\n"
    lines = lc.parse_lyrics(text)
    assert lc.plan_blocks(lines, [(4, 12), (24, 32)]) == [[0, 1], [2, 3]]


def test_fit_is_near_one_for_a_good_match_and_far_from_it_for_a_bad_one():
    lines = lc.parse_lyrics(THREE_SECTIONS)
    blocks = [(4, 14), (22, 32), (42, 52)]
    plan = lc.plan_blocks(lines, blocks)
    assert all(abs(f - 1.0) < 0.05 for f in lc.block_fit(lines, blocks, plan))
    lopsided = lc.block_fit(lines, [(0, 5), (10, 15), (20, 80)], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert lopsided[2] > 2.0 and lopsided[0] < 0.5
    assert lc.block_fit(lines, blocks, [[0, 1, 2], [], [3, 4, 5]])[1] is None


def test_the_summary_warns_when_lyrics_and_singing_look_mismatched():
    lines = lc.parse_lyrics("[V]\na b c\nd e f")
    lc.estimate_block(lines, 0.0, 10.0)
    odd = [{"start": 0.0, "end": 60.0, "first_line": 1, "last_line": 2, "fit": 6.0}]
    fine = [{"start": 0.0, "end": 10.0, "first_line": 1, "last_line": 2, "fit": 1.0}]
    assert "much longer than those lines should take" in lc.to_timed_text(lines, "t", 60.0, blocks=odd)
    assert "Check lines" not in lc.to_timed_text(lines, "t", 60.0, blocks=fine)
    assert "Singing found at" not in lc.to_timed_text(lines, "t", 60.0)


def test_sung_words_with_modest_confidence_are_not_flagged():
    lines = lc.parse_lyrics("[V]\na b c\nd e f")
    for l, conf in zip(lines, (0.4, 0.2)):
        l.start, l.end, l.confidence = 1.0 + l.index * 3, 3.0 + l.index * 3, conf
    lc.flag_lines(lines)
    assert lines[0].flags == [] and any("low confidence" in f for f in lines[1].flags)


def test_estimate_block_spreads_lines_in_order_inside_the_block():
    lines = lc.parse_lyrics("[V]\none two\nthree four five six\nseven")
    lc.estimate_block(lines, 10.0, 30.0)
    assert lines[0].start == 10.0 and abs(lines[-1].end - 30.0) < 0.01
    assert all(a.end <= b.start + 1e-6 for a, b in zip(lines, lines[1:]))
    assert all(len(l.words) == len(l.text.split()) for l in lines)
    assert (lines[1].end - lines[1].start) > (lines[0].end - lines[0].start)   # more words, more time


# ----------------------------------------------------------------------------------------------- end to end (stand-in model)

def test_each_section_is_aligned_inside_its_own_block():
    with tempfile.TemporaryDirectory() as d:
        wav = _song(BURSTS, 56.0, Path(d) / "voc.wav")
        lines = lc.parse_lyrics(THREE_SECTIONS)
        with patched_model(FakeModel()) as model:
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info["mode"] == "singing blocks" and len(info["blocks"]) == 3
        assert [(b["first_line"], b["last_line"]) for b in info["blocks"]] == [(1, 3), (4, 6), (7, 9)]
        for line, (s, e) in zip(lines, [BURSTS[0]] * 3 + [BURSTS[1]] * 3 + [BURSTS[2]] * 3):
            assert s - 1 <= line.start and line.end <= e + 1, (line.text, line.start, line.end)
        assert [l.start for l in lines] == sorted(l.start for l in lines)
        assert not any(lc.ESTIMATE_FLAG in l.flags for l in lines)
        assert all(c[0] == "slice" for c in model.calls[:3]), "each block is aligned on its own slice of audio"


class ListeningModel(FakeModel):
    """FakeModel that can also transcribe: returns the given texts, one per call (blocks are listened to in order)."""

    def __init__(self, texts):
        super().__init__()
        self._texts = list(texts)

    def transcribe(self, audio, **kw):
        return SimpleNamespace(text=self._texts.pop(0))


def test_the_singing_is_listened_to_even_when_the_breaks_are_clear():
    """The failure from "Time and Space": clear breaks used to skip listening, and block lengths alone paired the
    lyrics with the wrong stretches of singing. What was heard now decides, however clear the breaks look."""
    lyr = [f"w{3 * i + 1} w{3 * i + 2} w{3 * i + 3}" for i in range(12)]
    heard = [" ".join(lyr[0:6]), " ".join(lyr[6:9]), " ".join(lyr[9:12])]     # 6 + 3 + 3 lines, in three equal blocks
    with tempfile.TemporaryDirectory() as d:
        wav = _song(BURSTS, 56.0, Path(d) / "voc.wav")
        text = "[Song]\n" + "\n".join(lyr) + "\n"
        lines = lc.parse_lyrics(text)
        with patched_model(ListeningModel(heard)):
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info["evidence"] == "transcript"
        assert [(b["first_line"], b["last_line"]) for b in info["blocks"]] == [(1, 6), (7, 9), (10, 12)]   # lengths alone say 4+4+4
        with patched_model(FakeModel()):   # a model that cannot listen still falls back to lengths
            fallback = lc.align_vocals(wav, lc.parse_lyrics(text), model_name="x", device="cpu")
        assert [(b["first_line"], b["last_line"]) for b in fallback["blocks"]] == [(1, 4), (5, 8), (9, 12)]


def test_a_block_the_aligner_cannot_place_is_spread_over_the_singing_and_flagged():
    with tempfile.TemporaryDirectory() as d:
        wav = _song(BURSTS, 56.0, Path(d) / "voc.wav")
        lines = lc.parse_lyrics(THREE_SECTIONS)
        with patched_model(FakeModel(fail=True)):
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info["mode"] == "singing blocks"
        assert all(lc.ESTIMATE_FLAG in l.flags for l in lines)
        for line, (s, e) in zip(lines, [BURSTS[0]] * 3 + [BURSTS[1]] * 3 + [BURSTS[2]] * 3):
            assert s - 0.5 <= line.start < line.end <= e + 0.5, (line.text, line.start, line.end)   # never a pile-up
        assert len({round(l.start, 1) for l in lines}) == 9


def test_slow_verses_with_three_second_line_pauses_still_pair_with_the_right_section():
    """The failure from the second recording: each verse is 4 lines sung 1.5 s apart by 3.1 s pauses; the solos are
    9.6 s and 29.8 s. A fixed 3 s rule chopped each verse into line-sized pieces and mis-assigned them."""
    verse = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    text = ("[Verse 1]\n" + "\n".join(verse) + "\n[Guitar Fill]\n[Verse 2]\n" + "\n".join(verse) +
            "\n[Guitar Solo]\n[Verse 3]\n" + "\n".join(verse) + "\n")
    bursts, t = [], 4.0
    for part in range(3):
        for _ in range(4):
            bursts.append((t, t + 1.5))
            t += 1.5 + 3.1
        t += (9.6 if part == 0 else 29.8) - 3.1
    with tempfile.TemporaryDirectory() as d:
        wav = _song(bursts, t + 4, Path(d) / "voc.wav")
        lines = lc.parse_lyrics(text)
        with patched_model(FakeModel()):
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info["mode"] == "singing blocks" and len(info["blocks"]) == 3
        assert [(b["first_line"], b["last_line"]) for b in info["blocks"]] == [(1, 4), (5, 8), (9, 12)]
        assert lines[8].start > bursts[7][1] + 25   # Verse 3 begins after the long solo, not before it
        assert lines[4].start > bursts[3][1] + 5    # ...and Verse 2 after the fill
        assert [l.start for l in lines] == sorted(l.start for l in lines)
        assert info["break_length_s"] is not None and 3.2 < info["break_length_s"] < 9.6


def test_a_blocked_lameenc_does_not_stop_audio_from_being_read():
    """Windows Smart App Control can refuse lameenc's unsigned DLL; demucs only imports it for an MP3 writer we never use."""
    import types

    import core

    saved = sys.modules.get("lameenc", "absent")
    try:
        sys.modules["lameenc"] = None                         # makes "import lameenc" fail, like a blocked DLL
        core._tolerate_blocked_lameenc()
        stand_in = sys.modules["lameenc"]
        assert stand_in is not None and hasattr(stand_in, "Encoder")
        try:
            stand_in.Encoder()
            raise AssertionError("the stand-in encoder must refuse to encode")
        except RuntimeError as exc:
            assert "unavailable" in str(exc) and "ffmpeg" in str(exc)
        working = types.ModuleType("lameenc")                 # a lameenc that loads fine is left alone
        sys.modules["lameenc"] = working
        core._tolerate_blocked_lameenc()
        assert sys.modules["lameenc"] is working
    finally:
        if saved == "absent":
            sys.modules.pop("lameenc", None)
        else:
            sys.modules["lameenc"] = saved


def test_times_never_read_sixty_seconds():
    assert lc._clock(59.97) == "1:00.0" and lc._clock(179.96) == "3:00.0" and lc._clock(75.34) == "1:15.3"
    assert lc._clock(0.0) == "0:00.0" and lc._clock(-1) == "0:00.0" and lc._clock(125.0, tenths=False) == "2:05"
    assert lc._lrc_stamp(59.999) == "01:00.00" and lc._lrc_stamp(61.234) == "01:01.23" and lc._lrc_stamp(0) == "00:00.00"
    line = lc.Line(0, "x", start=119.97, end=122.0)
    assert "2:00.0 - 2:02.0" in lc.to_timed_text([line], "t", 130.0)


def test_a_song_with_no_gaps_is_aligned_in_one_pass_like_before():
    with tempfile.TemporaryDirectory() as d:
        wav = _song([(2.0, 30.0)], 32.0, Path(d) / "voc.wav")
        lines = lc.parse_lyrics("[V]\n" + "\n".join("la la la" for _ in range(4)))
        with patched_model(FakeModel()) as model:
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info["mode"] == "whole song" and info["blocks"] == [] and info["break_length_s"] is None
        assert model.calls[0][0] == "path"   # the whole file, exactly as the original code did


def test_run_writes_all_three_files_and_reports_where_the_singing_is():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        wav = _song(BURSTS, 56.0, d / "song.wav")
        with patched_model(FakeModel()):
            res = lyricsync.run(wav, THREE_SECTIONS, d / "out", model="x", device="cpu", vocals=wav)
        timed = Path(res["files"]["timed"]).read_text(encoding="utf-8")
        assert "Singing found at: 0:04.0-0:14.0 (lines 1-3)" in timed and "(lines 7-9)" in timed
        meta = json.loads(Path(res["files"]["json"]).read_text(encoding="utf-8"))
        assert meta["meta"]["alignment_mode"] == "singing blocks" and len(meta["meta"]["singing_blocks"]) == 3
        assert meta["lines"][3]["labels"] == ["Guitar Solo", "Verse 2"]
        assert Path(res["files"]["lrc"]).read_text(encoding="utf-8").count("[") >= 9


# ----------------------------------------------------------------------------------------------- runner

def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

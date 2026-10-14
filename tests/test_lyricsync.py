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


def test_a_song_with_no_gaps_is_aligned_in_one_pass_like_before():
    with tempfile.TemporaryDirectory() as d:
        wav = _song([(2.0, 30.0)], 32.0, Path(d) / "voc.wav")
        lines = lc.parse_lyrics("[V]\n" + "\n".join("la la la" for _ in range(4)))
        with patched_model(FakeModel()) as model:
            info = lc.align_vocals(wav, lines, model_name="x", device="cpu")
        assert info == {"mode": "whole song", "blocks": []}
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

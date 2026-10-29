"""Score lyric-timing results against reference answers, so "the tool improved" is a number and not an opinion.

  .venv\\Scripts\\python.exe tests\\score_lyricsync.py                      score every reference in test_data\\reference\\
  .venv\\Scripts\\python.exe tests\\score_lyricsync.py --run                re-run the tool on each song first (slow: needs the GPU/CPU)
  .venv\\Scripts\\python.exe tests\\score_lyricsync.py --result X_timing.json --reference test_data\\reference\\X.json

A reference is a small JSON file kept in test_data\\reference\\ (git-ignored: it holds lyrics and song paths):

  {"song": "...", "audio": "C:\\\\...\\\\song.mp3", "model": "large-v3", "lyrics_file": "song_lyrics.txt",   # relative to this file
   "notes": "how these times were established",
   "lines": [{"n": 1, "text": "first line", "start": 23.6, "tol": 0.5},    # n = line number among the sung lines
             {"n": 30, "text": "not sung line", "skip": true}]}           # skip = no reference time for this line

`tol` is how far (seconds) the tool may be from the reference and still count as right (default 1.0). A result is scored line by
line on its start time, matching by line number, so the result must have been made from the same lyrics (same lines, same order).
Without --run it scores the existing <song>_timing.json found next to the audio's stem_output, or the file given with --result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REFERENCE_DIR = ROOT / "test_data" / "reference"


def clock(t: float) -> str:
    m, s = divmod(round(t * 10) / 10, 60)
    return f"{int(m)}:{s:04.1f}"


def score(reference: dict, result: dict) -> dict:
    """Compare one result (the tool's _timing.json content) with one reference. Returns the numbers."""
    got = {i + 1: l for i, l in enumerate(result["lines"])}
    ref_lines = [l for l in reference["lines"] if not l.get("skip")]
    if len(result["lines"]) != len(reference["lines"]):
        raise SystemExit(f"{reference['song']}: the result has {len(result['lines'])} lines but the reference has "
                         f"{len(reference['lines'])}; they must come from the same lyrics.")
    errors, rows = [], []
    for r in ref_lines:
        line = got[r["n"]]
        if line.get("start") is None:
            err = float("inf")
        else:
            err = abs(line["start"] - r["start"])
        errors.append(err)
        rows.append((err, r["n"], r["text"], line.get("start"), r["start"], r.get("tol", 1.0)))
    finite = [e for e in errors if e != float("inf")]
    return {
        "song": reference["song"], "scored": len(errors),
        "median": statistics.median(finite) if finite else None,
        "mean": statistics.mean(finite) if finite else None,
        "within_tol": sum(1 for row in rows if row[0] <= row[5]),
        "over_1s": sum(1 for e in errors if e > 1.0), "over_3s": sum(1 for e in errors if e > 3.0),
        "unplaced": len(errors) - len(finite),
        "max": max(errors) if errors else None, "worst": sorted(rows, reverse=True)[:8],
        "warnings": len(result.get("meta", {}).get("warnings", [])),
        "flagged": sum(1 for l in result["lines"] if l.get("flags")),
    }


def show(s: dict) -> None:
    print(f"\n=== {s['song']}  ({s['scored']} lines with a reference time)")
    print(f"  within tolerance: {s['within_tol']}/{s['scored']}   median error: {s['median']:.2f} s   mean: {s['mean']:.2f} s   "
          f"max: {s['max']:.1f} s")
    print(f"  more than 1 s off: {s['over_1s']}   more than 3 s off: {s['over_3s']}   not placed at all: {s['unplaced']}")
    print(f"  the tool itself raised {s['warnings']} warning(s) and flagged {s['flagged']} line(s)")
    print("  worst lines:   n  error   tool -> reference")
    for err, n, text, got, want, tol in s["worst"]:
        if err <= tol:
            break
        got_s = "none" if got is None else clock(got)
        print(f"     {n:3d}  {err:5.1f}s  {got_s} -> {clock(want)}   {text[:40]}")


def run_tool(reference: dict, ref_path: Path, out_dir: Path) -> dict:
    import lyricsync
    lyrics = (ref_path.parent / reference["lyrics_file"]).read_text(encoding="utf-8-sig")
    res = lyricsync.run(Path(reference["audio"]), lyrics, out_dir, model=reference.get("model", "large-v3"), device="auto",
                        progress=lambda frac, msg: None)
    return json.loads(Path(res["files"]["json"]).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, action="append", help="a reference file (default: all in test_data/reference)")
    ap.add_argument("--result", type=Path, help="score this _timing.json (needs exactly one --reference)")
    ap.add_argument("--run", action="store_true", help="re-run the tool on each reference song and score the new result")
    a = ap.parse_args()
    refs = a.reference or sorted(REFERENCE_DIR.glob("*.json"))
    if not refs:
        print(f"No references found in {REFERENCE_DIR}.")
        return 1
    if a.result and len(refs) != 1:
        print("--result needs exactly one --reference.")
        return 1
    for ref_path in refs:
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        if a.result:
            result = json.loads(a.result.read_text(encoding="utf-8"))
        elif a.run:
            with tempfile.TemporaryDirectory() as d:
                result = run_tool(reference, ref_path, Path(d))
        else:
            existing = Path(reference["audio"]).parent / "stem_output" / f"{Path(reference['audio']).stem}_timing.json"
            if not existing.is_file():
                print(f"\n=== {reference['song']}: no existing result at {existing}; use --run or --result.")
                continue
            result = json.loads(existing.read_text(encoding="utf-8"))
        show(score(reference, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

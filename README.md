# Music File Instrument Remover

A local Windows desktop tool that removes drums (and eventually other stems) from audio files using [Facebook Research's Demucs](https://github.com/facebookresearch/demucs) model.

## Primary Use Case

Strip drum tracks from home demo recordings before uploading to Suno AI. Suno generates its own drum track during production, and certain drum loop patterns (e.g. from hardware loopers) can trigger its copyright scanner. This tool removes the drums cleanly using GPU-accelerated AI stem separation.

## Features

- **GUI** — simple tkinter interface, no command-line required for day-to-day use
- **CLI** — full command-line interface for scripting and automation
- **GPU-accelerated** — uses CUDA by default (NVIDIA GPU required for best performance)
- **High-quality separation** — uses Demucs `htdemucs_ft` model by default
- **Overwrite protection** — warns before overwriting existing output files
- **Persistent settings** — model, vocal separation, device, and default directories saved between sessions
- **Optional RoFormer vocals** — cleaner vocal stems for Remove Vocals, Full Separation and Lyric Timing (see below)
- **Hover help** — rest the mouse on a setting or mode for a short explanation

## Output

Two WAV files are produced per run:
- `{filename}_no_drums.wav` — everything except drums
- `{filename}_drums.wav` — isolated drum track

## Requirements

- Windows 10/11
- Python 3.13+
- NVIDIA GPU with CUDA support (recommended; CPU fallback available)
- ffmpeg (required for MP3 input)

## Installation

### 1. Install ffmpeg
Download and install from https://ffmpeg.org or via winget:
```
winget install --id Gyan.FFmpeg -e
```

### 2. Create a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install remaining dependencies
```powershell
pip install -r requirements.txt
```

### 5. (Optional) Install RoFormer for cleaner vocals
Only needed for **Vocal separation: RoFormer** (see below). The constraints file is essential: without it pip
replaces the GPU build of PyTorch with a CPU-only one, and everything quietly stops using the GPU.
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-roformer.txt -c constraints-roformer.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
Then check the GPU is still in use (should print `2.11.0+cu128 True`):
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
The model itself (`vocals_mel_band_roformer.ckpt`, about 900 MB) is kept in the `models\` folder. If it is missing
it is downloaded automatically the first time RoFormer is used.

### Rebuilding an exact environment
`venv_snapshots\` holds the exact package lists (`pip freeze`) of the working environment:
`pip_freeze_2026-11-24_after_roformer.txt` (current, with RoFormer) and `..._before_roformer.txt` (the setup before it).
To rebuild one in a fresh venv (Python 3.13):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r venv_snapshots\pip_freeze_2026-11-24_after_roformer.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
Then check the GPU as in step 5. The RoFormer model is downloaded again on first use (or copy `models\` across).

## Vocal separation: Demucs or RoFormer

**Advanced tab → Vocal separation** (CLI: `--vocals demucs|roformer`) chooses what separates the vocals:

| Choice | What it does |
|---|---|
| **Demucs (current)** | The default. The Demucs model does everything, as before. |
| **RoFormer (cleaner vocals)** | Mel-Band RoFormer separates the vocals. About 20 s per song on the GPU (about 2.5 min on the CPU), using about 4 GB of GPU memory. Between sung lines the vocal track is much cleaner (about 10 dB less leftover music in the song tested). |

It is used by:
- **Remove Vocals**: gives `<song>_vocals` and `<song>_no_vocals` (the instrumental).
- **Full Separation (4 stems)**: with RoFormer, RoFormer makes the vocals and the Demucs model then splits RoFormer's
  instrumental into drums, bass and other. That keeps the singing out of the other stems. Anything Demucs still
  calls "vocals" in the instrumental (usually a lead instrument) goes into `other`, so nothing is lost. About 35 s per song on the GPU.
- **Lyric Timing**: the vocal track it lines the lyrics up with. Each song's vocals are cached once per method.

**Removing drums, bass or other is not affected**: it always uses the Demucs model, and its output is unchanged.

The Main and Lyric Timing tabs show which method will be used; hover over any of these controls for details. If
RoFormer is selected but not installed, those tabs show a red warning, and a run stops with a plain message saying so.

The RoFormer weights (by Kimberley Jensen) are licensed **CC BY-NC-SA 4.0: non-commercial use only**.

## Usage

### GUI
```powershell
.\.venv\Scripts\python.exe gui.py
```

### CLI
```powershell
.\.venv\Scripts\python.exe drumremover.py --input "C:\Music\demo.mp3"
.\.venv\Scripts\python.exe drumremover.py --input "C:\Music\demo.mp3" --output "C:\Music\Output"
.\.venv\Scripts\python.exe drumremover.py --input "C:\Music\demo.mp3" --model htdemucs --device cpu
.\.venv\Scripts\python.exe drumremover.py --input "C:\Music\demo.mp3" --stem vocals --vocals roformer
.\.venv\Scripts\python.exe drumremover.py --input "C:\Music\demo.mp3" --stem all --vocals roformer
.\.venv\Scripts\python.exe drumremover.py --list-models
```
`--list-models` also lists the vocal separation choices. Without `--vocals`, the Advanced tab's saved setting is used.

## Project Structure

```
music-file-instrument-remover/
├── drumremover.py      # CLI entry point
├── gui.py              # GUI (tkinter wrapper around core.py)
├── core.py             # All separation logic, Demucs and RoFormer (shared by CLI and GUI)
├── config.py           # Settings persistence
├── tooltip.py          # Hover help for GUI controls
├── lyricsync.py        # Lyric timing CLI (lyricsync_core.py, lyricsync_tab.py = logic, GUI tab)
├── requirements.txt    # Python dependencies (excluding PyTorch)
├── requirements-roformer.txt / constraints-roformer.txt   # optional RoFormer install (keeps GPU PyTorch)
├── models/             # RoFormer model file (git-ignored, ~900 MB)
├── CLAUDE.md           # Project brief and architecture notes
└── logs/               # Log files (git-ignored)
```

## Backlog

Planned features (not yet implemented):

- Desktop icon / shortcut launcher
- Batch processing (multiple files or entire folder)
- Drag and drop support
- Waveform preview

## License

MIT

## Lyric timing (`lyricsync.py`)

Timestamps a lyrics text file against a song (e.g. for planning video/animation cuts):

```
python lyricsync.py "song.mp3" lyrics.txt
python lyricsync.py "song.mp3" lyrics.txt --words          # per-word tags in the .lrc
python lyricsync.py "song.mp3" lyrics.txt --vocals "song_vocals.wav"   # reuse an existing vocal stem
python lyricsync.py "song.mp3" lyrics.txt --vocal-method roformer       # separate the vocals with RoFormer
```

It separates the vocals with Demucs (or RoFormer: `--vocal-method roformer`, or the Advanced tab's Vocal separation
setting in the GUI), then force-aligns the lyrics to them with stable-ts (Whisper
`medium` by default; `--model large-v3` for harder songs). Lyrics format: one sung line per line;
`[Section]` labels, a `[title] ...` line and lines with no letters are skipped (so are echo lines written
in parentheses, such as `(Who will he see next?)` - they are not timed). A line of only punctuation such as `...`
or `----` is a rest marker: it is not timed either, but it tells the tool the singer pauses there, so a stretch of
singing may end at that point (see below); an optional leading
`m:ss` on a line is treated as a rough time and used as a cross-check. Outputs, next to the song
(or in `--output-dir`): `<song>_timed.txt` (paste into Claude chat), `<song>.lrc`, and
`<song>_timing.json` (always includes per-word times and confidence). Lines the aligner gave up on
are retried on the audio between their neighbours; anything still doubtful is marked `!`.
All logic is in `lyricsync_core.py`.

**How the timing is found.** The tool first finds where the singing actually is: stretches of the vocal track
separated by instrumental breaks (a silent intro, a fill, a guitar solo). What counts as a break is measured from
the song itself: the pauses between sung lines are all about the same length (2-3.3 s in the songs tried so far) and
breaks are much longer, so the cut-off is placed where the pauses jump in length (a fixed 3 s rule chopped a slow
song's verses into line-sized pieces). It then matches your lyric lines to
those stretches - mainly by which lyric words were actually heard in each stretch (see below), with the stretch's length
compared with how many words the lines have as a tie-breaker, preferring to split
where a new `[Section]` starts (especially after a label such as `[Guitar Solo]`, `[Guitar Fill]` or `[Intro]`,
which signal a break) or after a rest marker (`...`), and where your own rough times agree - and aligns each group of lines inside its own stretch, so the
speech model never has to cross a long instrumental part. The `_timed.txt` header lists where the singing was found
and which lines went with it, and warns when a stretch is much longer or shorter than its lines should take (a sign
the pairing may be wrong). A stretch the aligner cannot place is spread over the singing by word count and flagged
`estimated`. Songs with no clear gaps are aligned in one pass, as before. Sung words score low on Whisper's confidence
even when placed correctly, so a line is only marked `!` below 0.3.
When the pauses grow gradually instead of jumping (no clear break, e.g. 4.3, 6.3, 10.0, 10.4 s), every pause of
5 s or more is still treated as a break. Block lengths are only a weak guide to which lines are where (a chorus
chanted longer than its words, a reprise that is not in the lyrics, held notes and rests that make the pace very
uneven), and even clear-looking breaks can be paired wrongly by length alone, so the tool always listens to each
stretch first (a plain Whisper transcript, about half a minute on the GPU) and pairs lines with the stretches by which
lyric words were actually heard in each; a stretch that holds none of the lyrics gets none. If the model cannot
transcribe, it falls back to block lengths. After aligning, three corrections use the vocal track's loudness: each
line's end is pulled back to where the singing under it really stops (alignment can stretch an end across a long
pause); a line whose first word was stretched back across silence has its start moved to where the sound begins; and
a held word the aligner collapsed to a sliver (`time.....` sung for four seconds) is stretched over its note, never
into the neighbouring lines, and marked `held note`. (A line's start is only moved when the sound under it is
just the tail of the previous phrase and the next real start of singing follows at least 0.6 s of silence, so a short
first word followed by a brief breath is left alone.) The header says which path was used
(`Alignment: singing blocks` or `whole song`).
**Warnings.** So a result never looks more certain than it is, the tool checks its own output and prints `WARNING:` lines
in the `_timed.txt` header, in the GUI result box (and the status line), and in `<song>_timing.json` under
`meta.warnings` (each with the line numbers involved, ready for a future "fix these lines" step). It warns when three or
more lines in a row are squeezed into the same moment (the aligner gave up on them; those lines also get a `!`), when
12 s or more of singing-like sound sits between two lines with no lyric on it, and when singing continues 12 s or more
after the last lyric within its stretch. A warning usually means: add rough times (`m:ss` in front of a few lines)
and re-run, or check whether lyrics are missing or a line is not actually sung.
Echo lines written in parentheses, such as `(Who will he see next?)`, are not timed: they are sung, so they are given
to the aligner with the line before them (so a held last note is not shared out wrongly), but they get no line of
their own in the output.
Tests (no pytest or model needed): `.venv\Scripts\python.exe tests\test_lyricsync.py`.
Scoring against known-good timings: `.venv\Scripts\python.exe tests\score_lyricsync.py` (add `--run` to re-run the tool
on each song first, and `--vocal-method roformer` to make those runs use RoFormer vocals). It compares each line's start with a reference answer kept in `test_data\reference\` (git-ignored,
because it holds lyrics and song paths; the format is described at the top of the script) and prints how many lines are
within tolerance, the median and worst errors, and the worst lines. Add a reference for any song whose timing you have
corrected by hand, and a change to the tool is judged by whether these numbers improve without any song getting worse.

The GUI has the same thing as the **Lyric Timing** tab (paste lyrics, pick a song, copy the result).
Both use `lyricsync.run()`. Vocal stems are cached in `lyricsync_cache/` (git-ignored), one per song and vocal separation method, so a
re-run of the same song skips the separation step. After the first alignment pass, any line that failed, sits where the
vocal track is silent, or is far from its rough time is re-aligned on the audio between the
nearest good lines; a retry is kept only if it leaves fewer bad lines.

**Processor choice.** `--device auto` (the default) uses the GPU only when `nvidia-smi` reports enough
free memory for the job (about 4 GB for Demucs, 3.5 GB for Whisper `medium`, 6.5 GB for `large-v3`),
otherwise the CPU, so it will not fight a running animation job. `--device gpu|cpu` forces it. If
"auto" picks the GPU and it still runs out of memory, the run is retried on the CPU. The Lyric Timing
tab has the same Auto / GPU / CPU choice, shows the GPU's free memory (refreshed every 10 s), and asks
before running on the GPU when it is short of memory. This tab's choice is separate from the Advanced
tab's device setting, which still controls the Main tab.

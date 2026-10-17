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
- **Persistent settings** — model, device, and default directories saved between sessions

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
.\.venv\Scripts\python.exe drumremover.py --list-models
```

## Project Structure

```
music-file-instrument-remover/
├── drumremover.py      # CLI entry point
├── gui.py              # GUI (tkinter wrapper around core.py)
├── core.py             # All Demucs processing logic (shared by CLI and GUI)
├── config.py           # Settings persistence
├── requirements.txt    # Python dependencies (excluding PyTorch)
├── CLAUDE.md           # Project brief and architecture notes
└── logs/               # Log files (git-ignored)
```

## Backlog

Planned features (not yet implemented):

- Desktop icon / shortcut launcher
- Batch processing (multiple files or entire folder)
- Full 4-stem separation (drums, bass, vocals, other)
- Vocal removal
- Bass / guitar isolation
- Output format selector (WAV vs MP3)
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
```

It separates the vocals with Demucs, then force-aligns the lyrics to them with stable-ts (Whisper
`medium` by default; `--model large-v3` for harder songs). Lyrics format: one sung line per line;
`[Section]` labels, a `[title] ...` line and lines with no letters are skipped (so are echo lines written
in parentheses, such as `(Who will he see next?)` - they are not timed); an optional leading
`m:ss` on a line is treated as a rough time and used as a cross-check. Outputs, next to the song
(or in `--output-dir`): `<song>_timed.txt` (paste into Claude chat), `<song>.lrc`, and
`<song>_timing.json` (always includes per-word times and confidence). Lines the aligner gave up on
are retried on the audio between their neighbours; anything still doubtful is marked `!`.
All logic is in `lyricsync_core.py`.

**How the timing is found.** The tool first finds where the singing actually is: stretches of the vocal track
separated by instrumental breaks (a silent intro, a fill, a guitar solo). What counts as a break is measured from
the song itself: the pauses between sung lines are all about the same length (2-3.3 s in the songs tried so far) and
breaks are much longer, so the cut-off is placed where the pauses jump in length (a fixed 3 s rule chopped a slow
song's verses into line-sized pieces). It then matches your lyric sections to
those stretches - by how long each stretch is compared with how many words the section has, preferring to split
where a new `[Section]` starts (especially after a label such as `[Guitar Solo]`, `[Guitar Fill]` or `[Intro]`,
which signal a break) and where your own rough times agree - and aligns each section inside its own stretch, so the
speech model never has to cross a long instrumental part. The `_timed.txt` header lists where the singing was found
and which lines went with it, and warns when a stretch is much longer or shorter than its lines should take (a sign
the pairing may be wrong). A stretch the aligner cannot place is spread over the singing by word count and flagged
`estimated`. Songs with no clear gaps are aligned in one pass, as before. Sung words score low on Whisper's confidence
even when placed correctly, so a line is only marked `!` below 0.3.
When the pauses grow gradually instead of jumping (no clear break, e.g. 4.3, 6.3, 10.0, 10.4 s), every pause of
5 s or more is still treated as a break. Because block lengths are a weak guide in that case (a chorus chanted longer
than its words, a reprise that is not in the lyrics), the tool first listens to each stretch (a plain Whisper
transcript) and pairs sections with the stretches by which lyric words were actually heard in each; a stretch that
holds none of the lyrics gets none. After aligning, each line's end is pulled back to where the singing under it
really stops (alignment can stretch an end across a long pause). The header says which path was used
(`Alignment: singing blocks` or `whole song`).
Echo lines written in parentheses, such as `(Who will he see next?)`, are not timed: they are sung, so they are given
to the aligner with the line before them (so a held last note is not shared out wrongly), but they get no line of
their own in the output.
Tests (no pytest or model needed): `.venv\Scripts\python.exe tests\test_lyricsync.py`.

The GUI has the same thing as the **Lyric Timing** tab (paste lyrics, pick a song, copy the result).
Both use `lyricsync.run()`. Vocal stems are cached in `lyricsync_cache/` (git-ignored) so a re-run
of the same song skips Demucs. After the first alignment pass, any line that failed, sits where the
vocal track is silent, or is far from its rough time is re-aligned on the audio between the
nearest good lines; a retry is kept only if it leaves fewer bad lines.

**Processor choice.** `--device auto` (the default) uses the GPU only when `nvidia-smi` reports enough
free memory for the job (about 4 GB for Demucs, 3.5 GB for Whisper `medium`, 6.5 GB for `large-v3`),
otherwise the CPU, so it will not fight a running animation job. `--device gpu|cpu` forces it. If
"auto" picks the GPU and it still runs out of memory, the run is retried on the CPU. The Lyric Timing
tab has the same Auto / GPU / CPU choice, shows the GPU's free memory (refreshed every 10 s), and asks
before running on the GPU when it is short of memory. This tab's choice is separate from the Advanced
tab's device setting, which still controls the Main tab.

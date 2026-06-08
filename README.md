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

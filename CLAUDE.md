# Drum Remover — Project Briefing

## Project Overview

A local Windows desktop tool that removes drums (and potentially other stems) from audio files using Facebook Research's Demucs model. The primary use case is stripping drum tracks from demo recordings before uploading to Suno AI, which flags certain drum loop patterns as potentially copyrighted.

## Background & Context

The user is a songwriter who records multi-track demos at home using a Focusrite USB audio interface and Audacity. Demos are recorded in MP3 or WAV format. A Line 6 looper hardware unit is used to provide a drum loop during recording — this loop is sometimes flagged by Suno AI's copyright scanner, blocking uploads. The solution is to remove the drums before uploading to Suno, which generates its own drum track during production anyway.

The user has a CS background (BS Computer Science, Penn State 1984) and prior software development experience, now in a non-coding role. He is comfortable with technical tools but prefers clean, practical solutions over complexity. He will be using this tool regularly in retirement as part of an ongoing music creation hobby.

## Developer Environment

- **OS:** Windows 11
- **CPU:** AMD Ryzen 9 9950X (16-core)
- **RAM:** 64GB
- **GPU:** NVIDIA GeForce RTX 5060 Ti — use GPU acceleration by default
- **Python:** Already installed
- **Claude Code:** Already installed
- **Shell:** PowerShell

## Architecture Principle

**The GUI must be a wrapper around the CLI — not a separate implementation.** All core logic lives in the CLI. The GUI calls the CLI. This ensures the CLI is always the source of truth, makes unit testing straightforward, and keeps the two in sync automatically.

## P0 Feature — Drum Removal

The minimum viable feature. Must be fully working before any other features are started.

**Behavior:**
- User selects an audio file (MP3 or WAV)
- Tool runs Demucs stem separation with `--two-stems=drums`
- Outputs two files: `drums` and `no_drums`
- Default output location: same directory as the input file
- User is also presented with a "Save As" option to choose a different output directory
- If output file already exists, warn the user before overwriting — do not silently overwrite
- Use the `htdemucs` model (current Demucs default, highest quality)
- GPU acceleration enabled by default (RTX 5060 Ti available)

**Supported input formats:** MP3, WAV

**Output format:** WAV (Demucs default, uncompressed, best quality for Suno upload)

**Output naming convention:** `{original_filename}_no_drums.wav` and `{original_filename}_drums.wav`

---

## GUI Requirements

### General
- Simple, clean, low-friction — practical over fancy
- Launched from PowerShell command line (e.g., `python drumremover.py` or `drumremover`)
- Desktop icon launcher is a **backlog item**, not P0

### File Browser
- Must allow browsing the full Windows file system starting from "This PC"
- Must show both local drives and network mapped drives
- Default starting directory should be configurable (see Advanced tab)
- Standard Windows-style file open dialog is acceptable if it meets the above requirements

### Main Tab
- File selection (browse button + path display)
- "Remove Drums" button (P0 action)
- Progress bar with percentage and estimated time remaining
- Status messages (processing, complete, error)
- "Open Output Folder" button — appears after successful completion
- Output location display (shows where files will be saved)
- "Save As" option to override default output directory

### Advanced Tab
- Default starting directory for file browser
- Output naming convention (editable template)
- GPU / CPU toggle (default: GPU)
- Demucs model selector (default: htdemucs)
- Settings should persist between sessions (simple config file)

---

## CLI Requirements

The CLI must expose all core functions independently of the GUI. This enables unit testing and scripting.

**Minimum CLI interface:**

```
drumremover.py --input <filepath> [--output <directory>] [--model <model>] [--device cpu|gpu]
drumremover.py --help
drumremover.py --version
drumremover.py --list-models
```

**Examples:**
```
python drumremover.py --input "C:\Music\Demos\EGABass2.mp3"
python drumremover.py --input "C:\Music\Demos\EGABass2.mp3" --output "C:\Music\Output"
python drumremover.py --input "C:\Music\Demos\EGABass2.mp3" --device cpu
```

---

## Logging

- Write a log file to the output directory (or a configurable logs folder)
- Each operation logged with: timestamp, input file, model used, output location, duration, success/fail
- Errors logged with enough detail to diagnose
- Log file should be human-readable plain text

---

## Error Handling

- Clear, plain-English error messages — no raw Python tracebacks shown to the user
- Validate input file exists and is a supported format before starting
- Warn if output file already exists (do not silently overwrite)
- Graceful handling of GPU unavailability — fall back to CPU with a warning message
- Handle corrupt or unreadable audio files gracefully

---

## Backlog (Do Not Implement Yet — Document Only)

These are known future features. Do not implement in the initial build. Claude Code should acknowledge these exist and avoid architectural decisions that would make them hard to add later.

| Item | Description |
|------|-------------|
| Desktop icon launcher | Create a Windows shortcut / .bat file to launch the GUI without opening PowerShell |
| Batch processing | Select multiple files or an entire folder for processing in one run |
| Full 4-stem separation | Separate into drums, bass, vocals, other — all at once |
| Vocal removal | Isolate or remove vocals independently |
| Bass isolation | Extract bass stem only |
| Guitar isolation | Extract guitar/other stem only |
| Output format selector | Allow user to choose WAV vs MP3 output |
| Drag and drop | Drag audio files directly onto the GUI window |
| Waveform preview | Visual preview of input and output audio |
| Metadata preservation | Copy ID3/metadata tags from input file to output files |

---

## Dependencies

- **Python 3.x** — already installed
- **Demucs** — `pip install demucs` (install locally, not in Colab)
- **PyTorch with CUDA** — required for GPU acceleration on RTX 5060 Ti
- **tkinter** — preferred for GUI (ships with Python on Windows, no extra install)
- All dependencies should be installable via a `requirements.txt`

---

## Project Structure (Suggested)

```
drum-remover/
├── CLAUDE.md              # This file
├── drumremover.py         # CLI entry point
├── gui.py                 # GUI wrapper (calls CLI functions)
├── core.py                # Core Demucs logic
├── config.py              # Settings persistence
├── requirements.txt       # Python dependencies
├── logs/                  # Log files (default location)
└── README.md              # Basic usage instructions
```

---

## Getting Started Instruction for Claude Code

Read this file, confirm you understand the architecture (CLI-first, GUI as wrapper), ask any clarifying questions, then begin with the P0 feature: a working CLI that accepts an input file and produces a drumless WAV output using Demucs with GPU acceleration on Windows.

Do not start the GUI until the CLI is working and testable.

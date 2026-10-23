"""
Lyric Timing tab for the GUI. Like the rest of the GUI it only handles layout, user interaction and
threading; the work is done by lyricsync.run() (the same function the lyricsync.py CLI calls).
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import config
import lyricsync

MODELS = ["small", "medium", "large-v3"]

# Under pythonw (launch.bat) there is no console, and the progress bars in the speech-model
# libraries (e.g. the model download) crash when stdout/stderr are None. The replacement streams
# must be UTF-8: with the default Windows encoding the bars' block characters raise UnicodeEncodeError.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


class LyricTimingTab:
    def __init__(self, root: tk.Tk, frame: ttk.Frame, cfg: dict):
        self._root = root
        self._f = frame
        self._cfg = cfg
        self._audio: Path | None = None
        self._busy = False
        self._last_dir: Path | None = None
        self._full_text = ""
        self._build()

    # ------------------------------------------------------------------ layout

    def _build(self):
        f = self._f
        pad = {"padx": 12, "pady": 4}

        audio_frame = ttk.LabelFrame(f, text="Song (mp3 or wav)")
        audio_frame.pack(fill="x", **pad)
        self._audio_var = tk.StringVar(value="No file selected")
        ttk.Label(audio_frame, textvariable=self._audio_var, anchor="w", width=58,
                  relief="sunken", padding=4).pack(side="left", padx=6, pady=6)
        ttk.Button(audio_frame, text="Browse…", command=self._browse_audio).pack(side="left", padx=(0, 6))

        lyr_frame = ttk.LabelFrame(f, text="Lyrics (paste here, or load a text file)")
        lyr_frame.pack(fill="both", expand=False, **pad)
        self._lyrics = tk.Text(lyr_frame, height=9, width=70, wrap="word", font=("Consolas", 9), undo=True)
        sb = ttk.Scrollbar(lyr_frame, command=self._lyrics.yview)
        self._lyrics.configure(yscrollcommand=sb.set)
        self._lyrics.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="left", fill="y", pady=6, padx=(0, 2))
        ttk.Button(lyr_frame, text="Load file…", command=self._load_lyrics).pack(side="top", padx=6, pady=6)

        opt = ttk.Frame(f)
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Speech model:").pack(side="left", padx=(6, 4))
        self._model_var = tk.StringVar(value=self._cfg.get("lyric_model", "medium"))
        ttk.Combobox(opt, textvariable=self._model_var, values=MODELS, state="readonly", width=10).pack(side="left")
        self._words_var = tk.BooleanVar(value=bool(self._cfg.get("lyric_words", False)))
        ttk.Checkbutton(opt, text="Per-word times in .lrc", variable=self._words_var).pack(side="left", padx=16)
        ttk.Label(opt, text="(large-v3 is slower but handles harder songs)", foreground="gray").pack(side="left")

        dev = ttk.LabelFrame(f, text="Processor")
        dev.pack(fill="x", **pad)
        self._device_var = tk.StringVar(value=self._cfg.get("lyric_device", "auto"))
        for value, text in (("auto", "Auto (GPU only if it has room)"), ("gpu", "GPU"), ("cpu", "CPU (slower, leaves the GPU free)")):
            ttk.Radiobutton(dev, text=text, variable=self._device_var, value=value).pack(side="left", padx=10, pady=4)
        self._gpu_var = tk.StringVar(value="GPU: checking…")
        ttk.Label(dev, textvariable=self._gpu_var, foreground="gray").pack(side="right", padx=10)
        self._poll_gpu()

        self._run_btn = ttk.Button(f, text="Time the Lyrics", command=self._start)
        self._run_btn.pack(pady=(6, 2))
        self._progress = tk.DoubleVar(value=0)
        ttk.Progressbar(f, variable=self._progress, maximum=100).pack(fill="x", padx=12, pady=2)
        self._status = tk.StringVar(value="Ready. Output files go next to the song (or the default output folder).")
        ttk.Label(f, textvariable=self._status, anchor="w", relief="sunken", padding=4).pack(fill="x", padx=12, pady=4)

        res = ttk.LabelFrame(f, text="Result (Copy puts both parts on the clipboard, ready to paste into Claude chat/design)")
        res.pack(fill="both", expand=True, **pad)
        btns = ttk.Frame(res)
        btns.pack(side="right", fill="y", padx=6, pady=6)
        # Two panes the user can resize by dragging the divider: the timed lyrics, and the warnings
        # (which can be long) so they never push the lyrics out of view.
        panes = ttk.PanedWindow(res, orient="vertical")
        panes.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        self._result = self._scrolled_text(panes, "Timed lyrics", weight=3)
        self._warnings = self._scrolled_text(panes, "Flagged lines / warnings", weight=2)
        self._copy_btn = ttk.Button(btns, text="Copy", command=self._copy, state="disabled")
        self._copy_btn.pack(pady=2)
        self._open_btn = ttk.Button(btns, text="Open folder", command=self._open_folder, state="disabled")
        self._open_btn.pack(pady=2)

    @staticmethod
    def _scrolled_text(panes: ttk.PanedWindow, title: str, weight: int) -> tk.Text:
        """A read-only, word-wrapped text box with its own vertical scrollbar, added to `panes`."""
        box = ttk.LabelFrame(panes, text=title)
        text = tk.Text(box, height=5, width=70, wrap="word", font=("Consolas", 9), state="disabled")
        sb = ttk.Scrollbar(box, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        sb.pack(side="left", fill="y", pady=4, padx=(0, 2))
        panes.add(box, weight=weight)
        return text

    # ---------------------------------------------------------------- handlers

    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select song", initialdir=self._cfg.get("default_input_dir") or str(Path.home()),
            filetypes=[("Audio files", "*.mp3 *.wav *.MP3 *.WAV"), ("All files", "*.*")])
        if path:
            self._audio = Path(path)
            self._audio_var.set(path)

    def _load_lyrics(self):
        path = filedialog.askopenfilename(
            title="Select lyrics text file", initialdir=str(self._audio.parent) if self._audio else str(Path.home()),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self._lyrics.delete("1.0", "end")
            self._lyrics.insert("1.0", Path(path).read_text(encoding="utf-8-sig"))

    def _poll_gpu(self):
        """Show how much GPU memory is free, refreshed every 10 s. nvidia-smi runs on a worker thread
        so a slow reply never freezes the window."""
        def check():
            st = lyricsync.gpu_status()
            text = "GPU: not found" if st is None else f"GPU: {st[0]:.1f} of {st[1]:.1f} GB free"
            try:
                self._root.after(0, self._gpu_var.set, text)
            except Exception:
                pass  # window already closed
        threading.Thread(target=check, daemon=True).start()
        self._root.after(10000, self._poll_gpu)

    def _copy(self):
        self._root.clipboard_clear()
        self._root.clipboard_append(self._full_text.strip())
        self._status.set("Result copied to the clipboard.")

    def _open_folder(self):
        if self._last_dir:
            subprocess.Popen(["explorer", str(self._last_dir)])

    def _set_result(self, timed: str, warnings: str = ""):
        self._full_text = timed + ("\n" + warnings + "\n" if warnings else "")
        for box, text in ((self._result, timed), (self._warnings, warnings)):
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", text)
            box.configure(state="disabled")

    # --------------------------------------------------------------- processing

    def _start(self):
        if self._busy:
            return
        lyrics = self._lyrics.get("1.0", "end").strip()
        if not self._audio:
            messagebox.showwarning("No song", "Please choose a song first.")
            return
        if not lyrics:
            messagebox.showwarning("No lyrics", "Paste the lyrics or load a text file first.")
            return
        out_dir = Path(self._cfg["default_output_dir"]) if self._cfg.get("default_output_dir") else self._audio.parent
        existing = [out_dir / f"{self._audio.stem}{s}" for s in ("_timed.txt", ".lrc", "_timing.json")
                    if (out_dir / f"{self._audio.stem}{s}").exists()]
        if existing and not messagebox.askyesno(
                "Overwrite?", "These files already exist:\n\n" + "\n".join(p.name for p in existing) + "\n\nOverwrite?"):
            return
        model, words, device = self._model_var.get(), self._words_var.get(), self._device_var.get()
        demucs_model = self._cfg.get("model", "htdemucs_ft")
        if device == "gpu":
            st = lyricsync.gpu_status()
            need = lyricsync.needed_gb(model, lyricsync.cached_vocals_path(self._audio, demucs_model) is None)
            if st and st[0] < need and not messagebox.askyesno(
                    "GPU is busy",
                    f"Only {st[0]:.1f} GB of GPU memory is free and this needs about {need:.1f} GB.\n"
                    "Another program (such as the animation tool) may be using it; running now could "
                    "slow it down or make one of the jobs fail.\n\nRun on the GPU anyway?"):
                return
        for key, value in (("lyric_model", model), ("lyric_words", words), ("lyric_device", device)):
            self._cfg[key] = value
            config.set(key, value)

        self._busy = True
        self._run_btn.config(state="disabled")
        self._copy_btn.config(state="disabled")
        self._open_btn.config(state="disabled")
        self._progress.set(0)
        self._set_result("")
        self._status.set("Starting…")
        audio = self._audio

        def progress(frac, msg):
            if frac is not None:
                self._root.after(0, self._progress.set, frac * 100)
            self._root.after(0, self._status.set, msg)

        def work():
            try:
                res = lyricsync.run(audio, lyrics, out_dir, model=model, device=device, words=words,
                                    demucs_model=demucs_model, progress=progress)
                self._root.after(0, self._done, res)
            except Exception as exc:
                self._root.after(0, self._failed, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, res: dict):
        self._busy = False
        self._run_btn.config(state="normal")
        self._copy_btn.config(state="normal")
        self._open_btn.config(state="normal")
        self._last_dir = Path(res["output_dir"])
        self._progress.set(100)
        note = "" if not res["flagged"] else f"  {res['flagged']} line(s) marked ! - worth a second look."
        self._status.set(f"Done ✔  {res['lines']} lines timed on the {res['device_used'].upper()}.{note}")
        self._set_result(res["timed_text"], res["report"])

    def _failed(self, message: str):
        self._busy = False
        self._run_btn.config(state="normal")
        self._status.set(f"Error: {message}")
        messagebox.showerror("Lyric timing failed", message)

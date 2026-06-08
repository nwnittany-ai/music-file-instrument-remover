"""
Music File Instrument Remover — tkinter GUI

Wraps core.py. All processing logic lives in core.py; this file handles
only UI layout, user interaction, and threading.
"""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import config
import core

APP_TITLE = "Music File Instrument Remover"
VERSION = "0.2.0"
WIN_W, WIN_H = 620, 520


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_folder(path: Path) -> None:
    subprocess.Popen(["explorer", str(path)])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class DrumRemoverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} v{VERSION}")
        self.resizable(False, False)
        self.geometry(f"{WIN_W}x{WIN_H}")

        self._cfg = config.load()
        self._input_path: Path | None = None
        self._output_dir_override: Path | None = None
        self._processing = False

        self._build_menu()
        self._build_tabs()
        self._update_output_label()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About...", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            f"{APP_TITLE} v{VERSION}\n\n"
            "Uses Facebook Research Demucs for AI stem separation.\n"
            "GPU-accelerated via CUDA.",
        )

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._main_frame = ttk.Frame(nb)
        self._adv_frame  = ttk.Frame(nb)
        self._help_frame = ttk.Frame(nb)

        nb.add(self._main_frame, text="  Main  ")
        nb.add(self._adv_frame,  text="  Advanced  ")
        nb.add(self._help_frame, text="  CLI Reference  ")

        self._build_main_tab()
        self._build_advanced_tab()
        self._build_help_tab()

    # ------------------------------------------------------------------
    # Main tab
    # ------------------------------------------------------------------

    def _build_main_tab(self):
        f = self._main_frame
        pad = {"padx": 12, "pady": 6}

        # --- Input file ---
        input_frame = ttk.LabelFrame(f, text="Input File")
        input_frame.pack(fill="x", **pad)

        self._input_var = tk.StringVar(value="No file selected")
        ttk.Label(input_frame, textvariable=self._input_var, anchor="w",
                  width=58, relief="sunken", padding=4).pack(side="left", padx=6, pady=6)
        ttk.Button(input_frame, text="Browse…", command=self._browse_input).pack(
            side="left", padx=(0, 6), pady=6)

        # --- Stem selector ---
        stem_frame = ttk.LabelFrame(f, text="Separation Mode")
        stem_frame.pack(fill="x", **pad)

        self._stem_var = tk.StringVar(value=self._cfg.get("stem", core.DEFAULT_STEM))
        for i, (key, label) in enumerate(core.STEM_LABELS.items()):
            ttk.Radiobutton(
                stem_frame, text=label, variable=self._stem_var, value=key,
                command=self._on_stem_changed,
            ).grid(row=i // 2, column=i % 2, sticky="w", padx=16, pady=3)

        # --- Output format ---
        fmt_frame = ttk.LabelFrame(f, text="Output Format")
        fmt_frame.pack(fill="x", **pad)

        self._format_var = tk.StringVar(value=self._cfg.get("output_format", "wav"))
        ttk.Radiobutton(fmt_frame, text="WAV  (24-bit, best quality)",
                        variable=self._format_var, value="wav").pack(
            side="left", padx=16, pady=4)
        ttk.Radiobutton(fmt_frame, text="MP3  (320 kbps, ~10× smaller)",
                        variable=self._format_var, value="mp3").pack(
            side="left", padx=16, pady=4)

        # --- Output location ---
        out_frame = ttk.LabelFrame(f, text="Output Location")
        out_frame.pack(fill="x", **pad)

        self._output_label_var = tk.StringVar()
        ttk.Label(out_frame, textvariable=self._output_label_var, anchor="w",
                  width=58, relief="sunken", padding=4).pack(side="left", padx=6, pady=6)
        ttk.Button(out_frame, text="Save As…", command=self._browse_output).pack(
            side="left", padx=(0, 6), pady=6)

        # --- Action button ---
        self._run_btn = ttk.Button(
            f, text=self._action_label(), command=self._start_processing,
        )
        self._run_btn.pack(pady=(10, 4))

        # --- Progress bar ---
        prog_frame = ttk.Frame(f)
        prog_frame.pack(fill="x", padx=12)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame, variable=self._progress_var, maximum=100, length=580,
        )
        self._progress_bar.pack(fill="x", pady=(4, 2))

        self._pct_var = tk.StringVar(value="")
        ttk.Label(prog_frame, textvariable=self._pct_var, anchor="e").pack(fill="x")

        # --- Status message ---
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(f, textvariable=self._status_var, anchor="w",
                  relief="sunken", padding=4).pack(fill="x", padx=12, pady=(4, 6))

        # --- Open output folder button (hidden until success) ---
        self._open_folder_btn = ttk.Button(
            f, text="Open Output Folder", command=self._open_output_folder,
        )

    def _action_label(self) -> str:
        return core.STEM_LABELS.get(self._stem_var.get(), "Separate Stems")

    def _on_stem_changed(self):
        self._run_btn.config(text=self._action_label())
        self._hide_open_folder_btn()
        self._status_var.set("Ready.")
        self._progress_var.set(0)
        self._pct_var.set("")

    # ------------------------------------------------------------------
    # Help tab
    # ------------------------------------------------------------------

    def _build_help_tab(self):
        f = self._help_frame

        text = tk.Text(f, wrap="word", padx=12, pady=10, relief="flat",
                       font=("Consolas", 9), background=self.cget("background"))
        scroll = ttk.Scrollbar(f, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        # --- Tag styles ---
        text.tag_configure("h1",   font=("Segoe UI", 11, "bold"), spacing3=4)
        text.tag_configure("h2",   font=("Segoe UI", 10, "bold"), spacing1=8, spacing3=2)
        text.tag_configure("code", font=("Consolas", 9), background="#e8e8e8")
        text.tag_configure("body", font=("Segoe UI", 9))

        def h1(t):  text.insert("end", t + "\n", "h1")
        def h2(t):  text.insert("end", t + "\n", "h2")
        def code(t): text.insert("end", t + "\n", "code")
        def body(t): text.insert("end", t + "\n", "body")
        def blank():  text.insert("end", "\n")

        h1("Music File Instrument Remover — CLI Reference")
        body(f"Version {VERSION}")
        blank()

        h2("Launching the GUI")
        code('  .venv\\Scripts\\python.exe gui.py')
        blank()

        h2("CLI Syntax")
        code('  python drumremover.py --input <file> [options]')
        blank()

        h2("Required Argument")
        code('  --input  -i  <file>   Input MP3 or WAV file')
        blank()

        h2("Options")
        code('  --stem   -s  <stem>      Stem to separate (default: drums)')
        code('  --format -f  wav|mp3    Output format (default: wav)')
        code('  --output -o  <dir>      Output directory (default: same as input)')
        code('  --model  -m  <model>    Demucs model (default: htdemucs_ft)')
        code('  --device -d  gpu|cpu    Processing device (default: gpu)')
        code('  --list-stems          List available stems and exit')
        code('  --list-models         List available models and exit')
        code('  --version             Show version and exit')
        code('  --help                Show this help and exit')
        blank()

        h2("Available Stems")
        code('  drums    Remove or isolate drums')
        code('  vocals   Remove or isolate vocals (karaoke)')
        code('  bass     Remove or isolate bass')
        code('  other    Remove or isolate guitar/keys')
        code('  all      Full 4-stem separation (drums, bass, vocals, other)')
        blank()

        h2("Available Models")
        code('  htdemucs_ft  Best quality — recommended  (default)')
        code('  htdemucs     Fast, good quality')
        code('  mdx_extra    Alternative architecture, competitive quality')
        code('  mdx          Lighter/faster, lower quality')
        code('  mdx_q        Quantized mdx — fastest, lowest quality')
        code('  mdx_extra_q  Quantized mdx_extra')
        blank()

        h2("Examples")
        code('  python drumremover.py --input "C:\\Music\\demo.mp3"')
        code('  python drumremover.py --input "C:\\Music\\demo.mp3" --stem vocals')
        code('  python drumremover.py --input "C:\\Music\\demo.mp3" --stem all')
        code('  python drumremover.py --input "C:\\Music\\demo.mp3" --output "C:\\Music\\Output"')
        code('  python drumremover.py --input "C:\\Music\\demo.mp3" --model htdemucs --device cpu')
        blank()

        h2("Output Files")
        body('  Single stem (e.g. --stem drums):')
        code('    demo_no_drums.wav   Everything except drums')
        code('    demo_drums.wav      Isolated drums')
        blank()
        body('  Full separation (--stem all):')
        code('    demo_drums.wav')
        code('    demo_bass.wav')
        code('    demo_vocals.wav')
        code('    demo_other.wav')
        blank()

        h2("Notes")
        body('  • Output format is always 24-bit WAV (best quality for Suno uploads).')
        body('  • GPU acceleration requires an NVIDIA GPU with CUDA support.')
        body('  • MP3 input requires ffmpeg to be installed and on your PATH.')
        body('  • Settings in the Advanced tab are saved to drumremover_config.json.')
        body('  • Logs are written to the logs/ folder by default.')

        text.configure(state="disabled")  # read-only

    # ------------------------------------------------------------------
    # Advanced tab
    # ------------------------------------------------------------------

    def _build_advanced_tab(self):
        f = self._adv_frame

        row = 0

        def lbl(text, r):
            ttk.Label(f, text=text, anchor="e", width=22).grid(
                row=r, column=0, sticky="e", padx=(12, 4), pady=6)

        # Default input directory
        lbl("Default input folder:", row)
        self._def_input_var = tk.StringVar(value=self._cfg.get("default_input_dir", ""))
        ttk.Entry(f, textvariable=self._def_input_var, width=38).grid(
            row=row, column=1, sticky="ew", pady=6)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._browse_dir(self._def_input_var)).grid(
            row=row, column=2, padx=(4, 12), pady=6)
        row += 1

        # Default output directory
        lbl("Default output folder:", row)
        self._def_output_var = tk.StringVar(value=self._cfg.get("default_output_dir", ""))
        ttk.Entry(f, textvariable=self._def_output_var, width=38).grid(
            row=row, column=1, sticky="ew", pady=6)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._browse_dir(self._def_output_var)).grid(
            row=row, column=2, padx=(4, 12), pady=6)
        row += 1

        # Model
        lbl("Demucs model:", row)
        self._model_var = tk.StringVar(value=self._cfg.get("model", core.DEFAULT_MODEL))
        ttk.Combobox(f, textvariable=self._model_var,
                     values=core.list_models(), state="readonly", width=36).grid(
            row=row, column=1, sticky="ew", pady=6)
        row += 1

        # Device
        lbl("Processing device:", row)
        self._device_var = tk.StringVar(value=self._cfg.get("device", "gpu"))
        device_frame = ttk.Frame(f)
        device_frame.grid(row=row, column=1, sticky="w", pady=6)
        ttk.Radiobutton(device_frame, text="GPU (recommended)", variable=self._device_var,
                        value="gpu").pack(side="left", padx=(0, 16))
        ttk.Radiobutton(device_frame, text="CPU", variable=self._device_var,
                        value="cpu").pack(side="left")
        row += 1

        # Log directory
        lbl("Log folder:", row)
        self._log_dir_var = tk.StringVar(value=self._cfg.get("log_dir", ""))
        ttk.Entry(f, textvariable=self._log_dir_var, width=38).grid(
            row=row, column=1, sticky="ew", pady=6)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._browse_dir(self._log_dir_var)).grid(
            row=row, column=2, padx=(4, 12), pady=6)
        row += 1

        f.columnconfigure(1, weight=1)

        ttk.Button(f, text="Save Settings", command=self._save_settings).grid(
            row=row, column=0, columnspan=3, pady=16)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_input(self):
        initial = self._cfg.get("default_input_dir") or str(Path.home())
        path = filedialog.askopenfilename(
            title="Select audio file",
            initialdir=initial,
            filetypes=[("Audio files", "*.mp3 *.wav *.WAV *.MP3"), ("All files", "*.*")],
        )
        if path:
            self._input_path = Path(path)
            self._input_var.set(str(self._input_path))
            self._output_dir_override = None
            self._update_output_label()
            self._hide_open_folder_btn()
            self._status_var.set("Ready.")
            self._progress_var.set(0)
            self._pct_var.set("")

    def _browse_output(self):
        initial = (
            str(self._output_dir_override)
            or self._cfg.get("default_output_dir")
            or (str(self._input_path.parent) if self._input_path else str(Path.home()))
        )
        path = filedialog.askdirectory(title="Select output folder", initialdir=initial)
        if path:
            self._output_dir_override = Path(path)
            self._update_output_label()

    def _browse_dir(self, string_var: tk.StringVar):
        path = filedialog.askdirectory(title="Select folder",
                                       initialdir=string_var.get() or str(Path.home()))
        if path:
            string_var.set(path)

    def _update_output_label(self):
        if self._output_dir_override:
            self._output_label_var.set(str(self._output_dir_override))
        elif self._cfg.get("default_output_dir"):
            self._output_label_var.set(self._cfg["default_output_dir"])
        elif self._input_path:
            self._output_label_var.set(str(self._input_path.parent) + "  (same as input)")
        else:
            self._output_label_var.set("Same folder as input file")

    def _save_settings(self):
        self._cfg["default_input_dir"]  = self._def_input_var.get()
        self._cfg["default_output_dir"] = self._def_output_var.get()
        self._cfg["model"]              = self._model_var.get()
        self._cfg["device"]             = self._device_var.get()
        self._cfg["log_dir"]            = self._log_dir_var.get()
        config.save(self._cfg)
        self._update_output_label()
        messagebox.showinfo("Settings", "Settings saved.")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _start_processing(self):
        if self._processing:
            return
        if not self._input_path:
            messagebox.showwarning("No file", "Please select an input file first.")
            return

        stem          = self._stem_var.get()
        output_format = self._format_var.get()
        output_dir = self._output_dir_override or (
            Path(self._cfg["default_output_dir"]) if self._cfg.get("default_output_dir") else None
        )

        # Overwrite check
        output_paths = core.resolve_output_paths(self._input_path, output_dir, stem, output_format)
        existing = core.check_overwrite(output_paths)
        if existing:
            names = "\n".join(p.name for p in existing)
            if not messagebox.askyesno(
                "Overwrite?",
                f"The following file(s) already exist:\n\n{names}\n\nOverwrite?",
            ):
                return

        model  = self._cfg.get("model",  core.DEFAULT_MODEL)
        device = self._cfg.get("device", "gpu")

        self._processing = True
        self._run_btn.config(state="disabled")
        self._hide_open_folder_btn()
        self._progress_var.set(0)
        self._pct_var.set("")
        self._status_var.set("Starting…")

        def run():
            try:
                results = core.separate_stems(
                    input_path=self._input_path,
                    stem=stem,
                    output_dir=output_dir,
                    model=model,
                    device_preference=device,
                    output_format=output_format,
                    progress_callback=self._on_progress,
                )
                self.after(0, self._on_success, results)
            except core.SeparationError as e:
                self.after(0, self._on_error, str(e))
            except Exception as e:
                self.after(0, self._on_error, f"Unexpected error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _on_progress(self, fraction: float, message: str):
        pct = int(fraction * 100)
        self.after(0, self._progress_var.set, float(pct))
        self.after(0, self._pct_var.set, f"{pct}%")
        self.after(0, self._status_var.set, message)

    def _on_success(self, results: list[Path]):
        self._processing = False
        self._run_btn.config(state="normal")
        self._progress_var.set(100)
        self._pct_var.set("100%")
        count = len(results)
        self._status_var.set(f"Done! ✔  {count} file(s) written to {results[0].parent.name}")
        self._last_output_dir = results[0].parent
        self._show_open_folder_btn()

    def _on_error(self, message: str):
        self._processing = False
        self._run_btn.config(state="normal")
        self._status_var.set(f"Error: {message}")
        messagebox.showerror("Error", message)

    # ------------------------------------------------------------------
    # Open output folder button
    # ------------------------------------------------------------------

    def _show_open_folder_btn(self):
        self._open_folder_btn.pack(pady=(2, 8))

    def _hide_open_folder_btn(self):
        self._open_folder_btn.pack_forget()

    def _open_output_folder(self):
        if hasattr(self, "_last_output_dir"):
            _open_folder(self._last_output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = DrumRemoverApp()
    app.mainloop()

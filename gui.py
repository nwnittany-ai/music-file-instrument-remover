"""
Drum Remover — tkinter GUI

Wraps core.py. All processing logic lives in core.py; this file handles
only UI layout, user interaction, and threading.
"""

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import config
import core

APP_TITLE = "Drum Remover"
VERSION = "0.1.0"
WIN_W, WIN_H = 600, 480


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
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _show_about(self):
        messagebox.showinfo(
            "About Drum Remover",
            f"Drum Remover v{VERSION}\n\nUses Facebook Research Demucs for stem separation.\nGPU-accelerated via CUDA.",
        )

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._main_frame = ttk.Frame(nb)
        self._adv_frame  = ttk.Frame(nb)

        nb.add(self._main_frame, text="  Main  ")
        nb.add(self._adv_frame,  text="  Advanced  ")

        self._build_main_tab()
        self._build_advanced_tab()

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
                  width=55, relief="sunken", padding=4).pack(side="left", padx=6, pady=6)
        ttk.Button(input_frame, text="Browse…", command=self._browse_input).pack(
            side="left", padx=(0, 6), pady=6)

        # --- Output location ---
        out_frame = ttk.LabelFrame(f, text="Output Location")
        out_frame.pack(fill="x", **pad)

        self._output_label_var = tk.StringVar()
        ttk.Label(out_frame, textvariable=self._output_label_var, anchor="w",
                  width=55, relief="sunken", padding=4).pack(side="left", padx=6, pady=6)
        ttk.Button(out_frame, text="Save As…", command=self._browse_output).pack(
            side="left", padx=(0, 6), pady=6)

        # --- Action button ---
        self._remove_btn = ttk.Button(
            f, text="Remove Drums", command=self._start_processing,
            style="Accent.TButton",
        )
        self._remove_btn.pack(pady=(10, 4))

        # --- Progress bar ---
        prog_frame = ttk.Frame(f)
        prog_frame.pack(fill="x", padx=12)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame, variable=self._progress_var, maximum=100, length=560,
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
        # shown only after a successful run

    # ------------------------------------------------------------------
    # Advanced tab
    # ------------------------------------------------------------------

    def _build_advanced_tab(self):
        f = self._adv_frame
        pad = {"padx": 12, "pady": 8}

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
        model_cb = ttk.Combobox(f, textvariable=self._model_var,
                                values=core.list_models(), state="readonly", width=36)
        model_cb.grid(row=row, column=1, sticky="ew", pady=6)
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

        # Save button
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
            self._output_dir_override = None  # reset any prior Save As choice
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

        output_dir = self._output_dir_override or (
            Path(self._cfg["default_output_dir"]) if self._cfg.get("default_output_dir") else None
        )

        # Overwrite check
        no_drums_path, drums_path = core.resolve_output_paths(self._input_path, output_dir)
        existing = core.check_overwrite([no_drums_path, drums_path])
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
        self._remove_btn.config(state="disabled")
        self._hide_open_folder_btn()
        self._progress_var.set(0)
        self._pct_var.set("")
        self._status_var.set("Starting…")

        def run():
            try:
                no_drums, drums = core.remove_drums(
                    input_path=self._input_path,
                    output_dir=output_dir,
                    model=model,
                    device_preference=device,
                    progress_callback=self._on_progress,
                )
                self.after(0, self._on_success, no_drums)
            except core.DrumRemoverError as e:
                self.after(0, self._on_error, str(e))
            except Exception as e:
                self.after(0, self._on_error, f"Unexpected error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _on_progress(self, fraction: float, message: str):
        pct = int(fraction * 100)
        self.after(0, self._progress_var.set, pct * 1.0)
        self.after(0, self._pct_var.set, f"{pct}%")
        self.after(0, self._status_var.set, message)

    def _on_success(self, no_drums_path: Path):
        self._processing = False
        self._remove_btn.config(state="normal")
        self._progress_var.set(100)
        self._pct_var.set("100%")
        self._status_var.set(f"Done! ✔  {no_drums_path.name}")
        self._last_output_dir = no_drums_path.parent
        self._show_open_folder_btn()

    def _on_error(self, message: str):
        self._processing = False
        self._remove_btn.config(state="normal")
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

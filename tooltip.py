"""
Hover help for tkinter widgets: rest the mouse on a control for half a second to see what it does.
"""

import tkinter as tk


class Tooltip:
    DELAY_MS = 500

    def __init__(self, widget: tk.Widget, text: str, wrap: int = 420):
        self._widget = widget
        self._text = text
        self._wrap = wrap
        self._tip: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text: str) -> None:
        self._text = text

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self._widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after:
            self._widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        if self._tip or not self._text:
            return
        x = self._widget.winfo_rootx() + 16
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self._text, justify="left", wraplength=self._wrap,
                 background="#ffffe8", relief="solid", borderwidth=1, padx=6, pady=4,
                 font=("Segoe UI", 9)).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None


def add_tooltip(widget: tk.Widget, text: str) -> Tooltip:
    return Tooltip(widget, text)

"""
Persistent settings for Drum Remover — reads/writes a JSON config file.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "drumremover_config.json"

DEFAULTS = {
    "default_input_dir": str(Path.home()),
    "default_output_dir": "",          # empty = same dir as input
    "model": "htdemucs_ft",
    "device": "gpu",
    "output_name_template": "{stem}_no_drums",
    "log_dir": str(Path(__file__).parent / "logs"),
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                stored = json.load(f)
            cfg = {**DEFAULTS, **stored}
            return cfg
        except Exception:
            pass
    return dict(DEFAULTS)


def save(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get(key: str):
    return load().get(key, DEFAULTS.get(key))


def set(key: str, value) -> None:
    cfg = load()
    cfg[key] = value
    save(cfg)

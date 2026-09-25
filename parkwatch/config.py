"""YAML configuration with defaults. See configs/demo.yaml for a fully commented example."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict = {
    "site": {"name": "Car park", "capacity": 100, "initial_occupancy": 0, "overstay_limit_min": 60,
             "alert_unknown_arrival": False},
    "clock": {"start": None, "time_scale": 1.0},
    "models": {
        "device": None,
        "tracker": "configs/trackers/botsort_parking.yaml",
        "ground": {"weights": "yolo11s.pt", "imgsz": 640, "conf": 0.3,
                   "classes": ["car", "truck", "bus", "motorcycle", "vehicle"]},
        "overhead": {"weights": "yolo11s-obb.pt", "imgsz": 1024, "conf": 0.3,
                     "classes": ["small vehicle", "large vehicle", "vehicle"]},
    },
    "plates": {"format": "generic", "hint": "", "providers": ["llm", "local"], "max_attempts_per_vehicle": 2,
               "min_confidence": 0.4, "workers": 2,
               "llm": {"model": "gemini-2.5-flash-lite", "fallback_models": ["gemini-2.5-flash"],
                       "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "api": "chat.completions", "api_key_env": "GEMINI_API_KEY", "timeout_s": 20,
                       "cost_per_call_usd": 0.001, "request_params": {}},
               "local": {}},
    "linking": {"entry_to_lot_s": [0.5, 45.0], "lot_to_exit_s": [0.0, 60.0], "fuzzy_plate_edits": 1},
    "alerts": {"console": True, "webhook_url": None, "repeat_every_min": 0},
    "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 8000},
    "output": {"dir": "runs", "save_video": True},
    "ground_truth": None,
    "cameras": [],
}

CAMERA_DEFAULTS = {
    "entry": {"model": "ground", "direction": "any", "stride": 1, "dedup_seconds": 5.0, "dedup_px": 150},
    "exit": {"model": "ground", "direction": "any", "stride": 1, "dedup_seconds": 5.0, "dedup_px": 150},
    "lot": {"model": "overhead", "stride": 3, "stationary_seconds": 4.0, "stationary_px": 10, "moved_px": 30,
            "reid_radius_px": 45, "moving_lost_seconds": 3.0, "parked_lost_seconds": 120.0,
            "startup_grace_seconds": 3.0},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_path(p: str | None, base: Path | None = None) -> Path | None:
    """Resolve a path from the config: absolute, relative to CWD, to the config file, or to the repo root."""
    if p is None:
        return None
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        return path
    for root in (Path.cwd(), base, ROOT):
        if root is not None and (root / path).exists():
            return (root / path).resolve()
    return path


def _load_geometry(ref: str, base: Path) -> dict:
    """`file.json#key` -> dict. Keys can be nested with dots."""
    file, _, key = str(ref).partition("#")
    data = json.loads(resolve_path(file, base).read_text())
    for part in [k for k in key.split(".") if k]:
        data = data[part]
    return data


def load_config(path: str | Path) -> dict:
    path = Path(path)
    user = yaml.safe_load(path.read_text()) or {}
    cfg = deep_merge(DEFAULTS, user)
    base = path.resolve().parent
    cams = []
    for cam in cfg["cameras"]:
        role = cam.get("role")
        if role not in CAMERA_DEFAULTS:
            raise ValueError(f"camera {cam.get('id')}: role must be entry, exit or lot (got {role!r})")
        c = deep_merge(CAMERA_DEFAULTS[role], cam)
        c.setdefault("id", role)
        c.setdefault("name", c["id"])
        if c.get("geometry"):
            c = deep_merge(_load_geometry(c["geometry"], base), c)
        src = c.get("source")
        if isinstance(src, str) and not src.isdigit() and "://" not in src:
            c["source"] = str(resolve_path(src, base))
        cams.append(c)
    cfg["cameras"] = cams
    cfg["_base"] = str(base)
    if cfg.get("ground_truth"):
        cfg["ground_truth"] = str(resolve_path(cfg["ground_truth"], base))
    return cfg

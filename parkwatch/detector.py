"""YOLO detection + BoT-SORT tracking wrapper (axis-aligned or oriented boxes)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT, resolve_path

log = logging.getLogger("parkwatch.detector")


@dataclass
class Det:
    tid: int  # tracker id (short-lived, camera-local)
    cls: str
    conf: float
    xyxy: np.ndarray  # (4,)
    poly: np.ndarray | None = None  # (4,2) oriented box corners (OBB models)

    @property
    def center(self) -> tuple[float, float]:
        if self.poly is not None:
            return float(self.poly[:, 0].mean()), float(self.poly[:, 1].mean())
        x0, y0, x1, y1 = self.xyxy
        return float((x0 + x1) / 2), float((y0 + y1) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x0, _, x1, y1 = self.xyxy
        return float((x0 + x1) / 2), float(y1)


def _weights_path(name: str) -> str:
    p = resolve_path(name)
    if p is not None and p.exists():
        return str(p)
    for d in (ROOT / "models", ROOT / "weights"):
        if (d / name).exists():
            return str(d / name)
    return name  # let Ultralytics download official weights by name (e.g. yolo11s.pt)


class Detector:
    """One instance per camera: the tracker state lives inside the model's predictor."""

    def __init__(self, mcfg: dict, device=None, tracker: str | None = None):
        from ultralytics import YOLO

        self.weights = _weights_path(mcfg["weights"])
        self.model = YOLO(self.weights)
        self.imgsz, self.conf, self.device = int(mcfg.get("imgsz", 640)), float(mcfg.get("conf", 0.3)), device
        wanted = {c.lower() for c in mcfg.get("classes", [])}
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n.lower() in wanted] or None
        if wanted and not self.class_ids:
            log.warning("%s: none of %s in model classes; keeping all classes", self.weights, sorted(wanted))
        tr = resolve_path(tracker) if tracker else None
        self.tracker = str(tr) if tr is not None and Path(tr).exists() else "botsort.yaml"
        self.is_obb = getattr(self.model, "task", "") == "obb"

    def track(self, frame: np.ndarray) -> list[Det]:
        r = self.model.track(frame, persist=True, tracker=self.tracker, imgsz=self.imgsz, conf=self.conf,
                             classes=self.class_ids, device=self.device, verbose=False)[0]
        out: list[Det] = []
        boxes = r.obb if self.is_obb else r.boxes
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        if self.is_obb:
            polys = boxes.xyxyxyxy.cpu().numpy()
            xyxy = np.concatenate([polys.min(1), polys.max(1)], 1)
        else:
            polys, xyxy = None, boxes.xyxy.cpu().numpy()
        for i in range(len(ids)):
            out.append(Det(int(ids[i]), r.names[int(cls[i])], float(conf[i]), xyxy[i],
                           None if polys is None else polys[i]))
        return out

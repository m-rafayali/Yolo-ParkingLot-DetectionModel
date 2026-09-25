"""Video sources. Files are read on a shared media clock; live streams keep only the newest frame."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class FileSource:
    """Recorded footage. `advance_to(t)` moves to the frame shown at media time t (skipping as needed)."""

    live = False

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.idx = -1
        self.frame: np.ndarray | None = None
        self.done = False

    @property
    def duration(self) -> float:
        return self.n / self.fps if self.n else float("inf")

    def advance_to(self, t: float, decode: bool = True) -> bool:
        """Returns True if a new frame became current."""
        target = int(t * self.fps + 1e-6)
        if target <= self.idx or self.done:
            return False
        while self.idx < target - 1:  # skip without decoding
            if not self.cap.grab():
                self.done = True
                return False
            self.idx += 1
        ok = self.cap.grab()
        if not ok:
            self.done = True
            return False
        self.idx += 1
        if decode:
            ok, frame = self.cap.retrieve()
            if not ok:
                self.done = True
                return False
            self.frame = frame
        return True

    @property
    def mt(self) -> float:
        return self.idx / self.fps

    def close(self) -> None:
        self.cap.release()


class LiveSource:
    """RTSP / HTTP / webcam. A reader thread keeps the latest frame; media time = seconds since start."""

    live = True

    def __init__(self, src, t0: float):
        self.src = int(src) if str(src).isdigit() else src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise ConnectionError(f"cannot open stream {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.t0 = t0
        self.frame, self.idx, self._mt, self.done = None, -1, 0.0, False
        self._lock = threading.Lock()
        self._seen = -1
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        fails = 0
        while not self.done:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:  # reconnect
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(self.src)
                    fails = 0
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self.frame, self.idx, self._mt = frame, self.idx + 1, time.time() - self.t0

    duration = float("inf")

    def advance_to(self, t: float, decode: bool = True) -> bool:
        with self._lock:
            new = self.idx != self._seen
            self._seen = self.idx
        return new and self.frame is not None

    @property
    def mt(self) -> float:
        return self._mt

    def close(self) -> None:
        self.done = True
        self.cap.release()

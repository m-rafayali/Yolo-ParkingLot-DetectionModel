"""Site clock: converts media time (seconds since a stream started) to business time.

For recorded footage the clock follows the video, not the wall clock, so results are the
same whether the machine processes faster or slower than real time. `time_scale` lets a
short simulated clip represent a long real period (demo only; production uses 1).
"""

from __future__ import annotations

import time
from datetime import datetime


class SiteClock:
    def __init__(self, start: str | None = None, time_scale: float = 1.0):
        if start:
            self.start_epoch = datetime.fromisoformat(str(start)).timestamp()
        else:
            self.start_epoch = time.time()
        self.time_scale = float(time_scale)

    def business(self, media_t: float) -> float:
        """Epoch seconds on the site clock for a media timestamp."""
        return self.start_epoch + media_t * self.time_scale

    def span(self, media_seconds: float) -> float:
        """Business-time length of a media-time interval."""
        return media_seconds * self.time_scale

    @staticmethod
    def hms(epoch: float | None) -> str:
        return "--:--:--" if epoch is None else datetime.fromtimestamp(epoch).strftime("%H:%M:%S")

    @staticmethod
    def stamp(epoch: float | None) -> str | None:
        return None if epoch is None else datetime.fromtimestamp(epoch).isoformat(timespec="seconds")

    @staticmethod
    def dur(seconds: float | None) -> str:
        if seconds is None:
            return "-"
        s = int(round(max(0.0, seconds)))
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m:02d}m"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

"""Zero-dependency web dashboard: live MJPEG camera views + JSON state + remote event intake.

    GET  /                  dashboard page
    GET  /api/state         capacity, sessions, alerts, plate-reader stats (JSON)
    GET  /stream/<cam>      MJPEG of an annotated camera (or `mosaic`)
    GET  /evidence/<file>   plate crops and overstay snapshots
    POST /api/events        JSON event(s) from a remote edge device (same schema as events.jsonl)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

log = logging.getLogger("parkwatch.web")
HERE = Path(__file__).resolve().parent


class Dashboard:
    def __init__(self, host: str, port: int, out_dir: Path, central):
        self.host, self.port, self.out_dir, self.central = host, port, out_dir, central
        self._lock = threading.Lock()
        self.state: dict = {}
        self.frames: dict = {}
        self.version = 0
        self.viewers: Counter = Counter()
        self.remote: queue.Queue = queue.Queue()
        self.keep_alive = False
        self._jpeg: dict = {}

    # -------------------------------------------------------------- called by the runner
    def start(self) -> None:
        dash = self

        class Handler(_Handler):
            dashboard = dash

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="dashboard").start()
        shown = "localhost" if self.host in ("0.0.0.0", "127.0.0.1") else self.host
        log.info("dashboard: http://%s:%d", shown, self.port)

    def publish(self, snap: dict, plates: dict, views: dict) -> None:
        with self._lock:
            self.state = {"site": snap, "plates": plates, "cameras": sorted(views)}
            for k, v in views.items():
                self.frames[k] = v
            self.version += 1

    def publish_mosaic(self, m) -> None:
        with self._lock:
            self.frames["mosaic"] = m

    def has_viewers(self, key: str | None = None) -> bool:
        return bool(self.viewers[key]) if key else any(self.viewers.values())

    def drain_remote_events(self) -> list[dict]:
        out = []
        while not self.remote.empty():
            out.append(self.remote.get_nowait())
        return out

    def wait(self) -> None:
        log.info("footage finished; dashboard stays up - Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # -------------------------------------------------------------- used by handlers
    def jpeg(self, key: str) -> tuple[int, bytes | None]:
        with self._lock:
            frame, ver = self.frames.get(key), self.version
        if frame is None:
            return ver, None
        cached = self._jpeg.get(key)
        if cached and cached[0] == ver:
            return cached
        maxw = 1600 if key == "mosaic" else 960
        if frame.shape[1] > maxw:
            frame = cv2.resize(frame, (maxw, int(frame.shape[0] * maxw / frame.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        item = (ver, buf.tobytes() if ok else None)
        self._jpeg[key] = item
        return item


class _Handler(BaseHTTPRequestHandler):
    dashboard: Dashboard = None  # type: ignore[assignment]

    def log_message(self, *_):  # silence default access log
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        d, path = self.dashboard, self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            with d._lock:
                body = json.dumps(d.state, default=str).encode()
            return self._send(200, body, "application/json")
        if path.startswith("/evidence/"):
            name = Path(path.split("/", 2)[2]).name  # no directory traversal
            f = d.out_dir / "evidence" / name
            if f.is_file():  # evidence files never change: let the browser cache them
                return self._send(200, f.read_bytes(), "image/jpeg", "private, max-age=86400")
            return self._send(404, b"not found", "text/plain")
        if path.startswith("/stream/"):
            return self._stream(path.split("/", 2)[2])
        return self._send(404, b"not found", "text/plain")

    def _stream(self, key: str) -> None:
        d = self.dashboard
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        d.viewers[key] += 1
        last, sent_at = -1, 0.0
        try:
            while True:
                ver, jpg = d.jpeg(key)
                # browsers paint a part when the next one arrives: re-send at least every second
                if jpg is not None and (ver != last or time.time() - sent_at > 1.0):
                    last, sent_at = ver, time.time()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.06)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            d.viewers[key] -= 1

    def do_POST(self):  # noqa: N802
        if self.path != "/api/events":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"[]")
            events = body if isinstance(body, list) else [body]
            for ev in events:
                if not isinstance(ev, dict) or "type" not in ev:
                    raise ValueError("each event needs a 'type'")
                self.dashboard.remote.put(ev)
            return self._send(202, json.dumps({"accepted": len(events)}).encode(), "application/json")
        except (ValueError, TypeError) as e:
            return self._send(400, str(e).encode(), "text/plain")

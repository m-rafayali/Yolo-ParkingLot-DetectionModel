"""Runs all cameras on one clock, feeds the central service, and drives outputs (window, video, web)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

from . import annotate
from .alerts import AlertManager
from .central import CentralService
from .clock import SiteClock
from .config import resolve_path
from .detector import Detector
from .edge import GateCamera, LotCamera
from .plates import PlateFormat, PlateService
from .sources import FileSource, LiveSource
from .store import Store

log = logging.getLogger("parkwatch")
REMOTE_TYPES = {"entry", "exit", "plate", "lot_arrival", "parked", "unparked", "lot_departure", "lot_lost"}


class Camera:
    def __init__(self, edge, source):
        self.edge, self.source = edge, source
        self.id, self.role = edge.id, edge.role
        self.last_processed = -10**9
        self.view = None  # latest annotated frame
        self.fps = 0.0
        self._t = None


class Runner:
    def __init__(self, cfg: dict, realtime: bool = False, speed: float = 1.0, show: bool = False,
                 save_video: bool | None = None, web: bool | None = None, max_seconds: float | None = None,
                 out_dir: str | None = None):
        self.cfg, self.realtime, self.speed, self.show, self.max_seconds = cfg, realtime, speed, show, max_seconds
        live = any(str(c["source"]).isdigit() or "://" in str(c["source"]) for c in cfg["cameras"])
        ts = float(cfg["clock"].get("time_scale", 1.0))
        if live and ts != 1.0:
            log.warning("live sources: forcing clock.time_scale = 1")
            ts = 1.0
        self.live = live
        self.realtime = realtime or live
        self.clock = SiteClock(None if live else cfg["clock"].get("start"), ts)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = Path(out_dir) if out_dir else resolve_path(cfg["output"]["dir"]) / stamp
        (self.out / "evidence").mkdir(parents=True, exist_ok=True)
        self.store = Store(self.out)
        self.alerts = AlertManager(cfg["alerts"], self.out)
        self.central = CentralService(cfg, self.clock, self.store, self.alerts)
        self.central.snapshot_fn = self._lot_snapshot
        self.plates = PlateService(cfg["plates"], PlateFormat(cfg["plates"].get("format", "generic")))
        self._events: list[dict] = []
        t0 = time.time()
        self.cams: list[Camera] = []
        models = cfg["models"]
        for c in cfg["cameras"]:
            det = Detector(models[c.get("model", "ground")], device=models.get("device"), tracker=models.get("tracker"))
            log.info("camera %-6s role=%-5s model=%s  source=%s", c["id"], c["role"], Path(det.weights).name, c["source"])
            src = LiveSource(c["source"], t0) if (str(c["source"]).isdigit() or "://" in str(c["source"])) \
                else FileSource(c["source"])
            if c["role"] == "lot":
                edge = LotCamera(c, det, self.clock, self._events.append, self.out / "evidence")
            else:
                edge = GateCamera(c, det, self.clock, self._events.append, self.out / "evidence", self.plates)
            self.cams.append(Camera(edge, src))
        self.lot = next((c.edge for c in self.cams if c.role == "lot"), None)
        self.base_fps = max(c.source.fps for c in self.cams)
        self.save_video = cfg["output"].get("save_video", True) if save_video is None else save_video
        self.writer = None
        self.web = None
        if (cfg["dashboard"].get("enabled", True) if web is None else web):
            from .dashboard.server import Dashboard

            d = cfg["dashboard"]
            self.web = Dashboard(d.get("host", "127.0.0.1"), int(d.get("port", 8000)), self.out, self.central)
            self.web.start()

    # ------------------------------------------------------------------ helpers
    def _lot_snapshot(self, lid: str, kind: str) -> str | None:
        if self.lot is None:
            return None
        crop = self.lot.crop_of(lid)
        if crop is None:
            return None
        name = f"alert_{kind}_{lid}_{int(self.central.now_mt)}.jpg"
        cv2.imwrite(str(self.out / "evidence" / name), crop)
        return name

    def _process_camera(self, cam: Camera, t: float, need_view: bool) -> bool:
        src = cam.source
        new = src.advance_to(t, decode=True)
        if not new:
            return False
        mt = src.mt
        idx = src.idx
        if idx - cam.last_processed >= cam.edge.stride or src.live:
            t0 = time.time()
            cam.edge.process(src.frame, mt)
            cam.last_processed = idx
            dt = time.time() - t0
            cam.fps = 0.9 * cam.fps + 0.1 * (1.0 / max(dt, 1e-3)) if cam.fps else 1.0 / max(dt, 1e-3)
        if need_view:
            draw = annotate.draw_lot if cam.role == "lot" else annotate.draw_gate
            cam.view = draw(src.frame, cam.edge, self.central, self.clock.hms(self.central.now_t), cam.fps)
        return True

    def _drain(self, t: float) -> None:
        for ref, cam_id, res in self.plates.poll():
            ev = {"type": "plate", "ref": ref, "cam": cam_id, "mt": round(t, 3), "t": self.clock.business(t), **res.to_dict()}
            state = "valid" if res.valid else ("unverified" if res.plate else "unreadable")
            log.info("plate %-12s %-10s %-10s via %s%s", ref, res.plate or "-", state, res.source,
                     f" ({res.error})" if res.error else "")
            self._events.append(ev)
        if self._events:
            evs = sorted(self._events, key=lambda e: e.get("mt", 0))
            self._events.clear()
            for ev in evs:
                if ev["type"] in ("entry", "exit", "parked", "lot_arrival", "lot_departure", "unparked"):
                    log.info("%-13s %-6s %s", ev["type"], ev["cam"],
                             {k: ev[k] for k in ("event_id", "lid", "bay") if k in ev})
                self.central.handle(ev)
        if self.web:
            for ev in self.web.drain_remote_events():  # events POSTed by remote edge devices
                if ev.get("type") not in REMOTE_TYPES:
                    log.warning("ignored remote event of type %r", ev.get("type"))
                    continue
                ev.setdefault("mt", t)
                ev.setdefault("t", self.clock.business(ev["mt"]))
                try:
                    self.central.handle(ev)
                except (KeyError, TypeError, ValueError) as e:
                    log.warning("bad remote event %s: %s", ev, e)

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        dt = 1.0 / self.base_fps
        t, i = 0.0, 0
        wall0 = time.time()
        end = min((c.source.duration for c in self.cams), default=0)
        if self.max_seconds:
            end = min(end, self.max_seconds)
        log.info("running %s  (%s)", "LIVE" if self.live else f"{end:.0f}s of footage",
                 "real-time pacing" if self.realtime else "as fast as possible")
        last_print = 0.0
        try:
            while True:
                if self.realtime:
                    t = (time.time() - wall0) * self.speed
                else:
                    t = i * dt
                    i += 1
                if t > end:
                    break
                need_view = bool(self.show or self.save_video or (self.web and self.web.has_viewers()))
                fresh = [self._process_camera(cam, t, need_view) for cam in self.cams]
                if all(c.source.done for c in self.cams):
                    break
                if self.realtime and not any(fresh):
                    time.sleep(0.005)  # nothing new yet: don't spin
                self._drain(t)
                self.central.tick(t)
                if need_view:
                    self._output(t)
                if time.time() - last_print > 5:
                    last_print = time.time()
                    s = self.central
                    log.info("t=%6.1fs  site %s  free %d/%d  in %d out %d  overstay %d  | %s", t,
                             self.clock.hms(s.now_t), s.free, s.capacity, s.entries, s.exits,
                             sum(1 for x in s.sessions.values() if x.status == "OVERSTAY"),
                             "  ".join(f"{c.id}:{c.fps:.1f}fps" for c in self.cams))
                if self.show and (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            log.info("stopped by user")
        return self._finish(t)

    def _output(self, t: float) -> None:
        views = {c.id: c.view for c in self.cams if c.view is not None}
        roles = {c.id: c.role for c in self.cams}
        snap = self.central.snapshot()
        plates = self.plates.summary()
        if self.web:
            self.web.publish(snap, plates, views)
        if self.show or self.save_video or (self.web and self.web.has_viewers("mosaic")):
            m = annotate.mosaic(views, roles, snap, plates)
            if self.web:
                self.web.publish_mosaic(m)
            if self.save_video:
                if self.writer is None:
                    self.writer = cv2.VideoWriter(str(self.out / "annotated_mosaic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                  self.base_fps, (m.shape[1], m.shape[0]))
                self.writer.write(m)
            if self.show:
                cv2.imshow("ParkWatch", cv2.resize(m, (1600, 900)))

    def _finish(self, t: float) -> dict:
        log.info("footage finished - waiting for outstanding plate reads")
        self.plates.wait_idle(60)
        self._drain(t)
        self.central.tick(t)
        self.central.flush()
        rows = self.central.export_rows()
        self.store.export_csv(rows, self.out / "sessions.csv")
        summary = {"site": self.central.snapshot(), "plates": self.plates.summary(), "sessions": rows,
                   "time_scale": self.clock.time_scale, "start_epoch": self.clock.start_epoch}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        if self.writer is not None:
            self.writer.release()
        self.store.close()
        self.alerts.close()
        self.plates.close()
        for c in self.cams:
            c.source.close()
        if self.show:
            cv2.destroyAllWindows()
        report = print_report(self.central, self.plates.summary())
        if self.cfg.get("ground_truth") and not self.live:
            from .evaluate import evaluate

            evaluate(self.cfg, self.central, self.out)
        log.info("outputs in %s", self.out)
        if self.web and self.web.keep_alive:
            for c in self.cams:  # make sure the held dashboard shows the final annotated frames
                if c.source.frame is not None:
                    draw = annotate.draw_lot if c.role == "lot" else annotate.draw_gate
                    c.view = draw(c.source.frame, c.edge, self.central, self.clock.hms(self.central.now_t), c.fps)
            self.web.publish(self.central.snapshot(), self.plates.summary(), {c.id: c.view for c in self.cams if c.view is not None})
            self.web.wait()
        return report


def print_report(central: CentralService, plates: dict) -> dict:
    clk = central.clock
    print("\n" + "=" * 96)
    print(f"{'VID':6} {'PLATE':10} {'IN':9} {'OUT':9} {'DURATION':10} {'BAY':5} {'STATUS':16} {'MATCH':13} NOTE")
    print("-" * 96)
    for s in central.sessions.values():
        if s.status == "DUPLICATE":
            continue
        d = s.duration(central.now_t)
        dur = clk.dur(d) + ("+" if s.start_t is None and d is not None else "")
        print(f"{s.vid:6} {s.plate or '-':10} {clk.hms(s.entry_t) if s.entry_t else '-':9} "
              f"{clk.hms(s.exit_t) if s.exit_t else '-':9} {dur:10} {s.bay or '-':5} {s.status:16} "
              f"{s.matched_by or '-':13} {(s.notes[-1] if s.notes else '')[:40]}")
    print("-" * 96)
    llm = plates.get("llm", {})
    print(f"capacity {central.capacity}  occupied {central.occupied}  free {central.free}  "
          f"entries {central.entries}  exits {central.exits}  alerts {len(central.alert_log)}")
    print(f"plate reads: {plates['vehicles']}  LLM calls ok={llm.get('ok', 0)} errors={llm.get('errors', 0)} "
          f"est. cost ${plates.get('llm_cost_usd', 0):.4f}")
    print("=" * 96)
    return {"free": central.free, "entries": central.entries, "exits": central.exits}


"""Edge camera roles. Each turns frames into small JSON events; no identity logic lives here.

* GateCamera (entry / exit): a tracked vehicle crossing the counting line in the configured
  direction emits ONE event and ONE plate-read request with its best crops.
* LotCamera (overhead): a vehicle that stays still for N seconds becomes PARKED (with its bay);
  lost tracker ids are re-attached by *position* (a parked car cannot move), which is what
  keeps tracker-id churn from turning into phantom vehicles.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .clock import SiteClock
from .detector import Det, Detector
from .geometry import DIRECTIONS, crossed_line, point_in_poly, poly_centroid, scale_points

Emit = Callable[[dict], None]


def crop_box(frame: np.ndarray, xyxy, pad: float = 0.06, min_w: int = 400) -> np.ndarray:
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))
    c = frame[y0:y1, x0:x1].copy()
    if c.size and c.shape[1] < min_w:
        s = min_w / c.shape[1]
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return c


class _Base:
    def __init__(self, cfg: dict, detector: Detector, clock: SiteClock, emit: Emit, evidence_dir: Path):
        self.cfg, self.id, self.role, self.name = cfg, cfg["id"], cfg["role"], cfg.get("name", cfg["id"])
        self.detector, self.clock, self.emit, self.evidence_dir = detector, clock, emit, evidence_dir
        self.stride = int(cfg.get("stride", 1))
        self.last_dets: list[Det] = []
        self.last_frame: np.ndarray | None = None
        self._scaled_for: tuple | None = None

    def _event(self, etype: str, mt: float, **kw) -> dict:
        ev = {"type": etype, "cam": self.id, "role": self.role, "mt": round(mt, 3), "t": self.clock.business(mt)}
        ev.update(kw)
        self.emit(ev)
        return ev

    def _fit_geometry(self, frame: np.ndarray) -> None:
        """Scale pixel geometry if the stream resolution differs from the one it was drawn on."""
        H, W = frame.shape[:2]
        if self._scaled_for == (W, H):
            return
        self._scaled_for = (W, H)
        gw, gh = self.cfg.get("size") or (W, H)
        self._rescale(W / gw, H / gh)

    def _rescale(self, sx: float, sy: float) -> None:  # pragma: no cover - overridden
        pass


# ============================================================================ gate cameras
@dataclass
class _GateTrack:
    prev: tuple | None = None
    first: tuple | None = None
    last_xyxy: np.ndarray | None = None
    last_mt: float = 0.0
    crops: list = field(default_factory=list)  # [(score, mt, image)]
    event_id: str | None = None


class GateCamera(_Base):
    MAX_CROPS = 3

    def __init__(self, cfg, detector, clock, emit, evidence_dir, plates):
        super().__init__(cfg, detector, clock, emit, evidence_dir)
        if "line" not in cfg:
            raise ValueError(f"camera {self.id}: a gate camera needs a counting `line` (use tools/draw_zones.py)")
        self.line0 = cfg["line"]
        self.line = self.line0
        self.direction = cfg.get("direction", "any")
        self.plates = plates
        self.tracks: dict[int, _GateTrack] = {}
        self.recent: deque = deque(maxlen=20)  # (mt, point) of recent crossings, for duplicate suppression
        self.count = 0
        self._seq = itertools.count(1)

    def _rescale(self, sx, sy):
        self.line = scale_points(self.line0, sx, sy)

    def process(self, frame: np.ndarray, mt: float) -> list[Det]:
        self._fit_geometry(frame)
        self.last_frame = frame
        H, W = frame.shape[:2]
        dets = self.detector.track(frame)
        for d in dets:
            st = self.tracks.setdefault(d.tid, _GateTrack())
            st.last_mt = mt
            ref = d.bottom_center
            x0, y0, x1, y1 = d.xyxy
            if st.event_id is None:  # keep the best views until the vehicle is counted
                touching = x0 < 3 or y0 < 3 or x1 > W - 3 or y1 > H - 3
                score = (x1 - x0) * (y1 - y0) * (0.35 if touching else 1.0)
                if len(st.crops) < self.MAX_CROPS or score > st.crops[-1][0]:
                    st.crops.append((score, mt, crop_box(frame, d.xyxy)))
                    st.crops.sort(key=lambda c: -c[0])
                    del st.crops[self.MAX_CROPS:]
            if st.prev is not None and st.event_id is None and crossed_line(self.line, st.prev, ref, self.direction):
                if not self._duplicate(mt, ref):
                    self._count(d, st, mt, ref)
                else:
                    st.event_id = "duplicate"
            st.prev = ref
            st.first = st.first or ref
            st.last_xyxy = d.xyxy
        for tid, st in list(self.tracks.items()):
            if st.event_id is None and st.prev is not None and mt - st.last_mt > 0.6 and self._left_through_line(st, W, H):
                self._count(None, st, st.last_mt, st.prev, tid)
            if mt - st.last_mt > 5.0:
                del self.tracks[tid]
        self.last_dets = dets
        return dets

    def _left_through_line(self, st: _GateTrack, W: int, H: int) -> bool:
        """Lost right before the line while moving the counted way: it went through.

        Detectors often drop a vehicle when it is very close or cut off by the frame edge, exactly
        where the line is. Counting it (at worst a few seconds early) beats missing the visit;
        a genuine double count is caught later by the duplicate-plate check.
        """
        margin = float(self.cfg.get("lost_near_line_px", 60))
        x0, y0, x1, y1 = st.last_xyxy
        at_edge = x0 < 4 or y0 < 4 or x1 > W - 4 or y1 > H - 4
        a, b = np.asarray(self.line[0], float), np.asarray(self.line[1], float)
        p = np.asarray(st.prev, float)
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        dist = float(np.linalg.norm(p - (a + t * (b - a))))
        near = dist <= 0.4 * margin or (at_edge and dist <= margin)
        moved = np.asarray(st.prev, float) - np.asarray(st.first, float)
        if self.direction == "any":
            toward = np.linalg.norm(moved) > 20
        else:
            dx, dy = DIRECTIONS[self.direction]
            toward = moved[0] * dx + moved[1] * dy > 20
        return bool(near and toward)

    def _duplicate(self, mt, pt) -> bool:
        dt, dpx = float(self.cfg.get("dedup_seconds", 5.0)), float(self.cfg.get("dedup_px", 150))
        return any(mt - m < dt and np.hypot(pt[0] - p[0], pt[1] - p[1]) < dpx for m, p in self.recent)

    def _count(self, d: Det | None, st: _GateTrack, mt: float, ref, tid: int | None = None) -> None:
        n = next(self._seq)
        st.event_id = f"{self.id}-{n:04d}"
        self.count += 1
        self.recent.append((mt, ref))
        xyxy = d.xyxy if d is not None else st.last_xyxy
        crops = [c[2] for c in st.crops] or [crop_box(self.last_frame, xyxy)]
        ev_path = self.evidence_dir / f"{st.event_id}.jpg"
        cv2.imwrite(str(ev_path), crops[0])
        self._event(self.role, mt, event_id=st.event_id, track=d.tid if d is not None else tid,
                    bbox=[round(float(v), 1) for v in xyxy], evidence=ev_path.name,
                    **({} if d is not None else {"how": "lost-at-line"}))
        self.plates.submit(st.event_id, self.id, crops)

    def event_for_track(self, tid: int) -> str | None:
        st = self.tracks.get(tid)
        return st.event_id if st and st.event_id != "duplicate" else None


# ============================================================================ overhead lot camera
@dataclass
class LotIdentity:
    lid: str
    origin: str  # startup | arrival | appeared
    first_mt: float
    hist: deque = field(default_factory=lambda: deque(maxlen=400))
    state: str = "MOVING"  # MOVING | PARKED | GONE
    last_mt: float = 0.0
    last_pt: tuple = (0.0, 0.0)
    last_xyxy: np.ndarray | None = None
    last_poly: np.ndarray | None = None
    anchor: tuple | None = None
    parked_since_mt: float | None = None
    bay: str | None = None
    moving_since: float | None = None
    announced: bool = False
    in_entry_zone_at_birth: bool = False

    def speed(self) -> float:
        if len(self.hist) < 2:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = self.hist[-min(len(self.hist), 5)], self.hist[-1]
        return float(np.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-3))

    def predicted(self, mt: float) -> tuple:
        if len(self.hist) < 2:
            return self.last_pt
        (t0, x0, y0), (t1, x1, y1) = self.hist[-min(len(self.hist), 5)], self.hist[-1]
        k = (mt - t1) / max(t1 - t0, 1e-3)
        return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k


class LotCamera(_Base):
    def __init__(self, cfg, detector, clock, emit, evidence_dir):
        super().__init__(cfg, detector, clock, emit, evidence_dir)
        self.bays0 = cfg.get("bays") or {}
        self.entry_zone0, self.exit_zone0 = cfg.get("entry_zone"), cfg.get("exit_zone")
        self.bays, self.entry_zone, self.exit_zone = self.bays0, self.entry_zone0, self.exit_zone0
        self.idents: dict[str, LotIdentity] = {}
        self.tid2lid: dict[int, str] = {}
        self._seq = itertools.count(1)
        self.tracker_ids_seen: set[int] = set()

    def _rescale(self, sx, sy):
        self.bays = {k: scale_points(v, sx, sy) for k, v in self.bays0.items()}
        self.entry_zone = scale_points(self.entry_zone0, sx, sy) if self.entry_zone0 else None
        self.exit_zone = scale_points(self.exit_zone0, sx, sy) if self.exit_zone0 else None

    # ------------------------------------------------------------------ helpers
    def bay_at(self, pt) -> str | None:
        for name, poly in self.bays.items():
            if point_in_poly(pt, poly):
                return name
        best, bd = None, 1e9
        for name, poly in self.bays.items():
            c = poly_centroid(poly)
            d = np.hypot(pt[0] - c[0], pt[1] - c[1])
            if d < bd:
                best, bd = name, d
        return best if bd < 40 else None

    def occupied_bays(self) -> set[str]:
        return {i.bay for i in self.idents.values() if i.state == "PARKED" and i.bay}

    def live(self) -> list[LotIdentity]:
        return [i for i in self.idents.values() if i.state != "GONE"]

    def crop_of(self, lid: str) -> np.ndarray | None:
        i = self.idents.get(lid)
        if i is None or i.last_xyxy is None or self.last_frame is None:
            return None
        return crop_box(self.last_frame, i.last_xyxy, pad=0.25, min_w=240)

    # ------------------------------------------------------------------ main loop
    def process(self, frame: np.ndarray, mt: float) -> list[Det]:
        self._fit_geometry(frame)
        self.last_frame = frame
        dets = self.detector.track(frame)
        claimed: dict[str, Det] = {}
        fresh: list[Det] = []
        for d in dets:  # pass 1: tracker ids we already know
            self.tracker_ids_seen.add(d.tid)
            lid = self.tid2lid.get(d.tid)
            if lid and self.idents[lid].state != "GONE" and lid not in claimed:
                claimed[lid] = d
            else:
                fresh.append(d)
        for d in fresh:  # pass 2: new tracker ids -> re-identify by position, else new identity
            alias = self._alias_of(d, claimed)
            if alias is not None:  # second box on a car already tracked this frame (e.g. mid-turn)
                self.tid2lid[d.tid] = alias
                continue
            lid = self._reidentify(d, mt, claimed) or self._new_identity(d, mt)
            self.tid2lid[d.tid] = lid
            claimed[lid] = d
        for lid, d in claimed.items():
            i = self.idents[lid]
            c = d.center
            i.hist.append((mt, c[0], c[1]))
            i.last_mt, i.last_pt, i.last_xyxy, i.last_poly = mt, c, d.xyxy, d.poly
        self._update(mt, set(claimed))
        self.last_dets = dets
        return dets

    def _alias_of(self, d: Det, claimed: dict) -> str | None:
        """A new tracker id whose box sits on top of an identity already seen this frame is the same car."""
        c = d.center
        r = 0.8 * float(self.cfg.get("reid_radius_px", 45))
        for lid, other in claimed.items():
            oc = other.center
            if np.hypot(c[0] - oc[0], c[1] - oc[1]) < r:
                return lid
        return None

    def _reidentify(self, d: Det, mt: float, claimed: dict) -> str | None:
        c = d.center
        r0 = float(self.cfg.get("reid_radius_px", 45))
        best, bd = None, 1e9
        for i in self.idents.values():
            if i.state == "GONE" or i.lid in claimed or i.last_mt >= mt:
                continue
            if i.state == "PARKED":
                ref, radius = i.anchor, r0
            else:
                gap = mt - i.last_mt
                if gap > float(self.cfg.get("moving_lost_seconds", 3.0)):
                    continue
                ref, radius = i.predicted(mt), min(r0 + i.speed() * gap, 4 * r0)
            dist = np.hypot(c[0] - ref[0], c[1] - ref[1])
            if dist <= radius and dist < bd:
                best, bd = i.lid, dist
        return best

    def _new_identity(self, d: Det, mt: float) -> str:
        lid = f"L{next(self._seq):04d}"
        c = d.center
        in_entry = bool(self.entry_zone) and point_in_poly(c, self.entry_zone)
        if mt <= float(self.cfg.get("startup_grace_seconds", 3.0)):
            origin = "startup"
        else:
            origin = "arrival" if in_entry else "appeared"
        self.idents[lid] = LotIdentity(lid, origin, mt, in_entry_zone_at_birth=in_entry)
        return lid

    def _update(self, mt: float, seen: set[str]) -> None:
        win = float(self.cfg.get("stationary_seconds", 4.0))
        still_px = float(self.cfg.get("stationary_px", 10))
        moved_px = float(self.cfg.get("moved_px", 30))
        for i in list(self.idents.values()):
            if i.state == "GONE":
                continue
            if i.lid in seen:
                if not i.announced and i.origin == "arrival" and len(i.hist) >= 3:
                    i.announced = True
                    self._event("lot_arrival", i.first_mt, lid=i.lid)
                if i.state == "MOVING":
                    pts = [(t, x, y) for t, x, y in i.hist if t >= mt - win]
                    if pts and pts[0][0] <= mt - 0.85 * win and len(pts) >= 3:
                        xy = np.array([(x, y) for _, x, y in pts])
                        if np.abs(xy - xy.mean(0)).max() <= still_px:
                            i.state, i.anchor = "PARKED", tuple(xy.mean(0))
                            i.parked_since_mt = i.first_mt if i.origin == "startup" else pts[0][0]
                            i.bay = self.bay_at(i.anchor)
                            i.moving_since = None
                            self._event("parked", mt, lid=i.lid, bay=i.bay, since_mt=round(i.parked_since_mt, 3),
                                        since_t=self.clock.business(i.parked_since_mt), origin=i.origin)
                elif i.state == "PARKED":
                    if np.hypot(i.last_pt[0] - i.anchor[0], i.last_pt[1] - i.anchor[1]) > moved_px:
                        i.moving_since = i.moving_since or mt
                        if mt - i.moving_since >= 0.6:
                            self._event("unparked", mt, lid=i.lid, bay=i.bay,
                                        parked_s=self.clock.span(mt - i.parked_since_mt))
                            i.state, i.anchor, i.moving_since = "MOVING", None, None
                    else:
                        i.moving_since = None
            else:
                gap = mt - i.last_mt
                if i.state == "MOVING" and gap > float(self.cfg.get("moving_lost_seconds", 3.0)):
                    i.state = "GONE"
                    if self.exit_zone and point_in_poly(i.last_pt, self.exit_zone):
                        self._event("lot_departure", i.last_mt, lid=i.lid)
                    else:  # vanished mid-lot while moving: the central service may re-link it
                        self._event("lot_lost", i.last_mt, lid=i.lid)
                elif i.state == "PARKED" and gap > float(self.cfg.get("parked_lost_seconds", 120.0)):
                    self._event("unparked", mt, lid=i.lid, bay=i.bay, parked_s=self.clock.span(mt - i.parked_since_mt),
                                reason="lost")
                    i.state = "GONE"

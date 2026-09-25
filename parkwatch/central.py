"""Central service: the only place that owns identity, timers, counting and alerts.

Inputs are small JSON events from the edge cameras (and plate results). Identity is held in
three layers, most reliable first:
    1. plate (read once at a gate)          -> joins entry and exit
    2. lot position (a parked car can't move) -> joins tracker fragments on the lot camera
    3. tracker id                           -> only inside one camera
Entry sessions are joined to lot identities by arrival order (FIFO within a time window),
and lot departures back up the exit match when the exit plate is unreadable.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import asdict, dataclass, field

from .clock import SiteClock
from .plates.formats import best_match

OPEN_STATES = ("ENTERED", "PARKED", "LEAVING", "OVERSTAY")


@dataclass
class Session:
    vid: str
    origin: str  # gate | lot-startup | lot-arrival | lot-appeared | exit-only
    plate: str | None = None
    plate_conf: float = 0.0
    plate_valid: bool = False
    plate_source: str | None = None
    entry_event: str | None = None
    entry_mt: float | None = None
    entry_t: float | None = None
    entry_evidence: str | None = None
    lot_id: str | None = None
    bay: str | None = None
    parked_since_t: float | None = None
    parked_since_known: bool = True
    first_seen_t: float | None = None
    exit_event: str | None = None
    exit_mt: float | None = None
    exit_t: float | None = None
    exit_evidence: str | None = None
    exit_plate: str | None = None
    matched_by: str | None = None
    status: str = "ENTERED"
    overstay_alert_t: float | None = None
    notes: list = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.status in OPEN_STATES

    @property
    def start_t(self) -> float | None:
        if self.entry_t is not None:
            return self.entry_t
        return self.parked_since_t if self.parked_since_known else None

    def duration(self, now_t: float) -> float | None:
        s = self.start_t if self.start_t is not None else self.first_seen_t
        if s is None:
            return None
        return (self.exit_t if self.exit_t is not None else now_t) - s


class CentralService:
    def __init__(self, cfg: dict, clock: SiteClock, store=None, alerts=None):
        site = cfg["site"]
        self.name = site.get("name", "Car park")
        self.capacity = int(site["capacity"])
        self.initial = int(site.get("initial_occupancy", 0))
        self.limit_s = float(site["overstay_limit_min"]) * 60
        self.alert_unknown = bool(site.get("alert_unknown_arrival", False))
        self.repeat_s = float(cfg["alerts"].get("repeat_every_min", 0) or 0) * 60
        lk = cfg["linking"]
        self.win_arrival = tuple(lk["entry_to_lot_s"])
        self.win_departure = tuple(lk["lot_to_exit_s"])
        self.fuzzy = int(lk.get("fuzzy_plate_edits", 1))
        self.plate_timeout = 30.0
        self.clock, self.store, self.alerts = clock, store, alerts
        self.sessions: dict[str, Session] = {}
        self.by_event: dict[str, str] = {}
        self.by_lot: dict[str, str] = {}
        self.entries = self.exits = 0
        self.pending_arrivals: deque = deque()  # (mt, lid)
        self.pending_departures: deque = deque()  # (mt, lid)
        self.pending_exits: dict[str, dict] = {}
        self.lost_moving: list = []  # (mt, lid) lot identities lost while driving
        self.now_mt = 0.0
        self.alert_log: list[dict] = []
        self._vid = itertools.count(1)
        self._was_full = False
        self.snapshot_fn = None  # (lot_id, kind) -> evidence filename, set by the runner

    # ------------------------------------------------------------------ helpers
    @property
    def now_t(self) -> float:
        return self.clock.business(self.now_mt)

    @property
    def occupied(self) -> int:
        return max(0, min(self.capacity, self.initial + self.entries - self.exits))

    @property
    def free(self) -> int:
        return self.capacity - self.occupied

    def _new(self, origin: str, **kw) -> Session:
        s = Session(f"V{next(self._vid):04d}", origin, **kw)
        self.sessions[s.vid] = s
        self._save(s)
        return s

    def _save(self, s: Session) -> None:
        if self.store:
            self.store.upsert_session(s, self)

    def _alert(self, kind: str, s: Session | None, message: str, **extra) -> None:
        a = {"kind": kind, "t": self.now_t, "time": self.clock.hms(self.now_t), "vid": s.vid if s else None,
             "plate": s.plate if s else None, "bay": s.bay if s else None, "lot_id": s.lot_id if s else None,
             "message": message, **extra}
        if s is not None:
            a["evidence"] = [e for e in (s.entry_evidence, s.exit_evidence) if e]
            if s.lot_id and self.snapshot_fn and kind == "overstay":
                snap = self.snapshot_fn(s.lot_id, kind)
                if snap:
                    a["evidence"].append(snap)
        self.alert_log.append(a)
        if self.alerts:
            self.alerts.send(a)
        if self.store:
            self.store.add_alert(a)

    # ------------------------------------------------------------------ event intake
    def handle(self, ev: dict) -> None:
        if self.store:
            self.store.add_event(ev)
        getattr(self, f"_on_{ev['type']}", lambda e: None)(ev)

    def _on_entry(self, ev: dict) -> None:
        self.entries += 1
        s = self._new("gate", entry_event=ev["event_id"], entry_mt=ev["mt"], entry_t=ev["t"],
                      entry_evidence=ev.get("evidence"))
        self.by_event[ev["event_id"]] = s.vid
        self._link_arrivals()

    def _on_exit(self, ev: dict) -> None:
        self.exits += 1
        ev = dict(ev, received_mt=self.now_mt)
        self.pending_exits[ev["event_id"]] = ev

    def _on_plate(self, ev: dict) -> None:
        ref = ev["ref"]
        if ref in self.pending_exits:
            self._resolve_exit(self.pending_exits.pop(ref), ev)
            return
        vid = self.by_event.get(ref)
        if vid is None:
            return
        s = self.sessions[vid]
        if ev.get("plate"):
            dup = next((o for o in self.sessions.values() if o is not s and o.open and o.plate == ev["plate"]
                        and o.entry_mt is not None and abs(o.entry_mt - (s.entry_mt or 0)) < 60), None)
            if dup is not None:  # same car counted twice at the gate (tracker re-acquired it)
                self.entries -= 1
                s.status, s.matched_by = "DUPLICATE", dup.vid
                s.notes.append(f"duplicate of {dup.vid}")
                if s.lot_id and not dup.lot_id:  # keep the lot link on the surviving visit
                    dup.lot_id, self.by_lot[s.lot_id] = s.lot_id, dup.vid
                    self._save(dup)
                self._save(s)
                return
            for old in self.sessions.values():  # same plate still "inside": its exit was missed
                if old is not s and old.open and old.plate == ev["plate"]:
                    old.exit_t, old.exit_mt, old.status, old.matched_by = ev["t"], ev["mt"], "EXITED", "re-entry"
                    old.notes.append("exit not seen - closed when the same plate entered again (exit time is an upper bound)")
                    self._save(old)
            s.plate, s.plate_conf, s.plate_valid, s.plate_source = ev["plate"], ev.get("conf", 0), ev.get("valid", False), ev.get("source")
        else:
            s.notes.append("entry plate unreadable - review evidence image")
        self._save(s)

    def _on_lot_arrival(self, ev: dict) -> None:
        self.pending_arrivals.append((ev["mt"], ev["lid"]))
        self._link_arrivals()

    def _on_lot_lost(self, ev: dict) -> None:
        if ev["lid"] in self.by_lot:
            self.lost_moving.append((ev["mt"], ev["lid"]))

    def _relink_lost(self, lid: str, mt: float) -> bool:
        """A car the lot camera lost while it was driving re-appears as a 'new' car: give it its identity back."""
        for lost_mt, old in sorted(self.lost_moving, reverse=True):
            vid = self.by_lot.get(old)
            if vid and self.sessions[vid].open and 0 <= mt - lost_mt <= 30:
                self.lost_moving.remove((lost_mt, old))
                s = self.sessions[vid]
                s.lot_id = lid
                s.notes.append(f"lot track {old} re-linked as {lid}")
                self.by_lot[lid] = vid
                return True
        return False

    def _on_parked(self, ev: dict) -> None:
        lid = ev["lid"]
        vid = self.by_lot.get(lid)
        if vid is None and ev.get("origin") != "startup":
            self._relink_lost(lid, ev["mt"])
            vid = self.by_lot.get(lid)
        if vid is None:
            self.pending_arrivals = deque(a for a in self.pending_arrivals if a[1] != lid)
            known = ev.get("origin") != "startup"
            s = self._new(f"lot-{ev.get('origin', 'appeared')}", lot_id=lid, parked_since_known=known,
                          first_seen_t=ev["since_t"])
            s.notes.append("arrived before system start - arrival time unknown" if not known
                           else "no gate entry matched - plate unknown")
            self.by_lot[lid] = s.vid
        s = self.sessions[self.by_lot[lid]]
        if not s.open:
            return
        s.bay, s.parked_since_t = ev.get("bay"), ev["since_t"]
        if s.status != "OVERSTAY":
            s.status = "PARKED"
        self._save(s)

    def _on_unparked(self, ev: dict) -> None:
        vid = self.by_lot.get(ev["lid"])
        if vid is None:
            return
        s = self.sessions[vid]
        if s.open and s.status != "OVERSTAY":
            s.status = "LEAVING"
        s.notes.append(f"left bay {ev.get('bay')} after {self.clock.dur(ev.get('parked_s'))}")
        self._save(s)

    def _on_lot_departure(self, ev: dict) -> None:
        self.pending_departures.append((ev["mt"], ev["lid"]))
        # an exit may be waiting with an unreadable plate
        for eid, xev in list(self.pending_exits.items()):
            if xev.get("plate_done"):
                self._resolve_exit(self.pending_exits.pop(eid), xev.get("plate_ev"))

    # ------------------------------------------------------------------ linking
    def _link_arrivals(self) -> None:
        lo, hi = self.win_arrival
        keep = deque()
        while self.pending_arrivals:
            mt, lid = self.pending_arrivals.popleft()
            if lid in self.by_lot:
                continue
            cands = [s for s in self.sessions.values() if s.origin == "gate" and s.lot_id is None and s.open
                     and s.entry_mt is not None and lo <= mt - s.entry_mt <= hi]
            if cands:
                s = min(cands, key=lambda x: x.entry_mt)  # first in, first to reach the lot
                s.lot_id = lid
                self.by_lot[lid] = s.vid
                self._save(s)
            elif self.now_mt - mt < hi:
                keep.append((mt, lid))
        self.pending_arrivals = keep

    def _resolve_exit(self, xev: dict, pev: dict | None, force: bool = False) -> None:
        plate = (pev or {}).get("plate")
        s, how = None, None
        open_ = [o for o in self.sessions.values() if o.open]
        if plate:
            s = next((o for o in open_ if o.plate == plate), None)
            how = "plate"
            if s is None and self.fuzzy:
                m = best_match(plate, [o.plate for o in open_ if o.plate], self.fuzzy)
                s = next((o for o in open_ if o.plate == m), None) if m else None
                how = "fuzzy-plate"
        if s is None:  # fall back to the lot camera: which car just drove out of the lot?
            lo, hi = self.win_departure
            for mt, lid in list(self.pending_departures):
                vid = self.by_lot.get(lid)
                if vid and self.sessions[vid].open and lo <= xev["mt"] - mt <= hi:
                    s, how = self.sessions[vid], "lot-departure"
                    break
            if s is None and plate is None and not force and self.now_mt - xev["received_mt"] < self.plate_timeout:
                self.pending_exits[xev["event_id"]] = dict(xev, plate_done=True, plate_ev=pev)
                return  # wait a little for the lot camera to report the departure
        if s is None:
            s = self._new("exit-only")
            s.notes.append("no entry record: arrived before system start or entry missed")
        if plate and not s.plate:
            s.plate, s.plate_conf, s.plate_valid, s.plate_source = plate, pev.get("conf", 0), pev.get("valid"), pev.get("source")
        s.exit_event, s.exit_mt, s.exit_t = xev["event_id"], xev["mt"], xev["t"]
        s.exit_evidence, s.exit_plate, s.matched_by = xev.get("evidence"), plate, how
        if plate and s.plate and plate != s.plate:
            s.notes.append(f"exit read {plate} vs entry {s.plate}")
        self.pending_departures = deque(d for d in self.pending_departures if d[1] != s.lot_id)
        dur = s.duration(s.exit_t)
        over = s.start_t is not None and dur is not None and dur > self.limit_s
        s.status = "EXITED_OVERSTAY" if over else "EXITED"
        self._save(s)
        if over:
            self._alert("overstay_exit", s, f"{s.plate or s.vid} left at {self.clock.hms(s.exit_t)} after "
                                            f"{self.clock.dur(dur)} (limit {self.clock.dur(self.limit_s)})",
                        duration_s=dur)

    # ------------------------------------------------------------------ periodic work
    def tick(self, mt: float) -> None:
        self.now_mt = mt
        now = self.now_t
        for xev in list(self.pending_exits.values()):  # plate never came back: resolve without it
            if mt - xev["received_mt"] > self.plate_timeout:
                self._resolve_exit(self.pending_exits.pop(xev["event_id"]), xev.get("plate_ev"))
        self._link_arrivals()
        for s in self.sessions.values():
            if not s.open:
                continue
            start = s.start_t
            unknown = start is None
            if unknown:
                if not self.alert_unknown or s.first_seen_t is None:
                    continue
                start = s.first_seen_t
            dur = now - start
            if dur < self.limit_s:
                continue
            due = s.overstay_alert_t is None or (self.repeat_s and now - s.overstay_alert_t >= self.repeat_s)
            if s.status != "OVERSTAY":
                s.status = "OVERSTAY"
                self._save(s)
            if due:
                s.overstay_alert_t = now
                where = f" in bay {s.bay}" if s.bay else ""
                since = f"since {self.clock.hms(start)[:5]}" + (" or earlier" if unknown else "")
                self._alert("overstay", s, f"{s.plate or s.vid} on site {self.clock.dur(dur)}{where} ({since}); "
                                           f"limit {self.clock.dur(self.limit_s)}", duration_s=dur)
        full = self.free <= 0
        if full and not self._was_full:
            self._alert("lot_full", None, f"car park full ({self.occupied}/{self.capacity})")
        self._was_full = full

    def flush(self) -> None:
        """End of footage: close exits still waiting for a plate read or a lot departure."""
        for xev in list(self.pending_exits.values()):
            self._resolve_exit(self.pending_exits.pop(xev["event_id"]), xev.get("plate_ev"), force=True)

    # ------------------------------------------------------------------ views
    def session_row(self, s: Session) -> dict:
        now = self.now_t
        d = s.duration(now)
        return {"vid": s.vid, "plate": s.plate, "plate_valid": s.plate_valid, "plate_source": s.plate_source,
                "status": s.status, "origin": s.origin, "bay": s.bay, "lot_id": s.lot_id,
                "entry": self.clock.hms(s.entry_t) if s.entry_t else None,
                "exit": self.clock.hms(s.exit_t) if s.exit_t else None,
                "parked_since": self.clock.hms(s.parked_since_t) if s.parked_since_t else None,
                "duration": self.clock.dur(d) + ("+" if s.start_t is None and d is not None else ""),
                "duration_s": d, "matched_by": s.matched_by, "entry_evidence": s.entry_evidence,
                "exit_evidence": s.exit_evidence, "notes": s.notes[-2:]}

    def label_for_lot(self, lid: str) -> tuple[str | None, str | None]:
        vid = self.by_lot.get(lid)
        if not vid:
            return None, None
        s = self.sessions[vid]
        return (s.plate or s.vid), s.status

    def label_for_event(self, event_id: str) -> str | None:
        vid = self.by_event.get(event_id)
        if vid:
            s = self.sessions[vid]
            return s.plate or "reading plate..."
        for s in self.sessions.values():
            if s.exit_event == event_id:
                return s.exit_plate or s.plate or s.vid
        return "reading plate..." if event_id in self.pending_exits else None

    def snapshot(self) -> dict:
        rows = [self.session_row(s) for s in self.sessions.values() if s.status != "DUPLICATE"]
        rank = {"OVERSTAY": 0, "ENTERED": 1, "PARKED": 1, "LEAVING": 1}
        rows.sort(key=lambda r: (rank.get(r["status"], 2), -int(r["vid"][1:])))
        return {
            "site": self.name, "now": self.clock.hms(self.now_t), "media_t": round(self.now_mt, 1),
            "capacity": self.capacity, "occupied": self.occupied, "free": self.free,
            "entries": self.entries, "exits": self.exits, "limit": self.clock.dur(self.limit_s),
            "time_scale": self.clock.time_scale,
            "counts": {k: sum(1 for r in rows if r["status"] == k) for k in
                       ("ENTERED", "PARKED", "LEAVING", "OVERSTAY", "EXITED", "EXITED_OVERSTAY")},
            "sessions": rows, "alerts": self.alert_log[-30:][::-1],
        }

    def export_rows(self) -> list[dict]:
        out = []
        for s in self.sessions.values():
            d = asdict(s)
            d["duration"] = self.clock.dur(s.duration(self.now_t))
            d["entry_time"] = self.clock.stamp(s.entry_t)
            d["exit_time"] = self.clock.stamp(s.exit_t)
            d["notes"] = "; ".join(s.notes)
            out.append(d)
        return out

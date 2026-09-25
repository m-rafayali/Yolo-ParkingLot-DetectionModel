"""Persistence: SQLite (sessions / events / alerts) + JSONL event log + CSV export."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    vid TEXT PRIMARY KEY, plate TEXT, plate_valid INTEGER, plate_source TEXT, origin TEXT, status TEXT,
    entry_time TEXT, exit_time TEXT, duration_s REAL, bay TEXT, lot_id TEXT, matched_by TEXT,
    entry_evidence TEXT, exit_evidence TEXT, notes TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, cam TEXT, t REAL, json TEXT);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, vid TEXT, plate TEXT, t REAL,
                                   message TEXT, json TEXT);
"""


class Store:
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.db = sqlite3.connect(str(out_dir / "parking.db"), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.events_f = open(out_dir / "events.jsonl", "a", buffering=1)
        self._n = 0

    def add_event(self, ev: dict) -> None:
        line = json.dumps(ev, default=float)
        self.events_f.write(line + "\n")
        self.db.execute("INSERT INTO events(type, cam, t, json) VALUES (?,?,?,?)", (ev["type"], ev.get("cam"), ev.get("t"), line))
        self._tick()

    def add_alert(self, a: dict) -> None:
        self.db.execute("INSERT INTO alerts(kind, vid, plate, t, message, json) VALUES (?,?,?,?,?,?)",
                        (a["kind"], a.get("vid"), a.get("plate"), a["t"], a["message"], json.dumps(a, default=float)))
        self._tick()

    def upsert_session(self, s, central) -> None:
        clk = central.clock
        self.db.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.vid, s.plate, int(bool(s.plate_valid)), s.plate_source, s.origin, s.status, clk.stamp(s.entry_t),
             clk.stamp(s.exit_t), s.duration(central.now_t), s.bay, s.lot_id, s.matched_by, s.entry_evidence,
             s.exit_evidence, "; ".join(s.notes), clk.stamp(central.now_t)))
        self._tick()

    def _tick(self) -> None:
        self._n += 1
        if self._n % 25 == 0:
            self.db.commit()

    def export_csv(self, rows: list[dict], path: Path) -> None:
        cols = ["vid", "plate", "plate_valid", "plate_source", "status", "origin", "entry_time", "exit_time", "duration",
                "bay", "lot_id", "matched_by", "entry_evidence", "exit_evidence", "notes"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def close(self) -> None:
        self.db.commit()
        self.db.close()
        self.events_f.close()

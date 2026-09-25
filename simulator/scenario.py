"""Vehicle schedule + smooth trajectories for the synthetic car park.

A *plan* is a list of Visit objects. Each visit is turned into a timeline of
(t, x, y, heading) samples: approach -> stop at entry barrier -> drive into bay ->
dwell -> reverse out -> drive to exit barrier -> stop -> leave.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import world as W
from .vehicles import SPECS, CarModel, random_sg_plate

COLORS = {
    "white": (232, 232, 230), "silver": (190, 190, 186), "grey": (115, 115, 118), "black": (34, 34, 36),
    "blue": (150, 85, 35), "red": (38, 38, 185), "maroon": (40, 30, 110), "navy": (95, 45, 25),
    "beige": (165, 195, 210), "green": (60, 105, 55),
}
COLOR_WEIGHTS = {"white": 0.26, "silver": 0.14, "grey": 0.14, "black": 0.18, "blue": 0.08, "red": 0.08,
                 "maroon": 0.04, "navy": 0.04, "beige": 0.02, "green": 0.02}
SPAWN_X, DESPAWN_X = -58.0, 104.0
GATE_WAIT = 2.8  # seconds stopped at a barrier
TURN_R = 4.5


@dataclass
class Visit:
    plate: str
    kind: str
    color: str
    bay: str
    t_gate: float | None  # stop time at the ENTRY barrier; None = already parked at t=0
    t_leave: float | None  # time the car starts backing out of its bay; None = stays
    # filled by build()
    t: np.ndarray = field(default=None, repr=False)
    xyh: np.ndarray = field(default=None, repr=False)
    events: dict = field(default_factory=dict)
    model: CarModel | None = field(default=None, repr=False)

    def pose(self, t: float):
        if t < self.t[0] or t > self.t[-1]:
            return None
        x = np.interp(t, self.t, self.xyh[:, 0])
        y = np.interp(t, self.t, self.xyh[:, 1])
        h = np.interp(t, self.t, self.xyh[:, 2])
        return x, y, h


# ------------------------------------------------------------------ path primitives
def _straight(p0, p1, step=0.1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
    return np.linspace(p0, p1, n)


def _arc(center, r, a0, a1, step=0.1):
    n = max(3, int(abs(a1 - a0) * r / step) + 1)
    a = np.linspace(a0, a1, n)
    return np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], 1)


def _join(*segs):
    out = [segs[0]]
    for s in segs[1:]:
        out.append(s[1:] if np.allclose(out[-1][-1], s[0]) else s)
    return np.concatenate(out)


def _drive(pts, vmax, t0, reverse=False, stop_at_end=True, a_acc=1.6, a_dec=2.0, a_lat=1.1):
    """Time-parameterise a polyline with a trapezoid-ish speed profile. Returns (t, xyh)."""
    d = np.diff(pts, axis=0)
    ds = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    hd = np.arctan2(d[:, 1], d[:, 0])
    hd = np.unwrap(np.r_[hd, hd[-1]])
    curv = np.abs(np.gradient(hd)) / np.r_[ds, ds[-1]]
    vmax_arr = np.full(len(pts), float(vmax))
    v_lim = np.minimum(vmax_arr, np.sqrt(a_lat / np.maximum(curv, 1e-4)))
    v = v_lim.copy()
    v[0] = 0.0
    for i in range(len(pts) - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_acc * ds[i]))
    if stop_at_end:
        v[-1] = 0.0
    for i in range(len(pts) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_dec * ds[i]))
    dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-3)
    t = t0 + np.r_[0.0, np.cumsum(dt)]
    heading = hd + (np.pi if reverse else 0.0)
    return t, np.stack([pts[:, 0], pts[:, 1], heading], 1)


def _hold(xyh_last, t0, t1):
    return np.array([t0, t1]), np.array([xyh_last, xyh_last])


# ------------------------------------------------------------------ build timelines
def build(visit: Visit, duration: float) -> Visit:
    spec = SPECS[visit.kind]
    bay = W.BAYS[visit.bay]
    top = bay.row == "T"
    mouth_y = W.TOP_ROW[0] if top else W.BOT_ROW[1]
    park_y = bay.cy + (0.25 if top else -0.25)
    park_h = np.pi / 2 if top else -np.pi / 2
    entry_stop = W.ENTRY_BARRIER_X - 0.9 - spec.L / 2
    exit_stop = W.EXIT_BARRIER_X - 0.9 - spec.L / 2
    ts, xs = [], []

    def add(t, xyh):
        if ts:
            t, xyh = t[1:], xyh[1:]
        ts.append(t)
        xs.append(xyh)

    ev = {}
    if visit.t_gate is not None:
        approach = _straight((SPAWN_X, W.LANE_Y), (entry_stop, W.LANE_Y))
        t, x = _drive(approach, 6.5, 0.0)
        t = t - t[-1] + visit.t_gate
        add(t, x)
        add(*_hold(x[-1], visit.t_gate, visit.t_gate + GATE_WAIT))
        c = (bay.cx - TURN_R, W.LANE_Y + TURN_R) if top else (bay.cx - TURN_R, W.LANE_Y - TURN_R)
        a0, a1 = (-np.pi / 2, 0.0) if top else (np.pi / 2, 0.0)
        path = _join(_straight((entry_stop, W.LANE_Y), (bay.cx - TURN_R, W.LANE_Y)), _arc(c, TURN_R, a0, a1),
                     _straight((bay.cx, mouth_y), (bay.cx, park_y)))
        t, x = _drive(path, 5.0, visit.t_gate + GATE_WAIT)
        add(t, x)
        ev["entry_stop"] = visit.t_gate
        ev["entry_line"] = float(np.interp(W.ENTRY_BARRIER_X, x[:, 0] + spec.L / 2, t))  # front bumper passes barrier
        ev["parked"] = float(t[-1])
        ev["lot_in"] = float(np.interp(0.0, x[:, 0] + spec.L / 2, t))
    else:
        add(np.array([-5.0, 0.0]), np.array([[bay.cx, park_y, park_h]] * 2))
        ev["parked"] = None  # before recording started
    t_end_park = visit.t_leave if visit.t_leave is not None else duration + 5
    add(*_hold(xs[-1][-1], ts[-1][-1], max(t_end_park, ts[-1][-1] + 0.01)))
    if visit.t_leave is not None:
        c = (bay.cx - TURN_R, mouth_y)
        a0, a1 = (0.0, -np.pi / 2) if top else (0.0, np.pi / 2)
        path = _join(_straight((bay.cx, park_y), (bay.cx, mouth_y)), _arc(c, TURN_R, a0, a1))
        t, x = _drive(path, 1.8, visit.t_leave, reverse=True)
        add(t, x)
        path = _straight((bay.cx - TURN_R, W.LANE_Y), (exit_stop, W.LANE_Y))
        t, x = _drive(path, 6.0, t[-1])
        ev["lot_out"] = float(np.interp(40.0, x[:, 0] - spec.L / 2, t))  # rear leaves the lot
        add(t, x)
        ev["exit_stop"] = float(t[-1])
        add(*_hold(x[-1], t[-1], t[-1] + GATE_WAIT))
        t, x = _drive(_straight((exit_stop, W.LANE_Y), (DESPAWN_X, W.LANE_Y)), 6.5, t[-1] + GATE_WAIT,
                      stop_at_end=False)
        ev["exit_line"] = float(np.interp(W.EXIT_BARRIER_X, x[:, 0] + spec.L / 2, t))
        add(t, x)
    visit.t = np.concatenate(ts)
    visit.xyh = np.concatenate(xs)
    visit.xyh[:, 2] = np.unwrap(visit.xyh[:, 2])
    visit.events = ev
    visit.model = CarModel(visit.kind, COLORS[visit.color], visit.plate, sunroof=(sum(map(ord, visit.plate)) % 5 == 0))
    return visit


def barrier_openness(visits: list[Visit], t: float, which: str) -> float:
    key_stop, key_line = ("entry_stop", "entry_line") if which == "entry" else ("exit_stop", "exit_line")
    o = 0.0
    for v in visits:
        if key_stop not in v.events:
            continue
        t_open = v.events[key_stop] + 0.7
        t_close = v.events[key_line] + 2.2
        if t < t_open:
            continue
        o = max(o, min(1.0, (t - t_open) / 1.2) if t < t_close else max(0.0, 1 - (t - t_close) / 1.4))
    return o


def check_conflicts(visits: list[Visit], duration: float, min_gap=2.3) -> list[str]:
    """Report moments where a moving car gets closer than `min_gap` metres to another car."""
    msgs = []
    for t in np.arange(0.2, duration, 0.2):
        poses = []
        for v in visits:
            p, q = v.pose(t), v.pose(t - 0.2)
            if p is not None and -40 < p[0] < 95:
                poses.append((v.plate, p, q is None or np.hypot(p[0] - q[0], p[1] - q[1]) > 0.01))
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                (a, pa, ma), (b, pb, mb) = poses[i], poses[j]
                if (ma or mb) and np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < min_gap:
                    msgs.append(f"t={t:.1f}s {a} too close to {b}")
    return msgs


# ------------------------------------------------------------------ plans
def _car(rng, prefix=None):
    kind = rng.choice(["sedan", "sedan", "hatch", "suv", "suv", "mpv"])
    color = rng.choice(list(COLOR_WEIGHTS), p=list(COLOR_WEIGHTS.values()))
    return random_sg_plate(rng, prefix), str(kind), str(color)


def demo_plan(seed: int = 2026) -> list[Visit]:
    """The hand-authored 3-minute scenario used for the client demo.

    Times are video seconds. With the demo clock (1 video s = 1 simulated minute) the
    1-hour overstay limit is reached 60 s after a car's entry.
    """
    rng = np.random.default_rng(seed)
    visitors = [  # (t_gate, bay, t_leave)
        (6, "T01", 33), (15, "B02", 110), (26, "T06", None), (37, "B08", 70), (49, "T04", 76),
        (61, "B05", None), (74, "T01", 100), (88, "T11", None), (101, "B09", 131), (116, "B11", None),
        (131, "T08", None), (146, "B04", None), (160, "T02", None),
    ]
    parked = [("T03", None), ("T05", None), ("T07", None), ("T10", 142), ("T12", None), ("B01", None),
              ("B03", None), ("B06", 50), ("B07", None), ("B10", None)]
    plan = []
    for t_gate, bay, t_leave in visitors:
        plate, kind, color = _car(rng)
        plan.append(Visit(plate, kind, color, bay, float(t_gate), None if t_leave is None else float(t_leave)))
    for bay, t_leave in parked:
        plate, kind, color = _car(rng)
        plan.append(Visit(plate, kind, color, bay, None, None if t_leave is None else float(t_leave)))
    return plan


def random_plan(seed: int, duration: float) -> list[Visit]:
    """Randomised episode (different cars, bays and timings) used to build training data."""
    rng = np.random.default_rng(seed)
    bays = list(W.BAYS)
    rng.shuffle(bays)
    n_parked = int(rng.integers(6, 16))
    plan, busy = [], {}
    for bay in bays[:n_parked]:
        plate, kind, color = _car(rng)
        leave = float(rng.uniform(5, duration)) if rng.random() < 0.35 else None
        while leave is not None and any(abs(leave - v.t_leave) < 10 for v in plan if v.t_leave is not None):
            leave += 5.0
        plan.append(Visit(plate, kind, color, bay, None, leave))
        busy[bay] = (leave + 8) if leave is not None else 1e9
    t = float(rng.uniform(3, 8))
    while t < duration - 15:
        free = [b for b in bays if busy.get(b, -1) < t]
        if not free:
            break
        bay = str(rng.choice(free))
        plate, kind, color = _car(rng)
        leave = t + float(rng.uniform(15, 70)) if rng.random() < 0.5 else None
        while leave is not None and any(abs(leave - v.t_leave) < 10 for v in plan if v.t_leave is not None):
            leave += 5.0
        plan.append(Visit(plate, kind, color, bay, t, leave))
        busy[bay] = (leave + 8) if leave is not None else 1e9
        t += float(rng.uniform(9, 16))
    return plan

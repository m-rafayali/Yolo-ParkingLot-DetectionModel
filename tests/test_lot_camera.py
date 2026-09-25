"""Tracker ids churn (C4); lot identities must not."""
from pathlib import Path

import numpy as np

from parkwatch.clock import SiteClock
from parkwatch.config import CAMERA_DEFAULTS
from parkwatch.detector import Det
from parkwatch.edge import LotCamera


class FakeDetector:
    def __init__(self):
        self.script = []

    def track(self, frame):
        return self.script.pop(0)


def box(tid, x, y):
    return Det(tid, "vehicle", 0.9, np.array([x - 30, y - 60, x + 30, y + 60], float))


def test_parked_car_keeps_identity_when_tracker_id_changes(tmp_path: Path):
    events = []
    cfg = dict(CAMERA_DEFAULTS["lot"], id="lot", role="lot", bays={"T01": [[0, 0], [100, 0], [100, 200], [0, 200]]})
    det = FakeDetector()
    cam = LotCamera(cfg, det, SiteClock("2026-01-01 00:00:00"), events.append, tmp_path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    mt = 0.0
    for k in range(40):           # 8 s still at 5 fps -> parked
        det.script.append([box(7, 50, 100)])
        cam.process(frame, mt)
        mt += 0.2
    assert [e["type"] for e in events] == ["parked"] and events[0]["bay"] == "T01"
    for k in range(5):            # tracker drops the car, then re-acquires it with a NEW id, 20 px away
        det.script.append([] if k < 3 else [box(42, 60 + k, 110)])
        cam.process(frame, mt)
        mt += 0.2
    live = cam.live()
    assert len(live) == 1 and live[0].lid == events[0]["lid"] and live[0].state == "PARKED"
    assert cam.tid2lid[42] == events[0]["lid"]


def test_second_box_on_a_turning_car_is_an_alias_not_a_new_car(tmp_path: Path):
    events = []
    cfg = dict(CAMERA_DEFAULTS["lot"], id="lot", role="lot")
    det = FakeDetector()
    cam = LotCamera(cfg, det, SiteClock("2026-01-01 00:00:00"), events.append, tmp_path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    mt = 5.0
    for k in range(10):           # driving; halfway through the detector emits a 2nd overlapping box
        boxes = [box(15, 400 + 10 * k, 350)]
        if 4 <= k <= 5:
            boxes.append(box(18, 410 + 10 * k, 340))
        if k > 5:
            boxes = [box(18, 400 + 10 * k, 350)]   # ...and then only the new id survives
        det.script.append(boxes)
        cam.process(frame, mt)
        mt += 0.2
    assert len({lid for lid in cam.tid2lid.values()}) == 1
    assert len(cam.live()) == 1

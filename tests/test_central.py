import copy

from parkwatch.central import CentralService
from parkwatch.clock import SiteClock
from parkwatch.config import DEFAULTS, deep_merge

T0 = "2026-09-25 08:00:00"


def make(limit_min=60, scale=60, **site):
    cfg = deep_merge(copy.deepcopy(DEFAULTS), {"site": {"capacity": 10, "initial_occupancy": 2,
                                                        "overstay_limit_min": limit_min, **site}})
    clock = SiteClock(T0, scale)
    return CentralService(cfg, clock), clock


def ev(clock, type_, mt, **kw):
    return {"type": type_, "cam": "x", "mt": mt, "t": clock.business(mt), **kw}


def test_full_visit_with_overstay_and_exit():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "plate", 10.5, ref="entry-1", plate="SBA1234K", valid=True, conf=0.9))
    c.handle(ev(clk, "lot_arrival", 13, lid="L1"))
    c.handle(ev(clk, "parked", 20, lid="L1", bay="T01", since_t=clk.business(16), origin="arrival"))
    s = next(iter(c.sessions.values()))
    assert s.plate == "SBA1234K" and s.lot_id == "L1" and s.bay == "T01" and s.status == "PARKED"
    assert c.free == 10 - 3
    c.tick(69)                       # 59 video s = 59 site minutes: not yet
    assert s.status == "PARKED" and not c.alert_log
    c.tick(71)                       # 61 site minutes after entry -> overstay
    assert s.status == "OVERSTAY" and c.alert_log[-1]["kind"] == "overstay"
    c.handle(ev(clk, "unparked", 90, lid="L1", bay="T01", parked_s=4000))
    c.handle(ev(clk, "lot_departure", 95, lid="L1"))
    c.handle(ev(clk, "exit", 100, event_id="exit-1"))
    assert c.free == 10 - 2           # capacity is counted at the gates, before the plate is known
    c.handle(ev(clk, "plate", 100.4, ref="exit-1", plate="SBA1234K", valid=True))
    assert s.status == "EXITED_OVERSTAY" and s.matched_by == "plate"
    assert c.alert_log[-1]["kind"] == "overstay_exit"


def test_exit_with_unreadable_plate_matched_by_lot_departure():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "plate", 10, ref="entry-1", plate="SBA1234K", valid=True))
    c.handle(ev(clk, "lot_arrival", 13, lid="L1"))
    c.handle(ev(clk, "lot_departure", 40, lid="L1"))
    c.handle(ev(clk, "exit", 45, event_id="exit-1"))
    c.handle(ev(clk, "plate", 45, ref="exit-1", plate=None))
    s = c.sessions["V0001"]
    assert s.status == "EXITED" and s.matched_by == "lot-departure"


def test_fuzzy_exit_plate_and_duplicate_entry():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "plate", 10, ref="entry-1", plate="SHD1476M", valid=True))
    c.handle(ev(clk, "entry", 11, event_id="entry-2"))            # tracker re-acquired the same car
    c.handle(ev(clk, "plate", 11, ref="entry-2", plate="SHD1476M", valid=True))
    assert c.entries == 1 and c.sessions["V0002"].status == "DUPLICATE"
    c.handle(ev(clk, "exit", 30, event_id="exit-1"))
    c.handle(ev(clk, "plate", 30, ref="exit-1", plate="SHD1476N", valid=False))
    s = c.sessions["V0001"]
    assert s.status == "EXITED" and s.matched_by == "fuzzy-plate"


def test_vehicle_present_at_startup_is_not_alerted_by_default():
    c, clk = make()
    c.handle(ev(clk, "parked", 4, lid="L9", bay="B01", since_t=clk.business(0), origin="startup"))
    c.tick(200)
    s = c.sessions["V0001"]
    assert s.status == "PARKED" and not c.alert_log
    assert c.session_row(s)["duration"].endswith("+")   # shown as a lower bound


def test_lot_track_lost_while_driving_is_relinked_when_it_parks():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "plate", 10, ref="entry-1", plate="SBA1234K", valid=True))
    c.handle(ev(clk, "lot_arrival", 13, lid="L1"))
    c.handle(ev(clk, "lot_lost", 15, lid="L1"))              # lot camera lost the car mid-turn
    c.handle(ev(clk, "parked", 22, lid="L2", bay="T08", since_t=clk.business(18), origin="appeared"))
    s = c.sessions["V0001"]
    assert len(c.sessions) == 1 and s.lot_id == "L2" and s.bay == "T08" and s.status == "PARKED"


def test_reentry_closes_visit_whose_exit_was_missed():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "plate", 10, ref="entry-1", plate="SBA1234K", valid=True))
    c.handle(ev(clk, "entry", 150, event_id="entry-2"))           # exit camera never saw it leave
    c.handle(ev(clk, "plate", 150, ref="entry-2", plate="SBA1234K", valid=True))
    old, new = c.sessions["V0001"], c.sessions["V0002"]
    assert old.status == "EXITED" and old.matched_by == "re-entry" and new.open
    c.handle(ev(clk, "exit", 170, event_id="exit-1"))
    c.handle(ev(clk, "plate", 170, ref="exit-1", plate="SBA1234K", valid=True))
    assert new.status == "EXITED" and new.exit_event == "exit-1"


def test_duplicate_entry_hands_its_lot_link_to_the_real_visit():
    c, clk = make()
    c.handle(ev(clk, "entry", 10, event_id="entry-1"))
    c.handle(ev(clk, "lot_arrival", 13, lid="L1"))                # linked FIFO to entry-1 ...
    c.handle(ev(clk, "entry", 11, event_id="entry-2"))
    c.handle(ev(clk, "plate", 11.5, ref="entry-2", plate="SBA1234K", valid=True))
    c.handle(ev(clk, "plate", 12, ref="entry-1", plate="SBA1234K", valid=True))  # ... which turns out to be the duplicate
    c.handle(ev(clk, "parked", 20, lid="L1", bay="T01", since_t=clk.business(16), origin="arrival"))
    live = [s for s in c.sessions.values() if s.status != "DUPLICATE"]
    assert len(live) == 1 and live[0].bay == "T01" and live[0].status == "PARKED" and c.entries == 1

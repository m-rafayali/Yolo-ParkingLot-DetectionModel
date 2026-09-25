"""Score a run against the simulator's ground truth (what really happened in the synthetic clip)."""

from __future__ import annotations

import json
from pathlib import Path


def _nearest(sessions, attr: str, t: float, t_stop: float | None = None, tol: float = 3.0):
    """Session whose gate time falls between 'stopped at the barrier' and 'passed it' (+ tolerance)."""
    lo = (t_stop if t_stop is not None else t) - 1.0
    cands = [s for s in sessions if getattr(s, attr) is not None and lo <= getattr(s, attr) <= t + tol]
    return min(cands, key=lambda s: abs(getattr(s, attr) - t)) if cands else None


def evaluate(cfg: dict, central, out_dir: Path) -> dict:
    gt = json.loads(Path(cfg["ground_truth"]).read_text())
    scale, limit = central.clock.time_scale, central.limit_s
    end = central.now_mt
    sessions = [s for s in central.sessions.values() if s.status != "DUPLICATE"]
    rows, m = [], {"entries": 0, "entry_found": 0, "entry_plate_ok": 0, "exits": 0, "exit_found": 0,
                   "exit_plate_ok": 0, "closed_same_session": 0, "bay_ok": 0, "parked_expected": 0,
                   "overstay_expected": 0, "overstay_flagged": 0, "false_overstay": 0, "overstay_ambiguous": 0}
    for v in gt["vehicles"]:
        r = {"plate": v["plate"], "bay": v["bay"], "pre_parked": v["pre_parked"]}
        s_in = s_out = None
        if v.get("entry_line") is not None and v["entry_line"] < end - 1:
            m["entries"] += 1
            s_in = _nearest(sessions, "entry_mt", v["entry_line"], v.get("entry_stop"))
            r["entry"] = "found" if s_in else "MISSED"
            if s_in:
                m["entry_found"] += 1
                r["entry_read"] = s_in.plate
                m["entry_plate_ok"] += s_in.plate == v["plate"]
            if v.get("parked") is not None and v["parked"] < end - 5:
                m["parked_expected"] += 1
                r["bay_seen"] = s_in.bay if s_in else None
                m["bay_ok"] += bool(s_in and s_in.bay == v["bay"])
        if v.get("exit_line") is not None and v["exit_line"] < end - 1:
            m["exits"] += 1
            s_out = _nearest(sessions, "exit_mt", v["exit_line"], v.get("exit_stop"))
            r["exit"] = "found" if s_out else "MISSED"
            if s_out:
                m["exit_found"] += 1
                r["exit_read"] = s_out.exit_plate
                m["exit_plate_ok"] += s_out.exit_plate == v["plate"]
                r["matched_by"] = s_out.matched_by
                if s_in is not None:
                    m["closed_same_session"] += s_out is s_in
                    r["closed"] = "same visit" if s_out is s_in else "WRONG visit"
        if v.get("entry_line") is not None and v["entry_line"] < end - 1:
            stop = v["exit_line"] if v.get("exit_line") is not None and v["exit_line"] < end else end
            true_dur = (stop - v["entry_line"]) * scale
            expected = true_dur >= limit
            got = bool(s_in and s_in.status in ("OVERSTAY", "EXITED_OVERSTAY"))
            r["overstay"] = f"{'yes' if expected else 'no'}/{'yes' if got else 'no'}"
            if abs(true_dur - limit) < 3 * scale:
                m["overstay_ambiguous"] += 1
            elif expected:
                m["overstay_expected"] += 1
                m["overstay_flagged"] += got
            else:
                m["false_overstay"] += got
        rows.append(r)

    def pct(a, b):
        return f"{a}/{b}" + (f" ({100 * a / b:.0f}%)" if b else "")

    print("\nGROUND-TRUTH CHECK (synthetic clip: we know exactly what happened)")
    print(f"  entries detected        {pct(m['entry_found'], m['entries'])}")
    print(f"  entry plates correct    {pct(m['entry_plate_ok'], m['entries'])}")
    print(f"  exits detected          {pct(m['exit_found'], m['exits'])}")
    print(f"  exit plates correct     {pct(m['exit_plate_ok'], m['exits'])}")
    print(f"  visits closed correctly {pct(m['closed_same_session'], sum(1 for r in rows if 'closed' in r))}"
          "   (exit matched to the right entry)")
    print(f"  parked bay correct      {pct(m['bay_ok'], m['parked_expected'])}")
    print(f"  overstays flagged       {pct(m['overstay_flagged'], m['overstay_expected'])}"
          f"   false alarms: {m['false_overstay']}")
    missed = [r for r in rows if "MISSED" in (r.get("entry"), r.get("exit"))
              or r.get("closed") == "WRONG visit" or r.get("entry_read", r["plate"]) != r["plate"]]
    for r in missed:
        print(f"    check: {r}")
    out = {"metrics": m, "vehicles": rows}
    (out_dir / "evaluation.json").write_text(json.dumps(out, indent=1))
    return out

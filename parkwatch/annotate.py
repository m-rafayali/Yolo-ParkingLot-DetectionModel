"""Overlays for each camera and the combined control-room mosaic (OpenCV only)."""

from __future__ import annotations

import cv2
import numpy as np

GREEN, RED, AMBER, BLUE, GREY, WHITE = (90, 200, 90), (70, 70, 235), (0, 185, 255), (235, 170, 70), (170, 170, 170), (245, 245, 245)
DARK, PANEL = (22, 24, 28), (34, 37, 43)
STATUS_COLOR = {"PARKED": GREEN, "ENTERED": AMBER, "LEAVING": AMBER, "OVERSTAY": RED, "EXITED": GREY,
                "EXITED_OVERSTAY": (60, 110, 230)}
FONT, BOLD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX


def text(img, s, org, scale=0.55, color=WHITE, thick=1, font=FONT, bg=None, pad=4):
    (tw, th), base = cv2.getTextSize(s, font, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return tw


def header(img, title: str, clock: str, fps: float | None, extra: str = "") -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), DARK, -1)
    text(img, title, (14, 27), 0.7, WHITE, 1, BOLD)
    right = f"{clock}" + (f"   {fps:4.1f} fps" if fps else "")
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.6, 1)
    text(img, right, (img.shape[1] - tw - 14, 27), 0.6, (200, 230, 255))
    if extra:
        (ew, _), _ = cv2.getTextSize(extra, BOLD, 0.7, 1)
        text(img, extra, ((img.shape[1] - ew) // 2, 28), 0.7, (120, 255, 160), 1, BOLD)


def draw_gate(frame, cam, central, clock_str, fps) -> np.ndarray:
    img = frame.copy()
    col = GREEN if cam.role == "entry" else RED
    a, b = [tuple(int(v) for v in p) for p in cam.line]
    cv2.line(img, a, b, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.line(img, a, b, col, 4, cv2.LINE_AA)
    ok, va, vb = cv2.clipLine((0, 0, img.shape[1], img.shape[0] - 10), a, b)
    if ok:  # label the part of the line that is actually on screen
        mx, my = (va[0] + vb[0]) // 2, (va[1] + vb[1]) // 2
        text(img, f"{cam.role.upper()} LINE", (mx - 60, my - 14), 0.7, (0, 0, 0), 2, BOLD, bg=col)
    for d in cam.last_dets:
        eid = cam.event_for_track(d.tid)
        x0, y0, x1, y1 = [int(v) for v in d.xyxy]
        c = col if eid else (200, 200, 200)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        label = f"#{d.tid}"
        if eid:
            label += f"  {central.label_for_event(eid) or ''}"
        text(img, label, (x0 + 4, max(y0 - 8, 52)), 0.6, (0, 0, 0) if eid else WHITE, 1, BOLD, bg=c)
    arrow = "IN" if cam.role == "entry" else "OUT"
    header(img, cam.name, clock_str, fps, f"{arrow}: {cam.count}")
    return img


def draw_lot(frame, cam, central, clock_str, fps) -> np.ndarray:
    img = frame.copy()
    occ = cam.occupied_bays()
    if cam.bays:
        overlay = img.copy()
        for name, poly in cam.bays.items():
            pts = np.int32(poly)
            cv2.fillPoly(overlay, [pts], RED if name in occ else GREEN)
        img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
        for name, poly in cam.bays.items():
            cv2.polylines(img, [np.int32(poly)], True, RED if name in occ else GREEN, 2, cv2.LINE_AA)
    for zone, c in ((cam.entry_zone, GREEN), (cam.exit_zone, RED)):
        if zone:
            cv2.polylines(img, [np.int32(zone)], True, c, 1, cv2.LINE_AA)
    now = central.now_t
    by_tid = {d.tid: d for d in cam.last_dets}
    for tid, lid in cam.tid2lid.items():
        d = by_tid.get(tid)
        ident = cam.idents.get(lid)
        if d is None or ident is None or ident.state == "GONE":
            continue
        name, status = central.label_for_lot(lid)
        if ident.state == "PARKED":
            color = RED if status == "OVERSTAY" else GREEN
            since = ident.parked_since_mt
            tmr = central.clock.dur(now - central.clock.business(since)) if since is not None else ""
            if ident.origin == "startup":
                tmr += "+"
        else:
            color, tmr = AMBER, "moving"
        poly = d.poly if d.poly is not None else np.array([[d.xyxy[0], d.xyxy[1]], [d.xyxy[2], d.xyxy[1]],
                                                           [d.xyxy[2], d.xyxy[3]], [d.xyxy[0], d.xyxy[3]]])
        cv2.polylines(img, [np.int32(poly)], True, color, 2, cv2.LINE_AA)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        lab = f"{name or lid}"
        (lw, _), _ = cv2.getTextSize(lab, FONT, 0.45, 1)
        text(img, lab, (cx - lw / 2, cy - 4), 0.45, (0, 0, 0), 1, FONT, bg=color, pad=3)
        (tw, _), _ = cv2.getTextSize(tmr, FONT, 0.45, 1)
        text(img, tmr, (cx - tw / 2, cy + 16), 0.45, WHITE, 1, FONT, bg=DARK, pad=3)
    total = len(cam.bays) if cam.bays else None
    extra = f"BAYS FREE {total - len(occ)}/{total}" if total else f"PARKED {len(occ)}"
    header(img, cam.name, clock_str, fps, extra)
    return img


# ---------------------------------------------------------------------------- mosaic panels
def _kpi(img, x, y, w, h, label, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + 5, y + h), color, -1)
    text(img, label, (x + 16, y + 26), 0.5, (185, 190, 200))
    text(img, value, (x + 16, y + h - 16), 1.15, WHITE, 2, BOLD)


def kpi_panel(w, h, snap, plates) -> np.ndarray:
    img = np.full((h, w, 3), DARK, np.uint8)
    text(img, snap["site"].upper(), (18, 34), 0.62, WHITE, 1, BOLD)
    text(img, f"site clock {snap['now']}   limit {snap['limit']}", (18, 60), 0.5, (170, 200, 230))
    gw, gh, pad = (w - 3 * 16) // 2, 88, 16
    llm = plates.get("llm", {}) if plates else {}
    tiles = [
        ("FREE SPACES", f"{snap['free']}/{snap['capacity']}", GREEN if snap["free"] > 0 else RED),
        ("OVERSTAY NOW", str(snap["counts"]["OVERSTAY"]), RED if snap["counts"]["OVERSTAY"] else GREY),
        ("IN / OUT", f"{snap['entries']} / {snap['exits']}", BLUE),
        ("LLM CALLS  $", f"{llm.get('ok', 0)}  ${plates.get('llm_cost_usd', 0):.3f}" if plates else "-", AMBER),
    ]
    for k, (lab, val, col) in enumerate(tiles):
        _kpi(img, pad + (k % 2) * (gw + pad), 78 + (k // 2) * (gh + pad), gw, gh, lab, val, col)
    if plates:
        v = plates.get("vehicles", {})
        by = "  ".join(f"{r['name']}: {plates.get(r['name'], {}).get('ok', 0)}" + ("" if r["available"] else " (off)")
                       for r in plates.get("readers", []))
        text(img, f"plates  {v.get('requested', 0)} sent  {v.get('valid', 0)} checksum-valid  |  {by}",
             (18, h - 22), 0.47, (170, 180, 195))
    return img


def sessions_panel(w, h, snap, max_rows=None) -> np.ndarray:
    img = np.full((h, w, 3), DARK, np.uint8)
    text(img, "VEHICLES", (18, 32), 0.62, WHITE, 1, BOLD)
    cols = [("PLATE", 18), ("STATUS", 150), ("IN", 300), ("OUT", 390), ("TIME", 480), ("BAY", 570)]
    y = 62
    for name, x in cols:
        text(img, name, (x, y), 0.45, (150, 160, 175))
    rows = [r for r in snap["sessions"]]
    room = (h - 215 - 80) // 26 if max_rows is None else max_rows
    for r in rows[:room]:
        y += 26
        col = STATUS_COLOR.get(r["status"], GREY)
        plate = r["plate"] or (f"({r['vid']})")
        text(img, plate[:11], (18, y), 0.52, WHITE if r["plate"] else GREY, 1, BOLD if r["plate"] else FONT)
        text(img, r["status"].replace("EXITED_OVERSTAY", "OUT-LATE"), (150, y), 0.47, col, 1, BOLD)
        text(img, r["entry"] or "-", (300, y), 0.45)
        text(img, r["exit"] or "-", (390, y), 0.45)
        text(img, r["duration"] or "-", (480, y), 0.45, col)
        text(img, r["bay"] or "-", (570, y), 0.45)
    y = h - 215
    cv2.line(img, (16, y), (w - 16, y), (70, 74, 82), 1)
    text(img, "ALERTS", (18, y + 30), 0.62, (120, 140, 255), 1, BOLD)
    yy = y + 60
    for a in snap["alerts"][:4]:
        msg = f"{a['time']}  {a['message']}"
        while msg and yy < h - 10:  # wrap long messages onto a second line
            cut = len(msg) if len(msg) <= 74 else (msg.rfind(" ", 0, 74) if msg.rfind(" ", 0, 74) > 0 else 74)
            text(img, msg[:cut], (18, yy), 0.45, (150, 170, 255))
            msg, yy = "      " + msg[cut:].lstrip() if msg[cut:].strip() else "", yy + 22
        yy += 6
    return img


def mosaic(frames: dict, roles: dict, snap: dict, plates: dict) -> np.ndarray:
    """1920x1080 control-room view. frames/roles are keyed by camera id."""
    W, H = 1920, 1080
    canvas = np.full((H, W, 3), DARK, np.uint8)
    lot = next((c for c, r in roles.items() if r == "lot"), None)
    gates = [c for c, r in roles.items() if r != "lot"]
    if lot is not None:
        tiles = [(c, (0 + i * 640, 0, 640, 360)) for i, c in enumerate(gates[:2])]
        tiles.append((lot, (0, 360, 1280, 720)))
        canvas[0:360, 1280:1920] = kpi_panel(640, 360, snap, plates)
        canvas[360:1080, 1280:1920] = sessions_panel(640, 720, snap)
    else:
        n = max(1, len(gates))
        cols = 1 if n == 1 else 2
        tw, th = 1280 // cols, int(1280 // cols * 9 / 16)
        tiles = [(c, ((i % cols) * tw, (i // cols) * th, tw, th)) for i, c in enumerate(gates[:4])]
        canvas[0:360, 1280:1920] = kpi_panel(640, 360, snap, plates)
        canvas[360:1080, 1280:1920] = sessions_panel(640, 720, snap)
    for cid, (x, y, w, h) in tiles:
        f = frames.get(cid)
        if f is not None:
            canvas[y:y + h, x:x + w] = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    for x in (640, 1280):
        cv2.line(canvas, (x, 0), (x, 360), DARK, 3)
    cv2.line(canvas, (0, 360), (1920, 360), DARK, 3)
    return canvas

#!/usr/bin/env python3
"""Draw counting lines, bays and zones on a frame of YOUR footage; saves the geometry JSON the config uses.

    python tools/draw_zones.py --source gate.mp4 --camera entry --out configs/my_site_geometry.json
    python tools/draw_zones.py --source lot.mp4  --camera lot   --out configs/my_site_geometry.json --at 5

Keys (click points with the left mouse button):
    l  counting line (2 clicks)          gate cameras
    b  add a bay polygon (4 clicks)      lot camera, bays are named B01, B02, ...
    e  lot entry zone (4 clicks)         where arriving cars first appear in the lot view
    x  lot exit zone (4 clicks)          where leaving cars disappear from the lot view
    u  undo last shape    s  save    q  quit
Then reference it from the config:  geometry: my_site_geometry.json#entry
Needs the GUI build of OpenCV (pip install opencv-python).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

NEED = {"l": 2, "b": 4, "e": 4, "x": 4}
COL = {"l": (0, 255, 0), "b": (255, 200, 0), "e": (0, 200, 0), "x": (0, 0, 255)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, RTSP URL or webcam index")
    ap.add_argument("--camera", required=True, help="key to store the geometry under, e.g. entry / lot / exit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", type=float, default=0.0, help="seconds into the video to grab the frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read a frame from {args.source}")
    H, W = frame.shape[:2]
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    geo = data.get(args.camera, {"camera": args.camera, "size": [W, H]})
    geo["size"] = [W, H]
    geo.setdefault("bays", {})
    shapes: list[tuple[str, str | None]] = []
    mode, pts = None, []

    def redraw():
        img = frame.copy()
        if geo.get("line"):
            a, b = [tuple(int(v) for v in p) for p in geo["line"]]
            cv2.line(img, a, b, COL["l"], 3)
        for name, poly in geo["bays"].items():
            cv2.polylines(img, [np.int32(poly)], True, COL["b"], 2)
            c = np.int32(poly).mean(0).astype(int)
            cv2.putText(img, name, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL["b"], 1)
        for k, key in (("e", "entry_zone"), ("x", "exit_zone")):
            if geo.get(key):
                cv2.polylines(img, [np.int32(geo[key])], True, COL[k], 2)
        for p in pts:
            cv2.circle(img, p, 5, (0, 255, 255), -1)
        msg = f"mode: {mode or '-'}  ({len(pts)}/{NEED.get(mode, 0)})   l=line b=bay e=entry zone x=exit zone u=undo s=save q=quit"
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, msg, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("draw zones", img)

    def on_mouse(event, x, y, *_):
        nonlocal pts
        if event != cv2.EVENT_LBUTTONDOWN or mode is None:
            return
        pts.append((x, y))
        if len(pts) == NEED[mode]:
            p = [[float(a), float(b)] for a, b in pts]
            if mode == "l":
                geo["line"] = p
                shapes.append(("line", None))
            elif mode == "b":
                name = f"B{len(geo['bays']) + 1:02d}"
                geo["bays"][name] = p
                shapes.append(("bay", name))
            else:
                key = "entry_zone" if mode == "e" else "exit_zone"
                geo[key] = p
                shapes.append((key, None))
            pts = []
        redraw()

    cv2.namedWindow("draw zones", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("draw zones", on_mouse)
    redraw()
    while True:
        k = chr(cv2.waitKey(50) & 0xFF)
        if k in NEED:
            mode, pts = k, []
            redraw()
        elif k == "u" and shapes:
            kind, name = shapes.pop()
            if kind == "bay":
                geo["bays"].pop(name, None)
            else:
                geo.pop(kind, None)
            redraw()
        elif k == "s":
            data[args.camera] = geo
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=1))
            print(f"saved {out}#{args.camera}")
        elif k == "q":
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

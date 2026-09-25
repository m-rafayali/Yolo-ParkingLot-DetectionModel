#!/usr/bin/env python3
"""ParkWatch demo: gate + lot cameras -> events -> central service -> dashboard / CSV / SQLite.

Examples
    python run_demo.py                                  # synthetic 3-camera demo, as fast as the CPU allows
    python run_demo.py --realtime                       # paced like live CCTV (drops frames if the CPU is slow)
    python run_demo.py --realtime --show                # plus an OpenCV window
    python run_demo.py --config configs/my_site.yaml    # your own footage / RTSP cameras
Open http://localhost:8000 for the live dashboard.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "demo.yaml"))
    ap.add_argument("--realtime", action="store_true", help="pace playback to the wall clock (live behaviour)")
    ap.add_argument("--speed", type=float, default=1.0, help="with --realtime: 2 = play footage twice as fast")
    ap.add_argument("--show", action="store_true", help="open an OpenCV window with the control-room view")
    ap.add_argument("--no-video", action="store_true", help="do not write annotated_mosaic.mp4")
    ap.add_argument("--no-web", action="store_true", help="disable the web dashboard")
    ap.add_argument("--hold", action="store_true", help="keep the dashboard running after the footage ends")
    ap.add_argument("--max-seconds", type=float, default=None, help="only process the first N seconds")
    ap.add_argument("--out", default=None, help="output folder (default runs/<timestamp>)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("ultralytics", "httpx", "openai", "open_image_models", "fast_alpr", "fast_plate_ocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from parkwatch.config import load_config
    from parkwatch.runner import Runner

    cfg = load_config(args.config)
    missing = [c["source"] for c in cfg["cameras"]
               if not str(c["source"]).isdigit() and "://" not in str(c["source"]) and not Path(c["source"]).exists()]
    if missing:
        sys.exit("missing video(s):\n  " + "\n  ".join(missing) +
                 "\nGenerate the synthetic demo footage first:  python tools/make_demo_videos.py")
    if args.port:
        cfg["dashboard"]["port"] = args.port
    runner = Runner(cfg, realtime=args.realtime, speed=args.speed, show=args.show,
                    save_video=False if args.no_video else None, web=False if args.no_web else None,
                    max_seconds=args.max_seconds, out_dir=args.out)
    if runner.web:
        runner.web.keep_alive = args.hold
    runner.run()


if __name__ == "__main__":
    main()

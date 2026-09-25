#!/usr/bin/env python3
"""Generate the synthetic 3-camera car-park footage (the "digital twin" demo).

Produces, in --out (default assets/demo):
    entry_cam.mp4  lot_cam.mp4  exit_cam.mp4   same world, same cars, same clock
    ground_truth.json                          what really happened (for scoring the pipeline)
    scene_geometry.json                        counting lines, bay polygons, zones in pixel coords

Every car has a valid Singapore-format plate, so plate reads can be checked end to end:
the same plate is read at the entry gate, the car parks in a bay, and is read again at the exit.

    python tools/make_demo_videos.py                 # 3 min @ 15 fps (~3-5 min to render on a laptop)
    python tools/make_demo_videos.py --duration 60   # quick test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.episode import render_episode, write_geometry  # noqa: E402
from simulator.scenario import build, check_conflicts, demo_plan  # noqa: E402


def to_h264(path: Path) -> None:
    """Re-encode mp4v -> H.264 (smaller, plays in browsers) if an ffmpeg binary is available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            print("  (no ffmpeg found: keeping mp4v encoding; `pip install imageio-ffmpeg` for H.264)")
            return
    tmp = path.with_suffix(".h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "medium", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo"))
    ap.add_argument("--duration", type=float, default=180.0, help="seconds of footage")
    ap.add_argument("--fps", type=float, default=15.0, help="CCTV-style frame rate")
    ap.add_argument("--cams", default="entry,lot,exit")
    ap.add_argument("--no-h264", action="store_true", help="skip H.264 re-encode")
    ap.add_argument("--geometry-only", action="store_true", help="only rewrite scene_geometry.json")
    args = ap.parse_args()

    out = Path(args.out)
    if args.geometry_only:
        out.mkdir(parents=True, exist_ok=True)
        write_geometry(out)
        print(f"wrote {out / 'scene_geometry.json'}")
        return
    plan = demo_plan()
    conflicts = check_conflicts([build(v, args.duration) for v in demo_plan()], args.duration)
    if conflicts:
        print("warning: scenario has close calls:\n  " + "\n  ".join(conflicts[:5]))
    t0 = time.time()
    print(f"Rendering {args.duration:.0f}s @ {args.fps:g} fps for cameras: {args.cams}")
    res = render_episode(plan, args.duration, args.fps, out, cams=tuple(args.cams.split(",")))
    if not args.no_h264:
        for cam in args.cams.split(","):
            to_h264(out / f"{cam}_cam.mp4")
    print(f"Done in {time.time() - t0:.0f}s -> {out}")
    truth = res["truth"]["vehicles"]
    print(f"  {sum(v['pre_parked'] for v in truth)} cars already parked, "
          f"{sum(not v['pre_parked'] for v in truth)} arrive, "
          f"{sum('exit_line' in v for v in truth)} leave during the clip")


if __name__ == "__main__":
    main()

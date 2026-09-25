#!/usr/bin/env python3
"""Build auto-labelled YOLO datasets from randomised synthetic episodes.

The simulator knows exactly where every car is, so labels are free. Episodes use
*different* random seeds from the demo scenario, so the demo clip stays unseen
test data.

Outputs:
    datasets/twin_obb/   overhead camera, oriented boxes (YOLO-OBB format), class 0 = vehicle
    datasets/twin_det/   gate cameras, axis-aligned boxes (YOLO detect format), class 0 = vehicle

Then fine-tune, e.g.:
    python tools/train_twin.py --task obb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.episode import render_episode  # noqa: E402
from simulator.scenario import build, check_conflicts, random_plan  # noqa: E402


def write_split(samples, root: Path, split: str, task: str, W: int, H: int, prefix: str) -> int:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for s in samples:
        stem = f"{prefix}_{s['frame']:05d}"
        cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), s["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for o in s["objects"]:
            if task == "obb":
                pts = [f"{x / W:.6f} {y / H:.6f}" for x, y in o["obb"]]
                lines.append("0 " + " ".join(pts))
            else:
                x0, y0, x1, y1 = o["xyxy"]
                lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--duration", type=float, default=100.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--every", type=int, default=15, help="keep one frame every N (lot camera)")
    ap.add_argument("--gate-every", type=int, default=5, help="keep one frame every N (gate cameras)")
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    seed, done = args.seed, 0
    while done < args.episodes:
        seed += 1
        plan = random_plan(seed, args.duration)
        if check_conflicts([build(v, args.duration) for v in random_plan(seed, args.duration)], args.duration):
            continue
        split = "val" if done == args.episodes - 1 else "train"
        print(f"episode {done + 1}/{args.episodes} (seed {seed}, {len(plan)} cars) -> {split}")
        for cams, every, task, name in ((("lot",), args.every, "obb", "twin_obb"),
                                        (("entry", "exit"), args.gate_every, "det", "twin_det")):
            res = render_episode(random_plan(seed, args.duration), args.duration, args.fps,
                                 out / "_scratch", cams=cams, label_every=every, seed=seed, write_video=False,
                                 log=lambda *_: None)
            for cam in cams:
                samples = [s for s in res["labels"][cam] if s["objects"] or task == "obb"]
                n = write_split(samples, out / name, split, task, 1280, 720, f"s{seed}_{cam}")
                print(f"   {name:9s} {cam:5s}: {n} images")
        done += 1
    for name in ("twin_obb", "twin_det"):
        (out / name / "data.yaml").write_text(
            f"path: {(out / name).resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n")
    print(f"datasets written to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fine-tune YOLO on the auto-labelled synthetic datasets (site adaptation).

Why: stock COCO weights call top-down cars "cell phone", and stock DOTA-OBB weights are
tuned to aerial photos, not to this rendered site. A short fine-tune on labelled frames from
the *same cameras* fixes that - exactly what you would do for a real client site, with real
labelled frames instead of rendered ones.

    python tools/make_training_data.py            # 1. build datasets/ (about 5 min)
    python tools/train_twin.py --task obb         # 2. overhead model  -> models/twin_obb.pt
    python tools/train_twin.py --task det         #    (optional) gate model -> models/twin_det.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["obb", "det"], default="obb")
    ap.add_argument("--data", default=None, help="data.yaml (default datasets/twin_<task>/data.yaml)")
    ap.add_argument("--base", default=None, help="starting weights (default yolo11n-obb.pt / yolo11n.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = args.data or str(ROOT / "datasets" / f"twin_{args.task}" / "data.yaml")
    base = args.base or ("yolo11n-obb.pt" if args.task == "obb" else "yolo11n.pt")
    model = YOLO(base)
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=str(ROOT / "runs" / "train"), name=f"twin_{args.task}", exist_ok=True, patience=10,
                degrees=10 if args.task == "obb" else 0, fliplr=0.5, mosaic=1.0, close_mosaic=5, plots=False,
                workers=2)
    best = Path(model.trainer.best)
    out = ROOT / "models" / f"twin_{args.task}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

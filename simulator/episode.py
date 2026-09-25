"""Render one episode (a plan of visits) through the three virtual cameras."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import world as W
from .render import Post
from .scenario import Visit, barrier_openness, build

CAM_LABELS = {"entry": "CAM-01  ENTRY GATE", "lot": "CAM-02  CAR PARK OVERVIEW", "exit": "CAM-03  EXIT GATE"}
REGIONS = {"entry": ((-62, -6, 4, 28), 50), "exit": ((34, -6, 100, 28), 50), "lot": ((-8, -6, 48, 28.5), 40)}


class CameraRig:
    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.cam = W.make_cameras()[name]
        region, ppm = REGIONS[name]
        self.bg = W.ground_image(self.cam, W.paint_ground(region, ppm))
        gx = W.EXIT_BARRIER_X if name == "exit" else W.ENTRY_BARRIER_X
        self.gate = W.Gate(gx, "EXIT" if name == "exit" else "ENTRY", (40, 40, 190) if name == "exit" else (60, 130, 30))
        self.post = Post((self.cam.W, self.cam.H), seed=seed)

    # ------------------------------------------------------------------ geometry exported to the pipeline
    def _img(self, pts3):
        return [[round(float(u), 1), round(float(v), 1)] for u, v in self.cam.project(np.asarray(pts3, float))]

    def scene_geometry(self) -> dict:
        g: dict = {"camera": self.name, "size": [self.cam.W, self.cam.H]}
        if self.name in ("entry", "exit"):
            # at bumper height, 0.35 m past where the bumper stops: crossed as soon as the car drives off,
            # while the car is still fully in view (detectors drop cars cut off by the frame edge)
            gx = self.gate.gx - 0.9 + 0.35
            g["line"] = self._img([(gx, W.LANE_Y - 3.0, 0.3), (gx, W.LANE_Y + 3.0, 0.3)])
        else:
            g["bays"] = {b.name: self._img([(x, y, 0) for x, y in b.corners]) for b in W.BAYS.values()}
            y0, y1 = W.LANE_Y - 4.2, W.LANE_Y + 4.2
            g["entry_zone"] = self._img([(-1.5, y0, 0), (3.5, y0, 0), (3.5, y1, 0), (-1.5, y1, 0)])
            g["exit_zone"] = self._img([(35.5, y0, 0), (41.5, y0, 0), (41.5, y1, 0), (35.5, y1, 0)])
        return g

    # ------------------------------------------------------------------ frame rendering
    def _visible(self, x, y) -> bool:
        pc = self.cam.to_cam(np.array([[x, y, 0.8]]))[0]
        if pc[2] < 1.0:
            return False
        u, v = self.cam.project_cam(pc[None])[0]
        return -900 < u < self.cam.W + 900 and -900 < v < self.cam.H + 900

    def render(self, visits: list[Visit], t: float, labels: list | None = None) -> np.ndarray:
        cam = self.cam
        img = self.bg.copy()
        present = []
        for v in visits:
            p = v.pose(t)
            if p is not None and self._visible(p[0], p[1]):
                present.append((v, p))
        # soft ground shadows
        mask = np.zeros(img.shape[:2], np.uint8)
        for v, (x, y, h) in present:
            poly = v.model.shadow(x, y, h)
            P = np.c_[poly, np.zeros(len(poly))]
            if (cam.to_cam(P)[:, 2] > 0.3).all():
                cv2.fillPoly(mask, [np.round(cam.project(P) * 8).astype(np.int32)], 255, cv2.LINE_AA, shift=3)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.5)
        img = (img * (1 - 0.42 * mask[..., None].astype(np.float32) / 255)).astype(np.uint8)
        # depth-sorted drawing of cars + gate props
        items = []
        if self.name != "lot":
            which = "exit" if self.name == "exit" else "entry"
            for part in self.gate.static + self.gate.arm_faces(barrier_openness(visits, t, which)):
                items.append((W.part_depth(cam, part), 0, part))
        for v, p in present:
            items.append((float(np.linalg.norm(np.array([p[0], p[1], 0.7]) - cam.C)), 1, (v, p)))
        items.sort(key=lambda z: -z[0])
        for _, kind, obj in items:
            if kind == 0:
                W.draw_part(img, cam, obj)
            else:
                v, (x, y, h) = obj
                v.model.draw(img, cam, x, y, h)
        if labels is not None:
            for v, (x, y, h) in present:
                lab = self._label(v, x, y, h)
                if lab is not None:
                    labels.append(lab)
        out = self.post(img)
        self._overlay(out)
        return out

    def _overlay(self, img):
        (tw, th), _ = cv2.getTextSize(CAM_LABELS[self.name], cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
        cv2.rectangle(img, (10, 10), (26 + tw, 24 + th), (20, 20, 20), -1)
        cv2.putText(img, CAM_LABELS[self.name], (18, 17 + th), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1,
                    cv2.LINE_AA)
        txt = "SIMULATED FOOTAGE"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, txt, (img.shape[1] - tw - 16, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)

    def _label(self, v: Visit, x, y, h):
        """Ground-truth box for training/eval. Detect: xyxy. Lot: 4-point OBB."""
        s = v.model.spec
        c, sn = np.cos(h), np.sin(h)
        corners = np.array([[dx, dy, dz] for dx in (-s.L / 2, s.L / 2) for dy in (-s.W / 2, s.W / 2) for dz in (0, s.H)])
        R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        P = corners @ R.T + [x, y, 0]
        if (self.cam.to_cam(P)[:, 2] < 0.5).any():
            return None
        uv = self.cam.project(P)
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        full = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.cam.W), min(y1, self.cam.H)
        if cx1 <= cx0 or cy1 <= cy0:
            return None
        frac = (cx1 - cx0) * (cy1 - cy0) / max(full, 1e-6)
        if frac < 0.45 or (cx1 - cx0) < 14 or (cy1 - cy0) < 14:
            return None
        lab = {"plate": v.plate, "xyxy": [float(cx0), float(cy0), float(cx1), float(cy1)], "visible": float(frac)}
        if self.name == "lot":
            rect = cv2.minAreaRect(uv.astype(np.float32))
            lab["obb"] = cv2.boxPoints(rect).tolist()
        return lab


def write_geometry(out_dir: Path, cams=("entry", "lot", "exit")) -> dict:
    geometry = {name: CameraRig(name).scene_geometry() for name in cams}
    (out_dir / "scene_geometry.json").write_text(json.dumps(geometry, indent=1))
    return geometry


def render_episode(plan: list[Visit], duration: float, fps: float, out_dir: Path, cams=("entry", "lot", "exit"),
                   label_every: int = 0, seed: int = 0, write_video: bool = True, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    visits = [build(v, duration) for v in plan]
    n = int(duration * fps)
    geometry, labels_all = {}, {}
    for name in cams:
        rig = CameraRig(name, seed=seed)
        geometry[name] = rig.scene_geometry()
        writer = None
        if write_video:
            writer = cv2.VideoWriter(str(out_dir / f"{name}_cam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (rig.cam.W, rig.cam.H))
        cam_labels = []
        for f in range(n):
            t = f / fps
            want = label_every and f % label_every == 0
            if not write_video and not want:
                continue
            labels = [] if want else None
            frame = rig.render(visits, t, labels)
            if writer is not None:
                writer.write(frame)
            if want:
                cam_labels.append({"frame": f, "t": t, "objects": labels, "image": frame})
            if f % int(fps * 20) == 0:
                log(f"  [{name}] {f}/{n} frames")
        if writer is not None:
            writer.release()
        labels_all[name] = cam_labels
    truth = {
        "fps": fps, "duration": duration,
        "vehicles": [{"plate": v.plate, "kind": v.kind, "color": v.color, "bay": v.bay,
                      "pre_parked": v.t_gate is None, "t_leave_bay": v.t_leave, **v.events} for v in visits],
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    (out_dir / "scene_geometry.json").write_text(json.dumps(geometry, indent=1))
    return {"truth": truth, "geometry": geometry, "labels": labels_all, "visits": visits}

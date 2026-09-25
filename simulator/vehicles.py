"""Low-poly 3D car models (sedan / hatchback / SUV / MPV) with real, readable number plates."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from parkwatch.plates.formats import sg_checksum

from .render import Camera, Face, box_faces, draw_face, orient_normals, shadow_polygon


def random_sg_plate(rng: np.random.Generator, prefix: str | None = None) -> str:
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ"  # I and O are not issued
    if prefix is None:
        prefix = "S" + "".join(rng.choice(list(alphabet), 2))
    digits = str(int(rng.integers(1, 9999)))
    return f"{prefix}{digits}{sg_checksum(prefix, digits)}"


def split_plate(plate: str) -> str:
    """'SBA1234K' -> 'SBA 1234 K' for display on the physical plate."""
    i = 0
    while i < len(plate) and plate[i].isalpha():
        i += 1
    j = i
    while j < len(plate) and plate[j].isdigit():
        j += 1
    return " ".join(p for p in (plate[:i], plate[i:j], plate[j:]) if p)


def plate_texture(plate: str, style: str = "white") -> np.ndarray:
    """Render a 520x120 number plate image (condensed bold characters)."""
    bg, fg = {"white": ((245, 245, 245), (15, 15, 15)), "yellow": ((40, 200, 245), (15, 15, 15)),
              "black": ((20, 20, 20), (235, 235, 235))}[style]
    text = split_plate(plate)
    wide = np.full((120, 820, 3), bg, np.uint8)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, 9
    scale = 3.0
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    scale *= min(760 / tw, 78 / th)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(wide, text, ((820 - tw) // 2, (120 + th) // 2), font, scale, fg, thick, cv2.LINE_AA)
    img = cv2.resize(wide, (520, 120), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (3, 3), (516, 116), fg, 4)
    return img


@dataclass(frozen=True)
class CarSpec:
    L: float
    W: float
    H: float
    clear: float
    z_front: float
    z_belt: float
    z_rear: float
    x_ws: float  # windshield base
    x_rw: float  # rear window base
    roof_front: float
    roof_back: float
    wheel_r: float = 0.32
    tumble: float = 0.40  # cabin top is this much narrower than the body


SPECS = {
    "sedan": CarSpec(4.60, 1.80, 1.45, 0.22, 0.70, 0.90, 0.88, 0.95, -1.35, 0.05, -0.95),
    "hatch": CarSpec(4.10, 1.76, 1.50, 0.22, 0.72, 0.92, 0.90, 0.85, -1.92, 0.00, -1.72),
    "suv": CarSpec(4.70, 1.90, 1.72, 0.32, 0.92, 1.06, 1.04, 1.05, -2.22, 0.20, -2.05, 0.38, 0.32),
    "mpv": CarSpec(4.85, 1.85, 1.85, 0.25, 0.80, 1.00, 0.98, 1.35, -2.30, 0.55, -2.22, 0.34, 0.30),
}


def _quad_uv(q: np.ndarray, u0, u1, v0, v1) -> np.ndarray:
    """Sub-quad of quad q = [B0, B1, T1, T0] in bilinear (u along B0->B1, v along B->T)."""
    B0, B1, T1, T0 = q

    def at(u, v):
        return (B0 + (B1 - B0) * u) * (1 - v) + (T0 + (T1 - T0) * u) * v

    return np.array([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])


class CarModel:
    """Builds the local-frame geometry once; `parts(x, y, heading)` returns posed faces."""

    def __init__(self, kind: str, color, plate: str, sunroof: bool = False):
        self.kind, self.color, self.plate = kind, tuple(float(c) for c in color), plate
        self.spec = s = SPECS[kind]
        self.front_plate = plate_texture(plate, "white")
        self.rear_plate = plate_texture(plate, "yellow")
        col = self.color
        L2, W2 = s.L / 2, s.W / 2
        c, cx = 0.20, 0.28  # plan-view corner chamfers

        def hood_z(x):
            return s.z_front + (L2 - x) * (s.z_belt - s.z_front) / (L2 - s.x_ws)

        def trunk_z(x):
            return s.z_rear + (x + L2) * (s.z_belt - s.z_rear) / (s.x_rw + L2)

        zf2, zr2 = hood_z(L2 - cx), trunk_z(-L2 + cx)
        # ---------------- lower body (convex prism with wheel-arch cut-outs on the sides)
        wb = s.L * 0.60 / 2
        body: list[Face] = []
        for side in (1, -1):
            y = side * W2
            prof = [(-L2 + cx, s.clear)]
            for xa in (-wb, wb):
                ra = s.wheel_r + 0.06
                prof.append((xa - ra, s.clear))
                for a in np.linspace(np.pi, 0, 9):
                    prof.append((xa + ra * np.cos(a), s.wheel_r + ra * np.sin(a)))
                prof.append((xa + ra, s.clear))
            prof += [(L2 - cx, s.clear), (L2 - cx, zf2), (s.x_ws, s.z_belt), (s.x_rw, s.z_belt), (-L2 + cx, zr2)]
            V = np.array([[px, y, pz] for px, pz in prof])
            rocker = np.array([[-L2 + cx, y, s.clear], [L2 - cx, y, s.clear], [L2 - cx, y, s.clear + 0.08],
                               [-L2 + cx, y, s.clear + 0.08]])
            body.append(Face(V, col, decals=[Face(rocker, (40, 40, 40), "dark", outline=False)]))
        front = np.array([[L2, -W2 + c, s.clear], [L2, W2 - c, s.clear], [L2, W2 - c, s.z_front], [L2, -W2 + c, s.z_front]])
        fq = np.array([front[0], front[1], front[2], front[3]])
        decals = [
            Face(_quad_uv(fq, 0.02, 0.26, 0.62, 0.93), (225, 235, 240), "emissive", outline=True),
            Face(_quad_uv(fq, 0.74, 0.98, 0.62, 0.93), (225, 235, 240), "emissive", outline=True),
            Face(_quad_uv(fq, 0.30, 0.70, 0.64, 0.92), (30, 30, 32), "dark"),
            Face(_quad_uv(fq, 0.18, 0.82, 0.08, 0.22), (35, 35, 38), "dark"),
            Face(self._plate_quad(L2, 0.36, 0.48, front=True), (255, 255, 255), "texture", texture=self.front_plate),
        ]
        body.append(Face(front, col, decals=decals))
        rear = np.array([[-L2, W2 - c, s.clear], [-L2, -W2 + c, s.clear], [-L2, -W2 + c, s.z_rear], [-L2, W2 - c, s.z_rear]])
        rq = rear
        rdecals = [
            Face(_quad_uv(rq, 0.02, 0.27, 0.60, 0.92), (40, 40, 200), "emissive"),
            Face(_quad_uv(rq, 0.73, 0.98, 0.60, 0.92), (40, 40, 200), "emissive"),
            Face(_quad_uv(rq, 0.10, 0.90, 0.04, 0.16), (35, 35, 38), "dark"),
            Face(self._plate_quad(-L2, 0.42, 0.54, front=False), (255, 255, 255), "texture", texture=self.rear_plate),
        ]
        body.append(Face(rear, col, decals=rdecals))
        for side in (1, -1):  # chamfered corners
            body.append(Face(np.array([[L2, side * (W2 - c), s.clear], [L2 - cx, side * W2, s.clear],
                                       [L2 - cx, side * W2, zf2], [L2, side * (W2 - c), s.z_front]]), col))
            body.append(Face(np.array([[-L2, side * (W2 - c), s.clear], [-L2 + cx, side * W2, s.clear],
                                       [-L2 + cx, side * W2, zr2], [-L2, side * (W2 - c), s.z_rear]]), col))
        hood = np.array([[L2, -W2 + c, s.z_front], [L2, W2 - c, s.z_front], [L2 - cx, W2, zf2], [s.x_ws, W2, s.z_belt],
                         [s.x_ws, -W2, s.z_belt], [L2 - cx, -W2, zf2]])
        body.append(Face(hood, col))
        body.append(Face(np.array([[s.x_ws, -W2, s.z_belt], [s.x_ws, W2, s.z_belt], [s.x_rw, W2, s.z_belt],
                                   [s.x_rw, -W2, s.z_belt]]), col, outline=False))
        trunk = np.array([[s.x_rw, -W2, s.z_belt], [s.x_rw, W2, s.z_belt], [-L2 + cx, W2, zr2], [-L2, W2 - c, s.z_rear],
                          [-L2, -W2 + c, s.z_rear], [-L2 + cx, -W2, zr2]])
        body.append(Face(trunk, col))
        orient_normals(body, np.array([0, 0, (s.clear + s.z_belt) / 2]))
        # ---------------- cabin / greenhouse
        Wc, Wt = W2 - 0.06, W2 - s.tumble
        zb, H = s.z_belt, s.H
        B = {"rl": [s.x_rw, Wc, zb], "fl": [s.x_ws, Wc, zb], "fr": [s.x_ws, -Wc, zb], "rr": [s.x_rw, -Wc, zb]}
        T = {"rl": [s.roof_back, Wt, H], "fl": [s.roof_front, Wt, H], "fr": [s.roof_front, -Wt, H], "rr": [s.roof_back, -Wt, H]}
        B = {k: np.array(v, float) for k, v in B.items()}
        T = {k: np.array(v, float) for k, v in T.items()}
        glass = (58, 50, 45)
        cabin: list[Face] = []
        ws = np.array([B["fr"], B["fl"], T["fl"], T["fr"]])
        cabin.append(Face(ws, col, decals=[Face(_quad_uv(ws, 0.07, 0.93, 0.04, 0.94), glass, "glass", outline=False)]))
        rw = np.array([B["rl"], B["rr"], T["rr"], T["rl"]])
        cabin.append(Face(rw, col, decals=[Face(_quad_uv(rw, 0.08, 0.92, 0.06, 0.92), glass, "glass", outline=False)]))
        for side, (b0, b1, t1, t0) in (("l", ("rl", "fl", "fl", "rl")), ("r", ("rr", "fr", "fr", "rr"))):
            q = np.array([B[b0], B[b1], T[t1], T[t0]])
            wins = [Face(_quad_uv(q, 0.06, 0.47, 0.10, 0.88), glass, "glass", outline=False),
                    Face(_quad_uv(q, 0.53, 0.95, 0.10, 0.88), glass, "glass", outline=False)]
            cabin.append(Face(q, col, decals=wins))
        roof = np.array([T["rr"], T["fr"], T["fl"], T["rl"]])
        rdec = [Face(_quad_uv(roof, 0.35, 0.75, 0.25, 0.75), (40, 40, 45), "glass", outline=False)] if sunroof else []
        cabin.append(Face(roof, col, decals=rdec))
        orient_normals(cabin, np.array([(s.x_ws + s.x_rw) / 2, 0, (zb + H) / 2]))
        # ---------------- wheels (octagonal prisms)
        self.wheels = []
        for xa in (-wb, wb):
            for side in (1, -1):
                self.wheels.append(self._wheel(xa, side * (W2 - 0.14), s.wheel_r, side))
        # ---------------- mirrors
        self.mirrors = []
        for side in (1, -1):
            self.mirrors.append(box_faces((s.x_ws - 0.05, side * (W2 + 0.09), zb - 0.05), (0.16, 0.16, 0.12), 0, col))
        self.body, self.cabin = body, cabin
        # footprint + roof points for the ground shadow
        self.shadow_pts = np.array([[x, y, z] for x in (-L2, L2) for y in (-W2, W2) for z in (0.0, s.z_belt)]
                                   + [[x, y, H] for x in (s.roof_back, s.roof_front) for y in (-Wt, Wt)])

    def _plate_quad(self, x, z0, z1, front=True):
        y = -0.26 if front else 0.26
        # TL, TR, BR, BL as seen by a viewer facing the plate
        return np.array([[x, y, z1], [x, -y, z1], [x, -y, z0], [x, y, z0]])

    def _wheel(self, xa, yc, r, side):
        ang = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8
        w = 0.22
        yo, yi = yc + side * w / 2, yc - side * w / 2
        ring = np.stack([xa + r * np.cos(ang), r + r * np.sin(ang)], 1)
        outer = np.array([[px, yo, pz] for px, pz in ring])
        inner = np.array([[px, yi, pz] for px, pz in ring])
        hub = np.array([[xa + 0.55 * r * np.cos(a), yo + side * 0.005, r + 0.55 * r * np.sin(a)] for a in ang])
        faces = [Face(outer, (28, 28, 30), "dark", decals=[Face(hub, (165, 165, 170), "paint", outline=True)]),
                 Face(inner, (28, 28, 30), "dark")]
        for i in range(8):
            j = (i + 1) % 8
            faces.append(Face(np.array([outer[i], outer[j], inner[j], inner[i]]), (32, 32, 34), "dark", outline=False))
        orient_normals(faces, np.array([xa, yc, r]))
        return faces

    # -------------------------------------------------------------- posing / drawing
    @staticmethod
    def _xf(fc: Face, R: np.ndarray, t: np.ndarray) -> Face:
        return Face(fc.verts @ R.T + t, fc.color, fc.kind, fc.texture, [CarModel._xf(d, R, t) for d in fc.decals],
                    None if fc.normal is None else fc.normal @ R.T, fc.outline, fc.two_sided)

    def draw(self, img: np.ndarray, cam: Camera, x: float, y: float, heading: float) -> None:
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        t = np.array([x, y, 0.0])
        # far-side parts first; all cameras sit above roof height so body -> cabin order is safe
        for group in (*self.wheels, self.body, *self.mirrors, self.cabin):
            for fc in group:
                draw_face(img, self._xf(fc, R, t), cam)

    def shadow(self, x: float, y: float, heading: float) -> np.ndarray:
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return shadow_polygon(self.shadow_pts @ R.T + [x, y, 0.0])

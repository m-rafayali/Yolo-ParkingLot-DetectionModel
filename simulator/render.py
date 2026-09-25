"""Tiny software renderer used to build the synthetic car-park footage.

It is deliberately simple: flat-shaded convex polygons drawn with OpenCV in
painter's order, a textured ground plane warped through the camera homography,
and a CCTV-style post-process (blur, sensor noise, JPEG artefacts).

World frame: x = east, y = north, z = up, metres.
Camera frame (OpenCV): x = right, y = down, z = forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SUN = np.array([0.45, -0.35, 0.82])
SUN = SUN / np.linalg.norm(SUN)
SKY_BGR = np.array([215.0, 200.0, 185.0])


class Camera:
    """Pinhole camera defined by position, look-at target and horizontal FOV."""

    def __init__(self, pos, target, hfov_deg, size=(1280, 720), up=(0.0, 0.0, 1.0)):
        self.W, self.H = size
        self.C = np.asarray(pos, float)
        f = np.asarray(target, float) - self.C
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, float))
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        self.R = np.stack([r, d, f])
        self.f = (self.W / 2) / np.tan(np.radians(hfov_deg) / 2)
        self.K = np.array([[self.f, 0, self.W / 2], [0, self.f, self.H / 2], [0, 0, 1.0]])

    def to_cam(self, P: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(P) - self.C) @ self.R.T

    def project_cam(self, Pc: np.ndarray) -> np.ndarray:
        return np.stack([self.f * Pc[:, 0] / Pc[:, 2] + self.W / 2, self.f * Pc[:, 1] / Pc[:, 2] + self.H / 2], 1)

    def project(self, P: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixel coords (N,2). Points must be in front of the camera."""
        return self.project_cam(self.to_cam(P))

    def ground_homography(self) -> np.ndarray:
        """3x3 homography mapping ground-plane world (x, y, 1) to image pixels."""
        P = self.K @ np.hstack([self.R, -self.R @ self.C[:, None]])
        return P[:, [0, 1, 3]]


def clip_near(Pc: np.ndarray, near: float = 0.25):
    """Sutherland-Hodgman clip of a camera-space polygon against the near plane."""
    out = []
    n = len(Pc)
    for i in range(n):
        a, b = Pc[i], Pc[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


@dataclass
class Face:
    """A planar polygon with a material. Decals are drawn right after their parent face."""

    verts: np.ndarray  # (N,3) world
    color: tuple  # BGR 0..255
    kind: str = "paint"  # paint | glass | dark | light | emissive | texture
    texture: np.ndarray | None = None  # for kind == "texture" (4 verts, TL,TR,BR,BL order)
    decals: list = field(default_factory=list)
    normal: np.ndarray | None = None
    outline: bool = True
    two_sided: bool = False


def orient_normals(faces: list[Face], centroid: np.ndarray) -> None:
    """Give each face an outward normal (pointing away from the part centroid)."""
    for fc in faces:
        v = fc.verts
        n = np.cross(v[1] - v[0], v[2] - v[0])
        if np.linalg.norm(n) < 1e-9:
            n = np.cross(v[2] - v[0], v[-1] - v[0])
        n = n / (np.linalg.norm(n) + 1e-12)
        if np.dot(n, v.mean(0) - centroid) < 0:
            n = -n
        fc.normal = n
        for d in fc.decals:
            d.normal = n


def shade(fc: Face, cam: Camera) -> tuple:
    base = np.asarray(fc.color, float)
    n = fc.normal if fc.normal is not None else np.array([0, 0, 1.0])
    lam = max(0.0, float(np.dot(n, SUN)))
    if fc.kind == "glass":
        view = cam.C - fc.verts.mean(0)
        view /= np.linalg.norm(view) + 1e-9
        fres = 0.18 + 0.5 * (1 - abs(float(np.dot(view, n)))) ** 3 + 0.18 * max(0.0, n[2])
        c = base * (1 - fres) + SKY_BGR * fres * 0.75
    elif fc.kind in ("emissive", "texture"):
        c = base * (0.8 + 0.2 * lam)
    elif fc.kind == "dark":
        c = base * (0.7 + 0.3 * lam)
    else:
        c = base * (0.5 + 0.55 * lam)
    return tuple(float(x) for x in np.clip(c, 0, 255))


def _poly_pts(pts: np.ndarray) -> np.ndarray:
    return np.round(pts * 16).astype(np.int32)


def draw_face(img: np.ndarray, fc: Face, cam: Camera, cull: bool = True) -> bool:
    """Draw one face (+decals). Returns False when culled/clipped away."""
    if cull and not fc.two_sided and fc.normal is not None:
        if np.dot(fc.normal, cam.C - fc.verts.mean(0)) <= 0:
            return False
    Pc = cam.to_cam(fc.verts)
    if (Pc[:, 2] < 0.25).all():
        return False
    clipped = Pc if (Pc[:, 2] >= 0.25).all() else clip_near(Pc)
    if clipped is None:
        return False
    pts = cam.project_cam(clipped)
    if (pts[:, 0].max() < -50 or pts[:, 0].min() > cam.W + 50 or pts[:, 1].max() < -50 or pts[:, 1].min() > cam.H + 50):
        return False
    if abs(pts[:, 0].max()) > 1e5 or abs(pts[:, 1].max()) > 1e5:
        return False
    color = shade(fc, cam)
    if fc.kind == "texture" and fc.texture is not None and clipped is Pc:
        _draw_textured(img, fc.texture, pts, color)
    else:
        cv2.fillPoly(img, [_poly_pts(pts)], color, cv2.LINE_AA, shift=4)
        if fc.outline:
            edge = tuple(c * 0.72 for c in color)
            cv2.polylines(img, [_poly_pts(pts)], True, edge, 1, cv2.LINE_AA, shift=4)
    for d in fc.decals:
        draw_face(img, d, cam, cull=False)
    return True


def _draw_textured(img: np.ndarray, tex: np.ndarray, dst: np.ndarray, color) -> None:
    x0, y0 = np.floor(dst.min(0)).astype(int)
    x1, y1 = np.ceil(dst.max(0)).astype(int) + 1
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1c <= x0c or y1c <= y0c:
        return
    w, h = x1 - x0, y1 - y0
    if w < 6 or h < 3:
        cv2.fillPoly(img, [_poly_pts(dst)], tuple(float(c) for c in tex.reshape(-1, 3).mean(0)), cv2.LINE_AA, shift=4)
        return
    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, np.float32(dst - [x0, y0]))
    warped = cv2.warpPerspective(tex, M, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
    k = np.float32(color).mean() / 255.0
    roi = img[y0c:y1c, x0c:x1c].astype(np.float32)
    wv = warped[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0].astype(np.float32) * min(1.0, k + 0.1)
    a = mask[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0, None].astype(np.float32) / 255.0
    img[y0c:y1c, x0c:x1c] = (roi * (1 - a) + wv * a).astype(np.uint8)


def box_faces(center, size, yaw: float = 0.0, color=(128, 128, 128), kind: str = "paint") -> list[Face]:
    """Axis-aligned (then yawed) box. center = (x, y, z_bottom), size = (lx, ly, lz)."""
    cx, cy, z0 = center
    lx, ly, lz = size
    xs, ys = np.array([-lx / 2, lx / 2]), np.array([-ly / 2, ly / 2])
    corners = np.array([[x, y, z] for z in (0.0, lz) for y in ys for x in xs])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    V = corners @ rot.T + [cx, cy, z0]
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    faces = [Face(V[list(q)], color, kind) for q in idx]
    orient_normals(faces, V.mean(0))
    return faces


def shadow_polygon(points: np.ndarray) -> np.ndarray:
    """Project 3D points onto z=0 along the sun direction; return their 2D convex hull."""
    t = points[:, 2:3] / SUN[2]
    g = points[:, :2] - t * SUN[:2]
    hull = cv2.convexHull(g.astype(np.float32)).reshape(-1, 2)
    return hull


class Post:
    """CCTV look: slight blur, luminance noise, vignette, JPEG round-trip."""

    def __init__(self, size, seed=0, noise=3.0, blur=0.6, jpeg=72):
        W, H = size
        rng = np.random.default_rng(seed)
        self.noise = [rng.normal(0, noise, (H, W, 1)).astype(np.float32) for _ in range(6)]
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = (1 - 0.18 * np.clip(r - 0.35, 0, None) ** 2)[..., None].astype(np.float32)
        self.blur, self.jpeg, self.i = blur, jpeg, 0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.blur:
            img = cv2.GaussianBlur(img, (0, 0), self.blur)
        f = img.astype(np.float32) * self.vignette + self.noise[self.i % len(self.noise)]
        self.i += 1
        out = np.clip(f, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

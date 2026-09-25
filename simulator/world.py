"""Static geometry of the synthetic car park: ground textures, props, bays and cameras.

Layout (metres, x = east, y = north):

    entry road  ->  ENTRY GATE (x=-14)  ->  LOT (x 0..40, 24 bays)  ->  EXIT GATE (x=72)  ->  exit road
                     CAM-01 entry cam        CAM-02 overhead cam          CAM-03 exit cam

All traffic is one-way, eastbound, along the lane centre y = 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .render import Camera, Face, box_faces, draw_face

LANE_Y = 11.0
ENTRY_BARRIER_X = -14.0
EXIT_BARRIER_X = 72.0
BAY_W, BAY_D = 2.5, 5.0
BAY_X0 = 4.0
N_BAYS_PER_ROW = 12
TOP_ROW = (15.5, 20.5)  # y range
BOT_ROW = (1.5, 6.5)
SIZE = (1280, 720)


@dataclass(frozen=True)
class Bay:
    name: str
    cx: float
    cy: float
    row: str  # "T" (north, cars face north) or "B" (south, cars face south)

    @property
    def corners(self):
        y0, y1 = TOP_ROW if self.row == "T" else BOT_ROW
        return [(self.cx - BAY_W / 2, y0), (self.cx + BAY_W / 2, y0), (self.cx + BAY_W / 2, y1), (self.cx - BAY_W / 2, y1)]


def all_bays() -> list[Bay]:
    bays = []
    for row, (y0, y1) in (("T", TOP_ROW), ("B", BOT_ROW)):
        for k in range(N_BAYS_PER_ROW):
            bays.append(Bay(f"{row}{k + 1:02d}", BAY_X0 + BAY_W * (k + 0.5), (y0 + y1) / 2, row))
    return bays


BAYS = {b.name: b for b in all_bays()}


# ------------------------------------------------------------------ cameras
def make_cameras() -> dict[str, Camera]:
    ex, xx = ENTRY_BARRIER_X, EXIT_BARRIER_X
    return {
        # roadside ANPR-style cameras: ~3.6 m high, looking diagonally at approaching fronts
        "entry": Camera((ex + 7.5, LANE_Y - 5.0, 3.6), (ex - 6.0, LANE_Y + 0.4, 0.2), 34),
        "lot": Camera((20.0, 11.25, 32.0), (20.0, 11.25, 0.0), 64.0, up=(0.0, 1.0, 0.0)),
        "exit": Camera((xx + 7.5, LANE_Y - 5.0, 3.6), (xx - 6.0, LANE_Y + 0.4, 0.2), 34),
    }


# ------------------------------------------------------------------ ground texture
class Painter:
    """Paints the ground plane of a world region into a texture image."""

    def __init__(self, region, ppm, rng):
        self.x0, self.y0, self.x1, self.y1 = region
        self.ppm, self.rng = ppm, rng
        w, h = int((self.x1 - self.x0) * ppm), int((self.y1 - self.y0) * ppm)
        self.img = self._noise((h, w), (70, 110, 85), 10, 0.25)  # grass everywhere

    def _noise(self, shape, base, amp, blotch):
        h, w = shape
        n = self.rng.normal(0, amp, (h, w, 1)).astype(np.float32)
        lo = cv2.resize(self.rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32), (w, h))[..., None]
        img = np.float32(base)[None, None] * (1 + blotch * 0.25 * lo) + n
        return np.clip(img, 0, 255).astype(np.uint8)

    def px(self, pts):
        pts = np.asarray(pts, float)
        return np.stack([(pts[:, 0] - self.x0) * self.ppm, (self.y1 - pts[:, 1]) * self.ppm], 1)

    def poly(self, pts, color):
        cv2.fillPoly(self.img, [np.round(self.px(pts) * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

    def textured(self, pts, base, amp, blotch=1.0):
        mask = np.zeros(self.img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(self.px(pts)).astype(np.int32)], 255)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tex = self._noise((y1 - y0, x1 - x0), base, amp, blotch)
        m = mask[y0:y1, x0:x1, None] > 0
        self.img[y0:y1, x0:x1] = np.where(m, tex, self.img[y0:y1, x0:x1])

    def line(self, a, b, width_m, color):
        pa, pb = self.px([a, b])
        cv2.line(self.img, tuple(np.round(pa * 16).astype(int)), tuple(np.round(pb * 16).astype(int)), color,
                 max(1, int(round(width_m * self.ppm))), cv2.LINE_AA, shift=4)

    def rect(self, x0, y0, x1, y1, color):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], color)

    def text(self, s, x, y, height_m, color, angle=0):
        """Road text painted on the ground, readable when looking in direction `angle` (rad)."""
        font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(2, int(0.12 * height_m * self.ppm))
        scale = height_m * self.ppm / 22.0
        (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
        pad = thick * 2
        stamp = np.zeros((th + 2 * pad, tw + 2 * pad), np.uint8)
        cv2.putText(stamp, s, (pad, th + pad), font, scale, 255, thick, cv2.LINE_AA)
        stamp = cv2.resize(stamp, (stamp.shape[1], int(stamp.shape[0] * 2.2)))  # road text is elongated
        k = (int(round(np.degrees(angle) / 90)) - 1) % 4
        stamp = np.rot90(stamp, k)
        cx, cy = self.px([(x, y)])[0]
        h, w = stamp.shape
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        if x0 < 0 or y0 < 0 or x0 + w > self.img.shape[1] or y0 + h > self.img.shape[0]:
            return
        a = stamp[..., None].astype(np.float32) / 255 * 0.85
        roi = self.img[y0:y0 + h, x0:x0 + w].astype(np.float32)
        self.img[y0:y0 + h, x0:x0 + w] = (roi * (1 - a) + np.float32(color) * a).astype(np.uint8)

    def tree(self, x, y, r):
        cx, cy = self.px([(x, y)])[0]
        rp = r * self.ppm
        sh = (int(cx + 0.5 * rp), int(cy + 0.45 * rp))
        ov = self.img.copy()
        cv2.circle(ov, sh, int(rp), (30, 45, 35), -1, cv2.LINE_AA)
        self.img = cv2.addWeighted(ov, 0.55, self.img, 0.45, 0)
        for i, (dx, dy, rr, col) in enumerate([(0, 0, 1.0, (40, 95, 50)), (-0.25, -0.2, 0.7, (55, 120, 65)),
                                                (0.2, -0.3, 0.45, (70, 140, 80)), (-0.1, 0.25, 0.5, (45, 105, 55))]):
            cv2.circle(self.img, (int(cx + dx * rp), int(cy + dy * rp)), int(rr * rp), col, -1, cv2.LINE_AA)


def paint_ground(region, ppm, seed=7) -> Painter:
    rng = np.random.default_rng(seed)
    p = Painter(region, ppm, rng)
    asphalt, amp = (88, 90, 93), 9
    road_y0, road_y1 = LANE_Y - 2.6, LANE_Y + 2.6
    # sidewalks / curbs around the paved area
    p.textured([(-80, road_y0 - 1.6), (0, road_y0 - 1.6), (0, road_y1 + 1.6), (-80, road_y1 + 1.6)], (165, 168, 170), 6, 0.2)
    p.textured([(40, road_y0 - 1.6), (130, road_y0 - 1.6), (130, road_y1 + 1.6), (40, road_y1 + 1.6)], (165, 168, 170), 6, 0.2)
    p.textured([(-1.0, 0.3), (41.0, 0.3), (41.0, 22.2), (-1.0, 22.2)], (165, 168, 170), 6, 0.2)
    # asphalt: roads + lot
    p.textured([(-80, road_y0), (130, road_y0), (130, road_y1), (-80, road_y1)], asphalt, amp, 0.35)
    p.textured([(-0.6, 0.7), (40.6, 0.7), (40.6, 21.8), (-0.6, 21.8)], asphalt, amp, 0.35)
    for y in (road_y0, road_y1):  # kerb edge lines
        p.line((-80, y), (-0.6, y), 0.12, (150, 150, 150))
        p.line((40.6, y), (130, y), 0.12, (150, 150, 150))
    # landscaped islands at both ends of each bay row
    for x0, x1 in ((0.2, 3.6), (34.4, 40.2)):
        for y0, y1 in ((TOP_ROW[0] + 0.3, 21.5), (1.0, BOT_ROW[1] - 0.3)):
            p.textured([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (165, 168, 170), 5, 0.2)
            p.textured([(x0 + 0.25, y0 + 0.25), (x1 - 0.25, y0 + 0.25), (x1 - 0.25, y1 - 0.25), (x0 + 0.25, y1 - 0.25)],
                       (60, 115, 75), 12, 0.6)
    # bay markings
    white = (228, 230, 232)
    for row in (TOP_ROW, BOT_ROW):
        for k in range(N_BAYS_PER_ROW + 1):
            x = BAY_X0 + k * BAY_W
            p.line((x, row[0]), (x, row[1]), 0.12, white)
        back = row[1] if row is TOP_ROW else row[0]
        p.line((BAY_X0, back), (BAY_X0 + N_BAYS_PER_ROW * BAY_W, back), 0.12, white)
    for b in BAYS.values():
        yy = b.cy + (1.9 if b.row == "T" else -1.9)
        p.text(b.name, b.cx, yy, 0.32, (220, 222, 224), angle=np.pi / 2 if b.row == "T" else -np.pi / 2)
    # direction arrows on the aisle and roads
    for x in (-30, 1.5, 19, 37, 50, 88):
        p.poly([(x, LANE_Y - 0.12), (x + 2.0, LANE_Y - 0.12), (x + 2.0, LANE_Y - 0.4), (x + 2.9, LANE_Y),
                (x + 2.0, LANE_Y + 0.4), (x + 2.0, LANE_Y + 0.12), (x, LANE_Y + 0.12)], white)
    # gates: stop lines, hatching, painted words
    for gx, word in ((ENTRY_BARRIER_X, "ENTRY"), (EXIT_BARRIER_X, "EXIT")):
        p.line((gx - 0.6, road_y0), (gx - 0.6, road_y1), 0.35, white)
        for i in range(6):
            yy = road_y0 + 0.4 + i * 0.85
            p.line((gx - 0.2, yy), (gx + 0.6, yy + 0.5), 0.15, (40, 200, 235))
        p.text(word, gx - 8.0, LANE_Y, 0.9, white, angle=0)
        p.text("SLOW", gx - 16.0, LANE_Y, 0.9, white, angle=0)
    # centre dashes on roads
    for x in np.arange(-80, 130, 6.0):
        if -1 < x < 41:
            continue
        p.line((x, road_y0 + 0.25), (x + 3, road_y0 + 0.25), 0.1, (40, 200, 235))
        p.line((x, road_y1 - 0.25), (x + 3, road_y1 - 0.25), 0.1, (40, 200, 235))
    # trees on the verges and islands
    for (x, y, r) in [(1.9, 19.0, 1.4), (1.9, 3.5, 1.3), (37.3, 18.7, 1.6), (37.3, 3.4, 1.5), (-3.5, 2.0, 2.0),
                      (-3.8, 20.5, 2.2), (43.6, 1.5, 2.0), (44.0, 20.2, 2.3), (12, 23.8, 1.8), (26, 24.0, 1.9),
                      (8, -1.6, 1.7), (30, -1.8, 1.9)]:
        p.tree(x, y, r)
    return p


def ground_image(cam: Camera, painter: Painter) -> np.ndarray:
    T = np.array([[1 / painter.ppm, 0, painter.x0], [0, -1 / painter.ppm, painter.y1], [0, 0, 1.0]])
    H = cam.ground_homography() @ T
    return cv2.warpPerspective(painter.img, H, (cam.W, cam.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ props (3D boxes)
def sign_texture(text: str, bg, fg=(255, 255, 255)) -> np.ndarray:
    img = np.full((110, 360, 3), bg, np.uint8)
    font, thick = cv2.FONT_HERSHEY_DUPLEX, 5
    (tw, th), _ = cv2.getTextSize(text, font, 2.0, thick)
    s = min(320 / tw, 70 / th) * 2.0
    (tw, th), _ = cv2.getTextSize(text, font, s, thick)
    cv2.putText(img, text, ((360 - tw) // 2, (110 + th) // 2), font, s, fg, thick, cv2.LINE_AA)
    cv2.rectangle(img, (4, 4), (355, 105), fg, 4)
    return img


class Gate:
    """Barrier + booth + sign + hedges for one gate. The arm opening angle is animated."""

    def __init__(self, gx: float, word: str, color_bg):
        self.gx = gx
        road_y0 = LANE_Y - 2.6
        self.pivot = np.array([gx, road_y0 - 0.45, 1.0])
        self.static: list[list[Face]] = []
        self.static.append(box_faces((gx, road_y0 - 0.45, 0), (0.35, 0.35, 1.1), 0, (40, 190, 235)))  # post
        self.static.append(box_faces((gx - 1.3, road_y0 - 0.9, 0), (0.5, 0.45, 1.35), 0, (150, 150, 150)))  # ticket box
        self.static.append(box_faces((gx - 5.0, road_y0 - 3.4, 0), (3.2, 2.6, 2.7), 0, (175, 195, 210)))  # guard house
        self.static.append(box_faces((gx - 5.0, road_y0 - 3.4, 2.7), (3.6, 3.0, 0.18), 0, (70, 70, 80)))  # roof
        # sign on a post, facing approaching traffic (-x)
        sx = gx - 2.5
        self.static.append(box_faces((sx, LANE_Y + 3.6, 0), (0.12, 0.12, 2.3), 0, (120, 120, 120)))
        board = box_faces((sx, LANE_Y + 3.6, 2.1), (0.08, 1.5, 0.5), 0, color_bg)
        tex = sign_texture(word, color_bg)
        yl, yr = LANE_Y + 3.6 + 0.75, LANE_Y + 3.6 - 0.75
        quad = np.array([[sx - 0.045, yl, 2.6], [sx - 0.045, yr, 2.6], [sx - 0.045, yr, 2.1], [sx - 0.045, yl, 2.1]])
        for fc in board:
            if fc.normal is not None and fc.normal[0] < -0.9:
                fc.decals.append(Face(quad, (255, 255, 255), "texture", texture=tex))
        self.static.append(board)
        # hedges and lamp posts along the approach
        for x0 in np.arange(gx - 40, gx + 10, 7.0):
            if abs(x0 - (gx - 5)) < 4:
                continue
            self.static.append(box_faces((x0, road_y0 - 2.2, 0), (5.5, 0.9, 0.9), 0, (50, 100, 60)))
            self.static.append(box_faces((x0 + 1.0, LANE_Y + 4.3, 0), (5.5, 0.9, 0.9), 0, (50, 100, 60)))
        for x0 in np.arange(gx - 34, gx + 10, 14.0):
            self.static.append(box_faces((x0, LANE_Y + 3.2, 0), (0.14, 0.14, 4.5), 0, (110, 110, 115)))

    def arm_faces(self, openness: float) -> list[list[Face]]:
        """Striped arm, 4.8 m long, rotating up about the x axis. openness 0=closed, 1=open."""
        ang = openness * np.radians(84)
        d = np.array([0.0, np.cos(ang), np.sin(ang)])
        parts = []
        n = 8
        for i in range(n):
            a = self.pivot + d * (0.2 + 4.6 * i / n)
            b = self.pivot + d * (0.2 + 4.6 * (i + 1) / n)
            mid = (a + b) / 2
            L = np.linalg.norm(b - a)
            col = (40, 40, 210) if i % 2 == 0 else (240, 240, 240)
            fcs = box_faces((mid[0], mid[1], mid[2] - 0.05), (0.09, L, 0.1), 0, col)
            if ang > 1e-3:  # rotate the segment about its centre to follow the arm
                c, s = np.cos(ang), np.sin(ang)
                Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                ctr = np.array([mid[0], mid[1], mid[2]])
                for fc in fcs:
                    fc.verts = (fc.verts - ctr) @ Rx.T + ctr
                    fc.normal = fc.normal @ Rx.T
            parts.append(fcs)
        return parts

    @property
    def anchor(self):
        return np.array([self.gx, LANE_Y - 3.0, 1.0])


def draw_part(img, cam, faces):
    for fc in faces:
        draw_face(img, fc, cam)


def part_depth(cam: Camera, faces) -> float:
    c = np.mean([fc.verts.mean(0) for fc in faces], 0)
    return float(np.linalg.norm(c - cam.C))

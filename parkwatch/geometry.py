"""Small geometry helpers: line crossing, point-in-polygon."""

from __future__ import annotations

import numpy as np

DIRECTIONS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def _side(a, b, p) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1, p2, q1, q2) -> bool:
    d1, d2 = _side(q1, q2, p1), _side(q1, q2, p2)
    d3, d4 = _side(p1, p2, q1), _side(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def crossed_line(line, prev_pt, cur_pt, direction: str = "any") -> bool:
    """True when the movement prev_pt -> cur_pt crosses the counting line in the wanted direction."""
    a, b = line
    if not segments_intersect(prev_pt, cur_pt, a, b):
        return False
    if direction == "any":
        return True
    dx, dy = DIRECTIONS[direction]
    return (cur_pt[0] - prev_pt[0]) * dx + (cur_pt[1] - prev_pt[1]) * dy > 0


def point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def poly_centroid(poly) -> tuple[float, float]:
    p = np.asarray(poly, float)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def scale_points(pts, sx: float, sy: float):
    return [[float(x) * sx, float(y) * sy] for x, y in pts]

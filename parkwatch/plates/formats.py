"""Plate text normalisation and validation.

Validation matters because it lets the pipeline reject a misread *without* a second
paid LLM call. Singapore plates carry a checksum letter, so a single wrong character
is almost always caught.
"""

from __future__ import annotations

import re

_SG_WEIGHTS = (9, 4, 5, 4, 3, 2)
_SG_LETTERS = "AZYXUTSRPMLKJHGEDCB"
_SG_RE = re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z])$")

# OCR/LLM confusions, applied only where the plate grammar says a letter or digit must be.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}.items()}


def normalize(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def sg_checksum(prefix: str, digits: str) -> str:
    """Checksum letter for a Singapore plate, e.g. sg_checksum('SBA', '1234') -> 'K'."""
    letters = prefix[-2:] if len(prefix) >= 2 else prefix.rjust(2, "@")
    vals = [(ord(c) - 64) if c != "@" else 0 for c in letters] + [int(d) for d in digits.rjust(4, "0")]
    return _SG_LETTERS[sum(w * v for w, v in zip(_SG_WEIGHTS, vals)) % 19]


def sg_valid(plate: str) -> bool:
    m = _SG_RE.match(plate)
    return bool(m) and sg_checksum(m.group(1), m.group(2)) == m.group(3)


def _sg_repair(plate: str) -> str | None:
    """Try grammar-guided character swaps (0<->O, 1<->I, 8<->B ...) until the checksum passes."""
    if not 3 <= len(plate) <= 8:
        return None
    for split in range(1, 4):  # prefix length
        for dlen in range(1, 5):
            if split + dlen + 1 != len(plate):
                continue
            pre, dig, chk = plate[:split], plate[split:split + dlen], plate[-1]
            pre = "".join(_TO_LETTER.get(c, c) for c in pre)
            dig = "".join(_TO_DIGIT.get(c, c) for c in dig)
            chk = _TO_LETTER.get(chk, chk)
            cand = pre + dig + chk
            if sg_valid(cand):
                return cand
    return None


class PlateFormat:
    """`sg` = Singapore (regex + checksum), `generic` = 4-10 alphanumerics, `none` = accept anything."""

    def __init__(self, name: str = "generic"):
        if name not in ("sg", "generic", "none"):
            raise ValueError(f"unknown plate format {name!r}")
        self.name = name

    def validate(self, text: str | None) -> tuple[str | None, bool]:
        """Return (normalised plate or None, is_valid)."""
        p = normalize(text)
        if not p:
            return None, False
        if self.name == "sg":
            if sg_valid(p):
                return p, True
            fixed = _sg_repair(p)
            return (fixed, True) if fixed else (p, False)
        if self.name == "generic":
            return p, bool(re.fullmatch(r"[A-Z0-9]{4,10}", p))
        return p, True


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_match(plate: str, candidates, max_dist: int = 1) -> str | None:
    """Closest candidate within `max_dist` edits, only if unambiguous."""
    scored = sorted((edit_distance(plate, c), c) for c in candidates)
    if not scored or scored[0][0] > max_dist:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None
    return scored[0][1]


__all__ = ["PlateFormat", "best_match", "edit_distance", "normalize", "sg_checksum", "sg_valid"]

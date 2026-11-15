"""Shared helpers: class-name normalisation, box geometry, duplicate fingerprints."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class ClassMap:
    """The instrument taxonomy: canonical classes, their groups/departments, and label aliases."""

    IGNORE, EXCLUDE = "__ignore__", "__exclude_image__"

    def __init__(self, cfg_path: Path = ROOT / "configs" / "classes.yaml"):
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        classes = cfg["classes"]
        self.names: list[str] = list(classes)
        self.group = {n: c.get("group", "other") for n, c in classes.items()}
        self.depts = {n: list(c.get("depts", [])) for n, c in classes.items()}
        self._lookup = {}
        for n, c in classes.items():
            for raw in [n, *(c.get("aliases") or [])]:
                self._add(raw, n)
        for raw in cfg.get("ignore") or []:
            self._add(raw, self.IGNORE)
        for raw in cfg.get("exclude_images_with") or []:
            self._add(raw, self.EXCLUDE)

    def _add(self, raw: str, target: str):
        key = norm_key(raw)
        if self._lookup.get(key, target) != target:
            raise ValueError(f"label {raw!r} maps to both {self._lookup[key]!r} and {target!r}")
        self._lookup[key] = target

    def canonical(self, raw: str) -> str | None:
        """Canonical class, ClassMap.IGNORE / ClassMap.EXCLUDE, or None if the label is unknown."""
        return self._lookup.get(norm_key(raw))

    def department(self, dept: str) -> list[str]:
        return [n for n in self.names if dept in self.depts[n]]


def overlap_ratio(a, b) -> float:
    """Intersection area / smaller box area for xyxy boxes (1.0 = one box inside the other)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def has_overlap(boxes, thresh: float = 0.1) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if overlap_ratio(boxes[i], boxes[j]) >= thresh:
                return True
    return False


def open_upright(path: Path, max_side: int | None = None) -> Image.Image:
    """Open an image with its EXIF orientation applied (what annotators saw).

    max_side lets JPEG decoding skip straight to a reduced resolution (much faster for
    thumbnails of large camera photos).
    """
    im = Image.open(path)
    if max_side:
        im.draft("RGB", (max_side, max_side))
    return ImageOps.exif_transpose(im)


def dup_descriptor(im: Image.Image, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Layout fingerprint of an image under its 8 rotations/flips, for finding augmented copies.

    Returns (robust, zncc), each shaped (8, ...):
      robust - blurred size x size log-RGB thumbnail minus its median (invariant to blur and
               brightness gain, but keeps colour and layout so plain trays don't all look alike)
      zncc   - zero-mean unit-norm grey version, used for fast candidate retrieval
    """
    a = np.asarray(im.convert("RGB"))
    a = cv2.GaussianBlur(a, (0, 0), max(a.shape) / size / 2)
    t = np.log1p(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32))
    t -= np.median(t)
    rob, zn = [], []
    for k in range(4):
        r = np.rot90(t, k)
        for m in (r, r[:, ::-1]):
            rob.append(m.ravel())
            z = m.mean(2).ravel()
            z = z - z.mean()
            zn.append(z / (np.linalg.norm(z) + 1e-6))
    return np.stack(rob).astype(np.float32), np.stack(zn).astype(np.float32)


def aligned_distance(query: np.ndarray, refs: np.ndarray, q: float = 0.25) -> np.ndarray:
    """How far `query` is from being a pixel-aligned copy of each ref (lower = more alike).

    query: (D,) log-RGB descriptor (variant 0); refs: (K, 8, D) with all rotations/flips.
    The brightness gain is estimated per pair as the median log difference, then the
    q-quantile of the remaining |difference| is taken. A real copy matches almost exactly
    on most pixels even with blood stains or glare on a minority of them, whereas two
    different photos of a similar tray differ a little everywhere (lighting, shadows, noise).
    """
    d = refs - query
    d = np.abs(d - np.median(d, axis=-1, keepdims=True))
    k = int(q * d.shape[-1])
    return np.partition(d, k, axis=-1)[..., k].min(-1)

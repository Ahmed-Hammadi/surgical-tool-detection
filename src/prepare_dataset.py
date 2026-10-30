"""Convert a raw annotated dataset into a leakage-safe YOLO dataset.

Steps
1. Read annotations in YOLO txt, Pascal VOC xml or COCO json (auto-detected).
2. Map label strings onto the canonical classes in configs/classes.yaml.
3. Group near-duplicate images (augmented copies of one photo) using
   Roboflow-style filename stems plus a rotation/flip-invariant layout fingerprint
   that tolerates blur, lighting changes and local edits (blood stains, glare).
4. Split *groups* (not images) into train/val/test, balancing class counts,
   so no photo and its augmentations end up on both sides of the split.
5. Tag test images that contain overlapping instruments and write a separate
   overlap-only evaluation split.

Usage:
    python src/prepare_dataset.py --raw data/raw --out data/yolo
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_EXTS, ClassMap, dup_descriptor, has_overlap, open_upright, aligned_distance  # noqa: E402

SPLITS = ("train", "val", "test")
DUP_THRESH = 0.011  # copies score <~0.007, distinct photos >~0.019 (see README)


@dataclass
class Record:
    image: Path
    boxes: list = field(default_factory=list)  # (raw_label, x1, y1, x2, y2) in pixels
    width: int = 0
    height: int = 0


# --------------------------------------------------------------------- readers
def _image_index(raw: Path) -> dict[str, list[Path]]:
    idx = defaultdict(list)
    for p in raw.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx[p.name].append(p)
    return idx


def read_coco(raw: Path) -> list[Record]:
    images = _image_index(raw)
    recs = []
    for jf in raw.rglob("*.json"):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or not {"images", "annotations", "categories"} <= d.keys():
            continue
        cats = {c["id"]: c["name"] for c in d["categories"]}
        by_img = {im["id"]: Record(image=Path(), width=im.get("width", 0), height=im.get("height", 0)) for im in d["images"]}
        for im in d["images"]:
            cand = jf.parent / im["file_name"]
            if not cand.exists():
                hits = images.get(Path(im["file_name"]).name, [])
                cand = hits[0] if hits else None
            by_img[im["id"]].image = cand
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]].boxes.append((cats[a["category_id"]], x, y, x + w, y + h))
        recs += [r for r in by_img.values() if r.image]
    return recs


def read_voc(raw: Path) -> list[Record]:
    images = _image_index(raw)
    recs = []
    for xf in raw.rglob("*.xml"):
        try:
            root = ET.parse(xf).getroot()
        except ET.ParseError:
            continue
        if root.tag != "annotation":
            continue
        fname = root.findtext("filename") or ""
        cand = xf.with_name(fname) if fname else None
        if not cand or not cand.exists():
            hits = images.get(fname, []) or [p for s in IMG_EXTS for p in images.get(xf.stem + s, [])]
            cand = hits[0] if hits else None
        if cand is None:
            continue
        size = root.find("size")
        rec = Record(image=cand,
                     width=int(float(size.findtext("width") or 0)) if size is not None else 0,
                     height=int(float(size.findtext("height") or 0)) if size is not None else 0)
        for o in root.iter("object"):
            bb = o.find("bndbox")
            rec.boxes.append((o.findtext("name").strip(), *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        recs.append(rec)
    return recs


def _yolo_names(raw: Path) -> list[str] | None:
    for pat in ("data.yaml", "dataset.yaml", "*.yaml"):
        for f in raw.rglob(pat):
            d = yaml.safe_load(f.read_text()) or {}
            n = d.get("names") if isinstance(d, dict) else None
            if n:
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    for pat in ("classes.txt", "obj.names", "*.names"):
        for f in raw.rglob(pat):
            return [l.strip() for l in f.read_text().splitlines() if l.strip()]
    return None


def read_yolo(raw: Path) -> list[Record]:
    names = _yolo_names(raw)
    recs = []
    for img in sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS):
        parts = list(img.parts)
        cands = [img.with_suffix(".txt")]
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            cands.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
        lbl = next((c for c in cands if c.exists()), None)
        if lbl is None:
            continue
        if names is None:
            raise SystemExit("YOLO labels found but no class names (data.yaml / classes.txt / obj.names).")
        w, h = open_upright(img).size
        rec = Record(image=img, width=w, height=h)
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:5])
            if len(v) > 5:  # polygon/segment label -> bounding box
                xy = np.array(list(map(float, v[1:]))).reshape(-1, 2)
                (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
                cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            rec.boxes.append((names[c], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        recs.append(rec)
    return recs


def load_records(raw: Path, fmt: str) -> tuple[str, list[Record]]:
    readers = {"coco": read_coco, "voc": read_voc, "yolo": read_yolo}
    if fmt != "auto":
        return fmt, readers[fmt](raw)
    for name in ("coco", "voc", "yolo"):
        recs = readers[name](raw)
        if recs:
            return name, recs
    raise SystemExit(f"No COCO/VOC/YOLO annotations found under {raw}")


# -------------------------------------------------------------------- grouping
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def base_stem(path: Path) -> str:
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", path.stem)  # Roboflow export: IMG_1_jpg.rf.<hash>
    return s.lower()


@dataclass
class Fingerprints:
    robust: np.ndarray  # (N, 8, D)
    zncc: np.ndarray    # (N, 8, d)


def compute_fingerprints(recs: list[Record]) -> Fingerprints:
    rob, zn = zip(*(dup_descriptor(open_upright(r.image, max_side=256)) for r in recs))
    return Fingerprints(np.stack(rob), np.stack(zn))


def nearest(fp: Fingerprints, queries, refs, k: int = 10, chunk: int = 256):
    """For each query, the k most correlated refs (fast) re-scored by aligned_distance (precise).

    Returns a list of (query_idx, ref_idx, distance) for every retrieved candidate.
    """
    queries, refs = np.asarray(queries), np.asarray(refs)
    if not len(queries) or not len(refs):
        return []
    ref_z = fp.zncc[refs].reshape(len(refs) * 8, -1)
    out = []
    for s0 in range(0, len(queries), chunk):
        q = queries[s0:s0 + chunk]
        sim = (fp.zncc[q, 0] @ ref_z.T).reshape(len(q), len(refs), 8).max(2)
        sim[refs[None, :] == q[:, None]] = -np.inf  # never match an image to itself
        top = np.argsort(-sim, axis=1)[:, :k]
        for qi, cols in zip(q, top):
            d = aligned_distance(fp.robust[qi, 0], fp.robust[refs[cols]])
            out += [(int(qi), int(refs[c]), float(v)) for c, v in zip(cols, d)]
    return out


def group_records(recs: list[Record], fp: Fingerprints | None, thresh: float) -> tuple[list[int], list]:
    uf = _UF(len(recs))
    by_stem = defaultdict(list)
    for i, r in enumerate(recs):
        by_stem[base_stem(r.image)].append(i)
    for idxs in by_stem.values():
        if any(".rf." in recs[i].image.stem for i in idxs):
            for i in idxs[1:]:
                uf.union(idxs[0], i)
    pairs = []
    if fp is not None:
        idx = np.arange(len(recs))
        pairs = nearest(fp, idx, idx)
        for i, j, d in pairs:
            if d <= thresh:
                uf.union(i, j)
    return [uf.find(i) for i in range(len(recs))], pairs


def split_groups(group_ids, class_vecs, fracs, seed):
    """Greedy stratified group split: rarest-class groups placed first, each into
    the split furthest below its target share for that group's classes."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    gvec = {g: class_vecs[idx].sum(0) for g, idx in groups.items()}
    total = sum(gvec.values()) + 1e-9
    order = list(groups)
    rng.shuffle(order)
    rarity = 1.0 / total
    order.sort(key=lambda g: -(gvec[g] * rarity).max())
    cur = {s: np.zeros_like(total) for s in SPLITS}
    n_img = {s: 0 for s in SPLITS}
    n_total = len(group_ids)
    assign = {}
    for g in order:
        v = gvec[g]
        w = v / (v.sum() + 1e-9)
        best = max(SPLITS, key=lambda s: float((w * (fracs[s] * total - cur[s]) / total).sum())
                   + 0.5 * (fracs[s] - n_img[s] / n_total))
        assign[g] = best
        cur[best] += v
        n_img[best] += len(groups[g])
    return [assign[g] for g in group_ids]


# ---------------------------------------------------------------------- writer
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--classes", type=Path, default=Path(__file__).resolve().parent.parent / "configs/classes.yaml")
    ap.add_argument("--format", choices=["auto", "coco", "voc", "yolo"], default="auto")
    ap.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--dup-thresh", type=float, default=DUP_THRESH,
                    help="max fingerprint distance to treat two images as copies of one photo (negative disables)")
    ap.add_argument("--overlap-thresh", type=float, default=0.1, help="intersection/min-area to call two tools overlapping")
    ap.add_argument("--drop-unknown", action="store_true", help="drop boxes whose label is not in classes.yaml")
    ap.add_argument("--keep-empty", action="store_true", help="keep images with no boxes as background negatives")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cmap = ClassMap(a.classes)
    fmt, recs = load_records(a.raw, a.format)
    print(f"[read] {len(recs)} annotated images ({fmt})")

    unknown = Counter(b[0] for r in recs for b in r.boxes if cmap.canonical(b[0]) is None)
    if unknown and not a.drop_unknown:
        raise SystemExit(f"Unknown labels {dict(unknown)}. Add aliases in {a.classes} or pass --drop-unknown.")
    for r in recs:
        r.boxes = [(cmap.canonical(b[0]), *b[1:]) for b in r.boxes if cmap.canonical(b[0])]
    if not a.keep_empty:
        recs = [r for r in recs if r.boxes]

    nc = len(cmap.names)
    vecs = np.zeros((len(recs), nc))
    for i, r in enumerate(recs):
        for b in r.boxes:
            vecs[i, cmap.index(b[0])] += 1

    fp = compute_fingerprints(recs)
    gids, pairs = group_records(recs, fp if a.dup_thresh >= 0 else None, a.dup_thresh)
    n_groups = len(set(gids))
    sizes = sorted(Counter(gids).values(), reverse=True)
    print(f"[group] {len(recs)} images -> {n_groups} source-photo groups (largest: {sizes[:5]})")
    if sizes[0] > max(20, 0.02 * len(recs)):
        print(f"[group] WARNING: a group holds {sizes[0]} images - distinct photos are being chained "
              f"together. Inspect borderline_pairs.csv and lower --dup-thresh.")
    fr = dict(zip(SPLITS, np.array(a.split) / sum(a.split)))
    splits = split_groups(gids, vecs, fr, a.seed)

    if a.out.exists():
        shutil.rmtree(a.out)
    out = a.out.resolve()
    stats = {s: {"images": 0, "overlap_images": 0, "instances": Counter()} for s in SPLITS}
    lists = defaultdict(list)
    for i, (r, s) in enumerate(zip(recs, splits)):
        raw_im = Image.open(r.image)
        orientation = raw_im.getexif().get(0x0112, 1)
        # Boxes are in the frame the annotation tool displayed. Most tools apply the EXIF
        # rotation (upright frame); some don't (raw sensor frame). VOC/COCO size metadata
        # tells which, and the image is written in that frame with the EXIF tag dropped.
        raw_frame = orientation != 1 and (r.width, r.height) == raw_im.size != open_upright(r.image).size
        im = raw_im if raw_frame else open_upright(r.image)
        W, H = im.size
        sx = sy = 1.0
        if r.width and r.height and (r.width, r.height) != (W, H):
            sx, sy = W / r.width, H / r.height  # annotated on a resized copy
        name = f"{i:05d}_{re.sub(r'[^A-Za-z0-9_-]', '_', r.image.stem)[:60]}"
        dst = out / "images" / s / (name + r.image.suffix.lower())
        dst.parent.mkdir(parents=True, exist_ok=True)
        if orientation == 1:
            shutil.copy2(r.image, dst)
        else:
            im.convert("RGB").save(dst, quality=95)  # no EXIF tag -> every loader sees this frame
        lines, xyxy = [], []
        for c, x1, y1, x2, y2 in r.boxes:
            x1, x2 = sorted((max(0, min(W, x1 * sx)), max(0, min(W, x2 * sx))))
            y1, y2 = sorted((max(0, min(H, y1 * sy)), max(0, min(H, y2 * sy))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            xyxy.append((x1, y1, x2, y2))
            lines.append(f"{cmap.index(c)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            stats[s]["instances"][c] += 1
        lbl = out / "labels" / s / (name + ".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[s]["images"] += 1
        tag = "overlap" if has_overlap(xyxy, a.overlap_thresh) else "single"
        stats[s]["overlap_images"] += tag == "overlap"
        lists[f"{s}_{tag}"].append(str(dst))

    for k, v in lists.items():
        (out / f"{k}.txt").write_text("\n".join(sorted(v)) + "\n")
    names = dict(enumerate(cmap.names))
    base = {"path": str(out), "train": "images/train", "val": "images/val", "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump({**base, "test": "images/test"}, sort_keys=False))
    for tag in ("overlap", "single"):
        if lists.get(f"test_{tag}"):
            (out / f"data_test_{tag}.yaml").write_text(
                yaml.safe_dump({**base, "test": f"test_{tag}.txt"}, sort_keys=False))

    # Leakage audit: test images whose closest train image is within 1.25x the duplicate
    # threshold. Should be ~0; inspect borderline_pairs.csv if not.
    sp = np.array(splits)
    loose = 1.25 * max(a.dup_thresh, DUP_THRESH)
    best = defaultdict(lambda: np.inf)
    for q, _, d in nearest(fp, np.nonzero(sp == "test")[0], np.nonzero(sp == "train")[0]):
        best[q] = min(best[q], d)
    near = sum(d <= loose for d in best.values())
    print(f"[audit] test images with a near-duplicate in train: {near}/{int((sp == 'test').sum())}")

    # Pairs near the threshold, for a quick visual check of the grouping decision.
    with open(out / "borderline_pairs.csv", "w") as f:
        f.write("distance,grouped,image_a,image_b\n")
        seen = set()
        for i, j, d in sorted(pairs, key=lambda t: t[2]):
            key = (min(i, j), max(i, j))
            if key in seen or not (0.5 * a.dup_thresh <= d <= 2 * a.dup_thresh):
                continue
            seen.add(key)
            f.write(f"{d:.4f},{d <= a.dup_thresh},{recs[i].image},{recs[j].image}\n")

    report = {"format": fmt, "images": len(recs), "groups": n_groups, "largest_groups": sizes[:5], "dup_thresh": a.dup_thresh,
              "test_near_duplicates_in_train": near,
              "splits": {s: {**v, "instances": dict(v["instances"])} for s, v in stats.items()}}
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(f"[write] {out}")
    for s in SPLITS:
        v = stats[s]
        print(f"  {s:5s} images={v['images']:5d} overlap={v['overlap_images']:5d} "
              + " ".join(f"{c}={v['instances'].get(c, 0)}" for c in cmap.names))
    return report


if __name__ == "__main__":
    main()

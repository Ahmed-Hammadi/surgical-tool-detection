"""Generate a small synthetic stand-in for the surgical-tools dataset.

Only for smoke-testing the pipeline end to end; it is NOT a substitute for the
real data. Renders crude silhouettes of the 9 classes on tray-like backgrounds,
some scenes cluttered with overlapping tools, plus augmented copies of some
photos (blur / lighting / red "blood" spots / flips) to exercise the
leakage-safe grouping in prepare_dataset.py.

Usage:
    python src/make_synthetic.py --out data/raw_synthetic --n 300 --format voc
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ClassMap  # noqa: E402

STEEL = (180, 185, 192, 255)
DARK = (110, 115, 122, 255)


def _rings(d, y0=20, y1=60, x=10):
    d.ellipse([x, y0 - 14, x + 28, y0 + 14], outline=STEEL, width=5)
    d.ellipse([x, y1 - 14, x + 28, y1 + 14], outline=STEEL, width=5)


def draw_tool(cls: str, rng: random.Random) -> Image.Image:
    im = Image.new("RGBA", (300, 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if cls == "forceps":
        d.polygon([(10, 30), (290, 38), (290, 42), (10, 50)], fill=STEEL)
        d.polygon([(10, 34), (290, 39), (290, 41), (10, 46)], fill=(0, 0, 0, 0))
        for x in range(60, 160, 12):
            d.line([(x, 31), (x, 49)], fill=DARK, width=2)
    elif cls == "hemostat":
        _rings(d)
        d.line([(38, 20), (150, 40), (290, 38)], fill=STEEL, width=6)
        d.line([(38, 60), (150, 40), (290, 42)], fill=STEEL, width=6)
        for x in (60, 70, 80):
            d.line([(x, 26), (x, 54)], fill=DARK, width=3)  # ratchet
        for x in range(200, 288, 6):
            d.line([(x, 36), (x, 44)], fill=DARK, width=1)  # serrated jaws
    elif cls == "scalpel":
        d.rounded_rectangle([10, 32, 190, 48], 6, fill=DARK)
        d.polygon([(190, 33), (290, 44), (190, 48)], fill=(215, 220, 228, 255))
    elif cls == "mayo_scissors":
        _rings(d)
        d.polygon([(38, 16), (160, 36), (280, 30), (285, 36), (160, 44)], fill=STEEL)
        d.polygon([(38, 64), (160, 44), (280, 50), (285, 44), (160, 36)], fill=STEEL)
    elif cls == "syringe":
        d.rectangle([60, 25, 230, 55], fill=(235, 240, 245, 230), outline=(90, 90, 90, 255), width=2)
        for x in range(80, 230, 15):
            d.line([(x, 25), (x, 33)], fill=(40, 40, 40, 255), width=1)
        d.rectangle([10, 35, 60, 45], fill=(60, 60, 200, 255))
        d.rectangle([5, 22, 12, 58], fill=(60, 60, 200, 255))
        d.line([(230, 40), (295, 40)], fill=STEEL, width=2)
    elif cls == "bandage_scissors":
        _rings(d)
        d.line([(38, 20), (160, 40), (275, 36)], fill=STEEL, width=7)
        d.line([(38, 60), (160, 40), (280, 44)], fill=STEEL, width=7)
        d.ellipse([272, 38, 292, 52], fill=STEEL)  # blunt knob tip
    elif cls == "episiotomy_scissors":
        _rings(d)
        d.line([(38, 20), (150, 40), (230, 40), (285, 12)], fill=STEEL, width=7)
        d.line([(38, 60), (150, 40), (230, 44), (280, 18)], fill=STEEL, width=7)
    elif cls == "surgical_gloves":
        c = rng.choice([(150, 200, 235, 255), (235, 235, 225, 255), (120, 170, 120, 255)])
        d.rounded_rectangle([10, 15, 150, 65], 18, fill=c)
        for i, y in enumerate((14, 28, 42, 56)):
            d.rounded_rectangle([140, y - 5, 240 + (i % 2) * 20, y + 7], 6, fill=c)
        d.rounded_rectangle([100, 55, 160, 78], 8, fill=c)
    elif cls == "medical_cotton":
        im = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for _ in range(28):
            x, y, r = rng.randint(25, 135), rng.randint(25, 95), rng.randint(14, 24)
            g = rng.randint(238, 255)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(g, g, g - 4, 255))
    else:
        raise ValueError(cls)
    return im


def background(rng: random.Random, size: int) -> Image.Image:
    base = rng.choice([(40, 90, 110), (60, 110, 80), (200, 200, 205), (90, 60, 50), (30, 30, 35), (150, 160, 175)])
    im = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(0, 40)):  # fabric/tray texture
        x, y = rng.randint(0, size), rng.randint(0, size)
        c = tuple(max(0, min(255, v + rng.randint(-25, 25))) for v in base)
        d.line([(x, y), (x + rng.randint(-200, 200), y + rng.randint(-200, 200))], fill=c, width=rng.randint(1, 6))
    # Uneven lighting + low-frequency shading, unique per photo like a real camera shot.
    nrng = np.random.default_rng(rng.getrandbits(32))
    yy, xx = np.mgrid[0:size, 0:size] / size
    gx, gy = nrng.uniform(-0.35, 0.35, 2)
    field = 1 + gx * (xx - 0.5) + gy * (yy - 0.5)
    blobs = Image.fromarray((nrng.random((8, 8)) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    field *= 1 + 0.25 * (np.asarray(blobs, np.float32) / 255 - 0.5)
    return im.filter(ImageFilter.GaussianBlur(1)), field, nrng


def relight(im, field, nrng):
    arr = np.asarray(im, np.float32) * field[..., None] + nrng.normal(0, 3, (*field.shape, 1))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def scene(rng, names, size, cluttered):
    im, field, nrng = background(rng, size)
    owner = np.full((size, size), -1, dtype=np.int16)  # which tool is visible at each pixel
    boxes, areas = [], []
    n = rng.randint(3, 6) if cluttered else rng.randint(1, 3)
    placed = []
    for _ in range(n):
        cls = rng.choice(names)
        t = draw_tool(cls, rng)
        s = rng.uniform(0.9, 1.4) * size / 640
        t = t.resize((max(8, int(t.width * s)), max(8, int(t.height * s))))
        t = t.rotate(rng.uniform(0, 360), expand=True, resample=Image.BICUBIC)
        bb = t.getbbox()
        t = t.crop(bb)
        for _try in range(30):
            x, y = rng.randint(0, size - t.width), rng.randint(0, size - t.height)
            box = (x, y, x + t.width, y + t.height)
            ok = cluttered or all(box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3] for p in placed)
            if ok:
                break
        else:
            continue
        shadow = Image.new("RGBA", t.size, (0, 0, 0, 90))
        im.paste(shadow, (x + 4, y + 4), t.split()[3])
        im.paste(t, (x, y), t)
        mask = np.asarray(t.split()[3]) > 0
        owner[y:y + t.height, x:x + t.width][mask] = len(boxes)
        placed.append(box)
        boxes.append((cls, *box))
        areas.append(mask.sum())
    # drop tools that later tools hid almost entirely (label would be unlearnable)
    visible = np.bincount(owner[owner >= 0].ravel(), minlength=len(boxes))
    keep = [b for b, v, a in zip(boxes, visible, areas) if v >= 0.3 * a]
    return relight(im, field, nrng), keep


def augment(im, boxes, rng):
    k = rng.choice(["blur", "light", "blood", "flip"])
    if k == "blur":
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3)))
    elif k == "light":
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 1.5))
    elif k == "blood":
        im = im.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for _ in range(rng.randint(3, 8)):
            x, y, r = rng.randint(0, im.width), rng.randint(0, im.height), rng.randint(5, 25)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(130, 10, 15, 170))
    else:
        im = ImageOps.mirror(im)
        boxes = [(c, im.width - x2, y1, im.width - x1, y2) for c, x1, y1, x2, y2 in boxes]
    return im, boxes, k


def write(out: Path, fmt: str, items, names):
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    if fmt == "yolo":
        (out / "labels").mkdir(exist_ok=True)
        (out / "classes.txt").write_text("\n".join(names) + "\n")
    for idx, (fname, im, boxes) in enumerate(items):
        im.save(img_dir / fname, quality=92)
        W, H = im.size
        if fmt == "voc":
            objs = "".join(
                f"<object><name>{c.replace('_', ' ').title()}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>" for c, x1, y1, x2, y2 in boxes)
            (img_dir / Path(fname).with_suffix(".xml")).write_text(
                f"<annotation><filename>{fname}</filename><size><width>{W}</width><height>{H}</height>"
                f"<depth>3</depth></size>{objs}</annotation>")
        elif fmt == "yolo":
            (out / "labels" / Path(fname).with_suffix(".txt")).write_text("".join(
                f"{names.index(c)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
                for c, x1, y1, x2, y2 in boxes))
        else:
            coco["images"].append({"id": idx, "file_name": f"images/{fname}", "width": W, "height": H})
            for c, x1, y1, x2, y2 in boxes:
                coco["annotations"].append({"id": len(coco["annotations"]), "image_id": idx, "category_id": names.index(c) + 1,
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    if fmt == "coco":
        (out / "annotations.json").write_text(json.dumps(coco))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw_synthetic"))
    ap.add_argument("--n", type=int, default=300, help="number of distinct source photos")
    ap.add_argument("--aug-copies", type=float, default=0.5, help="fraction of photos that get an augmented copy")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--clutter", type=float, default=0.35, help="fraction of cluttered/overlapping scenes")
    ap.add_argument("--format", choices=["voc", "yolo", "coco"], default="voc")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    names = ClassMap().names
    items, copies = [], []
    for i in range(a.n):
        im, boxes = scene(rng, names, a.size, rng.random() < a.clutter)
        items.append((f"IMG_{i:04d}.jpg", im, boxes))
        if rng.random() < a.aug_copies:
            aim, aboxes, k = augment(im, boxes, rng)
            # random names on purpose: grouping must rely on image content, not filenames
            items.append((f"aug_{rng.getrandbits(32):08x}_{k}.jpg", aim, aboxes))
            copies.append((items[-1][0], items[-2][0], k))
    if a.out.exists():
        import shutil
        shutil.rmtree(a.out)
    write(a.out, a.format, items, names)
    # ground truth for testing the duplicate grouping (not an annotation file)
    (a.out / "copies.csv").write_text("copy,source,augmentation\n" + "".join(f"{c},{s},{k}\n" for c, s, k in copies))
    print(f"wrote {len(items)} images ({a.n} source photos) to {a.out} [{a.format}]")


if __name__ == "__main__":
    main()

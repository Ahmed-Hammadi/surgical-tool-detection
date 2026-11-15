"""Draft labels for your own photos, so adding a new instrument class takes minutes of
checking instead of hours of drawing boxes.

Capture protocol (see docs/adding_instruments.md): one folder per class, each photo showing
only that instrument type (one or several of them):

    data/capture/hohmann_retractor/IMG_001.jpg ...
    data/capture/bone_mallet/IMG_101.jpg ...
    data/capture/_mixed/IMG_201.jpg ...      # real trays with several types

For class folders the detector only has to find *where* the instruments are; the class comes
from the folder name, so it works for instruments the model has never seen. For `_mixed` the
model's own class guesses are kept and need checking.

The result is a YOLO dataset under data/raw/<name>/ that prepare_dataset.py reads directly,
plus review sheets (review/*.jpg) and a list of photos where nothing was found. Fix wrong
boxes in any YOLO-format editor (CVAT, Label Studio, Roboflow, labelImg) before training.

Usage:
    python src/prelabel.py --capture data/capture --name my_hospital
    python src/prelabel.py --capture data/capture --name my_hospital --weights runs/yolo11n_640/weights/best.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_EXTS, ClassMap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIXED = "_mixed"


def load_detector(weights: str, prompts: list[str]):
    """Our own trained detector, or an open-vocabulary YOLOE model prompted with text."""
    if "yoloe" in Path(weights).name.lower():
        from ultralytics import YOLOE

        model = YOLOE(weights)
        model.set_classes(prompts, model.get_text_pe(prompts))
        return model, True
    from ultralytics import YOLO

    return YOLO(weights), False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, default=ROOT / "data" / "capture")
    ap.add_argument("--name", required=True, help="dataset name; written to data/raw/<name>")
    ap.add_argument("--weights", default="yoloe-11s-seg.pt",
                    help="yoloe-*.pt finds objects from a text prompt (works before you have any model); "
                         "or your own best.pt once trained")
    ap.add_argument("--prompt", nargs="+", default=["surgical instrument", "metal tool"],
                    help="text prompts for YOLOE weights")
    ap.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a box is quicker than drawing one")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    cmap = ClassMap()
    folders = sorted(p for p in a.capture.iterdir() if p.is_dir()) if a.capture.is_dir() else []
    if not folders:
        sys.exit(f"no class folders under {a.capture}")
    bad = [f.name for f in folders if f.name != MIXED and cmap.canonical(f.name) not in cmap.names]
    if bad:
        sys.exit(f"folders {bad} are not classes in configs/classes.yaml (add the class or an alias first)")

    model, open_vocab = load_detector(a.weights, a.prompt)
    out = ROOT / "data" / "raw" / a.name
    if out.exists():
        sys.exit(f"{out} already exists - pick another --name or remove it")
    for sub in ("images", "labels", "review"):
        (out / sub).mkdir(parents=True)
    (out / "classes.txt").write_text("\n".join(cmap.names) + "\n")

    empty, n_img, n_box = [], 0, 0
    for folder in folders:
        cls = None if folder.name == MIXED else cmap.canonical(folder.name)
        if cls is None and open_vocab:
            print(f"skipping {MIXED}: a text-prompted model cannot name instrument types; use your trained weights")
            continue
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        for p in imgs:
            # agnostic NMS: two prompts/classes on one object must give one box
            r = model.predict(str(p), conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device,
                              agnostic_nms=True, verbose=False)[0]
            stem = f"{folder.name}__{p.stem}"
            lines = []
            for c, (cx, cy, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                name = cls or cmap.canonical(r.names[int(c)])
                if name in cmap.names:
                    lines.append(f"{cmap.names.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            shutil.copy2(p, out / "images" / (stem + p.suffix.lower()))
            (out / "labels" / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            r.save(str(out / "review" / (stem + ".jpg")))
            n_img += 1
            n_box += len(lines)
            if not lines:
                empty.append(str(p))
        print(f"{folder.name}: {len(imgs)} photos")
    (out / "needs_manual_labels.txt").write_text("\n".join(empty) + ("\n" if empty else ""))
    print(f"\n{n_img} photos, {n_box} draft boxes -> {out}")
    print(f"{len(empty)} photos with no box found: {out / 'needs_manual_labels.txt'}")
    print(f"Check the drafts in {out / 'review'}, correct them in a YOLO label editor, then run prepare_dataset.py")


if __name__ == "__main__":
    main()

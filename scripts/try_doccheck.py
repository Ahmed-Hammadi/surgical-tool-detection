"""Try DocCheck's pretrained Rona instrument detector (YOLOv5-L, 12 classes) on your own photos.

It is a quick look, not part of the pipeline: the model detects German-labelled forceps/scalpel/
scissors subtypes from a fixed overhead robot camera, so expect it to miss hemostats, syringes,
gloves and cotton and to struggle with hand-held cluttered trays. Each detection is also shown
mapped to our class names where a mapping exists.

Security: .pt files are Python pickles and can run code when loaded. Run this in Colab or a
sandbox, not on a machine with credentials.

Usage (needs internet access to huggingface.co and github.com):
    python scripts/try_doccheck.py --source my_tray_photos/
    python scripts/try_doccheck.py --source photo.jpg --conf 0.5
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_URL = ("https://huggingface.co/DocCheck/medical-instrument-detection/resolve/main/"
               "instrument_detector_model.pt?download=true")
DEFAULT_WEIGHTS = ROOT / "weights" / "doccheck" / "instrument_detector_model.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DocCheck label -> (English, our class or None)
LABELS = {
    "Anatomische Pinzette Standard": ("anatomical forceps, standard", "forceps"),
    "Anatomische Pinzette schlank": ("anatomical forceps, slim", "forceps"),
    "Chirurgische Pinzette": ("surgical forceps", "forceps"),
    "Splitterpinzette": ("splinter forceps", "forceps"),
    "Skalpellgriff Nr.3": ("scalpel handle no. 3", "scalpel"),
    "Skalpellgriff Nr.4": ("scalpel handle no. 4", "scalpel"),
    "Skalpell geballt": ("scalpel, bellied", "scalpel"),
    "Skalpell schmal": ("scalpel, narrow", "scalpel"),
    "Chirurgische Schere spitz/spitz": ("surgical scissors sharp/sharp", None),
    "Chirurgische Schere spitz/stumpf": ("surgical scissors sharp/blunt", None),
    "Praeparierschere Standard": ("dissecting scissors", None),
    "Praepariernadel": ("dissecting needle", None),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path, help="image or folder of images")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25, help="DocCheck themselves use 0.80")
    ap.add_argument("--imgsz", type=int, default=960, help="the model was trained at 960")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "doccheck_try")
    a = ap.parse_args(argv)

    if not a.weights.exists():
        a.weights.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {WEIGHTS_URL} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, a.weights)

    import torch

    # Original YOLOv5 checkpoints need the yolov5 code to unpickle; torch.hub fetches it.
    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(a.weights), trust_repo=True)
    model.conf = a.conf
    print("model classes:", list(model.names.values()))

    imgs = [a.source] if a.source.is_file() else sorted(p for p in a.source.iterdir() if p.suffix.lower() in IMG_EXTS)
    a.out.mkdir(parents=True, exist_ok=True)
    report = []
    for p in imgs:
        res = model(str(p), size=a.imgsz)
        df = res.pandas().xyxy[0]
        dets = []
        for _, d in df.iterrows():
            english, ours = LABELS.get(d["name"], (d["name"], None))
            dets.append({"label": d["name"], "english": english, "our_class": ours,
                         "conf": round(float(d["confidence"]), 3),
                         "xyxy": [round(float(d[k]), 1) for k in ("xmin", "ymin", "xmax", "ymax")]})
        res.render()
        from PIL import Image

        Image.fromarray(res.ims[0]).save(a.out / f"{p.stem}_doccheck.jpg")
        counts = Counter(d["our_class"] or f"(no match) {d['english']}" for d in dets)
        print(f"{p.name}: {len(dets)} detections  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        report.append({"image": str(p), "detections": dets})
    (a.out / "detections.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nannotated images + detections.json in {a.out}")


if __name__ == "__main__":
    main()

"""Detect and count surgical tools in images, a folder, a video or a webcam.

Usage:
    python src/predict.py --weights best.pt --source tray.jpg
    python src/predict.py --weights best.pt --source data/new_trays/ --save-json counts.json
    python src/predict.py --weights best.pt --source 0          # webcam
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU; raise for tightly packed trays")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--no-save", action="store_true", help="don't write annotated images")
    a = ap.parse_args(argv)

    from ultralytics import RTDETR, YOLO

    model = RTDETR(a.weights) if "rtdetr" in a.weights.lower() else YOLO(a.weights)
    imgsz = a.imgsz or (model.ckpt or {}).get("train_args", {}).get("imgsz", 640)
    src = int(a.source) if a.source.isdigit() else a.source
    out = []
    for r in model.predict(src, conf=a.conf, iou=a.iou, imgsz=imgsz, device=a.device,
                           save=not a.no_save, project=str(ROOT / "runs"), name="predict",
                           stream=True, verbose=False):
        names = r.names
        counts = Counter(names[int(c)] for c in r.boxes.cls.tolist())
        dets = [{"class": names[int(c)], "conf": round(float(s), 4), "xyxy": [round(v, 1) for v in b]}
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        out.append({"image": r.path, "total": sum(counts.values()), "counts": dict(counts), "detections": dets})
        print(f"{Path(r.path).name}: {sum(counts.values())} tools  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if a.save_json:
        a.save_json.write_text(json.dumps(out, indent=2))
        print(f"saved {a.save_json}")
    return out


if __name__ == "__main__":
    main()

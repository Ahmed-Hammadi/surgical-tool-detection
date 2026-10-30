"""Evaluate trained weights and write a markdown + JSON report.

Reports, for the full test split and separately for its overlapping-tools and
single/non-overlapping subsets:
  * mAP@0.5, mAP@0.5:0.95, precision, recall, F1 (overall and per class)
  * the most frequent class confusions (from the confusion matrix)
Plus, on the full test split:
  * tray counting accuracy (predicted vs true count per class per image)
  * inference speed (ms/image and FPS on this machine)

Usage:
    python src/evaluate.py --weights runs/yolo11n_640/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def run_val(model, data_yaml: Path, imgsz, device, out: Path, tag: str):
    # rect=False: score on the fixed square input the phone/exported model gets. Ultralytics'
    # default rect=True pads each image differently and can overstate deployed accuracy.
    m = model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device, plots=True, rect=False,
                  project=str(out), name=tag, exist_ok=True, verbose=False)
    names = m.names
    b = m.box
    per_class = []
    for i, c in enumerate(b.ap_class_index):
        p, r, ap50, ap = b.class_result(i)
        per_class.append({"class": names[int(c)], "precision": p, "recall": r, "f1": f1(p, r), "mAP50": ap50, "mAP50_95": ap})
    cm = m.confusion_matrix.matrix  # [predicted, true]; index nc = background
    labels = [names[i] for i in range(len(names))]
    conf = []
    # class-vs-class only: background rows/cols are dominated by the very low confidence
    # threshold (0.001) used to compute mAP, and misses/false alarms are covered by P and R
    nc = len(names)
    for pi in range(nc):
        for ti in range(nc):
            if pi != ti and cm[pi, ti] > 0:
                conf.append({"true": labels[ti], "predicted": labels[pi], "count": int(cm[pi, ti])})
    conf.sort(key=lambda d: -d["count"])
    return {
        "images": len(test_images(data_yaml)), "instances": int(m.nt_per_class.sum()),
        "precision": b.mp, "recall": b.mr, "f1": f1(b.mp, b.mr), "mAP50": b.map50, "mAP50_95": b.map,
        "per_class": per_class, "top_confusions": conf[:10], "plots_dir": str(m.save_dir),
    }


def test_images(data_yaml: Path) -> list[Path]:
    d = yaml.safe_load(data_yaml.read_text())
    t = Path(d["path"]) / d["test"]
    if t.is_dir():
        return sorted(p for p in t.iterdir() if p.is_file())
    return [Path(l) for l in t.read_text().split()]


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")


def counting_and_speed(model, imgs, imgsz, device, conf, n_speed=100):
    exact, abs_err, n_inst = 0, 0, 0
    times = []
    model.predict(str(imgs[0]), imgsz=imgsz, device=device, verbose=False)  # warm-up
    for k, img in enumerate(imgs):
        t0 = time.perf_counter()
        r = model.predict(str(img), imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
        if k < n_speed:
            times.append(time.perf_counter() - t0)
        pred = Counter(int(c) for c in r.boxes.cls.tolist())
        lp = label_path(img)
        true = Counter(int(l.split()[0]) for l in lp.read_text().splitlines() if l.strip()) if lp.exists() else Counter()
        diff = sum(abs(pred[c] - true[c]) for c in set(pred) | set(true))
        exact += diff == 0
        abs_err += diff
        n_inst += sum(true.values())
    ms = 1000 * float(np.median(times))
    return {
        "conf_threshold": conf,
        "tray_exact_count_accuracy": exact / len(imgs),
        "count_abs_error_per_image": abs_err / len(imgs),
        "count_error_rate_per_instance": abs_err / max(1, n_inst),
        "median_ms_per_image": ms, "fps": 1000 / ms,
    }


def fmt(x):
    return f"{100 * x:.1f}" if isinstance(x, float) else str(x)


def write_report(res: dict, path: Path):
    L = [f"# Evaluation report\n", f"Weights: `{res['weights']}`  \nImage size: {res['imgsz']}  \n"]
    L.append("## Overall\n\n| Split | Images | Instances | P | R | F1 | mAP50 | mAP50-95 |\n|---|---|---|---|---|---|---|---|")
    for tag, m in res["splits"].items():
        L.append(f"| {tag} | {m['images']} | {m['instances']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                 f"{fmt(m['mAP50'])} | {fmt(m['mAP50_95'])} |")
    main = res["splits"]["test"]
    L.append("\n## Per class (test)\n\n| Class | P | R | F1 | mAP50 | mAP50-95 |\n|---|---|---|---|---|---|")
    for c in main["per_class"]:
        L.append(f"| {c['class']} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | "
                 f"{fmt(c['mAP50'])} | {fmt(c['mAP50_95'])} |")
    L.append("\n## Most frequent class confusions (test)\n\n| True | Predicted as | Count |\n|---|---|---|")
    for c in main["top_confusions"]:
        L.append(f"| {c['true']} | {c['predicted']} | {c['count']} |")
    if "counting" in res:
        c = res["counting"]
        L.append(f"\n## Counting & speed (test, conf={c['conf_threshold']})\n")
        L.append(f"- Trays with every class counted exactly right: **{fmt(c['tray_exact_count_accuracy'])}%**")
        L.append(f"- Mean count error per image: {c['count_abs_error_per_image']:.3f} instruments")
        L.append(f"- Count errors per true instrument: {fmt(c['count_error_rate_per_instance'])}%")
        L.append(f"- Speed: {c['median_ms_per_image']:.1f} ms/image ({c['fps']:.1f} FPS) on `{res['device']}`")
    L.append(f"\nPlots (PR curves, confusion matrices): see `plots_dir` entries in `metrics.json`.\n")
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo" / "data.yaml")
    ap.add_argument("--imgsz", type=int, default=None, help="default: training image size")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold for counting")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/eval/<weights run name>")
    a = ap.parse_args(argv)

    from ultralytics import RTDETR, YOLO
    import torch

    model = RTDETR(a.weights) if "rtdetr" in str(a.weights).lower() else YOLO(a.weights)
    imgsz = a.imgsz or (model.ckpt or {}).get("train_args", {}).get("imgsz", 640)
    device = a.device or ("0" if torch.cuda.is_available() else "cpu")
    run_name = Path(a.weights).resolve().parent.parent.name
    out = (a.out or ROOT / "runs" / "eval" / run_name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    res = {"weights": str(a.weights), "imgsz": imgsz, "device": device, "splits": {}}
    res["splits"]["test"] = run_val(model, a.data, imgsz, device, out, "test")
    for tag in ("overlap", "single"):
        y = a.data.parent / f"data_test_{tag}.yaml"
        if y.exists():
            res["splits"][f"test_{tag}"] = run_val(model, y, imgsz, device, out, f"test_{tag}")
    res["counting"] = counting_and_speed(model, test_images(a.data), imgsz, device, a.conf)

    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    write_report(res, out / "report.md")
    print((out / "report.md").read_text())
    print(f"saved {out / 'report.md'}")
    return res


if __name__ == "__main__":
    main()

"""Export trained weights for on-device (phone) inference and check what quantization costs.

For every target it exports the model, then (where the format runs on this machine)
re-evaluates it on the test split and times it, so accuracy lost to quantization is
measured rather than assumed. Results go to runs/export/<run>/export_report.md and the
app-ready files plus labels/model card to mobile/<run>/.

Targets:
  litert-int8   Android, full INT8 (calibrated on train images): smallest, fastest on NPU/DSP
  litert-w8a32  Android, INT8 weights + FP32 activations: ~4x smaller, accuracy ~= FP32, GPU/CPU
  litert-fp32   Android, reference
  coreml-fp16   iOS, FP16 with built-in NMS (Vision framework ready, runs on the Neural Engine)
  coreml-w8a16  iOS, INT8 weights + FP16 activations: ~2x smaller again
  onnx          ONNX Runtime (Mobile / web / desktop), FP32 reference

Usage:
    python src/export.py --weights runs/yolo11n_640/weights/best.pt
    python src/export.py --weights best.pt --targets litert-int8 litert-w8a32
    # Core ML needs torch <= 2.7 (coremltools limit); use the separate env, see README:
    ../.venv-coreml/bin/python src/export.py --weights best.pt --targets coreml-fp16 coreml-w8a16
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    "litert-int8": dict(format="litert", quantize=8, platform="android"),
    "litert-w8a32": dict(format="litert", quantize="w8a32", platform="android"),
    "litert-fp32": dict(format="litert", platform="android"),
    "coreml-fp16": dict(format="coreml", quantize=16, nms=True, platform="ios"),
    "coreml-w8a16": dict(format="coreml", quantize="w8a16", nms=True, platform="ios"),
    "onnx": dict(format="onnx", simplify=True, platform="any"),
}
DEFAULT_TARGETS = ["litert-int8", "litert-w8a32", "coreml-fp16", "onnx"]
RUNS_HERE = {"litert", "onnx"}  # Core ML packages only execute on macOS/iOS


def size_mb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def evaluate(model, data, imgsz, n_speed=30):
    """mAP on the test split and median inference latency (ms) on this CPU."""
    # rect=False so the PyTorch reference sees the same fixed square input as the exports
    m = model.val(data=str(data), split="test", imgsz=imgsz, batch=1, rect=False, device="cpu",
                  plots=False, verbose=False)
    imgs = sorted((Path(yaml.safe_load(Path(data).read_text())["path"]) / "images" / "test").iterdir())[:n_speed]
    model.predict(str(imgs[0]), imgsz=imgsz, device="cpu", verbose=False)
    ms = [model.predict(str(p), imgsz=imgsz, device="cpu", verbose=False)[0].speed["inference"] for p in imgs]
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "ms": statistics.median(ms)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, choices=list(TARGETS))
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "yolo" / "data.yaml",
                    help="dataset yaml: INT8 calibration images (train) and evaluation (test)")
    ap.add_argument("--imgsz", type=int, default=None, help="default: training image size")
    ap.add_argument("--no-eval", action="store_true", help="export only")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    pt = YOLO(str(a.weights))
    imgsz = a.imgsz or (pt.ckpt or {}).get("train_args", {}).get("imgsz", 640)
    run = a.weights.resolve().parent.parent.name
    work = ROOT / "runs" / "export" / run
    bundle = ROOT / "mobile" / run
    work.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    report_path = work / "export_report.json"
    results = json.loads(report_path.read_text()) if report_path.exists() else {}

    if not a.no_eval and "pytorch" not in results:
        results["pytorch"] = {"file": a.weights.name, "size_mb": size_mb(a.weights), "platform": "reference",
                              **evaluate(pt, a.data, imgsz)}

    for t in a.targets:
        cfg = dict(TARGETS[t])
        platform = cfg.pop("platform")
        vdir = work / t  # one dir per target so exports don't overwrite each other
        shutil.rmtree(vdir, ignore_errors=True)
        vdir.mkdir(parents=True)
        src = shutil.copy2(a.weights, vdir / "model.pt")
        print(f"\n=== {t} ===")
        try:
            if cfg.get("quantize") == 8:
                cfg["data"] = str(a.data)  # full INT8 calibrates activation ranges on train images
            out = Path(YOLO(src).export(imgsz=imgsz, **cfg))
        except Exception as e:
            traceback.print_exc()
            results[t] = {"platform": platform, "error": f"{type(e).__name__}: {e}"[:300]}
            continue
        dst = bundle / f"surgical_tools_{t}{out.suffix}"  # .tflite / .onnx file or .mlpackage dir
        if dst.is_dir():
            shutil.rmtree(dst)
        if out.is_dir():
            shutil.copytree(out, dst)
        else:
            shutil.copy2(out, dst)
        res = {"file": dst.name, "size_mb": size_mb(out), "platform": platform}
        if not a.no_eval and cfg["format"] in RUNS_HERE:
            try:
                res.update(evaluate(YOLO(str(out), task="detect"), a.data, imgsz))
            except Exception as e:
                res["eval_error"] = f"{type(e).__name__}: {e}"[:300]
        elif cfg["format"] not in RUNS_HERE:
            res["note"] = "evaluate on a Mac/iPhone (Core ML does not run on Linux)"
        results[t] = res

    report_path.write_text(json.dumps(results, indent=2))
    names = pt.names
    io = {}  # actual tensor layouts, read back from an exported LiteRT file
    for f in sorted(bundle.glob("*.tflite")):
        try:
            from ai_edge_litert.interpreter import Interpreter

            it = Interpreter(model_path=str(f))
            i, o = it.get_input_details()[0], it.get_output_details()[0]
            io = {"input_litert": {"shape": i["shape"].tolist(), "dtype": i["dtype"].__name__,
                                   "layout": "NCHW" if i["shape"][1] == 3 else "NHWC"},
                  "output_litert": {"shape": o["shape"].tolist(), "dtype": o["dtype"].__name__}}
        except Exception:
            pass
        break
    (bundle / "labels.txt").write_text("\n".join(names[i] for i in range(len(names))) + "\n")
    (bundle / "model_card.json").write_text(json.dumps({
        "task": "detect", "classes": [names[i] for i in range(len(names))], "input_size": imgsz,
        "input": f"RGB, {imgsz}x{imgsz}, letterboxed (aspect kept, padded with 114 grey), float values 0-1",
        **io,
        "output_litert_onnx": f"[1, 4 + {len(names)}, N] = cx, cy, w, h (pixels of the input) + class scores; "
                              "apply confidence filter + NMS in the app",
        "output_coreml": "built-in NMS: confidence + coordinates (normalised), usable via VNCoreMLRequest",
        "recommended": {"conf": 0.4, "iou": 0.6},
        "source_weights": str(a.weights),
    }, indent=2))

    ref = results.get("pytorch", {})
    L = ["# Export report\n", f"Weights: `{a.weights}`, input {imgsz}x{imgsz}. "
         "Latency is this machine's CPU, only for comparing formats; phones with GPU/NPU delegates are faster.\n",
         "| Target | Platform | Size (MB) | mAP50 | mAP50-95 | Δ mAP50-95 vs PyTorch | CPU ms |",
         "|---|---|---|---|---|---|---|"]
    for t, r in results.items():
        if "error" in r:
            L.append(f"| {t} | {r['platform']} | export failed: {r['error'][:80]} | | | | |")
            continue
        if "mAP50_95" in r:
            delta = f"{100 * (r['mAP50_95'] - ref['mAP50_95']):+.1f}" if "mAP50_95" in ref else ""
            cells = f"{100 * r['mAP50']:.1f} | {100 * r['mAP50_95']:.1f} | {delta} | {r['ms']:.1f}"
        else:
            cells = f"{r.get('note') or r.get('eval_error', 'not evaluated')} | | |"
        L.append(f"| {t} | {r['platform']} | {r['size_mb']:.1f} | {cells} |")
    (work / "export_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\napp bundle: {bundle}\nreport: {work / 'export_report.md'}")
    return results


if __name__ == "__main__":
    main()

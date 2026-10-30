"""Train a detector with the settings in configs/train.yaml.

Usage:
    python src/train.py                                   # full config
    python src/train.py --set model=yolo11s.pt name=yolo11s_640
    python src/train.py --set imgsz=512 name=yolo11n_512    # even lighter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def parse_value(v: str):
    return yaml.safe_load(v)  # "50" -> 50, "true" -> True, "0.3" -> 0.3


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for kv in overrides:
        k, _, v = kv.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        cfg[k] = parse_value(v)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)

    from ultralytics import RTDETR, YOLO

    weights = cfg.pop("model")
    model = RTDETR(weights) if "rtdetr" in str(weights).lower() else YOLO(weights)
    if not Path(cfg["data"]).exists():
        sys.exit(f"{cfg['data']} not found - run src/prepare_dataset.py first")
    cfg["project"] = str((ROOT / cfg.get("project", "runs")).resolve())
    model.train(**cfg)
    best = Path(model.trainer.best)
    print(f"\nbest weights: {best}\nnext: python src/evaluate.py --weights {best}")
    return best


if __name__ == "__main__":
    main()

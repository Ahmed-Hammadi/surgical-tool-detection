#!/usr/bin/env bash
# End-to-end check on synthetic data: generate -> prepare -> train (few epochs) -> evaluate -> predict -> export.
# Proves the pipeline runs; the numbers it produces say nothing about real-data accuracy.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-15}
python src/make_synthetic.py --out data/raw_synthetic --n 400 --size 480 --format voc
python src/prepare_dataset.py --raw data/raw_synthetic --out data/yolo_synthetic
python src/train.py --set data=data/yolo_synthetic/data.yaml model=yolo11n.pt epochs="$EPOCHS" \
    imgsz=480 batch=16 patience=0 close_mosaic=3 name=smoke exist_ok=true workers=4 plots=false
python src/evaluate.py --weights runs/smoke/weights/best.pt --data data/yolo_synthetic/data.yaml
python src/predict.py --weights runs/smoke/weights/best.pt \
    --source "$(head -n1 data/yolo_synthetic/test_overlap.txt)" --save-json runs/smoke/predict_example.json
python src/export.py --weights runs/smoke/weights/best.pt --data data/yolo_synthetic/data.yaml \
    --targets litert-int8 litert-w8a32 onnx

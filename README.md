# Surgical Tool Detection

A lightweight multi-object detector for surgical instruments (detection + per-class
counting on instrument trays) designed to run **on a phone**: YOLO11n at 640 px, exported
to TFLite/LiteRT (Android) and Core ML (iOS), about 3–6 MB on device.

The target dataset is the Mendeley **SURGICAL TOOLS** dataset
([10.17632/cyghvmjrt3.1](https://data.mendeley.com/datasets/cyghvmjrt3/1)), described in
Data in Brief 2025 (SID-RAS). It has about 6,000 images in 9 classes, and 1,520 of them show
several tools, some overlapping. Related work: *Surgical Instrument Detection Algorithm Based
on Improved YOLOv7x*, Sensors 2023, 23(11):5037 (OSI26 dataset, 26 classes).

| id | class | id | class |
|---|---|---|---|
| 0 | forceps | 5 | bandage_scissors |
| 1 | hemostat | 6 | episiotomy_scissors |
| 2 | scalpel | 7 | surgical_gloves |
| 3 | mayo_scissors | 8 | medical_cotton |
| 4 | syringe | | |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-export.txt   # for src/export.py (phone formats)
```

Training needs a GPU to finish in reasonable time. A CPU works for the smoke test and for inference.

## 1. Get the data

Download the dataset zip from the Mendeley page and extract it anywhere under `data/raw/`.
The folder layout doesn't matter: `prepare_dataset.py` searches it recursively and detects the
annotation format on its own.

- **YOLO**: `.txt` labels next to the images or in a parallel `labels/` folder, with class names in `data.yaml`, `classes.txt` or `obj.names`. Polygon labels are converted to boxes.
- **Pascal VOC**: `.xml` files.
- **COCO**: a `.json` file with `images`, `annotations` and `categories`.

If the label spellings differ from the canonical names (for example `Haemostat` or `Mayo Scissor`),
add them under `aliases` in `configs/classes.yaml`. Unknown labels stop the script with a list of
what it found. Pass `--drop-unknown` to skip them instead.

## 2. Prepare a leakage-safe split

```bash
python src/prepare_dataset.py --raw data/raw --out data/yolo
```

The dataset was enlarged with blurred, relit and "blood"-stained copies of the same photos. A
random per-image split would put copies of one photo into both train and test, and that is one
likely reason published scores can reach about 99%. This script:

1. Groups each photo with its augmented copies. Candidates are found by thumbnail correlation
   across all 8 rotations/flips, then each candidate pair is checked for being a *pixel-aligned
   copy*. The check removes the brightness gain per pair and measures the 25th-percentile pixel
   difference. A copy matches almost exactly on most pixels, even with blur or blood stains on
   some of them. Two different shots of a similar tray differ slightly everywhere (lighting,
   shadows, noise). On the synthetic test set, copies score ≤ 0.007 and distinct photos
   ≥ 0.019, so the default `--dup-thresh` is 0.011. Roboflow `*.rf.<hash>` filename stems are
   grouped as well.
   **Check `borderline_pairs.csv` on the real data.** It lists image pairs between 0.5× and 2×
   the threshold. Open a few and adjust `--dup-thresh` if copies are missed or different photos
   are merged. The script prints a warning if grouping chains many photos into one huge group.
2. Splits **whole groups** 70/15/15, balancing per-class counts.
3. Runs a check that reports test images with a near-duplicate in train. This should be 0.
4. Writes `test_overlap.txt` and `test_single.txt` with matching `data_test_*.yaml`. Overlapping
   tools are the hard case, so they are scored separately.

`data/yolo/split_report.json` holds the per-split class counts.

## 3. Train

```bash
python src/train.py                                          # YOLO11n @ 640 (phone model), configs/train.yaml
python src/train.py --set model=yolo11s.pt name=yolo11s_640  # step up if nano mixes up scissor types
```

Default choices:

- `yolo11n` at `imgsz=640`: about 2.6M parameters, roughly 3 MB as an int8 file, real-time on recent phones. The model is trained at the same resolution the app will feed it.
- `epochs=200`: small models take longer to converge.
- `scale`, `translate`, `perspective` and `hsv_*`: these imitate hand-held phone photos taken at different heights, slight tilts and varied lighting.
- `degrees=180` and `flipud`: tools lie on a tray at any angle.
- `mosaic`, `copy_paste=0.3` and `mixup`: these create cluttered trays with overlapping tools.

### Training on a free GPU (Google Colab / Kaggle)

```bash
git clone <your repo> && cd surgical-tool-detection
pip install -r requirements.txt
pip install -r requirements-export.txt   # for src/export.py (phone formats)
# upload/extract the Mendeley zip into data/raw/, then:
python src/prepare_dataset.py
python src/train.py --set batch=-1      # batch=-1 auto-fits GPU memory
```

YOLO11n at 640 px for 200 epochs on ~4,200 training images takes about 2–3 hours on a T4.
You can resume an
interrupted run with `yolo train resume model=runs/<name>/weights/last.pt`.

## 4. Evaluate

```bash
python src/evaluate.py --weights runs/yolo11n_640/weights/best.pt
```

This writes `runs/eval/<run>/report.md` and `metrics.json`. They contain:

- P / R / F1 / mAP50 / **mAP50-95** for `test`, `test_overlap` and `test_single`
- per-class scores and the most frequent class confusions, plus confusion-matrix and PR-curve plots
- **tray counting accuracy**: the share of images where every class count is exactly right, and the mean count error
- ms/image and FPS on the current device

All scores are computed on the fixed square input the phone model receives (`rect=False`).
Ultralytics' default validation pads each image differently, and on the smoke-test data that
inflated mAP50-95 from 25 to 42. Training-time validation still uses the default, so expect
the numbers in `runs/<name>/results.csv` to be higher than the ones in the evaluation report.

Report mAP50-95 next to mAP50. mAP50 saturates near 99% in this domain, while mAP50-95 shows
how precisely the boxes fit.

## 5. Inference / counting

```bash
python src/predict.py --weights best.pt --source tray.jpg
python src/predict.py --weights best.pt --source folder/ --save-json counts.json
python src/predict.py --weights best.pt --source 0      # webcam
```

## 6. Mobile deployment

```bash
python src/export.py --weights runs/yolo11n_640/weights/best.pt
```

This exports each phone format and **re-evaluates every exported model on the test split**,
so you can see how much accuracy quantization costs. The report goes to
`runs/export/<run>/export_report.md`. The files the app needs go to `mobile/<run>/`, together
with `labels.txt` and `model_card.json` (input size, pre/post-processing, recommended thresholds).

| Target | Use it for | Notes |
|---|---|---|
| `litert-int8` | Android, smallest and fastest | Full INT8, calibrated on training images. Runs on NPU/DSP via NNAPI and on CPU (XNNPACK). Check the accuracy drop in the report. |
| `litert-w8a32` | Android, safest accuracy | INT8 weights with float activations: 4× smaller than FP32 with nearly identical accuracy. Use with the GPU delegate. |
| `coreml-fp16` | iOS | Runs on the Neural Engine. NMS is built into the model, so `VNCoreMLRequest` returns boxes directly. |
| `coreml-w8a16` | iOS, smaller | INT8 weights, FP16 compute. |
| `onnx` | ONNX Runtime Mobile / web | FP32 reference. |

**Core ML export** needs torch ≤ 2.7, because that's the newest version coremltools supports. Use a
separate environment:

```bash
python -m venv .venv-coreml && .venv-coreml/bin/pip install "torch==2.7.0" "torchvision==0.22.0" "ultralytics==8.4.194" "coremltools>=8" "numpy<=2.3.5" scikit-learn
.venv-coreml/bin/python src/export.py --weights runs/yolo11n_640/weights/best.pt --targets coreml-fp16 coreml-w8a16
```

**In the app:**

- Resize the camera frame to 640 with letterboxing: keep the aspect ratio and pad with grey (114). Scale pixel values to 0–1.
- LiteRT/ONNX input is **`[1, 3, 640, 640]` float32, channels-first (NCHW)**. Even the int8
  model takes and returns float32 because it quantizes internally. Most TFLite camera samples
  assume NHWC `[1, 640, 640, 3]`, so transpose the pixels if you reuse one. `model_card.json`
  records the exact shapes of your export.
- LiteRT/ONNX output is `[1, 13, 8400]`: box centre/size plus 9 class scores per candidate. Keep candidates scoring ≥ 0.4, run NMS at IoU 0.6, then count boxes per class. Core ML already does this step.
- For a live camera view, run detection on every 2nd–3rd frame and reuse the last boxes in between. For counting a tray, take one still photo instead: it's sharper and more accurate.
- **Fastest route to a cross-platform app:** the official Ultralytics Flutter plugin
  [`ultralytics_yolo`](https://pub.dev/documentation/ultralytics_yolo/0.6.14/). It loads the
  `.tflite` on Android and the `.mlpackage` on iOS, handles pre-processing and NMS, and reads the
  class names from the export's metadata. It is built around Ultralytics' own exports, but
  test it with a plugin version that matches the Ultralytics version you exported with (8.4.x here). Put the `.tflite` file in `assets/models/` and add the
  `.mlpackage` to the Xcode workspace. Check the plugin's README for the current version.

**If int8 loses too much accuracy**, in order of effort: use `litert-w8a32` / `coreml-w8a16`
instead, step up to `yolo11s`, or train with quantization-aware training (`--set quantize=8`;
needs an NVIDIA GPU).

## Smoke test (no download needed)

```bash
pytest -q                      # data pipeline unit tests
bash scripts/smoke_test.sh     # synthetic data -> prepare -> train -> evaluate -> predict
```

`src/make_synthetic.py` draws simple outlines of the 9 classes on tray-like backgrounds and
saves them as VOC, YOLO or COCO. It exists only to show the pipeline runs end to end. Scores
from synthetic data say nothing about real-data accuracy.

## Suggested experiment plan

1. YOLO11n compared with YOLO11s at 640: is the extra size worth it for the scissor classes?
2. 640 compared with 512 or 800 px: the accuracy-vs-latency trade-off on a real phone.
3. Accuracy loss of each export format (`src/export.py`), especially full INT8.
4. Ablation: with and without `copy_paste`, scored on `test_overlap`.
5. Compare the leakage-safe split against a random split (`--dup-thresh -1` on a copy) to
   measure how much leakage inflates the scores.
6. Optional: map OSI26 classes onto this class set and test generalisation across datasets.

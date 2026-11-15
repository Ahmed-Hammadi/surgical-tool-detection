# Handover

Written 2026-11-15 for the next working session. Read this first, then `README.md` and
`docs/adding_instruments.md`.

## Goal

A lightweight detector that finds and counts surgical instruments in phone photos of trays,
shipped inside a phone app (Android LiteRT/TFLite, iOS Core ML). Target: **30+ instrument
types** covering general, orthopaedic and paediatric sets.

## Where things stand

| Area | State |
|---|---|
| Code pipeline (prepare, train, evaluate, predict, export) | Built. 10 unit tests pass. End-to-end run verified on **synthetic drawings only**. |
| Taxonomy | 43 classes in `configs/classes.yaml`, tagged by family and department. |
| Real training with this repo | **Not done yet.** No real dataset has been run through `prepare_dataset.py`. |
| Earlier model | The owner trained a 7-class model on Kaggle before this repo; it worked. It is not in this repo. |
| Data for classes beyond the basic 9 | Not downloaded. Candidate sources listed below, all unverified. |
| Phone app | Not started. Export formats and a model card are ready for it. |

Nothing from the synthetic runs says anything about real accuracy.

## The owner's machine

- Windows, PowerShell. Use `;` or separate lines, not `&&`. Use `Expand-Archive`, not `unzip`.
- GPU: NVIDIA GTX 1650 Ti, 4 GB. Enough for YOLO11n at 640 px; expect roughly 6–10 hours
  for 200 epochs (an estimate, not measured).
- The repo was cloned under OneDrive (`C:\Users\ahmed\OneDrive\Documents\surgical-tool-detection`).
  Move it out, for example to `C:\dev\surgical-tool-detection`: OneDrive syncing slows
  training and can lock files.
- The existing venv uses Python 3.14, which may be too new for PyTorch. If `torch` fails to
  install or CUDA is not found, recreate the venv with Python 3.12.

## Setup on that machine

```powershell
py -3.12 -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```powershell
pip install -r requirements.txt
```

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. Then check the code:

```powershell
python -m pytest -q
```

## Next steps, in order

### 1. Get the data

Each dataset goes in its own folder under `data\raw\`. `data\` is git-ignored.

| Priority | Dataset | Link | Reported content |
|---|---|---|---|
| 1 | Mendeley SURGICAL TOOLS | https://data.mendeley.com/datasets/cyghvmjrt3/1 (Kaggle mirror: https://www.kaggle.com/datasets/jocelyndumlao/surgical-tools) | 6,000 images, the 9 basic classes |
| 2 | HOSPI-Tools | https://zenodo.org/records/5895068 (2.2 GB, CC BY 4.0, a newer version exists) | about 40,000 images, 360 instruments from orthopaedic and general surgery, boxes and masks |
| 3 | Labeled Surgical Tools and Images | https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools | about 2,600–3,000 images: scalpel, dissection clamp, mayo scissors |
| 4 | Surgical Tool Recognition (multiview) | https://huggingface.co/datasets/joonhaim/surgical-tool-recognition-full-multiview | 2,520 images, YOLO: clamp, needle holder, scalpel, shear, tweezer |
| 5 | Roboflow: surgical_tools (Zahoor Abbas) | https://universe.roboflow.com/zahoor-abbas-project/surgical_tools-6b7ze-kaxoz | about 1,500 images: allis, babcock, needle holder, sponge holder, several retractors |
| 6 | Roboflow: Surgery Tools (Nathan) | https://universe.roboflow.com/nathan-0dwqd/surgery-tools-qsdp7 | kocher, metzenbaum, needle holder, several retractors |

**None of these pages could be opened when this was written** (the build environment's
network blocked them). Sizes and class lists come from search results. For each one, check
the license, the actual class names and a sample of the labels before merging.

**HOSPI-Tools needs work before it can be used.** Its folder structure and label format are
unknown. First job: list its folders and label files, then either confirm
`prepare_dataset.py` reads it or write a small converter, and map its 360 fine-grained
instrument names onto the 43 classes by adding aliases in `configs/classes.yaml`. Its photos
were taken from 60–150 cm with a DSLR and webcam, so instruments may look smaller than in
hand-held phone photos.

No public dataset specific to paediatric surgery was found. Paediatric is handled as a tray
type (`--department pediatric`), since those sets are mostly smaller versions of general
instruments and a photo carries no scale.

### 2. Prepare

```powershell
python src/prepare_dataset.py
```

- If it stops with "Labels not in configs/classes.yaml", add each listed label as an alias
  of a class, under `ignore`, or under `exclude_images_with`.
- Read `data\yolo\coverage_report.md`: which classes are trained, which lack data.
- Check that `[audit] test images with a near-duplicate in train` is 0 or close to it, and
  open a few pairs from `data\yolo\borderline_pairs.csv`. The duplicate threshold
  (`--dup-thresh 0.011`) was tuned on synthetic data only.
- If the script warns about one huge group, lower `--dup-thresh`.

### 3. Train

```powershell
python src/train.py --set batch=-1 workers=2
```

If losses show `nan` in the first epoch (a known GTX 16-series half-precision problem), stop
and add `amp=False`. Resume an interrupted run with
`yolo train resume model=runs\yolo11n_640\weights\last.pt`.

### 4. Evaluate and export

```powershell
python src/evaluate.py --weights runs\yolo11n_640\weights\best.pt
```

```powershell
pip install -r requirements-export.txt
```

```powershell
python src/export.py --weights runs\yolo11n_640\weights\best.pt --targets litert-int8 litert-w8a32 onnx
```

Core ML export needs a separate environment with torch 2.7 (see README, "Mobile deployment").
LiteRT export has only been run on Linux; it is untested on Windows.

### 5. Then

- Compare with the earlier 7-class Kaggle model on the same phone photos.
- Read the class-confusion list in the evaluation report. Merge look-alike classes that stay
  confused, or move to `yolo11s.pt`.
- For classes with no public data (most orthopaedic tools), follow the capture protocol in
  `docs/adding_instruments.md` and draft labels with `src/prelabel.py`.

## Decisions already made

- **YOLO11n at 640 px** is the default: about 3 MB as int8. More classes add almost no size.
- **Leakage-safe split.** Augmented copies of one photo are grouped and kept on one side of
  the split. Published 99% scores on the Mendeley data likely benefit from leakage.
- **Evaluation uses a fixed square input** (`rect=False`), as on a phone. Scores logged
  during training use Ultralytics' default and read higher.
- **Classes are defined by shape, not size or department.** `mosquito_forceps` vs `hemostat`
  is size-only and will probably need merging.
- **A class is trained only with 40+ labelled boxes** (`--min-instances`). Images showing an
  untrained class are left out so the model is not taught that an instrument is background.
- **DocCheck's Rona model was tested and rejected**: 0 detections on the owner's phone photo,
  wrong classes, YOLOv5-Large is too big for a phone. `scripts/try_doccheck.py` remains only
  as a record.
- **License:** Ultralytics is AGPL-3.0. Publishing the app means open-sourcing it or buying
  their commercial license.

## Known gaps

- `src/prelabel.py` with the default text-prompted detector (YOLOE) ran but found nothing on
  synthetic drawings; untested on real photos. It downloads a text model of about 600 MB.
- Merged datasets only label their own classes. Unlabelled instruments in their photos teach
  the model to ignore them; skim each new dataset for this.
- The LiteRT model takes `[1, 3, H, W]` float32 input (channels first), unlike most TFLite
  camera samples. `mobile/<run>/model_card.json` records the exact shapes.

## File map

| Path | Purpose |
|---|---|
| `configs/classes.yaml` | 43-class taxonomy, aliases, ignore lists |
| `configs/train.yaml` | training settings |
| `src/prepare_dataset.py` | merge sources, map labels, split, coverage report |
| `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/export.py` | train, score, run, export for phones |
| `src/prelabel.py` | draft labels for own photos (one class per folder) |
| `src/make_synthetic.py`, `scripts/smoke_test.sh` | synthetic data and end-to-end check |
| `docs/adding_instruments.md` | how to add classes: data sources, capture, labelling |
| `tests/test_prepare.py` | unit tests for the data pipeline |

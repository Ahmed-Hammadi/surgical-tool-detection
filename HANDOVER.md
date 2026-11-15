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
| Kaggle notebook | `notebooks/kaggle_train.ipynb` written, not yet run on Kaggle. |
| Data for classes beyond the basic 9 | Not downloaded. Candidate sources listed below, all unverified. |
| Phone app | Not started. Export formats and a model card are ready for it. |

Nothing from the synthetic runs says anything about real accuracy.

## Where training happens: Kaggle

The owner trains on **Kaggle notebooks**, not on a local PC. Give commands as Kaggle notebook
cells (`!command`, Linux paths), not PowerShell.

- Notebook: `notebooks/kaggle_train.ipynb`. It clones this repo, installs requirements,
  prepares the data, trains, evaluates, exports and packs `results.zip`.
  **It has not been run on Kaggle yet**; expect to fix small things on the first run.
- Settings needed: Accelerator **GPU T4 x2** (or P100), Internet **On** (phone-verified account).
- Datasets are attached with **Add Input** and appear read-only under `/kaggle/input/<name>/`.
- Only `/kaggle/working/` is kept. Sessions last at most 12 hours; about 30 GPU hours a week.
- Long runs: **Save Version → Save & Run All (Commit)**, so the run survives a closed tab.
- Code changes go through GitHub: edit, push, and the notebook's first cell clones the latest.

## Next steps, in order

### 1. Get the data into Kaggle

| Priority | Dataset | How to get it on Kaggle | Reported content |
|---|---|---|---|
| 1 | Mendeley SURGICAL TOOLS | Add Input: `jocelyndumlao/surgical-tools` | 6,000 images, the 9 basic classes |
| 2 | HOSPI-Tools | Not on Kaggle. `wget` from https://zenodo.org/records/5895068 in the notebook (2.2 GB, CC BY 4.0, a newer version exists), or upload once as a private Kaggle Dataset | about 40,000 images, 360 instruments from orthopaedic and general surgery, boxes and masks |
| 3 | Labeled Surgical Tools and Images | Add Input: `dilavado/labeled-surgical-tools` | about 2,600–3,000 images: scalpel, dissection clamp, mayo scissors |
| 4 | Surgical Tool Recognition (multiview) | Download from https://huggingface.co/datasets/joonhaim/surgical-tool-recognition-full-multiview in the notebook | 2,520 images, YOLO: clamp, needle holder, scalpel, shear, tweezer |
| 5 | Roboflow: surgical_tools (Zahoor Abbas) | Export as YOLO from https://universe.roboflow.com/zahoor-abbas-project/surgical_tools-6b7ze-kaxoz (free account), upload as a Kaggle Dataset | about 1,500 images: allis, babcock, needle holder, sponge holder, several retractors |
| 6 | Roboflow: Surgery Tools (Nathan) | Same, from https://universe.roboflow.com/nathan-0dwqd/surgery-tools-qsdp7 | kocher, metzenbaum, needle holder, several retractors |

**None of these pages could be opened when this was written** (the build environment's
network blocked them). Sizes and class lists come from search results. For each one, check
the license, the actual class names and a sample of the labels before merging. In
particular, it is not confirmed that the Kaggle mirror of the Mendeley set includes
bounding-box labels; the notebook's "look at the input datasets" cell shows this.

**HOSPI-Tools needs work before it can be used.** Its folder structure and label format are
unknown. First job: list its folders and label files, then either confirm
`prepare_dataset.py` reads it or write a small converter, and map its 360 fine-grained
instrument names onto the 43 classes by adding aliases in `configs/classes.yaml`. Its photos
were taken from 60–150 cm with a DSLR and webcam, so instruments may look smaller than in
hand-held phone photos.

No public dataset specific to paediatric surgery was found. Paediatric is handled as a tray
type (`--department pediatric`), since those sets are mostly smaller versions of general
instruments and a photo carries no scale.

### 2. Prepare (notebook section 2)

```
!python src/prepare_dataset.py --raw /kaggle/input/surgical-tools /kaggle/input/<other> --out data/yolo
```

- Pass one folder per dataset. If a dataset's top folder holds several unrelated sub-folders
  and the script reports "No COCO/VOC/YOLO annotations found", point `--raw` at the
  sub-folder that holds the images and labels.
- If it stops with "Labels not in configs/classes.yaml", add each listed label as an alias
  of a class, under `ignore`, or under `exclude_images_with`, push, and re-run.
- Read `data/yolo/coverage_report.md`: which classes are trained, which lack data.
- Check that `[audit] test images with a near-duplicate in train` is 0 or close to it, and
  look at a few pairs from `data/yolo/borderline_pairs.csv`. The duplicate threshold
  (`--dup-thresh 0.011`) was tuned on synthetic data only.
- If the script warns about one huge group, lower `--dup-thresh`.

### 3. Train (notebook section 3)

```
!python src/train.py --set batch=-1 workers=4
```

YOLO11n at 640 px on a T4 should take a few hours for 200 epochs on the Mendeley set (an
estimate, not measured); more data takes proportionally longer, so watch the 12-hour limit
and lower `epochs` if needed. To resume a cut-off run, add the earlier version's output as
an input and run `!yolo train resume model=<path>/last.pt`.

### 4. Evaluate and export (notebook sections 4–6)

```
!python src/evaluate.py --weights runs/yolo11n_640/weights/best.pt
!pip install -q -r requirements-export.txt
!python src/export.py --weights runs/yolo11n_640/weights/best.pt --targets litert-int8 litert-w8a32 onnx
```

The export install has not been tried on Kaggle and may change the PyTorch version there. If
it breaks the session, skip it and export later from the saved `best.pt`. Core ML export
needs a separate environment with torch 2.7 (see README, "Mobile deployment").
Download `results.zip` from the notebook's Output panel.

### 5. Then

- Compare with the earlier 7-class Kaggle model on the same phone photos.
- Read the class-confusion list in the evaluation report. Merge look-alike classes that stay
  confused, or move to `yolo11s.pt`.
- For classes with no public data (most orthopaedic tools), follow the capture protocol in
  `docs/adding_instruments.md`: photograph, upload the photos as a Kaggle Dataset, and draft
  labels with `src/prelabel.py` in a notebook.

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
| `notebooks/kaggle_train.ipynb` | the Kaggle training notebook |
| `tests/test_prepare.py` | unit tests for the data pipeline |

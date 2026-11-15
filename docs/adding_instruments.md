# Growing the model to 30+ instruments (orthopaedic and paediatric sets)

The taxonomy in `configs/classes.yaml` lists 43 instrument classes. The code handles all of
them today. What decides how many the model actually recognises is **labelled photos**: a
class is trained once it has enough examples, and `prepare_dataset.py` reports the rest as
missing. This guide is the plan for filling those gaps.

## Where the classes stand

| Set | Classes | Labelled data today |
|---|---|---|
| Basic set | forceps, hemostat, scalpel, mayo/bandage/episiotomy scissors, syringe, gloves, cotton | Mendeley SURGICAL TOOLS (about 6,000 images) |
| Soft-tissue | metzenbaum and iris scissors, adson and debakey forceps, mosquito, allis, babcock, kocher, needle holder, towel clamp, sponge forceps, suction tip, probe, skin hook | Partly, in public sets listed below (unverified) |
| Retractors | senn, army-navy, langenbeck, deaver, weitlaner, hohmann | Partly, in public sets listed below (unverified) |
| Orthopaedic | mallet, osteotome, curette, periosteal elevator, rongeur, bone cutter, bone-holding forceps, rasp, bone hook, gigli saw, depth gauge, screwdriver, wire cutter, pliers | Almost none public: needs your own photos |

After every `prepare_dataset.py` run, open `data/yolo/coverage_report.md`. It lists each
class with its box count and status, and how many classes of each department are covered.

## Public datasets worth checking

These came up in a web search. Their pages could not be opened from the build environment,
so the class lists and licenses below are **unverified**: open each one, check the license
allows your use, and look at a sample of the labels before merging.

| Dataset | Reported content | Could add |
|---|---|---|
| Roboflow Universe, "surgical_tools" (Zahoor Abbas Project) | about 1,500 images; several retractor types (Deaver, Farabeuf, Langenbeck, self-retaining), needle holder, Rampley sponge holder | retractors, needle_holder, sponge_forceps |
| Roboflow Universe, "Surgery Tools" (Nathan) | Deaver, Harrington, Morris, Senn and self-retaining retractors, needle holder | retractors, needle_holder |
| Roboflow Universe, class searches for "allis forceps", "babcock tissue forceps", "halsted mosquito" | several small sets | allis, babcock, mosquito |
| Roboflow Universe, sets listing "Bone Mallet", "Bone-Cutter", "Bone-Nibbler", "Bone Chisel", "Bone File" | seen in search listings only | first orthopaedic classes |
| Hugging Face, `joonhaim/surgical-tool-recognition-full-multiview` | 2,520 images, YOLO format: clamp, needle_holder, scalpel, shear, tweezer | needle_holder, more scalpel/forceps |
| Kaggle, "Labeled Surgical Tools and Images" | 2,620 images: scalpel, straight dissection clamp, straight and curved mayo scissors | more basic-set data |
| DocCheck Rona dataset (Hugging Face) | fixed overhead camera, white drape: 4 forceps types, 4 scalpel types, dissecting scissors, needle | more forceps/scalpel data from a different camera |

To merge one, export it in YOLO, VOC or COCO format into its own folder and re-run:

```
data/raw/mendeley/...
data/raw/roboflow_retractors/...
```

```bash
python src/prepare_dataset.py
```

If a label spelling is new, the script stops and lists it. Add it in `configs/classes.yaml`
as an alias of a class, under `ignore` (not an instrument), or under `exclude_images_with`
(an instrument we have no class for).

**The partial-label trap.** A merged dataset only labels its own classes. If its photos also
show instruments it did not label (a hemostat lying next to the labelled retractor), the model
is taught that those are background and gets worse at finding them. Skim each new dataset's
images for this, and either label the extra instruments or leave those images out.

## Collecting your own photos (the only route for most orthopaedic tools)

**Target per class:** 150–300 photos to start, with the instrument appearing 300+ times in
total. Fewer than about 100 photos gives a class that works on the test set and fails in the
ward.

**Protocol that keeps labelling cheap:**

1. One session per instrument type. Put only that type in the frame, one to four pieces.
2. Take the photos with the phone the app will run on, hand-held, from 30–60 cm.
3. Between shots change something: rotate and flip the instrument, open and close hinged
   ones, move it in the frame, change the distance and the tilt of the phone.
4. Cover the conditions the app will meet: at least three backgrounds (green and blue drape,
   steel tray, white cloth), bright and dim light, a few with glare, a few slightly blurred.
5. If you have several sizes or makes of the same instrument, include all of them.
   Photos of one single physical instrument teach the model that one object, not the type.
6. Save each session into a folder named after the class:

   ```
   data/capture/hohmann_retractor/
   data/capture/bone_mallet/
   ```

7. Then photograph 100–200 real mixed trays (several types together, touching and
   overlapping) into `data/capture/_mixed/`. These matter most for real-world accuracy.

Do not photograph anything that identifies a patient, and follow your hospital's rules on
photography in clinical areas.

**Draft the labels automatically:**

```bash
python src/prelabel.py --capture data/capture --name my_hospital
```

For class folders the detector only has to find where the instruments are; the class comes
from the folder name, so this works for instruments no model has seen. It writes a YOLO
dataset to `data/raw/my_hospital/`, review pictures with the boxes drawn, and a list of
photos where it found nothing. The default detector is YOLOE, prompted with the text
"surgical instrument"; it needs `pip install git+https://github.com/ultralytics/CLIP.git` and
downloads a text model of about 600 MB on first use. It has only been run on synthetic
drawings so far, where it found nothing, so check its drafts on your first real photos
before relying on it.
Once you have a trained model, pass `--weights runs/<name>/weights/best.pt` instead, and use
that for the `_mixed` folder as well.

The drafts are a starting point. Open the dataset in a YOLO label editor (CVAT, Label Studio,
Roboflow, labelImg), delete wrong boxes, tighten loose ones and draw the missing ones. Mixed
trays must be labelled completely: every instrument in the photo, every time.

Then merge and retrain:

```bash
python src/prepare_dataset.py
python src/train.py --set batch=-1 workers=2
```

## Keeping it lightweight with 30+ classes

Adding classes costs almost nothing in size: each extra class adds a few hundred parameters
to the last layer. The limit is how well a small network separates look-alike instruments.

| Model | On-device size (int8) | When |
|---|---|---|
| YOLO11n @ 640 | about 3 MB | Start here |
| YOLO11s @ 640 | about 10 MB | If look-alike classes stay confused after you have enough data |
| YOLO11n @ 800 | about 3 MB, slower | If small instruments or fine tips are the problem |

Decide with the evaluation report, not in advance: `evaluate.py` lists the most frequent
class confusions.

## Look-alike classes: what to expect

Some pairs differ in details a phone photo barely shows. Expect these to be the weak spots:

- **hemostat vs mosquito_forceps**: same shape, different size. A photo has no scale, so
  this pair cannot be separated reliably. If the confusion stays high, merge them.
- **hemostat vs needle_holder vs kocher vs allis vs babcock**: the difference is in the jaws.
- **mayo vs metzenbaum vs iris scissors**: blade length relative to the handles.
- **forceps vs adson vs debakey**: tip shape.
- **osteotome vs periosteal_elevator vs bone_curette**: tip shape on a similar handle.

Three ways to handle them, in order of effort:

1. Train on families instead of types and check whether that is enough for the app:
   ```bash
   python src/prepare_dataset.py --granularity coarse --out data/yolo_coarse
   ```
   This gives 7 classes (forceps, clamp, scalpel, scissors, retractor, bone, other).
2. Merge a confused pair by making one class an alias of the other in `classes.yaml`.
3. Add close-up photos that show the distinguishing part, and move to YOLO11s.

**Paediatric sets** mostly contain smaller versions of the same patterns as adult sets. The
model detects the pattern, not the size, so "paediatric" is handled as a tray type: the app
(or `predict.py --department pediatric`) limits the result to the classes of that set.
`mobile/<run>/model_card.json` lists the class ids of each department for this purpose.

## Order of work

1. Train the basic set on Mendeley to get a working app end to end.
2. Add the public datasets that pass your check, aiming first at needle holder, towel clamp,
   allis, babcock and the retractors. This should bring the model to roughly 20 classes.
3. Photograph the orthopaedic set and the paediatric-specific instruments with the protocol
   above. This is what takes it past 30.
4. After each step, read the coverage report and the confusion list, and decide whether to
   merge classes, add photos or move to YOLO11s.

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import make_synthetic  # noqa: E402
import prepare_dataset  # noqa: E402
from common import ClassMap, has_overlap, overlap_ratio  # noqa: E402


def test_classmap_aliases():
    cm = ClassMap()
    assert cm.canonical("Mayo Scissors") == "mayo_scissors"
    assert cm.canonical("haemostats") == "hemostat"
    assert cm.canonical("Surgical-Gloves") == "surgical_gloves"
    assert cm.canonical("stethoscope") is None


def test_overlap_ratio():
    assert overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert overlap_ratio((0, 0, 10, 10), (2, 2, 4, 4)) == 1.0
    assert overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)
    assert has_overlap([(0, 0, 10, 10), (5, 0, 15, 10)])
    assert not has_overlap([(0, 0, 10, 10), (11, 0, 15, 10)])


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Same synthetic photos written in all three annotation formats."""
    root = tmp_path_factory.mktemp("synth")
    for fmt in ("voc", "yolo", "coco"):
        make_synthetic.main(["--out", str(root / fmt), "--n", "150", "--aug-copies", "0.6",
                             "--size", "320", "--format", fmt, "--seed", "1"])
    return root


@pytest.mark.parametrize("fmt", ["voc", "yolo", "coco"])
def test_readers_agree(synth, fmt):
    got_fmt, recs = prepare_dataset.load_records(synth / fmt, "auto")
    assert got_fmt == fmt
    ref_fmt, ref = prepare_dataset.load_records(synth / "voc", "voc")
    key = lambda rs: sorted((r.image.name, len(r.boxes)) for r in rs)  # noqa: E731
    assert key(recs) == key(ref)
    a = {r.image.name: sorted(b[1:] for b in r.boxes) for r in recs}
    b = {r.image.name: sorted(b[1:] for b in r.boxes) for r in ref}
    for name in a:
        for ba, bb in zip(a[name], b[name]):
            assert ba == pytest.approx(bb, abs=1.0)


def test_split_has_no_leakage(synth, tmp_path):
    out = tmp_path / "yolo"
    rep = prepare_dataset.main(["--raw", str(synth / "voc"), "--out", str(out), "--seed", "0"])
    # augmented copies are grouped with their source photo; allow a stray miss for copies
    # whose content is destroyed (e.g. brightness boost clipping most pixels to white)
    n_src = len(list((synth / "voc" / "images").glob("IMG_*.jpg")))
    assert n_src <= rep["groups"] <= n_src + max(1, n_src // 100)
    assert rep["largest_groups"][0] <= 3
    assert rep["test_near_duplicates_in_train"] <= 1

    # map written files back to the original names and check groups don't straddle splits
    _, recs = prepare_dataset.load_records(synth / "voc", "voc")
    gids, _ = prepare_dataset.group_records(recs, prepare_dataset.compute_fingerprints(recs), prepare_dataset.DUP_THRESH)
    split_of = {}
    for s in ("train", "val", "test"):
        for p in (out / "images" / s).iterdir():
            split_of[p.name.split("_", 1)[1].rsplit(".", 1)[0]] = s
    by_group = {}
    for r, g in zip(recs, gids):
        by_group.setdefault(g, set()).add(split_of[r.image.stem])
    assert all(len(s) == 1 for s in by_group.values())

    d = yaml.safe_load((out / "data.yaml").read_text())
    assert len(d["names"]) == 9 and d["test"] == "images/test"
    for s in ("train", "val", "test"):
        assert any((out / "images" / s).iterdir())
        for lbl in (out / "labels" / s).iterdir():
            for line in lbl.read_text().splitlines():
                c, *xywh = line.split()
                assert 0 <= int(c) < 9 and all(0 <= float(v) <= 1 for v in xywh)
    assert json.loads((out / "split_report.json").read_text())["images"] == len(recs)


def test_unknown_label_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    make_synthetic.main(["--out", str(raw), "--n", "3", "--size", "128", "--format", "voc"])
    xml = next((raw / "images").glob("*.xml"))
    xml.write_text(re.sub(r"<name>[^<]*</name>", "<name>Retractor</name>", xml.read_text(), count=1))
    with pytest.raises(SystemExit, match="Unknown labels"):
        prepare_dataset.main(["--raw", str(raw), "--out", str(tmp_path / "out")])
    prepare_dataset.main(["--raw", str(raw), "--out", str(tmp_path / "out"), "--drop-unknown"])


@pytest.mark.parametrize("frame", ["upright", "raw"])
def test_exif_rotated_photo_labels_stay_on_object(tmp_path, frame):
    """A phone photo stored sideways with EXIF orientation=6, annotated either in the upright
    frame (most tools) or the raw sensor frame. The written label must cover the object."""
    import numpy as np
    from PIL import Image, ImageOps

    raw = tmp_path / "raw"
    raw.mkdir()
    im = Image.new("RGB", (200, 100), (30, 90, 110))
    im.paste((255, 0, 0), (10, 20, 60, 40))
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(raw / "p.jpg", exif=exif, quality=100)

    shown = ImageOps.exif_transpose(Image.open(raw / "p.jpg")) if frame == "upright" else Image.open(raw / "p.jpg")
    a = np.asarray(shown.convert("RGB")).astype(int)
    ys, xs = np.nonzero((a[..., 0] > 200) & (a[..., 1] < 60))
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    (raw / "p.xml").write_text(
        f"<annotation><filename>p.jpg</filename><size><width>{shown.width}</width><height>{shown.height}</height></size>"
        f"<object><name>scalpel</name><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax>"
        f"<ymax>{box[3]}</ymax></bndbox></object></annotation>")

    out = tmp_path / "out"
    prepare_dataset.main(["--raw", str(raw), "--out", str(out), "--split", "0", "0", "1"])
    img = next((out / "images" / "test").iterdir())
    written = np.asarray(Image.open(img).convert("RGB")).astype(int)  # what YOLO loaders will see
    H, W = written.shape[:2]
    _, cx, cy, bw, bh = map(float, (out / "labels" / "test" / (img.stem + ".txt")).read_text().split())
    crop = written[int((cy - bh / 2) * H) + 2:int((cy + bh / 2) * H) - 2, int((cx - bw / 2) * W) + 2:int((cx + bw / 2) * W) - 2]
    assert crop.size and (crop[..., 0] > 200).mean() > 0.95

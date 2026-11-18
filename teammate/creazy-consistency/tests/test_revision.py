"""Revision check (check/revision.py) on synthetic fixes of a real Creazy output.

The before image is a Run 10b mug cut. Fixes are simulated: the clay lump on the left is the
region the seller asked to change (filled by inpainting), and other parts are damaged or kept
on purpose. No model, no API key.

  .venv/bin/python -m pytest tests/test_revision.py -q
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from check import revision

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "data" / "mug" / "outputs" / "run10b" / "AI-Generated-2c19c13d-d8f8-4ca5-94eb-8060068f0e45.png"
OTHER = ROOT / "data" / "mug" / "outputs" / "run10b" / "AI-Generated-06437da8-17b4-4f62-9b63-c198c262b4bf.png"

CLAY = (0, 195, 265, 410)  # the region the fix was asked for
PAUSE = (612, 350, 738, 392)  # the word "pause" on the mug
HOLE = (125, 562, 182, 590)  # the hole in the wooden tool
CAT = (430, 300, 690, 600)  # the cat line drawing
MUG = (360, 160, 830, 630)


@pytest.fixture(scope="module")
def before():
    return cv2.imread(str(BEFORE))


def inpaint(img, box):
    m = np.zeros(img.shape[:2], np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)


def jpeg(img, q=85):
    _, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def patch(img, box, fn):
    out = img.copy()
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = fn(img)[y0:y1, x0:x1]
    return out


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_a_clean_fix_passes_even_resized_and_shifted(before):
    fixed = inpaint(before, CLAY)
    assert revision.compare(before, fixed, [CLAY])["verdict"] == "pass"
    # Returned smaller, a few pixels off, re-encoded, with light noise.
    small = cv2.resize(jpeg(fixed), None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    small = cv2.warpAffine(small, np.float32([[1, 0, 3], [0, 1, 2]]), (small.shape[1], small.shape[0]), borderMode=cv2.BORDER_REFLECT)
    noise = np.random.default_rng(0).normal(0, 2, small.shape)
    small = np.clip(small + noise, 0, 255).astype(np.uint8)
    r = revision.compare(before, small, [CLAY])
    assert r["align"]["method"] == "similarity"
    assert r["inside"]["verdict"] == "changed"
    assert r["outside"]["areas"] == [] and r["verdict"] == "pass"


@pytest.mark.parametrize("damage, box, kind", [
    (lambda img: inpaint(img, PAUSE), PAUSE, "content"),  # a word lost (CS 10, 24)
    (lambda img: inpaint(img, HOLE), HOLE, "content"),  # a small part filled in
    (lambda img: patch(img, CAT, lambda x: cv2.GaussianBlur(x, (0, 0), 2.0)), CAT, "detail"),  # print redrawn soft (CS 38)
    (lambda img: patch(img, MUG, lambda x: np.clip(x.astype(np.int16) - 15, 0, 255).astype(np.uint8)), MUG, "tone"),  # product darker
])
def test_changes_outside_the_region_are_found(before, damage, box, kind):
    r = revision.compare(before, damage(inpaint(before, CLAY)), [CLAY])
    assert r["verdict"] == "review"
    hits = [a for a in r["outside"]["areas"] if overlaps(a["box"], box)]
    assert hits and hits[0]["kind"] == kind
    assert not any(overlaps(a["box"], (0, 0, 120, 120)) for a in r["outside"]["areas"])  # untouched corner


def test_unchanged_region_redrawn_picture_and_no_region(before):
    same = revision.compare(before, jpeg(before, 80), [CLAY])
    assert same["inside"]["verdict"] == "unchanged" and same["verdict"] == "review"
    assert any("반영되지" in n for n in same["notes_ko"])
    other = revision.compare(before, cv2.imread(str(OTHER)), [CLAY])
    assert other["verdict"] == "redrawn"
    listed = revision.compare(before, inpaint(before, PAUSE))
    assert listed["verdict"] == "changes" and "inside" not in listed
    assert overlaps(listed["outside"]["areas"][0]["box"], PAUSE)
    assert revision.compare(before, jpeg(before))["verdict"] == "unchanged"


def test_product_box_grades_background_texture_as_minor(before):
    fixed = inpaint(before, CLAY)
    table = (1000, 560, 1280, 700)  # flour and wood grain on the table, right of the mug
    soft = lambda img, box: patch(img, box, lambda x: cv2.GaussianBlur(x, (0, 0), 2.0))
    bg = revision.compare(before, soft(fixed, table), [CLAY], product_boxes=[MUG])
    assert bg["verdict"] == "pass" and bg["outside"]["level"] == "minor"
    assert all(a["on_product"] is False for a in bg["outside"]["areas"])
    on = revision.compare(before, soft(fixed, CAT), [CLAY], product_boxes=[MUG])
    assert on["verdict"] == "review" and on["outside"]["level"] == "major" and on["product"]["areas"] >= 1
    # Without a product box the same background change is not excused.
    assert revision.compare(before, soft(fixed, table), [CLAY])["verdict"] == "review"


EDITS = ROOT / "data" / "mug" / "outputs" / "run10b_edit"
PRODUCT = {"edit1_clay": [358, 131, 958, 641], "edit2_cookies": [483, 230, 907, 591], "edit3_succulent": [449, 198, 696, 493]}  # OWLv2 "a mug"


def test_real_creazy_area_edits():
    """Three AI area edits made in Creazy (5 credits each, edits.json). The edit returns the
    whole picture redrawn at 1.31 to 1.41x: background texture changes everywhere, the coffee
    surface inside the mug changes in edit 2, the removed pot's shadow goes in edit 3."""
    edits = {e["name"]: e for e in json.loads((EDITS / "edits.json").read_text())["edits"]}
    got = {}
    for name, e in edits.items():
        r = revision.compare(cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"])), [e["box"]], product_boxes=[PRODUCT[name]])
        assert r["inside"]["verdict"] == "changed" and r["align"]["size_ratio"] > 1.3
        assert any("배예요" in n for n in r["notes_ko"])
        got[name] = r
    assert got["edit1_clay"]["verdict"] == "pass" and got["edit1_clay"]["outside"]["level"] == "minor"
    assert got["edit2_cookies"]["verdict"] == "review" and got["edit2_cookies"]["product"]["areas"] >= 1
    shadow = [a for a in got["edit3_succulent"]["outside"]["areas"] if a["kind"] == "content"]
    assert shadow and not shadow[0]["on_product"] and overlaps(shadow[0]["box"], (420, 390, 460, 490))


def test_mask_print_check_and_files(before, tmp_path):
    # A minimal reference set: the real print is the cat crop of the before image.
    x0, y0, x1, y1 = CAT
    cv2.imwrite(str(tmp_path / "print.png"), before[y0:y1, x0:x1])
    (tmp_path / "refset.json").write_text(json.dumps({"prints": [{"path": "print.png"}], "print": {"path": "print.png"}}))
    cv2.imwrite(str(tmp_path / "before.png"), before)
    mask = np.zeros(before.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    cv2.imwrite(str(tmp_path / "after.png"), inpaint(before, CAT))
    r = revision.run(tmp_path / "before.png", tmp_path / "after.png", tmp_path / "out", mask_path=tmp_path / "mask.png", refset=tmp_path / "refset.json")
    assert r["region"]["mask"] and r["inside"]["verdict"] == "changed"
    assert r["print"]["verdict"] == "lost" and r["verdict"] == "review"
    assert any("사라졌어요" in n for n in r["notes_ko"])
    saved = json.loads((tmp_path / "out" / "revision.json").read_text())
    assert saved["verdict"] == "review" and (tmp_path / "out" / "revision.jpg").exists()
    with pytest.raises(Exception):
        revision.parse_box("10,10,5,20")

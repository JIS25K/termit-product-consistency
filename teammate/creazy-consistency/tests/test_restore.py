"""Keep the original outside the edited region (check/restore.py).

  .venv/bin/python -m pytest tests/test_restore.py -q
"""
import json
from pathlib import Path

import cv2
import numpy as np

from check import restore, revision

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "data" / "mug" / "outputs" / "run10b_edit"
PRODUCT = {"edit1_clay": [358, 131, 958, 641], "edit2_cookies": [483, 230, 907, 591], "edit3_succulent": [449, 198, 696, 493]}  # OWLv2 "a mug"
CLAY = (0, 195, 265, 410)
CAT = (430, 300, 690, 600)


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_synthetic_edit_redrawn_at_another_size():
    before = cv2.imread(str(EDITS / "../run10b/AI-Generated-2c19c13d-d8f8-4ca5-94eb-8060068f0e45.png"))
    m = np.zeros(before.shape[:2], np.uint8)
    x0, y0, x1, y1 = CLAY
    m[y0:y1, x0:x1] = 255
    edited = cv2.inpaint(before, m, 7, cv2.INPAINT_TELEA)
    cx0, cy0, cx1, cy1 = CAT
    edited[cy0:cy1, cx0:cx1] = cv2.GaussianBlur(edited, (0, 0), 2.0)[cy0:cy1, cx0:cx1]  # print redrawn soft
    after = cv2.resize(edited, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)  # returned bigger
    out, taken, info = restore.restore(before, after, [CLAY], product_boxes=[[358, 131, 958, 641]])
    assert out.shape == before.shape and info["size"]["after"][0] / info["size"]["before"][0] > 1.29
    assert np.array_equal(out[~taken], before[~taken])  # the original, to the pixel
    assert np.array_equal(out[cy0:cy1, cx0:cx1], before[cy0:cy1, cx0:cx1])  # the print came back
    assert not taken[cy0:cy1, cx0:cx1].any()
    inside = out[y0 + 20 : y1 - 20, x0 + 20 : x1 - 20].astype(int)
    assert np.abs(inside - before[y0 + 20 : y1 - 20, x0 + 20 : x1 - 20].astype(int)).mean() > 5  # the fix is there


def test_real_creazy_area_edits_restored():
    """The three Creazy area edits: as returned, the background and the print detail are
    redrawn; restored, only the painted part (and the removed pot's shadow next to it)
    differs from the original."""
    for e in json.loads((EDITS / "edits.json").read_text())["edits"]:
        before, after = cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"]))
        product = [PRODUCT[e["name"]]]
        out, taken, info = restore.restore(before, after, [e["box"]], product_boxes=product)
        assert np.array_equal(out[~taken], before[~taken])
        assert info["taken"] < 0.25
        r = revision.compare(before, out, mask=taken.astype(np.uint8) * 255, product_boxes=product)
        assert r["verdict"] == "pass" and r["outside"]["level"] == "same", (e["name"], r["notes_ko"])
        px0, py0, px1, py1 = product[0]
        assert taken[py0 + 30 : py1 - 30, px0 + 30 : px1 - 30].mean() < 0.01  # the mug itself is the original
        if e["name"] == "edit3_succulent":
            assert any(overlaps(b, (420, 390, 460, 490)) for b in info["nearby"])


CHAT_PRODUCT = {"chat1_succulent": [449, 198, 696, 493], "chat2_pen": [478, 116, 891, 476]}  # OWLv2 "a mug"
ASKED = {"chat1_succulent": (128, 110, 402, 506), "chat2_pen": (140, 240, 335, 315)}  # the pot, the pen


def test_find_region_without_the_painted_box():
    """The three area edits again, the painted box withheld: what was drawn or removed lands
    on the painted place, and nothing else is taken."""
    for e in json.loads((EDITS / "edits.json").read_text())["edits"]:
        before, after = cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"]))
        region, info = restore.find_region(before, after, [PRODUCT[e["name"]]])
        assert region is not None and len(info["boxes"]) == 1, (e["name"], info)
        assert overlaps(info["boxes"][0], e["box"])
        x0, y0, x1, y1 = e["box"]
        m = region > 0
        assert m[max(0, y0 - 80) : y1 + 80, max(0, x0 - 80) : x1 + 80].sum() / m.sum() > 0.95


def test_real_creazy_chat_edits_restored():
    """Two chat edits ("reference-based image generation", 5 credits each): the whole picture
    comes back redrawn at the same framing and the print is redrawn too (score 48 -> 31,
    50 -> 37). Found without a painted box and restored, only the removed object differs."""
    for e in json.loads((EDITS / "edits.json").read_text())["chat"]:
        before, after = cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"]))
        product = [CHAT_PRODUCT[e["name"]]]
        as_is = revision.compare(before, after, product_boxes=product)
        assert as_is["align"]["method"] in ("identity", "similarity") and as_is["outside"]["areas"]
        region, info = restore.find_region(before, after, product)
        assert region is not None and info["area"] < 0.1 and info["boxes"], (e["name"], info)
        assert all(overlaps(b, ASKED[e["name"]]) for b in info["boxes"])
        out, taken, _ = restore.restore(before, after, mask=region, product_boxes=product)
        assert np.array_equal(out[~taken], before[~taken])
        r = revision.compare(before, out, mask=taken.astype(np.uint8) * 255, product_boxes=product)
        assert r["verdict"] == "pass" and r["outside"]["level"] == "same", (e["name"], r["notes_ko"])
        px0, py0, px1, py1 = product[0]
        assert taken[py0 + 30 : py1 - 30, px0 + 30 : px1 - 30].mean() < 0.01


def test_find_region_keeps_only_what_was_pointed_at():
    """A change outside the box the seller pointed at, or on the product when nothing was
    pointed at, is not taken: the restored image is the original."""
    e = json.loads((EDITS / "edits.json").read_text())["chat"][1]
    before, after = cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"]))
    region, info = restore.find_region(before, after, where=[[1000, 0, 1280, 200]])
    assert not (region > 0).any() and info["left_out"]
    out, taken, _ = restore.restore(before, after, mask=region)
    assert np.array_equal(out, before)
    region, info = restore.find_region(before, after, product_boxes=[list(ASKED["chat2_pen"])])
    assert not (region > 0).any() and info["left_out"]


def test_find_region_gives_up_on_another_picture():
    a = cv2.imread(str(EDITS / "../run10b/AI-Generated-06437da8-17b4-4f62-9b63-c198c262b4bf.png"))
    b = cv2.imread(str(EDITS / "../run10b/AI-Generated-3a7de889-b943-491c-85af-6fa31612bdbc.png"))
    region, _ = restore.find_region(a, b)
    assert region is None

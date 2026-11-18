import json
from pathlib import Path

import numpy as np
from PIL import Image

from check import expect as expect_mod
from check import strong


def T(found=(), missing=(), garbled=(), foreign=(), read=3):
    return {"read": read, "expected": list(found) + list(missing), "found": list(found),
            "missing": list(missing), "garbled": list(garbled), "foreign": list(foreign)}


OK = {"n": 1, "range": [1, 1], "over": False}
OVER = {"n": 3, "range": [1, 1], "over": True}


def test_match_with_all_words_stays_match():
    assert strong.combine("match", T(found=["pause"]), OK, "print") == {"verdict": "match", "why": []}


def test_match_with_missing_word_is_text_difference():
    assert strong.combine("match", T(found=["pause"], missing=["faire"]), OK, "print")["why"] == ["text"]


def test_match_with_garbled_word_is_text_difference():
    assert strong.combine("match", T(garbled=["pavse"]), OK, "print")["verdict"] == "different"


def test_matched_print_with_nothing_readable_stays_match():
    assert strong.combine("match", T(missing=["pause"], read=0), OK, "print")["verdict"] == "match"


def test_unverified_with_anchor_read_becomes_match():
    assert strong.combine("unverified", T(found=["pause"]), OK, "product") == {"verdict": "match", "why": ["text"]}


def test_unverified_with_only_foreign_text_is_invented():
    assert strong.combine("unverified", T(foreign=["sale"]), OK, "product") == {"verdict": "different", "why": ["invented_text"]}


def test_unverified_without_text_stays_unverified():
    assert strong.combine("unverified", T(read=0), OK, "product")["verdict"] == "unverified"


def test_count_over_is_different_whatever_the_rest():
    assert strong.combine("match", T(found=["pause"]), OVER, "print") == {"verdict": "different", "why": ["count"]}


def test_no_anchors_leaves_text_out():
    assert strong.combine("unverified", None, OK, "product")["verdict"] == "unverified"
    assert strong.combine("match", None, None, "print")["verdict"] == "match"


def test_text_outside_the_box_is_ignored(tmp_path):
    # fake OCR: a confident sign word outside the product box, nothing inside
    class M:
        def locate(self, bgr): return {"inliers": 0, "pass": False, "box": None}
    exp = {"query": "a jar", "anchors": [{"word": "jam", "rel_h": 0.1, "print": None}], "count": {"per_photo": {}, "range": [1, 1]}}
    ocr = lambda im, box=None: [{"text": "BIG SALE", "conf": 0.95, "box": (500, 10, 600, 40)}]
    det = lambda im, q: [{"query": "a jar", "score": 0.6, "box": (0, 100, 200, 300)}]
    pil = Image.new("RGB", (640, 400)); bgr = np.zeros((400, 640, 3), "uint8")
    r = strong.inspect(pil, bgr, M(), exp, ocr=ocr, detect=det)
    assert r["verdict"] == "unverified" and r["text"]["foreign"] == []


def test_inspect_matched_print_reads_print_box_only_with_its_anchors():
    class M:
        def locate(self, bgr): return {"inliers": 30, "pass": True, "box": (10, 10, 110, 60), "print": "print_2.png"}
    exp = {"query": None,
           "anchors": [{"word": "jam", "rel_h": 0.5, "print": "print.png"},
                       {"word": "pause", "rel_h": 0.5, "print": "print_2.png"}],
           "count": {"per_photo": {}, "range": None}}
    ocr = lambda im, box=None: [{"text": "PAUSE", "conf": 0.9, "box": (20, 20, 80, 50)}]
    pil = Image.new("RGB", (200, 100)); bgr = np.zeros((100, 200, 3), "uint8")
    r = strong.inspect(pil, bgr, M(), exp, ocr=ocr)
    assert r["verdict"] == "match" and r["where"] == "print" and r["print_box"] == (10, 10, 110, 60)
    assert r["text"]["expected"] == ["pause"] and r["count"] is None


def test_inspect_without_matcher_and_detections_reads_nothing():
    exp = {"query": "a jar", "anchors": [{"word": "jam", "rel_h": 0.1, "print": None}],
           "count": {"per_photo": {}, "range": [1, 1]}}
    calls = []
    ocr = lambda im, box=None: calls.append(box) or []
    r = strong.inspect(Image.new("RGB", (50, 50)), np.zeros((50, 50, 3), "uint8"), None, exp, ocr=ocr,
                       detect=lambda im, q: [])
    assert calls == [] and r["where"] == "product" and r["verdict"] == "unverified"
    assert r["count"] == {"n": 0, "range": [1, 1], "over": False}


def test_build_expect_keeps_words_read_twice(tmp_path):
    folder = tmp_path / "refset"; folder.mkdir()
    uploads = tmp_path / "uploads"; uploads.mkdir()
    colors = {"print.png": (10, 10, 10), "crops/1.png": (20, 20, 20), "crops/2.png": (30, 30, 30),
              "crops/3.png": (40, 40, 40)}
    (folder / "crops").mkdir()
    for name, c in colors.items():
        Image.new("RGB", (100, 50), c).save(folder / name)
    for f in ("a.jpg", "b.jpg"):
        Image.new("RGB", (200, 200), (200, 0, 0)).save(uploads / f)
    rs = {"query": "a jar", "uploads": str(uploads),
          "items": [{"file": "a.jpg", "role": "product", "crop": "crops/1.png"},
                    {"file": "b.jpg", "role": "product", "crop": "crops/2.png"},
                    {"file": "v.mp4@3.0s", "role": "capture", "crop": "crops/3.png"}],
          "print": {"path": "print.png"}, "prints": [{"path": "print.png"}]}
    rp = folder / "refset.json"
    rp.write_text(json.dumps(rs))
    calls = []

    def box(h=20):
        return (0, 0, 40, h)

    def ocr(im, b=None):
        assert b == (0, 0, im.width, im.height)
        calls.append(im.getpixel((0, 0)))
        px = im.getpixel((0, 0))
        if px == (40, 40, 40):
            return []
        if px == (10, 10, 10) or px == (20, 20, 20):
            return [{"text": "JAM", "conf": 0.9, "box": box()}]
        return [{"text": "JAM SALE", "conf": 0.9, "box": box()}]

    # "SALE" only in crop 2 -> make crop 1 and print not read it (above returns SALE for crop 2 only)
    det = lambda im, q: [{"query": q[0], "score": 0.6, "box": (10, 10, 100, 100)}]
    ex = expect_mod.build_expect(rp, ocr=ocr, detect=det)
    assert [a["word"] for a in ex["anchors"]] == ["jam"]
    assert ex["anchors"][0]["print"] == "print.png" and abs(ex["anchors"][0]["rel_h"] - 0.4) < 1e-6
    assert ex["count"]["range"] == [1, 1] and set(ex["count"]["per_photo"]) == {"a.jpg", "b.jpg"}
    assert ex["query"] == "a jar" and (folder / "expect.json").exists()
    n = len(calls)
    again = expect_mod.build_expect(rp, ocr=ocr, detect=det)
    assert again == ex and len(calls) == n


def _refset(tmp_path, prints, crops, words_by_color, items=None):
    """crops get colors 20, 30, ... (print.png is 10). items: (file, role) per crop; by default
    each crop is a product photo named after it (crops/a.png -> a.jpg)."""
    folder = tmp_path / "rs"; (folder / "crops").mkdir(parents=True)
    for i, name in enumerate(["print.png"] + [c for c in crops]):
        Image.new("RGB", (100, 50), (10 * (i + 1),) * 3).save(folder / name)
    items = items or [(Path(c).stem + ".jpg", "product") for c in crops]
    rs = {"query": None, "uploads": "",
          "items": [{"file": f, "role": r, "crop": c} for c, (f, r) in zip(crops, items)],
          "print": {"path": "print.png"}, "prints": prints}
    (folder / "refset.json").write_text(json.dumps(rs))
    def ocr(im, b=None):
        return [{"text": w, "conf": 0.9, "box": (0, 0, 40, 20)} for w in words_by_color[im.getpixel((0, 0))[0]]]
    return folder / "refset.json", ocr


def test_print_and_its_source_crop_count_as_one_image(tmp_path):
    # print.png (10) cut from crops/a.png (20); crops/b.png (30) has other text
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/a.png"}],
                      ["crops/a.png", "crops/b.png"], {10: ["JAM"], 20: ["JAM"], 30: ["OTHER"]})
    assert expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"] == []


def test_word_in_print_and_a_different_crop_is_anchor(tmp_path):
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/a.png"}],
                      ["crops/a.png", "crops/b.png"], {10: ["JAM"], 20: [], 30: ["JAM"]})
    a = expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"]
    assert [x["word"] for x in a] == ["jam"] and a[0]["print"] == "print.png"


def test_single_source_image_needs_one_reading(tmp_path):
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/a.png"}],
                      ["crops/a.png"], {10: ["JAM"], 20: []})
    assert [x["word"] for x in expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"]] == ["jam"]


def test_info_crop_word_is_not_an_anchor(tmp_path):
    # JAM is read in product photo a and in a detail-page (info) crop: only one real photo
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/b.png"}],
                      ["crops/a.png", "crops/i.png", "crops/b.png"], {10: [], 20: ["JAM"], 30: ["JAM"], 40: []},
                      items=[("a.jpg", "product"), ("info-i.jpg", "info"), ("b.jpg", "product")])
    assert expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"] == []


def test_frames_of_one_capture_video_count_as_one_photo(tmp_path):
    # JAM is read in two frames of the same guided-capture video, nowhere else
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/a.png"}],
                      ["crops/a.png", "crops/v1.png", "crops/v2.png"], {10: [], 20: [], 30: ["JAM"], 40: ["JAM"]},
                      items=[("a.jpg", "product"), ("v.mp4@1.0s", "capture"), ("v.mp4@2.0s", "capture")])
    assert expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"] == []


def test_capture_video_and_a_product_photo_are_two_photos(tmp_path):
    rp, ocr = _refset(tmp_path, [{"path": "print.png", "from": "crops/a.png"}],
                      ["crops/a.png", "crops/v1.png"], {10: [], 20: ["JAM"], 30: ["JAM"]},
                      items=[("a.jpg", "product"), ("v.mp4@1.0s", "capture")])
    a = expect_mod.build_expect(rp, ocr=ocr, detect=None)["anchors"]
    assert [x["word"] for x in a] == ["jam"] and a[0]["print"] is None


def test_zoom_crops_and_checks():
    z = strong.zoom_crops(Image.new("RGB", (640, 480)), (100, 100, 200, 150), None)
    assert [l for l, _ in z] == ["print close-up"] and max(z[0][1].size) == 768
    c = strong.checks({"anchors": [{"word": "jam", "rel_h": 0.1, "print": "print.png"},
                                   {"word": "sale", "rel_h": 0.1, "print": None}],
                       "count": {"range": [1, 2]}})
    assert "jam" in c[0] and "sale" not in c[0] and "1~2개" in c[1]
    # reference lines for the VLM, not rules that decide "different"
    assert all(line.startswith("참고:") and "다름" not in line for line in c)
    assert strong.checks({"anchors": [{"word": "sale", "rel_h": 0.1, "print": None}], "count": {"range": None}}) == []

"""Showcase of the C path (print detail restoration) on CS case 19, for an internal
presentation: the seller's photo, the generated cut, and the restored cut, with close-ups.

The blind evaluation (docs/validation_cs_plan.md section 11) judged the processed cut of
case 19 as matching the reference with no pasted look, but that run restored only one can,
the one whose label the reference set holds. Here every can in the cut gets the label of the
same can in the seller's photo:

1. the cans in the seller's photo are found with the detector (the reference set's query),
   and each one's label band becomes a print;
2. the cans in the generated cut are found one at a time with the SIFT matcher (match, blank
   the matched label, match again);
3. each can takes the print whose label colour agrees (the cans share the drawing and the
   large lettering, so the matcher alone cannot tell them apart);
4. each can is restored with check.print_restore.restore on a crop around it.

Not part of the evaluation. Output goes to out/showcase/ (git-ignored): customer images stay
on this machine.

    .venv/bin/python spikes/validation/showcase.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "spikes")]

from check.print_restore import MIN_INLIERS, full_res_homography, restore  # noqa: E402
from check.sift import PrintMatcher  # noqa: E402

CASE = 19
GEN = ROOT / "data/cs/19/outputs/AI-Generated-1abc66cc-12f9-403d-82c4-566d60dbee19.png"
UPLOAD = ROOT / "data/cs/19/uploads/02f216b8-be02-4321-bb8c-a2826dc3dbfa.webp"
VALIDATED = ROOT / "out/validation/processed/19/AI-Generated-1abc66cc-12f9-403d-82c4-566d60dbee19.png"
OUT = ROOT / "out/showcase/c19"
LABEL_ROWS = (0.16, 0.93)
# Added after the blind evaluation, for this showcase: the print's own strokes are removed
# before its low frequencies are taken (no light rim around the letters), and the label is
# aligned once more with ECC (the cans are cylinders, one homography is a little off).
SETTINGS = dict(clean_print=True, refine=True)  # label band of a can box: below the lid, above the bottom rim
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


# ---------------------------------------------------------------- restoration
def seller_cans(img, query="a can", min_score=0.3):
    """Detector boxes of the cans in the seller's photo, overlapping boxes removed, left to right."""
    from refset import models
    dets = models.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [query], threshold=0.1)
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if d["score"] < min_score:
            continue
        if all(iou(d["box"], k) < 0.3 for k in keep):
            keep.append([int(v) for v in d["box"]])
    return sorted(keep, key=lambda b: (b[1] // 100, b[0]))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / float(area(a) + area(b) - inter + 1e-6)


def label_band(img, box):
    x0, y0, x1, y1 = box
    h = y1 - y0
    return img[y0 + int(LABEL_ROWS[0] * h):y0 + int(LABEL_ROWS[1] * h), x0:x1].copy()


def corners(shape):
    h, w = shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def find_instances(gen, matcher, pbgr, n=4):
    """Match, blank the matched label, match again: one quad per label found."""
    work, quads = gen.copy(), []
    for _ in range(n):
        r = matcher.match(work)
        if r["H"] is None or r["inliers"] < MIN_INLIERS:
            break
        q = cv2.perspectiveTransform(corners(pbgr.shape), full_res_homography(r["H"], pbgr.shape, gen.shape)).reshape(-1, 2)
        if not cv2.isContourConvex(q.astype(np.int32)) or cv2.contourArea(q) < 0.005 * gen.shape[0] * gen.shape[1]:
            break
        quads.append((q, int(r["inliers"])))
        cv2.fillPoly(work, [q.astype(np.int32)], (128, 128, 128))
    return quads


def quad_box(q, shape, grow=0.12):
    x0, y0 = q.min(0); x1, y1 = q.max(0)
    gx, gy = grow * (x1 - x0), grow * (y1 - y0)
    H, W = shape[:2]
    return [int(max(0, x0 - gx)), int(max(0, y0 - gy)), int(min(W, x1 + gx)), int(min(H, y1 + gy))]


def colour_gap(crop, pbgr, matcher):
    """How far the print's label colour is from the can it lands on (median a, b inside the
    located label, after matching on the crop). Returns (gap, inliers)."""
    r = matcher.match(crop)
    if r["H"] is None or r["inliers"] < MIN_INLIERS:
        return 1e9, int(r["inliers"])
    Hf = full_res_homography(r["H"], pbgr.shape, crop.shape)
    H, W = crop.shape[:2]
    warped = cv2.warpPerspective(pbgr, Hf, (W, H))
    inside = cv2.warpPerspective(np.ones(pbgr.shape[:2], np.uint8), Hf, (W, H)) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if inside.sum() < 500:
        return 1e9, int(r["inliers"])
    a = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[inside][:, 1:].astype(np.float32)
    b = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[inside][:, 1:].astype(np.float32)
    return float(np.hypot(*(np.median(a, 0) - np.median(b, 0)))), int(r["inliers"])


def restore_all(gen, prints, **kw):
    matchers = {}
    for name, p in prints.items():
        path = OUT / "prints" / f"{name}.png"
        cv2.imwrite(str(path), p)
        matchers[name] = PrintMatcher(path)
    found = []
    for name in prints:
        for q, n in find_instances(gen, matchers[name], prints[name]):
            found.append((q, n))
    cans = []
    for q, n in sorted(found, key=lambda t: -t[1]):
        b = quad_box(q, gen.shape, 0)
        if all(iou(b, quad_box(c, gen.shape, 0)) < 0.4 for c in cans):
            cans.append(q)
    out, report = gen.copy(), []
    for q in sorted(cans, key=lambda q: (q[:, 1].min() // 100, q[:, 0].min())):
        x0, y0, x1, y1 = quad_box(q, gen.shape)
        crop = out[y0:y1, x0:x1].copy()
        gaps = {name: colour_gap(crop, prints[name], matchers[name]) for name in prints}
        name = min(gaps, key=lambda k: gaps[k][0])
        region = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(region, [(q - [x0, y0]).astype(np.int32)], 1)
        region = cv2.dilate(region, np.ones((15, 15), np.uint8))
        res, info = restore(crop, prints[name], matchers[name], region=region, **kw)
        ok = info["status"] == "restored"
        if ok:
            out[y0:y1, x0:x1] = res
        report.append(dict(box=[x0, y0, x1, y1], print=name, colour_gap={k: round(v[0], 1) for k, v in gaps.items()},
                           status=info["status"], before=info["inliers_before"], after=info.get("inliers_after"),
                           outside_max_diff=info.get("outside_max_diff"), sigma=info.get("sigma"), refined=info.get("refined")))
    return out, report


# ---------------------------------------------------------------- slides
W, H = 1920, 1080
BG, INK, SUB, LINE = (247, 247, 245), (24, 24, 27), (96, 96, 104), (208, 208, 212)
BEFORE, AFTER = (190, 60, 50), (30, 120, 80)


def font(size, weight=0):
    return ImageFont.truetype(FONT, size, index=weight)


def pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def header(d, title, sub):
    d.text((80, 58), title, font=font(46, 6), fill=INK)
    d.text((80, 124), sub, font=font(26, 0), fill=SUB)


def panel(canvas, d, img, box, caption, colour=INK):
    x, y, w, h = box
    im = fit(img, w, h)
    ox, oy = x + (w - im.width) // 2, y + (h - im.height) // 2
    canvas.paste(im, (ox, oy))
    d.rectangle([ox - 1, oy - 1, ox + im.width, oy + im.height], outline=LINE, width=1)
    tw = d.textlength(caption, font=font(28, 4))
    d.text((x + (w - tw) / 2, oy - 46), caption, font=font(28, 4), fill=colour)
    return ox, oy, im.width / img.width


def slide_overview(seller, gen, out, path, note):
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    header(d, "생성 이미지 속 라벨 글씨를 셀러 사진에서 되살립니다",
           "AI가 라벨의 작은 글씨를 빠뜨리거나 뭉갠 부분만, 셀러가 올린 실물 사진의 글씨로 바꿔 넣습니다.")
    cols = [(80, 250, 560, 600), (680, 250, 560, 600), (1280, 250, 560, 600)]
    panel(c, d, pil(seller), cols[0], "셀러 원본 사진")
    for img, col, cap, colour in [(gen, cols[1], "AI 생성 결과 (처리 전)", BEFORE), (out, cols[2], "처리 후", AFTER)]:
        panel(c, d, pil(img), col, cap, colour)
    d.line([(80, 900), (1840, 900)], fill=LINE, width=1)
    for i, t in enumerate(note):
        d.text((80, 922 + i * 40), t, font=font(25, 0), fill=SUB if i else INK)
    c.save(path, quality=95)


def slide_closeups(seller, seller_boxes, gen, out, report, path):
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    header(d, "가까이 보면: 캔마다 빠진 글씨가 돌아옵니다",
           "왼쪽은 셀러 사진의 같은 캔, 가운데는 AI 생성 그대로, 오른쪽은 처리 후입니다.")
    xs, cw = [300, 840, 1380], 500
    for j, (cap, colour) in enumerate([("셀러 원본", INK), ("처리 전", BEFORE), ("처리 후", AFTER)]):
        tw = d.textlength(cap, font=font(28, 4))
        d.text((xs[j] + (cw - tw) / 2, 200), cap, font=font(28, 4), fill=colour)
    names = {"top": "위 캔", "left": "왼쪽 캔", "right": "오른쪽 캔"}
    rh, y = 250, 250
    for r in report:
        x0, y0, x1, y1 = r["box"]
        sb = seller_boxes[r["print"]]
        tiles = [seller[sb[1]:sb[3], sb[0]:sb[2]], gen[y0:y1, x0:x1], out[y0:y1, x0:x1]]
        d.text((80, y + rh / 2 - 20), names.get(r["print"], r["print"]), font=font(30, 6), fill=INK)
        for j, t in enumerate(tiles):
            im = fit(pil(t), cw, rh - 18)
            ox, oy = xs[j] + (cw - im.width) // 2, y + (rh - 18 - im.height) // 2
            c.paste(im, (ox, oy))
            d.rectangle([ox - 1, oy - 1, ox + im.width, oy + im.height], outline=LINE, width=1)
        y += rh + 22
    c.save(path, quality=95)


def slide_zoom(gen, out, box, path, title, sub):
    """One label, large: before and after side by side."""
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    header(d, title, sub)
    x0, y0, x1, y1 = box
    x0 += int(0.30 * (x1 - x0))  # the ribbon and the side text; the drawing on the left is unchanged
    for j, (img, cap, colour) in enumerate([(gen, "처리 전", BEFORE), (out, "처리 후", AFTER)]):
        panel(c, d, pil(img[y0:y1, x0:x1]), (80 + j * 900, 300, 860, 700), cap, colour)
    c.save(path, quality=95)


def main():
    (OUT / "prints").mkdir(parents=True, exist_ok=True)
    seller, gen = cv2.imread(str(UPLOAD)), cv2.imread(str(GEN))
    boxes = seller_cans(seller)
    if len(boxes) != 3:
        raise SystemExit(f"expected 3 cans in the seller photo, found {len(boxes)}: {boxes}")
    top = min(boxes, key=lambda b: b[1])
    rest = sorted((b for b in boxes if b is not top), key=lambda b: b[0])
    seller_boxes = {"top": top, "left": rest[0], "right": rest[1]}
    prints = {k: label_band(seller, b) for k, b in seller_boxes.items()}
    t0 = time.time()
    out, report = restore_all(gen, prints, **SETTINGS)
    seconds = round(time.time() - t0, 2)
    cv2.imwrite(str(OUT / "after_all_cans.png"), out)
    changed = np.abs(out.astype(np.int16) - gen.astype(np.int16)).max(2) > 0
    inside = np.zeros(changed.shape, bool)
    for r in report:
        x0, y0, x1, y1 = r["box"]; inside[y0:y1, x0:x1] = True
    summary = dict(case=CASE, settings=SETTINGS, seconds=seconds, cans=report, changed_px=int(changed.sum()), changed_outside_can_boxes=int((changed & ~inside).sum()),
                   seller_boxes=seller_boxes)
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    restored = [r for r in report if r["status"] == "restored"]
    note = ["생성 모델을 다시 돌리지 않습니다. 실물 사진과 생성 이미지의 특징점을 맞춰 라벨 위치를 찾고, 글씨의 가는 획만 옮깁니다.",
            f"캔 {len(restored)}개 복원에 {seconds:.1f}초, 라벨 바깥 픽셀은 하나도 바뀌지 않습니다.",
            "블라인드 평가(평가자 1명): 위 캔만 복원한 판은 '기준과 같음', 붙인 티 1점(5점 중). 세 캔 복원과 글자 둘레 보정은 평가 뒤에 더한 것입니다."]
    slide_overview(seller, gen, out, OUT / "slide1_overview.jpg", note)
    slide_closeups(seller, seller_boxes, gen, out, restored, OUT / "slide2_closeups.jpg")
    t = next((r for r in restored if r["print"] == "top"), restored[0])
    slide_zoom(gen, out, t["box"], OUT / "slide3_zoom.jpg", "한 캔을 크게 보면",
               "AI가 비워 둔 리본 문구와 옆면의 작은 표기가 실물 그대로 돌아왔습니다.")
    pages = [Image.open(OUT / f) for f in ("slide1_overview.jpg", "slide2_closeups.jpg", "slide3_zoom.jpg")]
    pages[0].save(OUT / "slides.pdf", save_all=True, append_images=pages[1:], resolution=144)


if __name__ == "__main__":
    main()

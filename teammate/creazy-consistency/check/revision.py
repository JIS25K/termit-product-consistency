"""Revision check: compare an image before and after a fix (docs/revision.md, step 6).

A fix in Creazy is a redraw too: the area edit regenerates the painted region, the chat edit
regenerates the whole image. Two things are checked against the image before the fix:

  outside  everything outside the edited region must stay as it was (CS 38: the orange grid
           vanished while removing wrinkles; CS 36: the image went away when only the text was
           to change; CS 34: a logo appeared and the length changed).
  inside   the edited region must actually change (CS 21, 32: the same picture came back).

With --refset, the real print is matched before and after as well, so a fix that loses the
real print (or one that brings it in with a real crop attached) is reported.

How
  1. Align: SIFT matches between the two images, a similarity transform (scale, rotation,
     shift) by RANSAC, the after image warped into the before frame. Creazy may return a
     fix at another size or a few pixels off; without this every edge would count as changed.
  2. Difference, three kinds, each averaged over a small window so resampling and JPEG noise
     on single edge pixels do not count:
       content  CIE76 colour difference (Lab) above DE: something was drawn or removed.
       detail   structural difference (1 - SSIM on lightness) above ST while the colour stays
                close: letters, lines or a pattern got blurred, redrawn or moved.
       tone     a wide area (TONE_AREA or more) whose colour moved by DE_TONE or more: the
                product or background got darker or changed hue.
  3. Regions: the edited region is widened by a margin (the edit blends into its border).
     Changed areas outside it are listed with their boxes in before-image pixels. When most
     of the image changed, the fix redrew the whole picture (verdict "redrawn").

Calibrated on synthetic edits of a Run 10b mug image (tests/test_revision.py): JPEG,
resampling, a 0.8x resize with a shift and light noise count as unchanged; a removed word,
a filled hole, a print blurred by 2 px and a mug darkened by about 6 dE count as changed.

Checked on three real Creazy area edits (data/mug/outputs/run10b_edit, 15 credits): the
"특정 부분만 생성하기" edit returns the whole picture redrawn at 1.31 to 1.41x. Outside the
brush, background texture (wood grain, flour, fabric weave, book spines) is redrawn in every
edit, the coffee surface inside the mug changed once, and the real print score fell by 23 to
48 percent (resampling alone: 2 to 18). That is why changes are graded against the product
box: background texture alone is "minor" and does not fail the fix.

Usage
  python -m check.revision --before b.png --after a.png --box 120,80,360,300 --out out/rev/x
  python -m check.revision --before b.png --after a.png --mask brush.png --refset out/mug/refset.json --out out/rev/x
  python -m check.revision --before b.png --after a.png --out out/rev/x     # no region: list every change

Without a region the report lists where the image changed; a person (or the seller's request)
says which of those were asked for.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .sift import PASS

WORK = 1024  # long side the comparison runs at, so the thresholds mean the same at any size
BLUR = 1.0  # px at WORK, before the colour difference
WIN = 7  # px at WORK, window the difference is averaged over
DE = 8.0  # mean CIE76 difference over the window that counts as changed (2.3 is just noticeable)
ST = 0.08  # mean 1 - SSIM over the window that counts as a changed detail (noise stays under 0.04)
DE_TONE = 4.0  # colour shift that counts over a wide area
TONE_AREA = 0.02  # share of the image a tone change must cover
REDRAWN = 0.5  # more than this share of the outside changed: the whole picture was redrawn
MIN_AREA = 0.0004  # changed area smaller than this share of the image is noise (about 20x20 px at 1024)
MARGIN = 0.02  # the edited region is widened by this share of the long side
INSIDE_MIN = 0.05  # less than this share of the region changed: the fix did not land
MIN_INLIERS = 25  # matches needed to trust the alignment
ON_PRODUCT = 0.3  # share of a changed area inside the product box that puts it on the product
PRINT_DROP = 0.2  # print score falling by this share while still matching: print details redrawn
# (resampling alone costs 2 to 18 percent; real redraws fell 23 to 48, the chat pen edit 26)


def _read(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def _work(img):
    s = WORK / max(img.shape[:2])
    return (cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()), min(1.0, s)


def align(before, after):
    """Warp `after` into the frame of `before` (both already at working size).
    Returns warped after, valid mask, info."""
    h, w = before.shape[:2]
    clahe = cv2.createCLAHE(2.0, (8, 8))
    sift = cv2.SIFT_create(nfeatures=4000)
    kb, db = sift.detectAndCompute(clahe.apply(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)), None)
    ka, da = sift.detectAndCompute(clahe.apply(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)), None)
    info = {"method": None, "inliers": 0}
    M = None
    if db is not None and da is not None and len(kb) >= 2 and len(ka) >= 2:
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
        if len(good) >= MIN_INLIERS:
            src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
            dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            n = int(inl.sum()) if inl is not None else 0
            info["inliers"] = n
            # A fix comes back at about the same framing; a collapsed or wild scale means the
            # matches sit on one small patch (the print alone) of two different pictures.
            if M is None or n < MIN_INLIERS or not 0.5 <= float(np.hypot(M[0, 0], M[1, 0])) <= 2.0:
                M = None
    if M is not None:
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        info.update(method="similarity", scale=round(scale, 4), shift=[round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)])
        same = after.shape[:2] == before.shape[:2] and abs(scale - 1) < 0.002 and abs(M[0, 2]) < 0.5 and abs(M[1, 2]) < 0.5
        # after -> before at working size, kept so check/restore.py can redo the warp at full size
        info["matrix"] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] if same else [[round(float(v), 6) for v in r] for r in M]
        if same:
            info["method"] = "identity"
            return after.copy(), np.ones((h, w), bool), info
        warped = cv2.warpAffine(after, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpAffine(np.full(after.shape[:2], 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
        # The blur and the averaging window reach about 8 px, so pixels that close to the
        # edge of the warped image see its black border and are left out.
        return warped, cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0, info
    # Too few matches: the picture was redrawn or recomposed. Only a same-shaped image can
    # still be compared, by stretching it to the same size.
    ra, rb = after.shape[1] / after.shape[0], w / h
    if abs(ra - rb) / rb < 0.02:
        info["method"] = "resize"
        info["matrix"] = [[round(w / after.shape[1], 6), 0.0, 0.0], [0.0, round(h / after.shape[0], 6), 0.0]]
        return cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA), np.ones((h, w), bool), info
    info["method"] = "none"
    return None, None, info


def difference(before, after):
    """Mean CIE76 colour difference over a WIN x WIN window, float32 map."""
    def lab(x):
        x = cv2.GaussianBlur(x, (0, 0), BLUR).astype(np.float32) / 255.0
        return cv2.cvtColor(x, cv2.COLOR_BGR2Lab)
    d = np.linalg.norm(lab(before) - lab(after), axis=2)
    return cv2.blur(d, (WIN, WIN))


def structure(before, after):
    """Mean 1 - SSIM of the lightness over a WIN x WIN window, float32 map."""
    def light(x):
        x = cv2.GaussianBlur(x, (0, 0), BLUR).astype(np.float32) / 255.0
        return cv2.cvtColor(x, cv2.COLOR_BGR2Lab)[..., 0] / 100.0
    x, y = light(before), light(after)
    g = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)
    mx, my = g(x), g(y)
    vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return cv2.blur(1.0 - ssim, (WIN, WIN))


def region_mask(shape, boxes=(), mask=None, scale=1.0):
    """Edited region at working size from boxes (before-image pixels) or a mask image."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0 * scale)) : int(np.ceil(y1 * scale)), max(0, int(x0 * scale)) : int(np.ceil(x1 * scale))] = 255
    if mask is not None:
        g = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        m |= (cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
    return m > 0


KIND_KO = {"content": "그려진 것이 바뀜", "detail": "글자, 선, 무늬 같은 세부가 달라짐", "tone": "색이 바뀜"}


def changed_maps(d, st):
    """Pixel maps of the three kinds of change (see the module notes)."""
    content = cv2.morphologyEx(((d > DE) | (st > ST)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    tone = cv2.morphologyEx(((d > DE_TONE) & ~content).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return content, tone


def changed_areas(content, tone, d, st, zone, scale, product=None):
    """Connected changed areas inside `zone` with boxes in before-image pixels, largest first.
    product: mask of the product at working size; each area then says whether it sits on it."""
    h, w = d.shape
    total = h * w
    out = []
    for mask, min_share, tone_kind in ((content & zone, MIN_AREA, False), (tone & zone, TONE_AREA, True)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < min_share * total:
                continue
            px = lab == i
            mean_de = float(d[px].mean())
            kind = "tone" if tone_kind else ("content" if mean_de >= DE else "detail")
            out.append({
                "kind": kind,
                "kind_ko": KIND_KO[kind],
                "box": [int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)],
                "area": round(area / total, 4),
                "mean_de": round(mean_de, 1),
                "mean_st": round(float(st[px].mean()), 3),
                "on_product": None if product is None else bool(product[px].mean() >= ON_PRODUCT),
            })
    out.sort(key=lambda r: -r["area"])
    return out


def print_check(refset_path, before, after):
    """Real print matched before and after the fix (check/sift.py, same threshold)."""
    from .run import make_matcher  # local import: check.run pulls in the VLM client

    refset_path = Path(refset_path)
    ref = json.loads(refset_path.read_text())
    matcher = make_matcher(ref, refset_path.parent)
    if matcher is None:
        return None
    b, a = matcher.score(before), matcher.score(after)
    if b["pass"] and not a["pass"]:
        verdict = "lost"
    elif a["pass"] and not b["pass"]:
        verdict = "gained"
    elif a["pass"]:
        verdict = "kept"
    else:
        verdict = "absent"
    return {"before": b["inliers"], "after": a["inliers"], "threshold": PASS, "verdict": verdict}


def compare(before, after, boxes=(), mask=None, product_boxes=None):
    """before, after: BGR images. boxes: edited region boxes in before-image pixels. mask: an
    image whose white part is the edited region (a brush mask). product_boxes: boxes of the
    product in the before image (run() finds them with OWLv2); with them, changes are graded:
      major  on the product, or something drawn or removed in the background
      minor  background texture or colour only (the area edit redraws the whole picture, see
             docs/revision.md), reported but not failing the fix
    Without product_boxes every change outside the region counts as major.
    Returns the report without file names, plus private arrays for drawing ("_vis")."""
    b, scale = _work(before)
    a, _ = _work(after)
    warped, valid, info = align(b, a)
    ratio = after.shape[1] / before.shape[1]
    info["size_ratio"] = round(ratio, 3)
    h, w = b.shape[:2]
    rep = {
        "size": {"before": [before.shape[1], before.shape[0]], "after": [after.shape[1], after.shape[0]]},
        "align": info,
        "params": {"work": WORK, "de": DE, "window": WIN, "min_area": MIN_AREA, "margin": MARGIN},
    }
    if warped is None:
        rep.update(
            verdict="redrawn",
            outside={"changed": None, "areas": [], "verdict": "redrawn"},
            notes_ko=["수정 전후 이미지의 구도가 달라 겹쳐 비교할 수 없어요. 고친 부분만이 아니라 이미지 전체가 다시 그려졌어요."],
        )
        return rep
    d = difference(b, warped)
    st = structure(b, warped)
    d[~valid] = 0
    st[~valid] = 0
    content, tone = changed_maps(d, st)
    changed = content | tone
    given = bool(len(boxes)) or mask is not None
    region = region_mask((h, w), boxes, mask, scale) if given else np.zeros((h, w), bool)
    pad = max(6, int(round(MARGIN * max(h, w))))
    near = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    keep_zone = valid & ~near
    rep["shift_de"] = round(float(np.median(d[keep_zone])) if keep_zone.any() else 0.0, 2)
    out_frac = float((changed & keep_zone).sum() / max(1, keep_zone.sum()))
    redrawn = out_frac > REDRAWN
    product = None
    if product_boxes is not None:
        product = region_mask((h, w), product_boxes, None, scale)
    out_areas = [] if redrawn else changed_areas(content, tone, d, st, keep_zone, scale, product)
    if product is None:
        major, minor = out_areas, []
    else:
        major = [a for a in out_areas if a["on_product"] or a["kind"] == "content"]
        minor = [a for a in out_areas if a not in major]
        zone = product & keep_zone
        rep["product"] = {
            "boxes": [list(map(int, x)) for x in product_boxes],
            "changed": round(float((changed & zone).sum() / max(1, zone.sum())), 4) if zone.any() else None,
            "areas": sum(bool(a["on_product"]) for a in out_areas),
        }
    level = "redrawn" if redrawn else ("major" if major else ("minor" if minor else "same"))
    rep["outside"] = {
        "changed": round(out_frac, 4),
        "level": level,
        "areas": out_areas,
        "verdict": "redrawn" if redrawn else ("changed" if out_areas else "kept"),
    }
    notes = []
    where = "고친 영역 밖" if given else "이미지"
    if redrawn:
        notes.append(f"{where}의 {out_frac:.0%}가 바뀌었어요. 고친 부분만이 아니라 이미지 대부분이 다시 그려졌어요.")
    if major:
        on = sum(bool(a["on_product"]) for a in major)
        kinds = [f"{KIND_KO[k]} {n}곳" for k in KIND_KO for n in [sum(a['kind'] == k for a in major)] if n]
        if product is not None:
            kinds.append(f"제품 위 {on}곳" if on else "제품 위는 아님")
        notes.append(f"{where}에서 {len(major)}곳이 바뀌었어요({', '.join(kinds)}). 빨간 상자를 수정 전과 비교해 주세요.")
    if minor:
        notes.append(f"배경의 질감이나 색이 미세하게 다시 그려진 곳이 {len(minor)}곳 있어요(노란 상자, 제품 위는 아님).")
    if not out_areas and not redrawn and given:
        notes.append("고친 영역 밖은 수정 전과 같아요.")
    if abs(ratio - 1) > 0.02:
        notes.append(f"수정본 크기({after.shape[1]}x{after.shape[0]})가 수정 전의 {ratio:.2f}배예요. 고친 영역만이 아니라 이미지 전체를 새로 만든 흔적이에요.")
    if given:
        inside = region & valid
        in_frac = float((changed & inside).sum() / max(1, inside.sum()))
        rep["region"] = {"boxes": [list(map(int, x)) for x in boxes], "mask": mask is not None, "area": round(float(region.mean()), 4), "margin_px": int(round(pad / scale))}
        rep["inside"] = {"changed": round(in_frac, 4), "verdict": "changed" if in_frac >= INSIDE_MIN else "unchanged"}
        if rep["inside"]["verdict"] == "unchanged":
            notes.append(f"고친 영역 안이 거의 그대로예요(바뀐 부분 {in_frac:.0%}). 요청이 반영되지 않았을 수 있어요.")
        verdict = "redrawn" if redrawn else ("pass" if not major and rep["inside"]["verdict"] == "changed" else "review")
    else:
        if not out_areas and not redrawn:
            notes.append("수정 전과 달라진 곳을 찾지 못했어요. 요청이 반영되지 않았을 수 있어요.")
        verdict = "redrawn" if redrawn else ("changes" if out_areas else "unchanged")
    if info["method"] == "resize" and not redrawn:
        notes.append("특징점이 적어 크기만 맞춰 비교했어요. 결과가 거칠 수 있어요.")
    rep.update(verdict=verdict, notes_ko=notes)
    rep["_vis"] = {"before": b, "after": warped, "region": region, "changed": changed, "areas": out_areas, "scale": scale, "product": product}
    return rep


def draw(vis, path):
    """before | after | after with every changed pixel tinted. The edited region is outlined
    in blue, the product box in green, changed areas outside the region boxed in red (on the
    product, or something drawn or removed) or yellow (background texture or colour only)."""
    b, a = vis["before"].copy(), vis["after"].copy()
    s = vis["scale"]
    for img in (b, a):
        for m, color in ((vis.get("product"), (60, 160, 40)), (vis["region"], (200, 120, 30))):
            if m is not None and m.any():
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, color, 2)
        for r in vis["areas"]:
            x0, y0, x1, y1 = (int(round(v * s)) for v in r["box"])
            minor = r["on_product"] is False and r["kind"] != "content"
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (30, 200, 230) if minor else (40, 40, 220), 2)
    tint = a.copy()
    tint[vis["changed"]] = (0.5 * tint[vis["changed"]] + 0.5 * np.array([40, 40, 220])).astype(np.uint8)
    gap = np.full((b.shape[0], 16, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([b, gap, a, gap, tint]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def parse_box(s):
    v = [int(float(x)) for x in s.split(",")]
    if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
        raise argparse.ArgumentTypeError(f"box must be x0,y0,x1,y1 with x1>x0, y1>y0: {s}")
    return v


def find_product(refset_path, before):
    """Product box in the before image: the best OWLv2 detection of the reference set's query,
    with the score floor of refset.build. Only the best one: "a mug" also finds the brush jar
    and the coffee server around the mug in the Run 10b cuts. None when the set has no query."""
    ref = json.loads(Path(refset_path).read_text())
    query = ref.get("query")
    if not query:
        return None
    from PIL import Image

    from refset import models

    img = Image.fromarray(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    dets = [x for x in models.detect(img, [query], threshold=0.1) if x["score"] >= 0.3]
    # Not found is "unknown", not "nowhere": without a box every change counts as major.
    return [list(x["box"]) for x in dets[:1]] or None


def run(before_path, after_path, out_dir, boxes=(), mask_path=None, refset=None, draw_vis=True, product_boxes=None):
    """refset: also find the product (OWLv2, unless product_boxes are given) and match the
    real print before and after."""
    before, after = _read(before_path), _read(after_path)
    mask = _read(mask_path) if mask_path else None
    if product_boxes is None and refset:
        product_boxes = find_product(refset, before)
    rep = compare(before, after, boxes, mask, product_boxes)
    vis = rep.pop("_vis", None)
    rep = {"before": str(before_path), "after": str(after_path), **rep}
    if refset:
        p = print_check(refset, before, after)
        if p:
            rep["print"] = p
            if p["verdict"] == "lost":
                rep["notes_ko"].append(f"수정 전에는 실물 프린트가 맞았는데(점수 {p['before']}) 수정 후에는 사라졌어요(점수 {p['after']}).")
                rep["verdict"] = {"pass": "review", "unchanged": "changes"}.get(rep["verdict"], rep["verdict"])
            elif p["verdict"] == "gained":
                rep["notes_ko"].append(f"수정 후 실물 프린트가 맞아요(점수 {p['before']} → {p['after']}).")
            elif p["verdict"] == "kept" and p["after"] <= (1 - PRINT_DROP) * p["before"]:
                # Resampling alone costs 2 to 18 percent on the Run 10b edits, so only a bigger
                # fall is reported (the chat pen edit fell 26 and was missed at 35). It is a
                # note to look closer, not a failure.
                p["redrawn"] = True
                rep["notes_ko"].append(f"실물 프린트는 맞지만 점수가 떨어졌어요(점수 {p['before']} → {p['after']}). 프린트의 획이나 글자가 미세하게 다시 그려졌을 수 있어요.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if vis is not None and draw_vis:
        draw(vis, out_dir / "revision.jpg")
        rep["vis"] = "revision.jpg"
    (out_dir / "revision.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser(description="Compare an image before and after a fix.")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--box", type=parse_box, action="append", default=[], help="edited region x0,y0,x1,y1 in before-image pixels (repeatable)")
    ap.add_argument("--mask", default=None, help="image whose white part is the edited region")
    ap.add_argument("--refset", default=None, help="refset.json: also match the real print before and after")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.before, a.after, a.out, a.box, a.mask, a.refset)
    print(json.dumps({k: r[k] for k in r if k in ("verdict", "align", "inside", "outside", "print", "shift_de", "notes_ko")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

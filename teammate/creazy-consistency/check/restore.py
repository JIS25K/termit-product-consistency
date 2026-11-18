"""Put the original back outside the edited region (docs/revision.md, section 4 step 5).

Creazy's area edit ("특정 부분만 생성하기") returns the whole picture redrawn at 1.3 to 1.4x:
background texture and the fine detail of the real print change outside the brush every time
(docs/revision.md 4.5). This takes from the edit only what was asked for and keeps every other
pixel of the image before the edit.

  1. Align the edit to the image before it (the same SIFT similarity as check/revision.py),
     at full resolution, and bring it to the original size.
  2. Take from the edit: the painted region, widened by the blending margin, plus changed
     background areas right next to it that are not on the product (a removed object's
     shadow, a plate rim). Everything else comes from the original.
  3. Match the colour of the taken part to the original along a ring around it (the edit
     shifts the overall tone by 1.5 to 2.5 dE), and feather the seam.
  4. Check the result with check/revision.py against the taken region: outside it the image
     must equal the original, and the real print score must come back.

A chat edit or a quick edit comes with no painted region; find_region (--auto) takes what was
drawn or removed in it, leaves out changes on the product (or outside the box the seller
pointed at, --where), and the same steps follow.

Usage
  python -m check.restore --before b.png --after a.png --box 0,178,302,440 --refset out/mug/refset.json --out out/restore/x
  python -m check.restore --before b.png --after a.png --auto --refset out/mug/refset.json --out out/restore/y
  python -m check.restore --edits data/mug/outputs/run10b_edit/edits.json --refset out/mug/refset.json --out out/restore
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import revision
from .revision import DE, MARGIN, MIN_AREA, ON_PRODUCT, REDRAWN, _read, _work, align, changed_maps, difference, region_mask, structure

NEAR = 0.05  # changed background areas within this share of the long side belong to the edit
NEAR_MAX = 0.02  # ... unless they are bigger than this share of the image
FEATHER = 0.006  # seam blur, share of the long side (about 8 px at 1280)


def full_matrix(info, sa, sb):
    """The after -> before transform found at working size, for full-size images."""
    m = np.array(info["matrix"], np.float64)
    out = m.copy()
    out[:, :2] *= sa / sb
    out[:, 2] /= sb
    return out


def warp_to(before, after, m):
    """after warped into the frame of before. A downscale is done with INTER_AREA first so
    the redrawn texture does not alias."""
    H, W = before.shape[:2]
    s = float(np.hypot(m[0, 0], m[1, 0]))
    if s < 0.98:
        after = cv2.resize(after, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = m.copy()
        m[:, :2] /= s
    return cv2.warpAffine(after, m.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def nearby_changes(bw, aw, region_w, product_w):
    """Changed background areas next to the region (working size), as a mask and boxes."""
    h, w = region_w.shape
    d, st = difference(bw, aw), structure(bw, aw)
    content, _ = changed_maps(d, st)
    pad = max(6, int(round(MARGIN * max(h, w))))
    near_px = int(round(NEAR * max(h, w)))
    disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    margin = cv2.dilate(region_w.astype(np.uint8), disk(pad)) > 0
    close = cv2.dilate(region_w.astype(np.uint8), disk(near_px)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats((content & ~margin).astype(np.uint8), 8)
    keep = np.zeros((h, w), bool)
    boxes = []
    for i in range(1, n):
        x, y, bw_, bh, area = (int(v) for v in stats[i])
        if area < MIN_AREA * h * w or area > NEAR_MAX * h * w:
            continue
        px = lab == i
        if close[px].mean() < 0.5:
            continue
        if product_w is not None and product_w[px].mean() >= ON_PRODUCT:
            continue  # a change on the product is never taken from the edit
        keep |= px
        boxes.append([x, y, x + bw_, y + bh])
    return keep, boxes


def _fill_holes(m):
    """Holes inside a mask filled (flood fill of the background from the border)."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (pad | (flood == 0))[1:-1, 1:-1] > 0


def _share(px, zone):
    """How much of the pixels `px` lie in `zone`, or of `zone` lies in `px`, whichever is more."""
    both = float((px & zone).sum())
    return max(both / max(1.0, float(px.sum())), both / max(1.0, float(zone.sum())))


def find_region(before, after, product_boxes=None, where=None):
    """The edited region of an edit that came without one. A chat edit or a quick edit
    ("빠른 편집") redraws the whole picture and does not say where the asked change is.
    Taken: the content changes of the edit against the original, that is something drawn or
    removed (mean dE at least revision.DE). Texture, fine detail and tone changes are the
    redraw itself and are left out.
      where  boxes the seller pointed at (before-image pixels). Only changes on them are
             taken, so a change somewhere else (a new logo, a lost pattern) is covered.
      else   changes on the product are left out: a request about the scene must not change
             the product (CS 38, 34). With no product box every content change is taken.
    The picked changes are closed and hole-filled so no piece of a removed object stays from
    the original. Returns (mask at the size of before, info); the mask is None when the
    composition changed or most of the picture changed (revision.REDRAWN), and there is no
    original to keep around the change."""
    bw, sb = _work(before)
    aw, _ = _work(after)
    warped, valid, info = align(bw, aw)
    if warped is None:
        return None, {"align": info, "boxes": []}
    d, st = difference(bw, warped), structure(bw, warped)
    d[~valid] = 0
    st[~valid] = 0
    content, tone = changed_maps(d, st)
    if float((content | tone)[valid].mean()) > REDRAWN:
        return None, {"align": info, "boxes": []}  # most of the picture changed: recomposed
    h, w = d.shape
    product = region_mask((h, w), product_boxes, None, sb) if product_boxes else None
    target = region_mask((h, w), where, None, sb) if where else None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(content.astype(np.uint8), 8)
    keep = np.zeros((h, w), bool)
    boxes, skipped = [], []
    for i in range(1, n):
        x, y, bw_, bh, area = (int(v) for v in stats[i])
        if area < MIN_AREA * h * w:
            continue
        px = lab == i
        if float(d[px].mean()) < DE:
            continue  # fine detail redrawn, not an object drawn or removed
        box = [int(x / sb), int(y / sb), int((x + bw_) / sb), int((y + bh) / sb)]
        if target is not None:
            if _share(px, target) < 0.3:
                skipped.append(box)
                continue
        elif product is not None and product[px].mean() >= ON_PRODUCT:
            skipped.append(box)
            continue
        keep |= px
        boxes.append(box)
    pad = max(6, int(round(MARGIN * max(h, w))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    keep = _fill_holes(cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, disk) > 0)
    H, W = before.shape[:2]
    full = cv2.resize(keep.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    return full, {"align": info, "boxes": boxes, "left_out": skipped, "area": round(float(keep.mean()), 4)}


def restore(before, after, boxes=(), mask=None, product_boxes=None):
    """before, after: BGR images. boxes / mask: the painted region in before-image pixels.
    Returns (restored image at the size of before, taken mask at full size, info)."""
    bw, sb = _work(before)
    aw, sa = _work(after)
    warped_w, valid_w, info = align(bw, aw)
    if warped_w is None:
        raise ValueError("the edit changed the composition; there is no original to keep around it")
    H, W = before.shape[:2]
    m = full_matrix(info, sa, sb)
    warped = warp_to(before, after, m)

    h, w = bw.shape[:2]
    region_w = region_mask((h, w), boxes, mask, sb)
    product_w = region_mask((h, w), product_boxes, None, sb) if product_boxes else None
    extra_w, extra_boxes = nearby_changes(bw, warped_w, region_w, product_w)

    region = region_mask((H, W), boxes, mask, 1.0)
    extra = cv2.resize(extra_w.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    pad = max(6, int(round(MARGIN * max(H, W))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    src = (region | extra).astype(np.uint8)
    taken = cv2.dilate(src, disk) > 0
    if product_boxes:
        # Inside the product box the blending margin shrinks to a few pixels, so the edit's
        # redrawn product never comes in beyond the brush. Not a hard cut at the box edge:
        # the box is a rectangle around the product and a cut there leaves a visible seam
        # (the removed pot's shadow in edit 3 runs into the mug's box).
        small = max(3, pad // 4)
        near = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * small + 1, 2 * small + 1))) > 0
        box = region_mask((H, W), product_boxes, None, 1.0)
        taken = np.where(box, near, taken)
    sigma = max(1.0, FEATHER * max(H, W))
    alpha = cv2.GaussianBlur(taken.astype(np.float32), (0, 0), sigma)
    alpha[region] = 1.0
    alpha[alpha < 1 / 255] = 0.0  # far from the seam the original is kept to the pixel

    # Colour match on a ring just outside the taken part, where both images show the same scene.
    ring = (cv2.dilate(taken.astype(np.uint8), disk) > 0) & (alpha == 0)
    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_a = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab).astype(np.float32)
    shift = (lab_b[ring].mean(0) - lab_a[ring].mean(0)) if ring.sum() > 100 else np.zeros(3, np.float32)
    lab_a += shift
    matched = cv2.cvtColor(np.clip(lab_a, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

    a3 = alpha[..., None]
    out = (a3 * matched.astype(np.float32) + (1 - a3) * before.astype(np.float32)).round().astype(np.uint8)
    out[alpha == 0] = before[alpha == 0]
    taken_full = alpha > 0
    info = {
        "align": info,
        "size": {"before": [W, H], "after": [after.shape[1], after.shape[0]]},
        "taken": round(float(taken_full.mean()), 4),
        "region": round(float(region.mean()), 4),
        "nearby": [[int(v / sb) for v in b] for b in extra_boxes],
        "color_shift_lab": [round(float(v), 2) for v in shift],
    }
    return out, taken_full, info


def run(before_path, after_path, out_dir, boxes=(), mask_path=None, refset=None, product_boxes=None, auto=False, where=None):
    """auto: the edit came without a painted region (chat edit, quick edit); find it with
    find_region, narrowed to `where` (boxes the seller pointed at) when given."""
    before, after = _read(before_path), _read(after_path)
    if product_boxes is None and refset:
        product_boxes = revision.find_product(refset, before)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = None
    if auto:
        region, found = find_region(before, after, product_boxes, where)
        if region is None:
            as_is = revision.run(before_path, after_path, out_dir / "as_is", (), None, refset, product_boxes=product_boxes)
            rep = {
                "before": str(before_path), "after": str(after_path), "auto": found,
                "as_is": {"verdict": as_is["verdict"], "print": as_is.get("print")}, "restored": None,
                "notes_ko": ["수정 전후의 구도가 달라 원본으로 덮을 수 없어요. 영역 편집으로 다시 하거나 실물 사진으로 교체해야 해요."],
            }
            (out_dir / "restore.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
            return rep
        mask_path = out_dir / "auto_region.png"
        cv2.imwrite(str(mask_path), region)
        boxes = ()
    mask = _read(mask_path) if mask_path else None
    restored, taken, info = restore(before, after, boxes, mask, product_boxes)
    if found:
        info["auto"] = found
    cv2.imwrite(str(out_dir / "restored.png"), restored)
    cv2.imwrite(str(out_dir / "taken.png"), taken.astype(np.uint8) * 255)
    # The edit as it came back, and the restored image, both checked against the original.
    as_is = revision.run(before_path, after_path, out_dir / "as_is", boxes, mask_path, refset, product_boxes=product_boxes)
    check = revision.run(before_path, out_dir / "restored.png", out_dir / "check", (), out_dir / "taken.png", refset, product_boxes=product_boxes)
    if found:
        notes = [f"칠한 영역이 없어 수정 전후에서 무언가 그려지거나 지워진 곳 {len(found['boxes'])}곳을 찾아 그 부분과 경계({info['taken']:.1%})만 가져오고, 나머지는 원본 픽셀 그대로예요."]
        if found["left_out"]:
            why = "가리킨 곳 밖" if where else "제품 위"
            notes.append(f"{why}에서 바뀐 곳 {len(found['left_out'])}곳은 요청과 상관없는 변화로 보고 원본으로 덮었어요.")
        if not found["boxes"]:
            notes.append("그려지거나 지워진 곳을 찾지 못했어요. 요청이 반영되지 않았을 수 있어요.")
    else:
        notes = [f"수정본에서 칠한 곳과 그 경계({info['taken']:.1%})만 가져오고 나머지는 원본 픽셀 그대로예요."]
    if info["nearby"]:
        notes.append(f"칠한 곳 바로 옆에서 함께 바뀐 곳 {len(info['nearby'])}곳(지운 물건의 그림자 등)도 수정본을 따랐어요. 제품 위는 따르지 않아요.")
    if as_is.get("print") and check.get("print"):
        notes.append(f"실물 프린트 점수: 원본 {as_is['print']['before']}, 크리에이지 수정본 {as_is['print']['after']}, 원본 덮기 {check['print']['after']}.")
    rep = {
        "before": str(before_path), "after": str(after_path), **info,
        "as_is": {"verdict": as_is["verdict"], "level": as_is["outside"].get("level"), "areas": len(as_is["outside"]["areas"]), "print": as_is.get("print")},
        "restored": {"verdict": check["verdict"], "level": check["outside"].get("level"), "areas": len(check["outside"]["areas"]), "print": check.get("print"), "outside_changed": check["outside"]["changed"]},
        "notes_ko": notes,
    }
    (out_dir / "restore.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    b, _ = _work(before)
    a, _ = _work(warp_to(before, after, full_matrix(info["align"], _work(after)[1], _work(before)[1])))
    r, _ = _work(restored)
    gap = np.full((b.shape[0], 16, 3), 255, np.uint8)
    cv2.imwrite(str(out_dir / "restore.jpg"), np.hstack([b, gap, a, gap, r]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rep


def main():
    ap = argparse.ArgumentParser(description="Keep the original outside the edited region of a Creazy area edit.")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--box", type=revision.parse_box, action="append", default=[])
    ap.add_argument("--mask", default=None)
    ap.add_argument("--auto", action="store_true", help="no painted region (chat edit, quick edit): find what was drawn or removed")
    ap.add_argument("--where", type=revision.parse_box, action="append", default=[], help="with --auto: the box the seller pointed at")
    ap.add_argument("--edits", default=None, help="edits.json instead of --before/--after: 'edits' with a painted box, 'chat' found with --auto")
    ap.add_argument("--refset", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    if a.edits:
        base = Path(a.edits).parent
        spec = json.loads(Path(a.edits).read_text())
        for e in spec.get("edits", []):
            jobs.append((e["name"], base / e["before"], base / e["after"], [e["box"]], False, None))
        for e in spec.get("chat", []):
            jobs.append((e["name"], base / e["before"], base / e["after"], (), True, e.get("where")))
    else:
        jobs.append(("", a.before, a.after, a.box, a.auto, a.where or None))
    for name, b, af, boxes, auto, where in jobs:
        r = run(b, af, Path(a.out) / name, boxes, a.mask, a.refset, auto=auto, where=where)
        if r["restored"] is None:
            print(name or af, "| as is:", r["as_is"]["verdict"], "| not restored")
        else:
            print(name or af, "| as is:", r["as_is"]["verdict"], r["as_is"]["level"], r["as_is"]["areas"], "areas",
                  "| restored:", r["restored"]["verdict"], r["restored"]["level"], r["restored"]["areas"], "areas, outside changed", r["restored"]["outside_changed"])
        for n in r["notes_ko"]:
            print("  ", n)


if __name__ == "__main__":
    main()

"""Print regions for restoration, cut from the product itself.

The reference set's print regions (refset/build.py) come from keypoint density over the
whole crop, and two problems showed up in the CS validation (docs/validation_cs_plan.md
sections 11 and 12):

- a crop that also shows the retail box behind the product gives a print that covers both,
  and restoration writes the box's lettering onto the generated background (case 10);
- regions whose label matches one already kept are skipped as duplicates, so a product sold
  in several variants (colours) keeps one variant's label, and restoration renames the others
  (case 28).

Here every product photo of the set gives its own print: the detector box of the set's query
inside the crop (the product, not what stands behind it), then the keypoint-density box inside
that. Only identical regions are dropped. check.print_restore.prints_for reads the result like
a reference set.

    .venv/bin/python -m check.print_set --refset out/cs_all/10/refset/refset.json --out out/validation/prints_v3/10
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MIN_SCORE = 0.2   # detector score for "this is the product"
LIMIT = 40        # prints per set (checked after each photo); matching time grows with each


def product_boxes(img: Image.Image, query: str, min_score=MIN_SCORE, overlap=0.3):
    """Detector boxes of the product in a crop, strongest first, overlapping ones removed.
    A photo of several units (three cans) gives one box per unit."""
    from refset import models
    keep = []
    for d in sorted(models.detect(img, [query], threshold=0.1), key=lambda d: -d["score"]):
        if d["score"] < min_score:
            continue
        b = [int(v) for v in d["box"]]
        if all(_iou(b, k) < overlap for k in keep) and not any(_inside(b, k) or _inside(k, b) for k in keep):
            keep.append(b)
    return keep


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / float(area(a) + area(b) - inter + 1e-6)


def _inside(a, b, share=0.85):
    """Most of box a lies inside box b (a group box around single units, or the reverse)."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter >= share * (a[2] - a[0]) * (a[3] - a[1])


def density_boxes(dens, pad=0.06, share=0.2):
    """Print boxes from a keypoint-density map: one per dense component (a product label, the
    label of the box behind it), then their union when there are several. Same thresholds as
    refset.build.print_box_from_density, which returns only the union."""
    from refset.build import pad_box
    h, w = dens.shape
    blur = cv2.GaussianBlur(dens, (0, 0), 0.04 * min(h, w))
    if blur.max() <= 0:
        return []
    mask = (blur >= 0.10 * blur.max()).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    sums = [(i, float(blur[lab == i].sum())) for i in range(1, n)
            if 0.15 * w <= cents[i][0] <= 0.85 * w and 0.10 * h <= cents[i][1] <= 0.90 * h]
    if not sums:
        return []
    best = max(v for _, v in sums)
    keep = sorted([(i, v) for i, v in sums if v >= share * best], key=lambda t: -t[1])
    box = lambda i: (stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])
    out = [pad_box(box(i), (w, h), pad) for i, _ in keep]
    if len(keep) > 1:
        bs = [box(i) for i, _ in keep]
        out.append(pad_box((min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)), (w, h), pad))
    return out


def agreement(image, print_bgr, matcher, match):
    """How well the print's detail lines up with the image where the print is (high-pass
    luminance correlation over the surfaces the match vouches for). A label of another variant
    (another colour name) lines up everywhere except that name, and scores lower."""
    from check.print_restore import full_res_homography, region_mask, supported_surface
    Hf = full_res_homography(match["H"], print_bgr.shape, image.shape)
    H, W = image.shape[:2]
    warped = cv2.warpPerspective(print_bgr, Hf, (W, H))
    _, outline, _ = region_mask(Hf, print_bgr.shape, image.shape)
    sup = cv2.warpPerspective(supported_surface(print_bgr, matcher, match).astype(np.uint8), Hf, (W, H)) > 0
    m = sup & outline
    if m.sum() < 500:
        return -1.0
    hp = lambda x: (lambda L: L - cv2.GaussianBlur(L, (0, 0), 3))(cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32))
    a, b = hp(image)[m], hp(warped)[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-6))


def best_print(image, multi, parts, images, near=0.3, top_n=4, tie=0.02):
    """The print to restore with. Candidates: the prints that match with at least `near` of
    the best inlier count (at most top_n). Among them, the one whose detail agrees best with
    the image wins (agreement): the inlier count favours the photo the generator copied the
    composition from, which may show another variant of the product. Within `tie` of the best
    agreement, a single label ("component") wins over a union of several regions.
    parts: {name: "component" | "union"}; images: {name: colour image}.
    Returns (name, matcher, match result, info) where info lists the candidates."""
    from check.print_restore import MIN_INLIERS
    feats = multi.matchers[0].features(image)
    scored = sorted(((n, m, m.match_features(*feats)) for n, m in zip(multi.names, multi.matchers)), key=lambda t: -t[2]["inliers"])
    top = scored[0][2]["inliers"]
    cands = [t for t in scored if t[2]["inliers"] >= max(MIN_INLIERS, near * top) and t[2]["H"] is not None][:top_n] or scored[:1]
    agree = {t[0]: agreement(image, images[t[0]], t[1], t[2]) if t[2]["H"] is not None else -1.0 for t in cands}
    best = max(agree.values())
    pick = max((t for t in cands if agree[t[0]] >= best - tie), key=lambda t: (parts.get(t[0]) != "union", agree[t[0]]))
    info = [dict(print=t[0], inliers=int(t[2]["inliers"]), agreement=round(agree[t[0]], 3)) for t in cands]
    return pick[0], pick[1], pick[2], info


def load(prints_dir):
    """(MultiPrintMatcher, {name: colour image}, {name: part}) for a folder written by build()."""
    from check.print_restore import prints_for
    d = json.loads((Path(prints_dir) / "prints.json").read_text())
    multi, images = prints_for(Path(prints_dir) / "prints.json")
    return multi, images, {p["path"]: p.get("part", "component") for p in d["prints"]}


def build(refset_path, out_dir, query=None, limit=LIMIT):
    """Write one print per product unit of every product photo in the set, and prints.json
    (readable by check.print_restore.prints_for). Returns the list of print entries."""
    from refset.build import detail_map
    ref = json.loads(Path(refset_path).read_text())
    base, out = Path(refset_path).parent, Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    query = query or ref.get("query") or "a product"
    items = sorted((i for i in ref["items"] if i.get("role") == "product" and i.get("crop")), key=lambda i: -(i.get("detail") or 0))
    prints, seen = [], set()
    for it in items:
        crop = Image.open(base / it["crop"]).convert("RGB")
        boxes = product_boxes(crop, query) or [[0, 0, crop.width, crop.height]]
        for b in boxes:
            unit = crop.crop(b)
            dens, _ = detail_map(unit)
            parts = density_boxes(dens)
            for k, pb in enumerate(parts):
                region = unit.crop(pb)
                if min(region.size) < 48:
                    continue
                h = hashlib.sha1(np.asarray(region.resize((64, 64))).tobytes()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                name = f"print_{len(prints) + 1}.png"
                region.save(out / name)
                box = [b[0] + pb[0], b[1] + pb[1], b[0] + pb[2], b[1] + pb[3]]
                part = "union" if len(parts) > 1 and k == len(parts) - 1 else "component"
                prints.append({"path": name, "from": it["crop"], "box": box, "unit": b, "part": part, "method": "product+density"})
        if len(prints) >= limit:
            break
    (out / "prints.json").write_text(json.dumps({"query": query, "refset": str(refset_path), "prints": prints}, ensure_ascii=False, indent=1))
    return prints


def main():
    ap = argparse.ArgumentParser(description="Print regions cut from the product in every product photo of a reference set")
    ap.add_argument("--refset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query")
    a = ap.parse_args()
    for p in build(a.refset, a.out, a.query):
        print(p["path"], p["from"], p["box"])


if __name__ == "__main__":
    main()

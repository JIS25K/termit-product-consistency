"""Restore the fine detail of the real print on a generated image (a simple, model-free take on
the FlowFixer idea: keep the generated picture, bring back only the product's detail).

The generated image G stays the base: the product's angle, size, light, shadow and the
background are G's. From the seller's print (the reference set's print.png) only the high
frequencies are taken (lines, letter outlines, pattern), and only inside the print region.
Every pixel outside that region is left exactly as it was.

  1. Locate the print in G with PrintMatcher (SIFT + RANSAC homography). The matcher works on
     resized images (reference at REF_SIDE, output at OUT_SIDE); the homography is brought
     back to the original pixels of print.png and G.
  2. Gate: at least MIN_INLIERS inliers, otherwise "no_print". An invented or missing print
     cannot be located, and restoring it is not attempted (regenerate, or composite the real
     product instead).
  3. Warp the colour print onto G. Optional fine alignment (ECC) for curved products, where a
     homography is only an approximation; skipped when it does not converge.
  4. Frequency split on Lab L (G's colour kept): restored L = low(G) + high(print), the
     high-frequency layer of G inside the region (blurred or wrong lines) is dropped. G's own
     strokes are filled in before its low frequencies are taken (otherwise they come back as a
     soft ghost), and only the strokes are touched: the real print's thin lines (top-hat) and
     the generated strokes being removed, where the print and G show the same surface. A step
     edge in the print crop (the product's outline, the table line) is not copied. "mul"
     scales the print's detail by the scene's local brightness instead (ink is reflectance).
     Sigma is a fraction of the print's width in G; several fractions are tried and the one
     whose result matches the real print best (inliers) is kept.
  5. Blend inside the warped print outline shrunk a little, with a feathered edge; the mask is
     zero outside the outline, so the image outside is unchanged.
  6. Check: inliers after vs before, and the largest pixel change outside the mask (must be 0).

A reference set with several print regions (front label, back label) is handled by
restore_best: the region that matches the image best is restored.

Usage
  python -m check.print_restore --refset out/mug/refset.json --images data/mug/outputs/run9/*.png --out out/print_restore/run9
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from check.sift import OUT_SIDE, REF_SIDE, MultiPrintMatcher, PrintMatcher

MIN_INLIERS = 12
FRACTIONS = (0.008, 0.015, 0.03)  # sigma as a fraction of the print's width in G


def full_res_homography(H, print_shape, image_shape):
    """PrintMatcher's H maps the reference resized to REF_SIDE onto the output resized to
    OUT_SIDE (only when larger). Return the same mapping between original pixels."""
    s_ref = REF_SIDE / max(print_shape[:2])
    s_out = min(1.0, OUT_SIDE / max(image_shape[:2]))
    return np.diag([1 / s_out, 1 / s_out, 1.0]) @ H @ np.diag([s_ref, s_ref, 1.0])


def region_mask(Hf, print_shape, image_shape, shrink=0.03, feather=None):
    """Warped print outline, shrunk by a fraction of its size, with a feathered edge that stays
    inside the outline. Returns (soft mask, hard outline, print width in G)."""
    h, w = print_shape[:2]
    H, W = image_shape[:2]
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), Hf).reshape(-1, 2)
    outline = np.zeros((H, W), np.uint8)
    cv2.fillPoly(outline, [np.round(quad).astype(np.int32)], 1)
    width = float(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    k = max(3, int(shrink * width)) | 1
    inner = cv2.erode(outline, np.ones((k, k), np.uint8))
    f = feather or max(1.5, 0.01 * width)
    soft = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), f) * outline
    return soft, outline.astype(bool), width


def refine_ecc(warped, image, outline):
    """Fine alignment of the warped print to G inside the outline (affine ECC on edges).
    Returns the corrected warped print, or None when ECC does not converge."""
    a = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    m = cv2.erode(outline.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), m, 5)
    except cv2.error:
        return None
    if np.abs(warp[:, 2]).max() > 0.05 * max(image.shape[:2]):  # a runaway solution
        return None
    return cv2.warpAffine(warped, warp, image.shape[1::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def clean_generated(image, outline, width):
    """G with its own print strokes removed inside the outline (filled from the surface around
    them). Taking G's low frequencies from this keeps the generated lines from coming back as a
    soft ghost next to the real ones. Strokes: strong local contrast against a small blur."""
    ys, xs = np.nonzero(outline)
    pad = 8
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1
    crop = image[y0:y1, x0:x1]
    L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    hp = L - cv2.GaussianBlur(L, (0, 0), max(2.0, 0.01 * width))
    inside = outline[y0:y1, x0:x1]
    mad = float(np.median(np.abs(hp[inside]))) + 1e-3
    ink = (np.abs(hp) > max(6.0, 4.0 * mad)) & inside
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8))
    out = image.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(crop, ink, max(3, int(0.01 * width)), cv2.INPAINT_TELEA)
    return out, int(ink.sum())


def same_surface(base, warped, outline, width, max_ab=8.0, max_dl=None):
    """Where the warped print and G show the same surface: their low-frequency colour (a, b)
    agrees. The print crop can include things around the print (the product's edge, the table
    under it); there the colours differ and no detail is taken. Brightness may differ (scene
    light), so only a and b are compared. Returns a soft 0..1 map."""
    sig = max(3.0, 0.03 * width)
    a = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), sig)
    b = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), sig)
    da, db = a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]
    # the scene light tints the whole product (a warm room shifts a and b everywhere): take
    # the typical offset out first, compare what is left
    da -= np.median(da[outline]); db -= np.median(db[outline])
    agree = (np.hypot(da, db) < max_ab) & outline
    if max_dl is not None:  # also brightness, after the typical offset: a dark label against a white bottle
        dl = a[..., 0] - b[..., 0]
        dl -= np.median(dl[outline])
        agree &= np.abs(dl) < max_dl
    agree = agree.astype(np.float32)
    agree = cv2.erode(agree, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(agree, (0, 0), max(1.5, 0.01 * width))


def print_ink(warped, outline, width):
    """Thin strokes of the warped print (lines, letter outlines, pattern): bright and dark
    top-hats, which answer thin lines but not a step edge such as the product's outline or the
    table line that a print crop may contain."""
    L = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[..., 0]
    k = max(5, int(0.025 * width)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = np.maximum(cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se)).astype(np.float32)
    mad = float(np.median(th[outline])) + 1e-3
    ink = (th > max(8.0, 4.0 * mad)) & outline
    return cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def supported_surface(print_bgr, matcher, match, k=5):
    """The parts of the print the match vouches for, in print coordinates (bool mask).

    The print is cut into surfaces: colour clusters of the blurred print (the blur melts small
    lettering into the surface under it), each closed over thin lines (a rule across a label)
    and split into connected parts. A part counts when inlier keypoints fall on it; holes in
    the counted parts (lettering, a ribbon inside a label) count too. A print crop that also
    caught the box behind the product has no inliers on the box when the generated image does
    not show it, so the box's lettering is left out."""
    h, w = print_bgr.shape[:2]
    s = matcher.ref.shape[1] / float(w)
    pts = np.float32([matcher.kr[m.queryIdx].pt for m in match["matches"]]) / s
    if len(pts) == 0:
        return np.ones((h, w), bool)
    width = float(max(h, w))
    lab = cv2.cvtColor(cv2.GaussianBlur(print_bgr, (0, 0), max(1.5, 0.01 * width)), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    _, labels, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    kc = max(3, int(0.03 * width)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    xi = np.clip(pts[:, 0].astype(int), 0, w - 1); yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
    need = max(3, int(0.05 * len(pts)))
    keep = np.zeros((h, w), np.uint8)
    for c in range(k):
        m = cv2.morphologyEx((labels == c).astype(np.uint8), cv2.MORPH_CLOSE, se)
        n, comp = cv2.connectedComponents(m, connectivity=8)
        counts = np.bincount(comp[yi, xi], minlength=n)
        for i in range(1, n):
            if counts[i] >= need:
                keep[comp == i] = 1
    # fill holes: everything not reachable from the border through unkept pixels
    inv = (1 - keep).astype(np.uint8)
    n, comp = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]])).tolist())
    holes = (inv > 0) & ~np.isin(comp, list(border))
    keep = (keep > 0) | holes
    return cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def split_merge(image, warped, sigma, mode="add", low_source=None, print_low_source=None):
    """Lab L: low frequencies from G (or low_source, G with its strokes removed), high
    frequencies from the warped print; a and b from G. print_low_source (the print with its
    strokes removed) gives the print's low frequencies: without it the blur of the print is
    darker next to each stroke, and the difference leaves a light rim around the letters."""
    gl = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pl = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    src = gl if low_source is None else cv2.cvtColor(low_source, cv2.COLOR_BGR2LAB).astype(np.float32)
    g_low = cv2.GaussianBlur(src[..., 0], (0, 0), sigma)
    psrc = pl if print_low_source is None else cv2.cvtColor(print_low_source, cv2.COLOR_BGR2LAB).astype(np.float32)
    p_low = cv2.GaussianBlur(psrc[..., 0], (0, 0), sigma)
    if mode == "mul":
        L = g_low * np.clip((pl[..., 0] + 1) / (p_low + 1), 0.3, 1.8)
    else:
        L = g_low + (pl[..., 0] - p_low)
    out = gl.copy()
    out[..., 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def restore(image, print_bgr, matcher, mode="add", refine=False, fractions=FRACTIONS, min_inliers=MIN_INLIERS, region=None,
            clean_print=False, support=False, edge_safe=False):
    """Restore the print detail on image. Returns (best attempt, info). info["status"] is
    "restored", "no_print" (gate; the image is returned unchanged), or "not_better" (no sigma
    raised the score; the caller should keep the original, the attempt is returned to look at).
    clean_print removes the print's own strokes before taking its low frequencies, the same
    way as for G (no light rim around the letters). support keeps restoration on the print's
    surfaces that the match vouches for (supported_surface). edge_safe takes only G's thin
    strokes out (not the edge of a label or of the product, which a slightly different layout
    would leave smeared) and also compares brightness when deciding where the surfaces are
    the same. All three off by default: the CS validation ran without them."""
    r0 = matcher.match(image)
    info = dict(inliers_before=int(r0["inliers"]), min_inliers=min_inliers, mode=mode, clean_print=clean_print, support=support, edge_safe=edge_safe)
    if r0["H"] is None or r0["inliers"] < min_inliers:
        return image, {**info, "status": "no_print", "route": "regenerate or composite the real product"}
    Hf = full_res_homography(r0["H"], print_bgr.shape, image.shape)
    H, W = image.shape[:2]
    warped = cv2.warpPerspective(print_bgr, Hf, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    soft, outline, width = region_mask(Hf, print_bgr.shape, image.shape)
    info.update(print_width_px=round(width, 1))
    if refine:
        w2 = refine_ecc(warped, image, outline)
        info["refined"] = w2 is not None
        if w2 is not None:
            warped = w2
    clean, ink_px = clean_generated(image, outline, width)
    warped_clean = clean_generated(warped, outline, width)[0] if clean_print else None
    agree = same_surface(clean, warped, outline, width, max_dl=45.0 if edge_safe else None)
    # touch only the real print's strokes and the generated strokes (to remove them), on the
    # same surface; the rest of the outline keeps G's pixels
    g_strokes = np.abs(clean.astype(np.int16) - image.astype(np.int16)).max(axis=2) > 3
    if edge_safe:
        g_strokes &= print_ink(image, outline, width)
    touch = print_ink(warped, outline, width) | g_strokes
    touch = cv2.GaussianBlur(cv2.dilate(touch.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), max(1.5, 0.006 * width))
    soft = soft * agree * touch
    if support:
        sup = supported_surface(print_bgr, matcher, r0)
        sup_w = cv2.warpPerspective(sup.astype(np.uint8), Hf, (W, H), flags=cv2.INTER_NEAREST) > 0
        info["supported_share"] = round(float(sup.mean()), 3)
        soft = soft * cv2.GaussianBlur(sup_w.astype(np.float32), (0, 0), max(1.5, 0.006 * width))
    if region is not None:  # the product in G (a detector box): a print crop that also caught packaging or background must not write onto them
        soft = soft * cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.0)
    info.update(generated_stroke_px=ink_px, same_surface_share=round(float(agree[outline].mean()), 3),
                touched_share=round(float((soft > 0.5)[outline].mean()), 3))
    best = None
    tried = []
    for f in fractions:
        sigma = max(1.0, f * width)
        merged = split_merge(image, warped, sigma, mode, low_source=clean, print_low_source=warped_clean)
        out = (image.astype(np.float32) * (1 - soft[..., None]) + merged.astype(np.float32) * soft[..., None])
        out = np.where(outline[..., None], np.clip(np.round(out), 0, 255), image).astype(np.uint8)
        n = int(matcher.match(out)["inliers"])
        tried.append(dict(fraction=f, sigma=round(sigma, 2), inliers=n))
        if best is None or n > best[0]:
            best = (n, out, f, sigma)
    n, out, f, sigma = best
    info.update(tried=tried, fraction=f, sigma=round(sigma, 2), inliers_after=n,
                outside_max_diff=int(np.abs(out.astype(np.int16) - image.astype(np.int16))[~outline].max(initial=0)),
                mask_px=int(outline.sum()))
    return out, {**info, "status": "restored" if n > info["inliers_before"] else "not_better"}


def matcher_for(refset_path):
    ref = json.loads(Path(refset_path).read_text())
    base = Path(refset_path).parent
    p = (ref.get("prints") or [{"path": ref["print"]["path"]}])[0]["path"]
    return PrintMatcher(base / p), cv2.imread(str(base / p))


def prints_for(refset_path):
    """All print regions of a reference set: (MultiPrintMatcher, {name: colour image})."""
    ref = json.loads(Path(refset_path).read_text())
    base = Path(refset_path).parent
    items = ref.get("prints") or [{"path": ref["print"]["path"]}]
    pairs = [(p["path"], base / p["path"]) for p in items]
    return MultiPrintMatcher(pairs), {n: cv2.imread(str(f)) for n, f in pairs}


def restore_best(image, multi, print_images, **kw):
    """A reference set with several print regions (front label, back label, ...): restore with
    the region that matches the image best, scored by that region's own matcher."""
    name, m, _ = multi._best(image)
    out, info = restore(image, print_images[name], m, **kw)
    return out, {**info, "print": name}


def main():
    ap = argparse.ArgumentParser(description="Restore the real print's detail on generated images")
    ap.add_argument("--refset", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["add", "mul"], default="add")
    ap.add_argument("--refine", action="store_true", help="fine alignment with ECC after the homography")
    a = ap.parse_args()
    multi, print_images = prints_for(a.refset)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            continue
        res, info = restore_best(img, multi, print_images, mode=a.mode, refine=a.refine)
        if info["status"] != "no_print":
            cv2.imwrite(str(out / (Path(p).stem + ("_restored.png" if info["status"] == "restored" else "_attempt.png"))), res)
        report[Path(p).name] = info
        print(Path(p).name, info["status"], info["inliers_before"], "->", info.get("inliers_after"), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

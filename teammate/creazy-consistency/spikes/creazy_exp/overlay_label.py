"""Condition B, step 4 for P1 (sesame-oil bottle): put the real label back on the generated bottle.

Used only on a kept cut where our checker found the label text missing or garbled, and only when
the real label does not have to be enlarged more than MAX_UP times (the seller's photo is small,
180x530 px, so a big close-up would turn blurry; that cut keeps the generated label).

  1. Real label: the cream rectangle of the seller's photo (holes from the dark print filled).
  2. Generated label: the largest cream region inside the bottle box (OWLv2), as a rotated
     rectangle (minAreaRect).
  3. Which way round: the real label is tried in 4 rotations; the one whose edges correlate best
     with the generated label wins (the generator keeps the layout roughly, scrambles the text).
  4. Warp the real label onto the rectangle, keep the generated label's shading (low-frequency
     brightness ratio, so the cylinder's light and shadow stay), feather the edge.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REAL = ROOT / "out" / "creazy_exp_run" / "creazy_exp" / "inputs" / "P1" / "1.jpg"
MAX_UP = 2.0


def cream(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    return ((L > 200) & (b > 8) & (b < 40) & (a < 6)).astype(np.uint8)


def off_border(m):
    """Drop regions connected to the image border (a cream backdrop is as light as the label)."""
    n, lab = cv2.connectedComponents(m)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return (np.isin(lab, edge[edge > 0], invert=True) & (lab > 0)).astype(np.uint8)


def largest(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    return (lab == i).astype(np.uint8)


def filled(m):
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return m | (1 - ff)


def real_label():
    """The label in the seller's photo: inside the bottle box (OWLv2), the longest run of rows in
    the lower half where the centre is not oil-coloured (oil b* 50-70, label and print below 35).
    Colour alone cannot do it: the photo's backdrop is the same cream as the label."""
    from refset.models import detect
    im = cv2.imread(str(REAL))
    d = detect(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), ["a bottle"], threshold=0.1)
    x0, y0, x1, y1 = d[0]["box"] if d else (0, 0, im.shape[1], im.shape[0])
    b = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 2].astype(np.int16) - 128
    cx = (x0 + x1) // 2
    col = (b[:, cx - 20:cx + 20] < 35).mean(1) > 0.6
    runs, start = [], None
    for y, v in enumerate(col):
        if v and start is None:
            start = y
        if not v and start is not None:
            runs.append((start, y)); start = None
    if start is not None:
        runs.append((start, len(col)))
    lower = [r for r in runs if r[0] > (y0 + y1) // 3]
    ly0, ly1 = max(lower, key=lambda r: r[1] - r[0])
    return im[ly0:ly1, x0 + 3:x1 - 3]


def label_colour_mask(gen, lab_img):
    """Pixels coloured like label paper under scene light: the real label is a pale cream
    (L 244, b* 21) but generated scenes light it warmer and darker (L 180-230, b* 25-45), so
    the test is on hue (neutral a*, yellow b*) and a floor on lightness, not on distance."""
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = g[..., 0], g[..., 1] - 128, g[..., 2] - 128
    return ((L > 150) & (np.abs(a) < 9) & (b > 12) & (b < 55)).astype(np.uint8)


def generated_label(gen, lab_img):
    from refset.models import detect
    pil = Image.fromarray(cv2.cvtColor(gen, cv2.COLOR_BGR2RGB))
    dets = detect(pil, ["a bottle"], threshold=0.15)
    if not dets:
        return None, None, None
    bx0, by0, bx1, by1 = dets[0]["box"]
    px, py = int(0.25 * (bx1 - bx0)), int(0.25 * (by1 - by0))  # a hand often cuts the box short
    H, W = gen.shape[:2]
    box = (max(bx0 - px, 0), max(by0 - py, 0), min(bx1 + px, W), min(by1 + py, H))
    x0, y0, x1, y1 = box
    m = np.zeros(gen.shape[:2], np.uint8)
    m[y0:y1, x0:x1] = off_border(label_colour_mask(gen, lab_img))[y0:y1, x0:x1]
    m = largest(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    if m is None:
        return None, None, box
    m = filled(m)
    return cv2.minAreaRect(cv2.findNonZero(m)), m, box


def cap_centre(gen, box):
    """Centre of the red cap: the largest strongly red blob inside the bottle box (the red
    sauce in a bowl lies outside the bottle box or is smaller)."""
    x0, y0, x1, y1 = box
    lab = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.int16)
    red = ((lab[..., 1] - 128 > 45) & (lab[..., 0] > 60)).astype(np.uint8)
    sub = np.zeros_like(red); sub[y0:y1, x0:x1] = red[y0:y1, x0:x1]
    m = largest(cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    if m is None:
        return None
    ys, xs = np.nonzero(m)
    return np.float32([xs.mean(), ys.mean()])


def ordered(rect, cap):
    """Corners [top-left, top-right, bottom-right, bottom-left] of the label as printed, with the
    top edge facing the cap and the order clockwise on screen (not mirrored: the label on the
    front of the bottle is seen directly)."""
    q = cv2.boxPoints(rect).astype(np.float32)
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise on screen (y points down)
    best = None
    for k in range(4):
        qq = np.roll(q, -k, axis=0)
        top_mid = (qq[0] + qq[1]) / 2
        d = np.linalg.norm(top_mid - cap)
        if best is None or d < best[0]:
            best = (d, qq)
    return best[1]


def place(gen, lab_img, quad):
    h, w = lab_img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, quad)
    H, W = gen.shape[:2]
    warped = cv2.warpPerspective(lab_img, M, (W, H), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H))
    return warped, mask


def overlay(gen):
    """Returns (new image, info). info["applied"] says whether the label was replaced and why not."""
    lab_img = real_label()
    rect, gm, box = generated_label(gen, lab_img)
    if rect is None:
        return gen, {"applied": False, "why": "no generated label found"}
    (cx, cy), (rw, rh), _ = rect
    bx0, by0, bx1, by1 = box
    if rw * rh < 0.08 * (bx1 - bx0) * (by1 - by0):
        return gen, {"applied": False, "why": "label region too small for the bottle (label not facing the camera)"}
    up = max(max(rw, rh) / max(lab_img.shape[:2]), min(rw, rh) / min(lab_img.shape[:2]))
    if up > MAX_UP:
        return gen, {"applied": False, "why": f"real label would be enlarged {up:.1f}x (limit {MAX_UP})", "scale": round(up, 2)}
    cap = cap_centre(gen, box)
    if cap is None:
        return gen, {"applied": False, "why": "cap not found, label direction unknown"}
    quad = ordered(rect, cap)
    warped, mask = place(gen, lab_img, quad)
    # keep the generated label's shading (cylinder light) and anything in front of it (fingers):
    # the paste is limited to the generated label's own region
    g = cv2.GaussianBlur(gen.astype(np.float32), (0, 0), 12)
    r = cv2.GaussianBlur(warped.astype(np.float32), (0, 0), 12)
    shade = np.clip((g.mean(-1) + 1) / (r.mean(-1) + 1), 0.6, 1.3)[..., None]
    out_lab = np.clip(warped.astype(np.float32) * shade, 0, 255)
    a = cv2.erode(mask, np.ones((3, 3), np.uint8)) * cv2.erode(gm.astype(np.float32), np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
    out = (gen.astype(np.float32) * (1 - a) + out_lab * a).astype(np.uint8)
    return out, {"applied": True, "scale": round(up, 2), "covered": round(float((a > 0.5).mean()), 4)}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out, info = overlay(cv2.imread(src))
    cv2.imwrite(dst, out)
    print(info)

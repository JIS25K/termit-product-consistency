"""Product count for the experiment (not the checker's own count.py, which the CS scoring uses).

OWLv2 "a lotion bottle" boxes, then two rules:
  - a box that lies almost entirely inside another box is a part or a picture on the product
    (the label of this bottle shows a small baby bottle, which OWLv2 boxes too),
  - a box counts as the seller's product only when its crop matches the real photo's label
    (SIFT + RANSAC homography with at least MIN_INLIERS agreeing points). Scene light changes
    the body colour too much for a colour test, and other brands' bottles must not count.
A product whose label faces away is not counted, so the count is a floor: two or more means the
product was drawn more than once.
"""
import cv2
import numpy as np
from PIL import Image

INSIDE = 0.9
MIN_INLIERS = 8
REL_BEST = 0.3
_REAL = {}


def _inside(b, o):
    w = min(b[2], o[2]) - max(b[0], o[0]); h = min(b[3], o[3]) - max(b[1], o[1])
    a = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return max(0, w) * max(0, h) / a


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(0, w) * max(0, h)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def _real(path):
    if path not in _REAL:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        g = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        m = (cv2.resize(rgba[..., 3], None, fx=2, fy=2) > 128).astype(np.uint8) * 255
        sift = cv2.SIFT_create(nfeatures=4000)
        _REAL[path] = (sift, *sift.detectAndCompute(g, m))
    return _REAL[path]


def label_inliers(crop, real_path):
    sift, k1, d1 = _real(real_path)
    if crop.shape[0] < 400:
        f = 400 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
    if d2 is None or len(k2) < 4:
        return 0
    good = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 4:
        return 0
    p = np.float32([k1[m.queryIdx].pt for m in good]); q = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p, q, cv2.RANSAC, 6.0)
    return int(inl.sum()) if inl is not None else 0


def product_boxes(bgr, real_path, threshold=0.15):
    from refset.models import detect
    dets = detect(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ["a lotion bottle"], threshold=threshold)
    boxes = [tuple(map(int, d["box"])) for d in dets]
    boxes = [b for b in boxes if not any(o != b and _inside(b, o) >= INSIDE for o in boxes)]
    H, W = bgr.shape[:2]
    scored = []
    for x0, y0, x1, y1 in boxes:
        pw, ph = int(0.1 * (x1 - x0)), int(0.05 * (y1 - y0))
        crop = bgr[max(y0 - ph, 0):min(y1 + ph, H), max(x0 - pw, 0):min(x1 + pw, W)]
        scored.append(((x0, y0, x1, y1), label_inliers(crop, real_path)))
    if not scored:
        return []
    # a small enlarged crop of another bottle can yield a few chance inliers: a match counts only
    # when it reaches REL_BEST of the best match in the same image
    best = max(n for _, n in scored)
    out = []
    for b, n in sorted(scored, key=lambda x: -x[1]):
        if n < max(MIN_INLIERS, REL_BEST * best):
            continue
        if any(_iou(b, o) >= 0.5 or _inside(b, o) >= 0.5 or _inside(o, b) >= 0.5 for o, _ in out):
            continue  # the same bottle boxed twice (whole bottle and label part)
        out.append((b, n))
    return out

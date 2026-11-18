"""Pick, for one generated output, the real photo whose view matches the generated product best:
scale each real cutout to the generated product's extents (width fit, bottom aligned, centred)
and compare silhouettes with a tolerant IoU (both masks dilated). Real views are the full
upload cutouts plus the panels of the multi-view upload (out/exp13/views/)."""
import os, sys, json, numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, place as P
from compose_lama import main_part
from refset import models

VIEWS = ["u0", "u1", "v_oblL", "v_frontR", "v_smallR", "v_side"]

def real_view(name):
    if name in ("u0", "u1"):
        im, a, _ = P.real(int(name[1]))
    else:
        im = cv2.imread(f"{P.ROOT}/out/exp13/views/{name}.png")
        a = np.load(f"{P.ROOT}/out/exp13/views/{name}_alpha.npy")
    return im, a

def gen_part(scene):
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    dets = models.detect(pil, ["a shelf", "a bookshelf"], threshold=0.15)[:3]
    if not dets:
        return None, None, None, None
    box = E.box_mask(scene.shape, [d["box"] for d in dets])
    M, core, d, bg = E.shape_mask(scene, box)
    part = main_part((d > 20).astype(np.uint8) & (core > 0)) if (d > 20).any() else main_part(core)
    return part, M, core, (d, bg, dets)

def fit_mask(a, ext, shape, fit="width"):
    op = (a > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(op)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    gx0, gy0, gx1, gy1 = ext
    s = (gx1 - gx0) / (x1 - x0) if fit == "width" else (gy1 - gy0) / (y1 - y0)
    w, h = max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))
    crop = cv2.resize(op[y0:y1, x0:x1] * 255, (w, h), interpolation=cv2.INTER_AREA) > 127
    out = np.zeros(shape[:2], np.uint8)
    px, py = (gx0 + gx1) // 2 - w // 2, gy1 - h
    sx0, sy0, sx1, sy1 = max(px, 0), max(py, 0), min(px + w, shape[1]), min(py + h, shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0:sy1, sx0:sx1] = crop[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
    return out

def tol_iou(a, b, r):
    k = np.ones((r, r), np.uint8)
    da, db = cv2.dilate(a, k) > 0, cv2.dilate(b, k) > 0
    return float((da & db).sum()) / max(float((da | db).sum()), 1.0)

def score_views(scene, part, views=VIEWS):
    ys, xs = np.nonzero(part)
    ext = (int(np.percentile(xs, 0.5)), int(ys.min()), int(np.percentile(xs, 99.5)) + 1, int(np.percentile(ys, 99.5)) + 1)
    r = max(5, int(0.02 * (ext[2] - ext[0]))) | 1
    out = {}
    for v in views:
        im, a = real_view(v)
        out[v] = max((tol_iou(fit_mask(a, ext, scene.shape, f), part.astype(np.uint8), r), f) for f in ("width", "height"))
    return out, ext


def crop_on_grey(im, a, pad=0.06):
    """Product crop on a neutral grey ground, so the embedding sees the product's pose."""
    m = (a > 0.5)
    ys, xs = np.nonzero(m)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    p = int(pad * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, im.shape[1]), min(y1 + p, im.shape[0])
    aa = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[y0:y1, x0:x1, None]
    c = (im[y0:y1, x0:x1].astype(np.float32) * aa + 128 * (1 - aa)).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

def dino_scores(scene, gen_mask, views=VIEWS):
    """Cosine similarity of DINOv2 embeddings: generated product crop vs each real view crop."""
    g = crop_on_grey(scene, gen_mask.astype(np.float32))
    crops = [crop_on_grey(*real_view(v)) for v in views]
    e = np.asarray(models.embed([g] + crops), dtype=np.float32)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    return {v: float(e[0] @ e[i + 1]) for i, v in enumerate(views)}

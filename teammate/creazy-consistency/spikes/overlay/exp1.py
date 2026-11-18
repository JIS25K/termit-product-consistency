from pathlib import Path
"""Spike on CS 1 (swing-handle tumbler): same scene, real product, on the plain-background cuts."""
import os, sys, numpy as np, cv2, time
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cutout as K, compose as C
from refset import models

ROOT = str(Path(__file__).resolve().parents[2])
CASE = f"{ROOT}/data/cs/1"
UPS = sorted(f for f in os.listdir(f"{CASE}/uploads") if not f.startswith("info-"))
OUTS = sorted(f for f in os.listdir(f"{CASE}/outputs") if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
OUT = f"{ROOT}/out/exp1"

def real_cut(i):
    im, bg, raw = K.cutout(f"{CASE}/uploads/{UPS[i]}", lo=6, hi=26)
    strong = (raw > 0.6).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, 8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    # fill holes inside the tumbler body (cream on white has weak spots), keep the handle gap
    filled = m.copy(); h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8); inv = (1 - m).copy()
    cv2.floodFill(inv, ff, (0, 0), 0)
    holes = inv.astype(bool)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 4)
    for j in range(1, n2):
        if st2[j, cv2.CC_STAT_AREA] < 0.01 * m.sum():
            filled[lab2 == j] = 1
    a = cv2.GaussianBlur(filled.astype(np.float32), (3, 3), 0.8)
    ys, xs = np.nonzero(filled)
    return im, a, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

def gen_box(scene):
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    d = models.detect(pil, ["a tumbler", "a cup", "a mug"], threshold=0.15)
    return (d[0]["box"] if d else None), d[:3]

def place_h(scene, im, a, rb, gb):
    x0, y0, x1, y1 = rb; gx0, gy0, gx1, gy1 = gb
    s = (gy1 - gy0) / (y1 - y0)
    crop, ca = im[y0:y1, x0:x1], a[y0:y1, x0:x1]
    w, h = int((x1 - x0) * s), int((y1 - y0) * s)
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA); ca = cv2.resize(ca, (w, h))
    px, py = (gx0 + gx1) // 2 - w // 2, gy1 - h
    H, W = scene.shape[:2]
    fg = np.zeros_like(scene, np.float32); al = np.zeros((H, W), np.float32)
    sx0, sy0, sx1, sy1 = max(px, 0), max(py, 0), min(px + w, W), min(py + h, H)
    fg[sy0:sy1, sx0:sx1] = crop[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
    al[sy0:sy1, sx0:sx1] = ca[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
    return fg, al

def base_shadow(scene, al, dark=0.45):
    op = (al > 0.5).astype(np.uint8); ys, xs = np.nonzero(op)
    yb = ys.max(); row = np.nonzero(op[yb - 4:yb + 1].any(0))[0]
    cx, half = int(row.mean()), max(8, (row.max() - row.min()) // 2)
    sh = np.zeros(al.shape, np.float32)
    cv2.ellipse(sh, (cx, yb), (int(half * 1.15), max(4, half // 5)), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(3, half / 6)) * (1 - al)
    return (scene.astype(np.float32) * (1 - dark * sh[..., None])).astype(np.uint8)

def run(k, ri, prompt, seed=0, pad=30):
    scene = cv2.imread(f"{CASE}/outputs/{OUTS[k]}")
    gb, dets = gen_box(scene)
    m = np.zeros(scene.shape[:2], np.uint8)
    for d in dets:
        if d["score"] >= 0.15:
            x0, y0, x1, y1 = d["box"]; m[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = 1
    plate = C.erase(scene, m, prompt, "cup, tumbler, mug, bottle, object, product, text, watermark", seed=seed)
    im, a, rb = real_cut(ri)
    fg, al = place_h(plate, im, a, rb, gb)
    fg = C.tint(fg, al, plate, strength=0.4)
    comp = (base_shadow(plate, al).astype(np.float32) * (1 - al[..., None]) + fg * al[..., None]).astype(np.uint8)
    cv2.imwrite(f"{OUT}/ours_O{k}_U{ri}.png", comp); cv2.imwrite(f"{OUT}/plate_O{k}.jpg", plate)
    return gb

if __name__ == "__main__":
    ri = int(sys.argv[1])
    for k, prompt in [(22, "an empty seamless white studio background and floor, soft studio light, product photo background"),
                      (32, "an empty seamless warm beige studio background and floor, soft light, product photo background")]:
        t = time.time(); gb = run(k, ri, prompt)
        print(f"O{k} box={gb} {round(time.time()-t)}s", flush=True)

def plain_fill(scene, mask):
    """Plain studio background: refit the smooth background surface from the pixels outside
    the mask and paint it inside (no generative model, nothing can be invented)."""
    bg = K.fit_bg(scene.astype(np.float32), mask == 0)
    noise = np.random.default_rng(0).normal(0, 1.2, scene.shape).astype(np.float32)
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 4)[..., None]
    return np.clip((bg + noise) * soft + scene * (1 - soft), 0, 255).astype(np.uint8)

def drop_floor_glint(im, a, frac=0.07):
    """Remove the white floor reflection under the base that the key picked up."""
    ys = np.nonzero(a > 0.5)[0]; y0 = ys.max() - int(frac * (ys.max() - ys.min()))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 225) & (hsv[..., 1] < 18)
    out = a.copy(); band = np.zeros_like(white); band[y0:] = True
    out[band & white] = 0
    return out

def run_plain(k, ri, pad=30):
    scene = cv2.imread(f"{CASE}/outputs/{OUTS[k]}")
    gb, dets = gen_box(scene)
    m = np.zeros(scene.shape[:2], np.uint8)
    for d in dets:
        if d["score"] >= 0.15:
            x0, y0, x1, y1 = d["box"]; m[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = 1
    plate = plain_fill(scene, m)
    im, a, rb = real_cut(ri)
    a = drop_floor_glint(im, a)
    fg, al = place_h(plate, im, a, rb, gb)
    fg = C.tint(fg, al, plate, strength=0.4)
    comp = (base_shadow(plate, al).astype(np.float32) * (1 - al[..., None]) + fg * al[..., None]).astype(np.uint8)
    cv2.imwrite(f"{OUT}/ours_plain_O{k}_U{ri}.png", comp)
    return comp

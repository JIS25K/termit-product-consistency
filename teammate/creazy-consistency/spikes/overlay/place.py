from pathlib import Path
"""Spike: place the real shelf cutout where the generated shelf stands in a Creazy output."""
import os, sys, json, numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from refset import models

ROOT = str(Path(__file__).resolve().parents[2])
UPS = sorted(f for f in os.listdir(f"{ROOT}/data/cs/13/uploads") if not f.startswith("info-"))
OUTS = sorted(f for f in os.listdir(f"{ROOT}/data/cs/13/outputs") if f.endswith(".png"))

def real(i):
    im = cv2.imread(f"{ROOT}/data/cs/13/uploads/{UPS[i]}")
    a = np.load(f"{ROOT}/out/exp13/u{i}_alpha_final.npy")
    ys, xs = np.nonzero(a > 0.5)
    return im, a, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

def gen_box(scene_bgr):
    pil = Image.fromarray(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
    dets = models.detect(pil, ["a shelf", "a bookshelf"], threshold=0.15)
    return dets[0]["box"] if dets else None, dets[:3]

def place(scene, im, a, rbox, gbox, fit="width"):
    """Scale the real cutout so its width (or height) matches the generated shelf's box,
    bottom-aligned to the box's bottom (floor contact) and centred horizontally."""
    x0, y0, x1, y1 = rbox
    gx0, gy0, gx1, gy1 = gbox
    s = (gx1 - gx0) / (x1 - x0) if fit == "width" else (gy1 - gy0) / (y1 - y0)
    crop, ca = im[y0:y1, x0:x1], a[y0:y1, x0:x1]
    w, h = int((x1 - x0) * s), int((y1 - y0) * s)
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ca = cv2.resize(ca, (w, h), interpolation=cv2.INTER_LINEAR)
    cx = (gx0 + gx1) // 2
    px, py = cx - w // 2, gy1 - h
    H, W = scene.shape[:2]
    fg = np.zeros_like(scene, dtype=np.float32); al = np.zeros(scene.shape[:2], np.float32)
    sx0, sy0 = max(px, 0), max(py, 0); sx1, sy1 = min(px + w, W), min(py + h, H)
    fg[sy0:sy1, sx0:sx1] = crop[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
    al[sy0:sy1, sx0:sx1] = ca[sy0 - py:sy1 - py, sx0 - px:sx1 - px]
    return fg, al, (px, py, px + w, py + h)

if __name__ == "__main__":
    picks = [int(x) for x in sys.argv[1:]] or [0, 13]
    info = {}
    for k in picks:
        scene = cv2.imread(f"{ROOT}/data/cs/13/outputs/{OUTS[k]}")
        g, dets = gen_box(scene)
        info[k] = {"file": OUTS[k], "size": scene.shape[:2], "gen_box": g, "dets": [(d["query"], round(d["score"], 2), d["box"]) for d in dets]}
        print(k, info[k], flush=True)
        for ri in (0, 1):
            im, a, rb = real(ri)
            fg, al, pb = place(scene, im, a, rb, g)
            comp = scene.astype(np.float32) * (1 - al[..., None]) + fg * al[..., None]
            vis = comp.astype(np.uint8).copy()
            cv2.rectangle(vis, tuple(g[:2]), tuple(g[2:]), (0, 0, 255), 2)
            cv2.imwrite(f"{ROOT}/out/exp13/place_O{k}_u{ri}.jpg", vis)
    json.dump(info, open(f"{ROOT}/out/exp13/place_info.json", "w"), default=int)

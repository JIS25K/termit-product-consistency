"""CS 1 (tumbler) on a complex-background cut: LaMa erase with the product-shaped mask, real
cutout fitted to the generated product's width and base line, contact shadow."""
import os, sys, json, numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, exp1 as X, compose as C
from compose_lama import main_part
from refset import models

def gen_ext_body(part, d, thr=20):
    st = (d > thr) & (part > 0)
    ys, xs = np.nonzero(st)
    return int(np.percentile(xs, 0.3)), int(ys.min()), int(np.percentile(xs, 99.7)) + 1, int(np.percentile(ys, 99.7)) + 1

def trim_base_shadow(im, a, frac=0.08, bmin=4.0):
    """The studio shadow disc under the base is neutral grey; the body is cream (b* > 0).
    Drop neutral pixels in the bottom band of the cutout."""
    ys = np.nonzero(a > 0.5)[0]
    y0 = ys.max() - int(frac * (ys.max() - ys.min()))
    b = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32) - 128
    out = a.copy()
    # per column, cut from the first run of 3 neutral rows inside the bottom band (the body's
    # curved bottom edge); the warm-tinted shadow halo below it goes too
    Lc = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    neutral = (b < bmin)
    for c in np.nonzero((a[y0:] > 0.5).any(0))[0]:
        col = neutral[y0:, c]
        run = np.convolve(col.astype(np.int8), np.ones(3, np.int8), mode="valid")
        hit = np.nonzero(run >= 3)[0]
        if len(hit):
            out[y0 + hit[0]:, c] = 0
    # the disc is also wider than the body: in a taller band keep only the body's columns
    y1 = ys.max() - int(2 * frac * (ys.max() - ys.min()))
    xs = np.nonzero(a[y1 - 5] > 0.5)[0]
    if len(xs):
        runs = np.split(xs, np.nonzero(np.diff(xs) > 3)[0] + 1)
        body = max(runs, key=len)
        out[y1:, :max(body.min() - 3, 0)] = 0
        out[y1:, body.max() + 4:] = 0
    # keep the largest blob only
    op = (out > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(op)
    if n > 2:
        big = 1 + int(np.argmax(st[1:, 4]))
        out[(lab != big) & (lab > 0)] = 0
    return out

def place_w(scene, im, a, gen):
    """Width fit (body + handle), bottom-aligned, centred."""
    ys, xs = np.nonzero(a > 0.5)
    rb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    s = (gen[2] - gen[0]) / (rb[2] - rb[0])
    h = (rb[3] - rb[1]) * s
    gb = (gen[0], int(gen[3] - h), gen[2], gen[3])
    fg, al = X.place_h(scene, im, a, rb, gb)
    return fg, al, s

def run(k, ri, queries=("a tumbler", "a cup", "a mug"), side=512, tag="", detail=False):
    scene = cv2.imread(f"{X.CASE}/outputs/{X.OUTS[k]}")
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    dets = models.detect(pil, list(queries), threshold=0.15)[:3]
    box = E.box_mask(scene.shape, [d["box"] for d in dets], pad=(30, 30, 30, 30))
    M, core, d, bg = E.shape_mask(scene, box, hull=True)
    im, a, rb = X.real_cut(ri)
    cm = E.color_mask(scene, box, im, a)
    # join colour pixels that touch the product's own part, then grow like the shape mask
    n, lab = cv2.connectedComponents(cm | core)
    hit = np.unique(lab[core > 0]); hit = hit[hit > 0]
    obj = np.isin(lab, hit).astype(np.uint8)
    g = max(15, int(0.1 * scene.shape[1])) | 1
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((1, g), np.uint8))
    M = cv2.dilate(obj, np.ones((21, 21), np.uint8))
    color_px = int(cm.sum())
    # the product's reflection on a shiny surface under it is part of the product too:
    # cover a band under the base, and measure how strong the reflection was
    g0 = gen_ext_body(main_part(core), d)
    hh = g0[3] - g0[1]
    rows = core[max(g0[3] - 12, 0):g0[3]]
    cols = np.nonzero(rows.any(0))[0]
    bx0, bx1 = (int(cols.min()), int(cols.max())) if len(cols) else (g0[0], g0[2])
    ry0, ry1 = g0[3] - 5, min(g0[3] + int(0.15 * hh), scene.shape[0])
    # stop the band at the first strong scene structure under the base (a display, an edge)
    gr = E.grad(scene)[:, bx0:bx1] > 40
    for y in range(g0[3] + 8, ry1):
        if gr[y].mean() > 0.3:
            ry1 = y - 4
            break
    ext = np.zeros_like(M); ext[ry0:ry1, max(bx0 - 10, 0):bx1 + 10] = 1
    bstar = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32)
    wdt = bx1 - bx0
    side_px = np.concatenate([bstar[ry0 + 5:ry1, max(bx0 - wdt, 0):max(bx0 - 15, 1)].ravel(), bstar[ry0 + 5:ry1, bx1 + 15:bx1 + wdt].ravel()])
    refl = float(bstar[ry0 + 5:ry1, bx0:bx1].mean() - side_px.mean()) if side_px.size else 0.0
    if refl > 1.5:
        M = M | cv2.dilate(ext, np.ones((11, 11), np.uint8))
    plate = E.lama_lowres(scene, M, side)
    plate = E.flat_match(scene, plate, M)[0]
    chk = E.residuals(plate, M); left = E.leftover(scene, plate, M)
    gen = gen_ext_body(main_part(core), d)
    a = trim_base_shadow(im, a)  # (drop_floor_glint cut the bright lower body; not used)
    fg, al, s = place_w(plate, im, a, gen)
    fg = C.tint(fg, al, plate, strength=0.4)
    body = (cm > 0) & (main_part(core) > 0)
    base, drop = E.transfer_shadow(scene, bg, main_part(core), body, plate, al, area=M)
    base = base.astype(np.float32)
    if refl > 1.5:  # faint mirrored copy under the base, fading out
        ys = np.nonzero(al > 0.5)[0]; yb = int(ys.max())
        n = int(0.15 * (ys.max() - ys.min()))
        ra = np.zeros_like(al); rf = np.zeros_like(fg)
        for i in range(1, min(n, al.shape[0] - yb)):
            ra[yb + i] = al[yb - i] * (1 - i / n)
            rf[yb + i] = fg[yb - i]
        ra = cv2.GaussianBlur(ra, (0, 0), 2.5) * min(0.35, max(0.1, refl / 10)) * (1 - al)
        rf = cv2.GaussianBlur(rf, (0, 0), 2.5)
        base = base * (1 - ra[..., None]) + rf * ra[..., None]
    dinfo = None
    if detail:  # crispness and grain of the scene (detail.py); the seller photo is smoother
        import detail as D
        fg, al, dinfo = D.match_detail(fg, al, base, scene, main_part(core))
    comp = (base * (1 - al[..., None]) + fg * al[..., None]).astype(np.uint8)
    cv2.imwrite(f"{X.OUT}/lama_plate_O{k}.png", plate)
    cv2.imwrite(f"{X.OUT}/ours_lama_O{k}_U{ri}{tag}.png", comp)
    meta = dict(k=k, ri=ri, contact_drop=round(drop, 1), color_px=color_px, reflection=round(refl, 2), dets=[(x["query"], round(x["score"], 2), x["box"]) for x in dets], gen_ext=gen,
                scale=round(s, 3), mask_px=int(M.sum()), hot_frac=chk["hot_frac"], blobs=len(chk["blobs"]), leftover=left, detail=dinfo)
    return scene, plate, comp, M, meta

def run_plain(k, ri, tag="", detail=False):
    """Plain studio background: polynomial fill instead of LaMa, same cutout and shadow steps."""
    scene = cv2.imread(f"{X.CASE}/outputs/{X.OUTS[k]}")
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    dets = models.detect(pil, ["a tumbler", "a cup", "a mug"], threshold=0.15)[:3]
    gb = dets[0]["box"]
    m = E.box_mask(scene.shape, [d["box"] for d in dets], pad=(30, 30, 30, 30))
    plate = X.plain_fill(scene, m)
    _, core, d, bg = E.shape_mask(scene, m, hull=True)
    im, a, rb = X.real_cut(ri)
    cm = E.color_mask(scene, m, im, a)
    a = trim_base_shadow(im, a)
    ys, xs = np.nonzero(a > 0.5); rb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    fg, al = X.place_h(plate, im, a, rb, gb)
    fg = C.tint(fg, al, plate, strength=0.4)
    body = (cm > 0) & (main_part(core) > 0)
    base, drop = E.transfer_shadow(scene, plate, main_part(core), body, plate, al, area=m)
    base = base.astype(np.float32)
    dinfo = None
    if detail:  # crispness and grain of the scene (detail.py)
        import detail as D
        fg, al, dinfo = D.match_detail(fg, al, base, scene, main_part(core))
    comp = (base * (1 - al[..., None]) + fg * al[..., None]).astype(np.uint8)
    cv2.imwrite(f"{X.OUT}/ours_plain3_O{k}_U{ri}{tag}.png", comp)
    return scene, plate, comp, dict(k=k, ri=ri, contact_drop=round(drop, 1), box=gb, detail=dinfo)

if __name__ == "__main__":
    k, ri = int(sys.argv[1]), int(sys.argv[2])
    scene, plate, comp, M, meta = run(k, ri)
    print(json.dumps(meta, default=int))
    v = scene.copy(); v[M > 0] = (0.5 * v[M > 0] + [0, 0, 127]).astype(np.uint8)
    S = X.OUT
    cv2.imwrite(f"{S}/cs1_O{k}_U{ri}.jpg", np.hstack([cv2.resize(x, (420, 420)) for x in (scene, v, plate, comp)]))

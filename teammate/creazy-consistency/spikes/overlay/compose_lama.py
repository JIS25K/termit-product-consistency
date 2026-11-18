"""Same scene, real product: LaMa erase with a product-shaped mask, fit the real cutout to the
generated product's extents (outer legs, feet line), tint, contact shadows, optional props."""
import sys, os, json, numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, place as P, compose as C
sys.path.insert(0, P.ROOT)
from refset import models

def main_part(core):
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    i = 1 + int(np.argmax(st[1:, 4]))
    return (lab == i).astype(np.uint8)

def extents(m):
    ys, xs = np.nonzero(m)
    return int(np.percentile(xs, 0.2)), int(ys.min()), int(np.percentile(xs, 99.8)) + 1, int(np.percentile(ys, 99.7)) + 1

def gen_extents(part, d, thr=20):
    """Generated product extents from strong differences only (chrome, wood edges), so soft
    floor shadow does not push the feet line down. Feet = lowest strong row near each end."""
    st = (d > thr) & (part > 0)
    ys, xs = np.nonzero(st)
    x0, x1 = int(np.percentile(xs, 0.3)), int(np.percentile(xs, 99.7)) + 1
    feet = []
    for a, b in ((x0, x0 + 60), (x1 - 60, x1)):
        yy = np.nonzero(st[:, a:b].sum(1) >= 2)[0]
        if len(yy): feet.append(int(yy.max()))
    return x0, int(ys.min()), x1, int(np.mean(feet)) + 1

def fit_place(scene, im, a, gen_ext):
    rx0, ry0, rx1, ry1 = extents(a > 0.5)
    gx0, gy0, gx1, gy1 = gen_ext
    s = (gx1 - gx0) / (rx1 - rx0)
    gb = (int(round(gx0 + 0 * s)), 0, 0, 0)
    # reuse place(): give it a target box whose width/bottom/centre are the generated extents
    return P.place(scene, im, a, (rx0, ry0, rx1, ry1), (gx0, gy1 - int((ry1 - ry0) * s), gx1, gy1)), s

def props(scene, core, bg, de=12.0):
    """Scene props standing on the generated product (speaker, plant): OWLv2 boxes, and inside
    them the strong-difference parts that are not the product's own main part."""
    ls, main, strong = E.shape_mask.parts
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    dets = models.detect(pil, ["a speaker", "a potted plant", "a vase", "a book", "a decorative object"], threshold=0.2)
    other = ((strong > 0) & (ls != main)).astype(np.uint8)
    pm = np.zeros(core.shape, np.uint8); boxes = []
    for x in dets:
        x0, y0, x1, y1 = x["box"]
        if (x1 - x0) * (y1 - y0) > 0.25 * core.size:
            continue
        sub = cv2.morphologyEx(other[y0:y1, x0:x1], cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ff = sub.copy(); h, w = ff.shape
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        sub = sub | (1 - ff)
        # cut away the generated board under the prop: per column, rows at or below the
        # first main-part pixel in the lower half of the box
        mp = (ls[y0:y1, x0:x1] == main)
        h2 = (y1 - y0) // 2
        for c in range(x1 - x0):
            r = np.nonzero(mp[h2:, c])[0]
            if len(r):
                sub[h2 + r[0]:, c] = 0
        pm[y0:y1, x0:x1] |= sub
        boxes.append((x["query"], round(x["score"], 2), x["box"]))
    return cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 0.8), boxes

def clean_under(a, run=21, wide=0.7, im=None, dark=170):
    """Below the lowest board keep only thin legs: the cutout also caught the studio floor's
    shadow strip under the board, which would show as pale blocks around the feet."""
    op = (a > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(op)
    w = xs.max() - xs.min() + 1
    rows = np.nonzero(op.sum(1) > wide * w)[0]
    if im is not None:
        # a board is wood-coloured; a pale floor strip the cutout caught is also wide but is not
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        wood = op & (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 35)
        wrows = np.nonzero(wood.sum(1) > 0.6 * wide * w)[0]
        if len(wrows):
            rows = rows[rows <= wrows.max() + 3]
    if not len(rows):
        return a
    runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    yb = int(runs[-1][-1]) + 2
    thin = op & (1 - cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)))
    if im is not None:
        # legs are chrome: each leg column has dark reflections somewhere under the board;
        # the floor-shadow strip is pale and even
        L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        Lu = np.where(op[yb + 8:] > 0, L[yb + 8:], 255)  # skip the dark underside edge
        cols = (Lu.min(0) < dark).astype(np.uint8)
        cols = cv2.dilate(cols[None], np.ones((1, 5), np.uint8))[0]
        thin = thin & cols[None, :]
    out = a.copy()
    out[yb:] *= thin[yb:]
    return out

def clean_under_cols(a, im, dark=170):
    """View-agnostic version of clean_under: per column, below the lowest wood-coloured pixel
    (the lowest board's edge, slanted in an oblique view) keep only thin chrome legs. Removes the
    pale studio floor the cutout caught under the board."""
    op = (a > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(op)
    w = xs.max() - xs.min() + 1
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    wood = op & (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 45)
    run = max(9, int(0.03 * w)) | 1
    thin = op & (1 - cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)))
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
    out = a.copy()
    for c in np.nonzero(wood.any(0))[0]:
        yb = np.nonzero(wood[:, c])[0].max() + 1
        if yb >= a.shape[0]:
            continue
        colkeep = thin[yb:, c] & (L[yb:, c].min() < dark if op[yb:, c].any() else False)
        out[yb:, c] *= colkeep
    return out

def legs(al, run=41):
    """Thin vertical parts of the placed real product (legs): opaque pixels that are not part
    of a wide horizontal run. They are drawn again over the props (front legs occlude them)."""
    op = (al > 0.5).astype(np.uint8)
    wide = cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    return al * ((op & (1 - wide)) > 0)

def board_top(al, x0, x1, y_feet, frac=(0.80, 0.97)):
    """y of the lowest board's top edge in the placed real shelf: the first opaque row, scanning up
    from the feet, of a wide horizontal run between the props' x-range."""
    band = al[:, x0:x1] > 0.5
    rows = band.mean(1)
    ys = np.nonzero(rows > 0.9)[0]
    ys = ys[ys < y_feet]
    if not len(ys):
        return None
    # the lowest wide run of rows
    runs = np.split(ys, np.nonzero(np.diff(ys) > 1)[0] + 1)
    return int(runs[-1][0])

def board_run_end(al, x0, x1, bt):
    rows = (al[:, x0:x1] > 0.5).mean(1)
    y = bt
    while y + 1 < len(rows) and rows[y + 1] > 0.9:
        y += 1
    return y

def run(k=0, ri=0, side=512, with_props=False, view=None):
    scene = cv2.imread(f"{P.ROOT}/data/cs/13/outputs/{P.OUTS[k]}")
    info = json.load(open(f"{P.ROOT}/out/exp13/place_info.json"))
    if str(k) in info:
        boxes = [d[2] for d in info[str(k)]["dets"]]
    else:
        _, dets = P.gen_box(scene)
        boxes = [x["box"] for x in dets]
    box = E.box_mask(scene.shape, boxes)
    M, core, d, bg = E.shape_mask(scene, box)
    plate = E.lama_lowres(scene, M, side)
    plate = E.flat_match(scene, plate, M)[0]
    chk = E.residuals(plate, M); left = E.leftover(scene, plate, M)
    gen = gen_extents(main_part(core), d)
    if view:  # a view picked by select_view (full upload or a panel of the multi-view upload)
        import select_view as V
        im, a = V.real_view(view)
        im = np.ascontiguousarray(im)
    else:
        im, a, _ = P.real(ri)
    if not view or view in ("u0", "u1"):
        a = clean_under(a, im=im)
    else:  # panels of the multi-view upload, possibly oblique
        a = clean_under_cols(a, im)
    (fg, al, pb), s = fit_place(plate, im, a, gen)
    if s > 1.5:  # a small panel blown up: unsharp mask to win back some edge contrast
        bl = cv2.GaussianBlur(fg, (0, 0), 1.2 * s / 2)
        fg = np.clip(fg + 0.6 * (fg - bl), 0, 255)
    fg = C.tint(fg, al, plate)
    base = C.shadows(plate, al)
    comp = (base.astype(np.float32) * (1 - al[..., None]) + fg * al[..., None])
    meta = dict(k=k, ri=ri, dropped_px=E.shape_mask.dropped, gen_ext=gen, placed=pb, scale=round(s, 3), mask_px=int(M.sum()),
                hot_frac=chk["hot_frac"], blobs=len(chk["blobs"]), leftover=left)
    if with_props:
        pm, pboxes = props(scene, core, bg)
        ys, xs = np.nonzero(pm > 0.5)
        if len(ys):
            gy_bottom = int(np.percentile(ys, 99.5))
            bt = board_top(al, int(xs.min()), int(xs.max()), pb[3])
            dy = (bt + 0.45 * (board_run_end(al, int(xs.min()), int(xs.max()), bt) - bt) - gy_bottom) if bt else 0
            dy = int(round(dy))
            M2 = np.float32([[1, 0, 0], [0, 1, dy]])
            pm2 = cv2.warpAffine(pm, M2, pm.shape[::-1]); sc2 = cv2.warpAffine(scene, M2, pm.shape[::-1]).astype(np.float32)
            comp = comp * (1 - pm2[..., None]) + sc2 * pm2[..., None]
            lg = legs(al)[..., None]
            comp = comp * (1 - lg) + fg * lg
            meta.update(props=pboxes, props_dy=dy, board_top=bt)
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    tag = f"O{k}_" + (view if view else f"u{ri}") + ("_props" if with_props else "")
    cv2.imwrite(f"{P.ROOT}/out/exp13/lama_plate_O{k}.png", plate)
    cv2.imwrite(f"{P.ROOT}/out/exp13/ours4_{tag}.png", comp)
    return scene, plate, comp, M, meta

if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    outs = []
    for ri, wp in ((0, False), (0, True), (1, False)):
        scene, plate, comp, M, meta = run(k, ri, with_props=wp)
        print(json.dumps(meta, default=int), flush=True)
        outs.append(comp)
    S = f"{P.ROOT}/out/exp13"
    t = [cv2.resize(x, (480, int(480 * x.shape[0] / x.shape[1]))) for x in [scene, plate] + outs[:2]]
    cv2.imwrite(f"{S}/ours4_O{k}.jpg", np.vstack([np.hstack(t[:2]), np.hstack(t[2:])]))
    t2 = cv2.resize(outs[2], (480, int(480 * outs[2].shape[0] / outs[2].shape[1])))
    cv2.imwrite(f"{S}/ours4_O{k}_u1.jpg", t2)

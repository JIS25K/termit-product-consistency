"""Erase the generated product with LaMa at low resolution, then check the filled hole for
structures that were not there before (list-free: edge energy vs the ring around the hole)."""
import numpy as np, cv2
import lama

def box_mask(shape, boxes, pad=(40, 30, 50, 45)):
    """Union of detection boxes, padded left, top, right, bottom (bottom covers feet and shadow)."""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    l, t, r, b = pad
    for x0, y0, x1, y1 in boxes:
        m[max(y0 - t, 0):min(y1 + b, H), max(x0 - l, 0):min(x1 + r, W)] = 1
    return m

def lama_lowres(bgr, mask, side=512):
    s = min(1.0, side / max(bgr.shape[:2]))
    sb = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sm = cv2.resize(mask, (sb.shape[1], sb.shape[0]), interpolation=cv2.INTER_NEAREST)
    o, _ = lama.inpaint(sb, sm)
    up = cv2.resize(o, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)[..., None]
    return (up * soft + bgr * (1 - soft)).astype(np.uint8)

def grad(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

def residuals(filled, mask, ring_px=40, q=99, min_area=120):
    """Strong-edge blobs inside the hole. Threshold = q-th percentile of edge strength in the
    ring around the hole (real scene structure such as the baseboard and plank seams sets it)."""
    g = grad(filled)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))  # skip the blend seam
    ring = (cv2.dilate(mask, np.ones((ring_px, ring_px), np.uint8)) > 0) & (mask == 0)
    thr = max(np.percentile(g[ring], q), 8.0) if ring.any() else 8.0  # hole may reach the image edges
    hot = ((g > thr) & (inner > 0)).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot)
    blobs = [dict(box=(int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])),
                  area=int(st[i, 4])) for i in range(1, n) if st[i, 4] >= min_area]
    frac = float(hot.sum()) / max(int(inner.sum()), 1)
    return dict(thr=float(thr), hot_frac=round(frac, 4), blobs=blobs, hot=hot)

def erase(bgr, mask, side=512, rounds=3, grow=25):
    """Erase, check, and if a blob reaches the hole edge, grow the mask over it and retry
    (a leftover foot or leg end outside the mask is what LaMa extends back into the hole)."""
    log = []
    m = mask.copy()
    for r in range(rounds):
        out = lama_lowres(bgr, m, side)
        res = residuals(out, m)
        log.append(dict(round=r, mask_px=int(m.sum()), thr=res["thr"], hot_frac=res["hot_frac"], blobs=res["blobs"]))
        edge = (m > 0) & (cv2.erode(m, np.ones((31, 31), np.uint8)) == 0)
        touch = [b for b in res["blobs"] if edge[b["box"][1]:b["box"][3], b["box"][0]:b["box"][2]].any()]
        if not touch:
            break
        for b in touch:
            x0, y0, x1, y1 = b["box"]
            H, W = m.shape
            m[max(y0 - grow, 0):min(y1 + grow, H), max(x0 - grow, 0):min(x1 + grow, W)] = 1
    return out, m, log

def shape_mask(bgr, box, side=384, de=12.0, weak_de=6.0, grow=21, min_area=200, hull=False):
    """Product-shaped mask without a segmentation model: LaMa fills the whole box once to
    estimate the background, and pixels that differ from that estimate (Lab dE) are the object
    (shelf, props, their contact shadows). Real background structure the box cut through,
    such as a picture-frame corner, is reconstructed by LaMa and so stays out of the mask."""
    bg = lama_lowres(bgr, box, side)
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.sqrt(((a - b) ** 2).sum(-1)) * (box > 0)
    # hysteresis: weak differences (clear acrylic, soft shadow) count only when they touch
    # a strong one (chrome, wood edge, props)
    strong = cv2.morphologyEx((d > de).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    weak = cv2.morphologyEx((d > weak_de).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) | strong
    # seeds: the largest strong part (the product) and strong parts within 40 px of it
    # (props standing on it). Small parts the box cut through (picture-frame corner,
    # armchair edge: touching the box edge with strong edges just outside) are not seeds.
    edge = (box > 0) & (cv2.erode(box, np.ones((9, 9), np.uint8)) == 0)
    outside = (cv2.dilate(box, np.ones((13, 13), np.uint8)) > 0) & (box == 0) & (grad(bgr) > 40)
    ns, ls, sts, _ = cv2.connectedComponentsWithStats(strong)
    main = 1 + int(np.argmax(sts[1:, 4]))
    dist = cv2.distanceTransform((ls != main).astype(np.uint8), cv2.DIST_L2, 5)
    seeds = np.zeros_like(strong); dropped = 0
    for i in range(1, ns):
        comp = ls == i
        near = cv2.dilate((comp & edge).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        cut = sts[i, 4] < 5000 and (near & outside).sum() > 5
        if i == main or (not cut and dist[comp].min() <= 40):
            seeds[comp] = 1
        else:
            dropped += int(sts[i, 4])
    self_drop = dropped
    shape_mask.parts = (ls, main, strong)
    # weak differences (clear acrylic, soft shadow) join only within 30 px of a seed
    obj = seeds.copy()
    for _ in range(30):
        obj = cv2.dilate(obj, np.ones((3, 3), np.uint8)) & weak
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(obj)
    keep = np.zeros_like(obj)
    for i in range(1, n):
        if st[i, 4] >= min_area:
            keep[lab == i] = 1
    ff = keep.copy(); h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    keep = keep | (1 - ff)  # fill enclosed holes
    if hull:  # compact products (tumbler, bottle): close horizontal gaps up to 10% of the
        # width, so a low-contrast body side is covered but the open loop of a handle (and the
        # background edge seen through it) stays out of the mask
        g = max(15, int(0.1 * bgr.shape[1])) | 1
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((1, g), np.uint8))
    shape_mask.dropped = self_drop
    return cv2.dilate(keep, np.ones((grow, grow), np.uint8)), keep, d, bg

def leftover(orig, filled, mask):
    """Share of the original's strong edges inside the hole that survive in the fill.
    The hole was the product, so surviving edges are leftover product structure."""
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    go, gf = grad(orig), grad(filled)
    so = (go > 40) & inner
    if so.sum() == 0:
        return 0.0
    return round(float(((gf > 40) & so).sum()) / float(so.sum()), 4)


def color_mask(bgr, box, real_im, real_a, k=8, ring=41):
    """Product-aware mask: pixels in the box whose Lab colour is closer to the real product's
    colours (k-means of the real cutout) than to the scene colours just outside the box.
    Uses what we know about the product, so a cream body on a light wall is still found."""
    def lab(x):
        return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    def centers(px):
        px = px[np.random.default_rng(0).choice(len(px), min(len(px), 20000), replace=False)]
        _, _, c = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        return c
    pr = centers(lab(real_im)[real_a > 0.9])
    rg = (cv2.dilate(box, np.ones((ring, ring), np.uint8)) > 0) & (box == 0)
    bc = centers(lab(bgr)[rg])
    L = lab(bgr)
    dp = np.min(((L[..., None, :] - pr[None, None]) ** 2).sum(-1), -1)
    db = np.min(((L[..., None, :] - bc[None, None]) ** 2).sum(-1), -1)
    m = ((dp < db) & (np.sqrt(dp) < 12) & (box > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m



def flat_match(src, filled, mask, scale=8, win=15, tau=6.0, blur=24):
    """Pull the low-frequency colour of the filled hole toward a smooth membrane spanned by the
    pixels around it, but only where those surroundings are flat (a plain wall). LaMa sometimes
    leaves a faint tint of the erased product in a large hole; texture inside the fill is kept.
    Membrane and flatness are both carried into the hole by Telea inpainting at 1/scale size."""
    H, W = mask.shape
    m = (mask > 0).astype(np.uint8)
    x = src.astype(np.float32)
    mean = cv2.blur(x, (win, win)); sq = cv2.blur(x * x, (win, win))
    spread = np.sqrt(np.clip(sq - mean * mean, 0, None).mean(-1))
    small = (W // scale, H // scale)
    ms = cv2.dilate(cv2.resize(m, small, interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    xs = cv2.resize(src, small, interpolation=cv2.INTER_AREA)
    sp = np.clip(cv2.resize(spread, small, interpolation=cv2.INTER_AREA) * 8, 0, 255).astype(np.uint8)
    B = cv2.resize(cv2.inpaint(xs, ms, 5, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    spi = cv2.resize(cv2.inpaint(sp, ms, 5, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 8
    w = np.clip(1 - spi / tau, 0, 1) * m
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 6)[..., None]
    F = filled.astype(np.float32)
    g = lambda a: cv2.GaussianBlur(a, (0, 0), blur)
    out = F + w * (g(B) - g(F))
    return np.clip(out, 0, 255).astype(np.uint8), w[..., 0]


def contact_strength(scene, bg, part, band=(-3, 8), q=75):
    """How dark the generated product's contact shadow is: L drop (background estimate minus
    scene) in a thin band at the generated base, inside the base's columns. Used to draw the
    real product's shadow as dark as the scene's own lighting made it."""
    ys, xs = np.nonzero(part)
    yb = int(np.percentile(ys, 99.5))
    cols = np.nonzero(part[max(yb - 10, 0):yb + 1].any(0))[0]
    if not len(cols):
        return 0.0, yb
    Ls = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Lb = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    y0, y1 = max(yb + band[0], 0), min(yb + band[1], scene.shape[0])
    reg = (Lb - Ls)[y0:y1, cols.min():cols.max() + 1]
    reg = reg[part[y0:y1, cols.min():cols.max() + 1] == 0]
    return float(np.percentile(reg, q)) if reg.size else 0.0, yb


def contact_shadow(plate, al, drop, ambient=0.35):
    """Two-layer shadow under a standing product: a tight contact ellipse whose centre lowers L
    by `drop` (measured from the generated scene) and a wider, softer ambient layer."""
    op = (al > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(op)
    yb = ys.max()
    row = np.nonzero(op[yb - 4:yb + 1].any(0))[0]
    cx, half = int(row.mean()), max(8, (row.max() - row.min()) // 2)
    c = np.zeros(al.shape, np.float32); a = np.zeros(al.shape, np.float32)
    cv2.ellipse(c, (cx, yb), (int(half * 1.02), max(3, half // 8)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(a, (cx, yb + max(2, half // 12)), (int(half * 1.5), max(6, half // 4)), 0, 0, 360, 1.0, -1)
    c = cv2.GaussianBlur(c, (0, 0), max(2, half / 14)); a = cv2.GaussianBlur(a, (0, 0), max(6, half / 4))
    sh = np.clip(c + ambient * a, 0, 1) * (1 - al)
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = drop / max(c.max(), 1e-6)
    lab[..., 0] = np.clip(lab[..., 0] - k * sh * (255 / 100) * (100 / 255), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def base_line(mask):
    """Lowest point of a standing product's base, from the per-column bottoms of the central 60%
    of its base columns (stray pixels at the sides, such as leftover studio floor, do not count).
    Returns (y, x0, x1) with x0..x1 the base columns."""
    ys, xs = np.nonzero(mask)
    yb = ys.max()
    band = mask[max(yb - int(0.08 * (yb - ys.min())), 0):yb + 1]
    cols = np.nonzero(band.any(0))[0]
    x0, x1 = cols.min(), cols.max()
    w = x1 - x0
    bottoms = [np.nonzero(mask[:, c])[0].max() for c in range(int(x0 + 0.2 * w), int(x1 - 0.2 * w) + 1) if mask[:, c].any()]
    return int(np.percentile(bottoms, 90)), int(x0), int(x1)


def transfer_shadow(scene, bg, part, body, plate, al, below=0.12, area=None, side=0.4):
    """Copy the generated product's own contact shadow (L drop against the background estimate,
    in the product region but outside its body colour) and move it under the real product,
    scaled from the generated base width to the real base width. The scene's lighting made that
    shadow, so the real product gets the same softness and darkness."""
    # shadow-like = same chromaticity as the background estimate, only darker; the body is the
    # rest of the product part (a colour model of the product mistakes shaded body for shadow)
    ls = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    shadowlike = (np.abs(ls[..., 1] - lb[..., 1]) < 3) & (np.abs(ls[..., 2] - lb[..., 2]) < 4) & (ls[..., 0] < lb[..., 0] - 2)
    body = (part > 0) & ~shadowlike
    body = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    by, bx = np.nonzero(body)
    if not len(by):
        return plate, 0.0
    h = by.max() - by.min()
    gyb, gx0, gx1 = base_line(body)
    Ls = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Lb = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    # where the generated shadow can be: the erased area (it was removed with the product),
    # within the base's columns widened by `side` of the base width on each side
    zone = (area > 0) if area is not None else (part > 0)
    region = zone & shadowlike & (cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    region[: max(gyb - int(0.05 * h), 0)] = False
    region[min(gyb + int(below * h), scene.shape[0]):] = False
    bw = gx1 - gx0
    region[:, :max(int(gx0 - side * bw), 0)] = False
    region[:, int(gx1 + side * bw) + 1:] = False
    sh = np.clip(Lb - Ls, 0, None) * region
    # keep only shadow that touches the base (drop lines such as a backdrop horizon or a
    # handle edge that also read darker than the background estimate)
    n, lab = cv2.connectedComponents((sh > 3).astype(np.uint8))
    touch = np.unique(lab[max(gyb - 3, 0):gyb + 5, gx0:gx1 + 1]); touch = touch[touch > 0]
    keep = cv2.dilate(np.isin(lab, touch).astype(np.uint8), np.ones((7, 7), np.uint8))
    sh = cv2.GaussianBlur(sh * keep, (0, 0), 1.5)
    ryb, rx0, rx1 = base_line(al > 0.5)
    s = (rx1 - rx0) / max(gx1 - gx0, 1)
    A = np.float32([[s, 0, rx0 - gx0 * s], [0, s, ryb - gyb * s]])
    sh2 = cv2.warpAffine(sh, A, (al.shape[1], al.shape[0])) * (1 - al)
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] - sh2, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), float(sh.max())

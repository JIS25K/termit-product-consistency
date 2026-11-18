"""Spike: cut the real shelf out of a plain-background product photo, keeping the clear
acrylic see-through. Background = smooth quadratic surface fitted to pixels outside the
product; alpha = colour distance from that surface; small holes in opaque parts filled."""
import sys, numpy as np, cv2

def fit_bg(im, keep):
    h, w = keep.shape
    ys, xs = np.nonzero(keep)
    sel = np.random.default_rng(0).choice(len(ys), min(40000, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    X = np.stack([np.ones_like(xs), xs / w, ys / h, (xs / w) ** 2, (ys / h) ** 2, xs * ys / (w * h)], 1).astype(np.float64)
    gy, gx = np.mgrid[0:h, 0:w]
    G = np.stack([np.ones_like(gx), gx / w, gy / h, (gx / w) ** 2, (gy / h) ** 2, gx * gy / (w * h)], -1).reshape(-1, 6)
    bg = np.zeros_like(im)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(X, im[ys, xs, c].astype(np.float64), rcond=None)
        bg[..., c] = (G @ coef).reshape(h, w)
    return bg

def cutout(path, lo=10, hi=40):
    im = cv2.imread(path).astype(np.float32)
    # rough product mask: strong deviation from a heavy blur, dilated generously
    rough = np.linalg.norm(im - cv2.GaussianBlur(im, (0, 0), 25), axis=2) > 12
    rough = cv2.dilate(rough.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    bg = fit_bg(im, ~rough)
    for _ in range(2):  # refit with the product excluded more precisely
        d = np.linalg.norm(im - bg, axis=2)
        prod = cv2.dilate((d > lo).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        bg = fit_bg(im, ~prod)
    d = np.linalg.norm(im - bg, axis=2)
    alpha = np.clip((d - lo) / (hi - lo), 0, 1)
    # fill small holes inside opaque parts (wood grain, chrome highlights)
    hard = (alpha > 0.5).astype(np.uint8)
    inv = 1 - hard
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 1500:
            alpha[lab == i] = 1.0
    return im, bg, alpha

if __name__ == "__main__":
    im, bg, a = cutout(sys.argv[1])
    cv2.imwrite(sys.argv[2], (a * 255).astype(np.uint8))
    np.save(sys.argv[2].replace(".png", "_bg.npy"), bg)

def clean(alpha, top_frac=0.3):
    """Opaque frame = largest strong component; the clear acrylic panel keeps its keyed alpha
    (edges and caps visible, see-through inside); faint haze elsewhere (wall/floor seams,
    floor shadow) is dropped."""
    strong = (alpha > 0.75).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    frame = (lab == big)
    x, y, w, h = stats[big, :4]
    out = np.where(frame, 1.0, 0.0).astype(np.float32)
    # acrylic: the band at the top of the product where faint alpha sits between the caps
    band = np.zeros_like(alpha, bool)
    y1 = y + int(h * top_frac)
    band[y:y1, x:x + w] = True
    acr = band & (alpha > 0.15)
    if acr.any():
        ys, xs = np.nonzero(acr)
        ay0, ay1 = ys.min(), ys.max()
        rows = np.nonzero((alpha[ay0:ay1 + 1] > 0.3).sum(1) > 40)[0]
        if len(rows):
            ay0, ay1 = ay0 + rows.min(), ay0 + rows.max()
        ax0, ax1 = xs.min(), xs.max()
        box = (slice(max(ay0 - 3, 0), ay1 + 4), slice(max(ax0 - 3, 0), ax1 + 4))
        out[box] = np.maximum(out[box], alpha[box])
    out = cv2.GaussianBlur(out, (3, 3), 0.8)
    return out, (x, y, w, h)

def drop_wall(im, a, s_max=18, v_min=185):
    """Remove wall/floor patches the smooth background model missed (bright, grey-white),
    except inside the chrome poles (vertical structures)."""
    hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < s_max) & (hsv[..., 2] > v_min)
    opaque = (a > 0.5).astype(np.uint8)
    poles = cv2.morphologyEx(opaque, cv2.MORPH_OPEN, np.ones((70, 1), np.uint8))
    poles = cv2.dilate(poles, np.ones((1, 5), np.uint8)) > 0
    out = a.copy()
    kill = white & ~poles
    kill = cv2.morphologyEx(kill.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    out[kill] = 0
    # re-keep the largest component only (drops islands left behind)
    n, lab, st, _ = cv2.connectedComponentsWithStats((out > 0.5).astype(np.uint8), 8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        islands = (out > 0.5) & (lab != big)
        out[islands] = 0
    return out

def acrylic_box(alpha, bb, top_frac=0.3):
    x, y, w, h = bb
    band = np.zeros_like(alpha, bool); band[y:y + int(h * top_frac), x:x + w] = True
    acr = band & (alpha > 0.15)
    ys, xs = np.nonzero(acr)
    ay0, ay1 = ys.min(), ys.max()
    rows = np.nonzero((alpha[ay0:ay1 + 1] > 0.3).sum(1) > 40)[0]
    if len(rows):
        ay0, ay1 = ay0 + rows.min(), ay0 + rows.max()
    return max(ay0 - 4, 0), ay1 + 5, max(xs.min() - 4, 0), xs.max() + 5

def final_alpha(im, raw):
    """raw = keyed alpha from cutout(). Frame from clean()+drop_wall(), acrylic panel kept
    from the keyed alpha, floor seen under the lowest board removed except the legs."""
    a, bb = clean(raw)
    y0, y1, x0, x1 = acrylic_box(raw, bb)
    frame = drop_wall(im, a)
    # below the lowest wide board keep only the legs (vertical structures)
    opaque = frame > 0.5
    x, y, w, h = bb
    wide = np.nonzero(opaque[:, x:x + w].sum(1) > 0.5 * w)[0]
    if len(wide):
        bottom = wide.max()
        legs = cv2.morphologyEx(opaque.astype(np.uint8), cv2.MORPH_OPEN, np.ones((25, 1), np.uint8)) > 0
        legs = cv2.dilate(legs.astype(np.uint8), np.ones((1, 3), np.uint8)) > 0
        below = np.zeros_like(opaque); below[bottom + 1:] = True
        frame[below & ~legs] = 0
    frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], raw[y0:y1, x0:x1])
    return cv2.GaussianBlur(frame, (3, 3), 0.8), (y0, y1, x0, x1)

def trim_floor(im, a):
    """Under the lowest wood board keep only the legs: find the last row with many
    wood-coloured pixels, then drop grey pixels below it that are not leg columns."""
    hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2HSV)
    wood = (hsv[..., 1] > 35) & (hsv[..., 0] > 8) & (hsv[..., 0] < 30) & (a > 0.5)
    rows = np.nonzero(wood.sum(1) > 0.25 * wood.sum(1).max())[0]
    bottom = rows.max()
    opaque = (a > 0.5).astype(np.uint8)
    legs = cv2.morphologyEx(opaque[bottom + 1:], cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    # legs: columns that are opaque in most rows just above the board top too (pole columns)
    above = opaque[max(bottom - 200, 0):bottom - 60]
    pole_cols = above.mean(0) > 0.6
    pole_cols = cv2.dilate(pole_cols.astype(np.uint8)[None], np.ones((1, 3), np.uint8))[0] > 0
    out = a.copy()
    sub = out[bottom + 1:]
    sub[:, ~pole_cols] = 0
    return out, bottom

def board_edge(im, a, guess, span=70):
    """Row of the lowest board's bottom edge: the strongest horizontal edge just above the
    wood-tinted floor shadow."""
    g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    cols = (a[guess - 30] > 0.5)
    lo, hi = guess - span, guess + 3
    prof = [gy[r, cols].mean() for r in range(lo, hi)]
    return lo + int(np.argmax(prof))

def trim_floor2(im, a):
    _, guess = trim_floor(im, a)
    edge = board_edge(im, a, guess)
    opaque = (a > 0.5).astype(np.uint8)
    above = opaque[max(edge - 200, 0):edge - 60]
    pole_cols = above.mean(0) > 0.6
    pole_cols = cv2.dilate(pole_cols.astype(np.uint8)[None], np.ones((1, 3), np.uint8))[0] > 0
    out = a.copy()
    out[edge + 2:, ~pole_cols] = 0
    # legs end where the floor starts: cut each leg where its column turns grey-white
    return out, edge

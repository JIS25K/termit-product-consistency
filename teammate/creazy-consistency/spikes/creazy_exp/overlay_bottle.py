"""Condition B, overlay step for a round bottle: wrap the real photo's front onto the generated
bottle, at the angle the generator drew it.

Model: the bottle body is a cylinder of radius R (from the photo's silhouette). A point of the
photo at column x_r has the angle t = asin((x_r - cx) / R) on the cylinder. In the generated image
the same point appears at
    x_g = cx' + s * R * sin(t + phi),   y_g = cy' + s * y_r
(phi: how far the generator turned the bottle; s: scale; cx', cy': position; orthographic, no
tilt). SIFT matches between the photo and the generated image fix (phi, s, cx', cy'): for each
phi on a 2 degree grid, RANSAC on (s, cx', cy'); the phi with most inliers wins.

Then every generated pixel on the body is mapped back to the photo and the real texture is
drawn there, with the generated image's own low-frequency brightness (scene light, cylinder
shading). Only the straight body is covered (the shoulder and pump keep the generated pixels).

Skipped (and the reason written down) when:
  - fewer than MIN_INLIERS matches agree (the label is hidden, too small, or redrawn beyond
    recognition),
  - the photo would be enlarged more than MAX_UP times (blurry),
  - a hand or another object covers more than MAX_OCC of the body area.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
EXP = ROOT / "out" / "creazy_exp2"
SRC = EXP / "inputs" / "P3" / "original.png"
MIN_INLIERS = 15
MAX_UP = 2.0
MAX_OCC = 0.35  # occluders (skin) stay on top; above this the cut is mostly hidden
MAX_TEX_ANGLE = 75  # degrees: the photo's texture near the silhouette edge is too stretched


_LABEL = {}


def label_mask(bgr, a, cx, R, yb0, yb1):
    """The printed label on the seller's photo: body pixels that are not the bottle's own colour
    (measured on the plain lower body), holes filled, largest region. Only this is pasted."""
    key = id(bgr)
    if key in _LABEL:
        return _LABEL[key]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h = yb1 - yb0
    plain = lab[int(yb1 - 0.12 * h):yb1 - 4, int(cx - R / 2):int(cx + R / 2)].reshape(-1, 3)
    body = np.median(plain, 0)
    m = ((np.linalg.norm(lab - body, axis=-1) > 18) & (a > 0.5)).astype(np.uint8)
    m[:yb0] = 0; m[yb1:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lb == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)
    ff = m.copy(); hh, ww = m.shape
    cv2.floodFill(ff, np.zeros((hh + 2, ww + 2), np.uint8), (0, 0), 1)
    m = (m | (1 - ff)).astype(np.float32)
    _LABEL[key] = m
    return m


def real():
    rgba = np.asarray(Image.open(SRC).convert("RGBA"))
    bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    a = rgba[..., 3].astype(np.float32) / 255
    rows = np.nonzero((a > 0.5).any(1))[0]
    w = {y: np.nonzero(a[y] > 0.5)[0] for y in rows}
    wmax = max(v[-1] - v[0] for v in w.values() if len(v))
    body = [y for y in rows if len(w[y]) and (w[y][-1] - w[y][0]) > 0.97 * wmax]
    cx = float(np.median([(w[y][0] + w[y][-1]) / 2 for y in body]))
    R = wmax / 2
    return bgr, a, cx, R, min(body), max(body)


def matches(rbgr, ra, gbgr, up=2.0):
    """SIFT matches photo -> generated; the photo is enlarged first (it is small) and only its
    product pixels are used."""
    sift = cv2.SIFT_create(nfeatures=6000)
    rg = cv2.cvtColor(cv2.resize(rbgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    rm = (cv2.resize(ra, None, fx=up, fy=up) > 0.5).astype(np.uint8) * 255
    k1, d1 = sift.detectAndCompute(rg, rm)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(gbgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = []
    for m in cv2.BFMatcher().knnMatch(d1, d2, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    p = np.float32([k1[m.queryIdx].pt for m in good]) / up
    q = np.float32([k2[m.trainIdx].pt for m in good])
    return p, q


def fit_cylinder(p, q, cx, R, iters=300, tol=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arcsin(np.clip((p[:, 0] - cx) / R, -1, 1))
    best = (0, None)
    for phi_deg in range(-80, 81, 2):
        phi = np.deg2rad(phi_deg)
        sx = R * np.sin(t + phi)
        # unknowns (s, cx', cy'): x_g = cx' + s*sx ; y_g = cy' + s*y_r
        A = np.zeros((2 * len(p), 3)); b = np.zeros(2 * len(p))
        A[0::2, 0] = sx; A[0::2, 1] = 1; b[0::2] = q[:, 0]
        A[1::2, 0] = p[:, 1]; A[1::2, 2] = 1; b[1::2] = q[:, 1]
        for _ in range(iters):
            i = rng.choice(len(p), 2, replace=False)
            rows = np.concatenate([2 * i, 2 * i + 1])
            sol, *_ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
            if sol[0] <= 0:
                continue
            r = (A @ sol - b).reshape(-1, 2)
            inl = np.hypot(r[:, 0], r[:, 1]) < tol
            if inl.sum() > best[0]:
                rows = np.concatenate([2 * np.nonzero(inl)[0], 2 * np.nonzero(inl)[0] + 1])
                sol2, *_ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
                best = (int(inl.sum()), (phi_deg, *sol2))
    return best


def refine_elevation(p, q, cx, R, par, tol=5.0):
    """Looking down (or up) at the bottle: a point at angle a on the cylinder (a = t + turn) and
    photo row y_r appears at
        x_g = cx' + s * R * sin(a),   y_g = cy' + s * (y_r * cos(e) + R * cos(a) * sin(e))
    so the label's horizontal lines bend into arcs (lower in the middle when looking down). Starting
    from the e = 0 fit, the matches that agree with it are refitted for turn +-6 degrees and e in
    -20..40 degrees; the pair with the smallest median error wins."""
    phi0, s0, cx0, cy0 = par
    t = np.arcsin(np.clip((p[:, 0] - cx) / R, -1, 1))
    def solve(phi, e, idx):
        a = t[idx] + np.deg2rad(phi); er = np.deg2rad(e)
        A = np.zeros((2 * len(idx), 3)); b = np.zeros(2 * len(idx))
        A[0::2, 0] = R * np.sin(a); A[0::2, 1] = 1; b[0::2] = q[idx, 0]
        A[1::2, 0] = p[idx, 1] * np.cos(er) + R * np.cos(a) * np.sin(er); A[1::2, 2] = 1; b[1::2] = q[idx, 1]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = (A @ sol - b).reshape(-1, 2)
        return sol, np.hypot(r[:, 0], r[:, 1])
    _, r0 = solve(phi0, 0, np.arange(len(p)))
    idx = np.nonzero(r0 < 2 * tol)[0]
    if len(idx) < MIN_INLIERS:
        return (phi0, s0, cx0, cy0, 0)
    best = None
    for phi in range(int(phi0) - 6, int(phi0) + 7):
        for e in range(-20, 42, 2):
            sol, r = solve(phi, e, idx)
            if sol[0] <= 0:
                continue
            med = float(np.median(r))
            if best is None or med < best[0]:
                best = (med, (phi, float(sol[0]), float(sol[1]), float(sol[2]), e))
    return best[1]


def silhouette_fix(gen, tex, ok, par, roll, centre, R, yb0, yb1, elev=0):
    """When the label is small or redrawn, the SIFT fit pins its position but not the bottle's
    turn and width well. The generated bottle's own silhouette does: in the upright frame, the
    body's left and right edges at the label's middle row give the axis and the radius. The
    real label's centre sits on the photo's axis, so its offset from that axis is the turn."""
    phi, s, cxg, cyg = par
    er = np.deg2rad(elev)
    ym = (yb0 + yb1) / 2
    gl = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB).astype(np.float32)
    body_px = ok & (rl[..., 2] - 128 < -8)
    if body_px.sum() < 50:
        return None
    blue = (np.linalg.norm(gl - np.median(gl[body_px], 0), axis=-1) < 20) | ok
    M = cv2.getRotationMatrix2D(centre, roll, 1.0)  # turn the picture back upright
    up = cv2.warpAffine(blue.astype(np.uint8), M, (gen.shape[1], gen.shape[0]), flags=cv2.INTER_NEAREST)
    y_mid = int(round(cyg + s * (ym * np.cos(er) + R * np.cos(np.deg2rad(phi)) * np.sin(er))))
    if not (0 <= y_mid < up.shape[0]):
        return None
    xl = cxg + s * R * np.sin(np.deg2rad(phi))  # the label centre in the upright frame
    row = up[max(y_mid - 3, 0):y_mid + 4].max(0)
    xs = np.nonzero(row)[0]
    if not len(xs):
        return None
    runs = np.split(xs, np.nonzero(np.diff(xs) > 2)[0] + 1)
    run = min(runs, key=lambda r: 0 if r[0] <= xl <= r[-1] else min(abs(r[0] - xl), abs(r[-1] - xl)))
    axis, half = (run[0] + run[-1]) / 2, (run[-1] - run[0]) / 2
    s2 = half / R
    if not (0.65 * s < s2 < 1.6 * s):
        return None
    phi2 = float(np.rad2deg(np.arcsin(np.clip((xl - axis) / (s2 * R), -0.95, 0.95))))
    cyg2 = (cyg + s * (ym * np.cos(er) + R * np.cos(np.deg2rad(phi)) * np.sin(er))
            - s2 * (ym * np.cos(er) + R * np.cos(np.deg2rad(phi2)) * np.sin(er)))
    return (round(phi2), s2, axis, cyg2)


def wrap(gen, rbgr, ra, cx, R, yb0, yb1, phi_deg, s, cxg, cyg, roll_deg=0.0, centre=(0.0, 0.0), elev_deg=0.0):
    """Real texture at every generated pixel of the body. roll_deg: the bottle's axis is tilted in
    the picture by this much around `centre` (cxg, cyg are in the frame turned back upright)."""
    H, W = gen.shape[:2]
    phi = np.deg2rad(phi_deg)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    if roll_deg:
        r = np.deg2rad(-roll_deg); c0, c1 = centre
        dx, dy = xs - c0, ys - c1
        xs, ys = (c0 + dx * np.cos(r) - dy * np.sin(r)).astype(np.float32), (c1 + dx * np.sin(r) + dy * np.cos(r)).astype(np.float32)
    u = (xs - cxg) / (s * R)
    on = np.abs(u) < 0.999
    a = np.arcsin(np.clip(u, -0.999, 0.999))
    t = a - phi
    xr = cx + R * np.sin(t)
    er = np.deg2rad(elev_deg)
    yr = ((ys - cyg) / s - R * np.cos(a) * np.sin(er)) / np.cos(er)
    ok = on & (np.abs(np.rad2deg(t)) < MAX_TEX_ANGLE) & (yr >= yb0 + 2) & (yr <= yb1 - 2)
    mapx, mapy = xr.astype(np.float32), yr.astype(np.float32)
    tex = cv2.remap(rbgr, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ta = cv2.remap(ra, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
    ok &= ta > 0.9
    lm = label_mask(rbgr, ra, cx, R, yb0, yb1)
    wrap.label = cv2.remap(lm, mapx, mapy, cv2.INTER_LINEAR, borderValue=0) > 0.5
    return tex, ok


def harmonize(tex, gen, keep, s, R, strength=0.85, finish=True):
    """Make the real texture look like it was shot in the generated scene, using the generated
    product's own pixels at the same place as the target (they carry the scene's rendering):
      1. tone and colour: Lab mean and spread matched over the body (contrast, saturation, colour
         cast), robust to outliers, ratios clamped so noise is not amplified;
      2. light across the bottle: the generated image's low-frequency brightness;
      3. sharpness: the real photo is blurred until its edge energy matches the generated one;
      4. grain: noise added up to the generated image's noise level.
    The product's own colours stay in their relations; strength < 1 keeps some of the original."""
    gl = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    tl = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = keep

    def stats(x):
        v = x[k]
        lo, hi = np.percentile(v, [2, 98])
        v = v[(v >= lo) & (v <= hi)]
        return float(v.mean()), float(v.std() + 1e-3)

    out = tl.copy()
    # 1. L: contrast and level
    mg, sg = stats(gl[..., 0]); mt, st = stats(tl[..., 0])
    cr = float(np.clip(sg / st, 0.6, 1.2))
    out[..., 0] = (tl[..., 0] - mt) * cr + mg
    # a*, b*: one chroma scale for both (keeps hue relations), then the scene's colour cast
    ag, bg = gl[..., 1] - 128, gl[..., 2] - 128
    at, bt = tl[..., 1] - 128, tl[..., 2] - 128
    chroma_g = float(np.median(np.hypot(ag, bg)[k])); chroma_t = float(np.median(np.hypot(at, bt)[k]))
    sat = float(np.clip((chroma_g + 1) / (chroma_t + 1), 0.5, 1.2))
    for c, (xg, xt) in ((1, (ag, at)), (2, (bg, bt))):
        out[..., c] = 128 + (xt - float(np.median(xt[k]))) * sat + float(np.median(xg[k]))
    out = strength * out + (1 - strength) * tl
    # 2. light and colour of light across the bottle, by frequency: the coarse layer (scene light,
    # the cylinder's shading, highlights, colour cast) comes from the generated product at the same
    # place; the fine layer (print, letters, texture) stays the real photo's
    sig = max(3.0, 0.05 * s * 2 * R)
    m = k.astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x * m, (0, 0), sig) / (cv2.GaussianBlur(m, (0, 0), sig) + 1e-6)
    ratio = np.clip((blur(gl[..., 0]) + 1) / (blur(out[..., 0]) + 1), 0.65, 1.45)
    out[..., 0] = out[..., 0] * ratio
    for c in (1, 2):
        out[..., c] = out[..., c] + np.clip(blur(gl[..., c]) - blur(out[..., c]), -12, 12)
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # 3. sharpness: edge energy (Laplacian spread) over the body
    def energy(img):
        g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F)[k].std())
    eg, blur_sigma = energy(gen.astype(np.float32)), 0.0
    if finish and energy(res) > 1.15 * eg:
        for sgm in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
            cand = cv2.GaussianBlur(res, (0, 0), sgm)
            blur_sigma = sgm
            if energy(cand) <= 1.15 * eg:
                break
        res = cand
    # 4. grain: high-pass residual spread
    def noise(img):
        g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.median(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32))[k]) * 1.48)
    ng, nt = noise(gen.astype(np.float32)), noise(res)
    add = float(np.sqrt(max(0.0, ng ** 2 - nt ** 2)))
    if finish and add > 0.3:
        res = res + np.random.default_rng(0).normal(0, add, res.shape[:2])[..., None].astype(np.float32)
    return np.clip(res, 0, 255), {"contrast": round(cr, 2), "saturation": round(sat, 2), "blur": blur_sigma, "grain": round(add, 2)}


def skin(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    return (L > 90) & (a > 8) & (a < 35) & (b > 10) & (b < 40)


def locate(gen, rbgr, ra, cx, R):
    """Find the seller's product, match the real photo against an enlarged crop of it (a small
    product has too few features at its own size), estimate the tilt (similarity fit), then the
    cylinder turn and scale in the frame turned back upright."""
    import count_product
    boxes = count_product.product_boxes(gen, str(SRC))
    if not boxes:
        return None, {"why": "product label not found in the picture"}
    (x0, y0, x1, y1), _ = max(boxes, key=lambda b: b[1])
    H, W = gen.shape[:2]
    pw, ph = int(0.25 * (x1 - x0)), int(0.08 * (y1 - y0))
    X0, Y0, X1, Y1 = max(x0 - pw, 0), max(y0 - ph, 0), min(x1 + pw, W), min(y1 + ph, H)
    crop = gen[Y0:Y1, X0:X1]
    f = max(1.0, 700 / crop.shape[0])
    p, qc = matches(rbgr, ra, cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
    if len(p) < MIN_INLIERS:
        return None, {"why": f"too few matches ({len(p)})"}
    q = qc / f + np.float32([X0, Y0])
    M, inl = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    roll = float(np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))) if M is not None else 0.0
    centre = ((X0 + X1) / 2, (Y0 + Y1) / 2)
    r = np.deg2rad(-roll)
    d = q - np.float32(centre)
    qu = np.stack([centre[0] + d[:, 0] * np.cos(r) - d[:, 1] * np.sin(r), centre[1] + d[:, 0] * np.sin(r) + d[:, 1] * np.cos(r)], 1)
    n, par = fit_cylinder(p, qu, cx, R, tol=5.0 / f + 2.0)
    if par is None or n < MIN_INLIERS:
        return None, {"why": f"label not found reliably ({n} agreeing matches)"}
    phi, s_, cxg, cyg, e = refine_elevation(p, qu, cx, R, par, tol=5.0 / f + 2.0)
    return ((phi, s_, cxg, cyg), roll, centre, (X0, Y0, X1, Y1), e), {"inliers": n, "matches": int(len(p))}


def overlay(gen, finish=True):
    """finish=False skips the blur and grain of the scene match: the check after the fix compares
    label geometry, and blur and grain lower SIFT matches whatever the placement."""
    rbgr, ra, cx, R, yb0, yb1 = real()
    loc, info = locate(gen, rbgr, ra, cx, R)
    if loc is None:
        return gen, {"applied": False, **info}
    (phi, s, cxg, cyg), roll, centre, box, elev = loc
    info.update(turn_deg=phi, tilt_deg=round(roll, 1), look_down_deg=elev, scale=round(float(s), 3))
    if s > MAX_UP:
        return gen, {**info, "applied": False, "why": f"photo would be enlarged {s:.1f}x (limit {MAX_UP})"}
    tex, ok = wrap(gen, rbgr, ra, cx, R, yb0, yb1, phi, s, cxg, cyg, roll, centre, elev)
    fix = silhouette_fix(gen, tex, ok, (phi, s, cxg, cyg), roll, centre, R, yb0, yb1, elev)
    if fix is not None and (abs(fix[0] - phi) > 8 or abs(fix[1] - s) / s > 0.08):
        info["silhouette_fix"] = {"turn_deg": fix[0], "scale": round(float(fix[1]), 3)}
        phi, s, cxg, cyg = fix
        info.update(turn_deg=phi, scale=round(float(s), 3))
        tex, ok = wrap(gen, rbgr, ra, cx, R, yb0, yb1, phi, s, cxg, cyg, roll, centre, elev)
    # safety: the wrapped body must lie inside the product's own box; a fit that spills out of it
    # (onto a face, a hand, the wall) is wrong, and pasting it would make the picture worse
    # Clip to the generated bottle's own outline. Colour cannot give it (a bathroom wall can be the
    # same light blue as the bottle), so GrabCut finds it: the wrapped area well inside is the
    # bottle, far outside is background, the band between is decided by GrabCut's colour models.
    # When the generator drew the bottle a little shorter or narrower than the real proportions,
    # the wrapped body would run past it; those pixels are dropped. Much spill (skin excepted,
    # hands are kept on top later) means the placement itself is wrong and the cut is left as it was.
    ys_, xs_ = np.nonzero(ok)
    pad = 40
    cy0, cy1 = max(ys_.min() - pad, 0), min(ys_.max() + pad, gen.shape[0])
    cx0, cx1 = max(xs_.min() - pad, 0), min(xs_.max() + pad, gen.shape[1])
    okc = ok[cy0:cy1, cx0:cx1].astype(np.uint8)
    gcm = np.full(okc.shape, cv2.GC_PR_BGD, np.uint8)
    gcm[cv2.dilate(okc, np.ones((9, 9), np.uint8)) > 0] = cv2.GC_PR_FGD
    gcm[cv2.erode(okc, np.ones((25, 25), np.uint8)) > 0] = cv2.GC_FGD
    gcm[cv2.dilate(okc, np.ones((45, 45), np.uint8)) == 0] = cv2.GC_BGD
    bottle = np.zeros(ok.shape, bool)
    try:
        b1, f1 = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(np.ascontiguousarray(gen[cy0:cy1, cx0:cx1]), gcm, None, b1, f1, 4, cv2.GC_INIT_WITH_MASK)
        bottle[cy0:cy1, cx0:cx1] = (gcm == cv2.GC_FGD) | (gcm == cv2.GC_PR_FGD)
    except cv2.error:
        bottle = ok.copy()
    sk = skin(gen)
    spill = float((ok & ~bottle & ~sk).sum()) / max(int(ok.sum()), 1)
    info["spill"] = round(spill, 3)
    if spill > 0.15:
        return gen, {**info, "applied": False, "why": f"placement spills out of the product ({1 - spill:.0%} inside)"}
    ok = ok & (bottle | sk)
    # occluders: skin-like in the generated image where the real texture is not skin-like
    occ = ok & skin(gen) & ~skin(tex)
    occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    frac = float(occ.sum()) / max(int(ok.sum()), 1)
    info["covered_by_hand"] = round(frac, 3)
    if frac > MAX_OCC:
        return gen, {**info, "applied": False, "why": f"a hand covers {frac:.0%} of the body"}
    keep = ok & ~cv2.dilate(occ.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    # paste the label only (and a thin margin of body around it): the generated body has the
    # scene's own shading and outline, and pasting the whole body spilled past it when the
    # generator drew the bottle in slightly different proportions
    keep = keep & (cv2.dilate(wrap.label.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    gl = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    out_tex, hinfo = harmonize(tex, gen, keep, s, R, finish=finish)
    info["harmonize"] = hinfo
    # soft edge: wider across the top and bottom of the body (where the shoulder and base keep the
    # generated pixels) than at the silhouette sides
    k = cv2.erode(keep.astype(np.float32), np.ones((7, 3), np.uint8))
    alpha = cv2.GaussianBlur(k, (0, 0), sigmaX=1.2, sigmaY=4.0)[..., None]
    out = (gen.astype(np.float32) * (1 - alpha) + out_tex * alpha).astype(np.uint8)
    # leftovers: when the generator drew its label bigger than the real one, parts of it stay
    # visible around the pasted label. Inside the bottle (its blue body with the label hole
    # filled), pixels that are neither covered, nor body-coloured, nor skin are erased with LaMa.
    # Only a thin band next to the pasted label and inside the generated bottle's outline counts
    # (a wider search found hair and wall inside the outline's filled holes and blurred them).
    rl = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB).astype(np.float32)
    body_px = keep & (rl[..., 2] - 128 < -8)  # bluish in the real texture: the bottle body
    if body_px.sum() > 50:
        body_g = np.median(gl[body_px], 0)
        blue = np.linalg.norm(gl - body_g, axis=-1) < 20
        band = cv2.dilate(keep.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        inside_bottle = cv2.erode(bottle.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        # leftovers look like label paint: within reach of the real label's own colours (hair, a
        # sleeve or a shadow next to the bottle do not)
        lab_px = rl[keep].reshape(-1, 3)
        pal = lab_px[np.random.default_rng(0).choice(len(lab_px), min(len(lab_px), 20000), replace=False)].astype(np.float32)
        _, _, centres = cv2.kmeans(pal, 8, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
        dist = np.min(np.linalg.norm(gl[..., None, :] - centres[None, None], axis=-1), -1)
        left = band & inside_bottle & ~keep & ~blue & ~skin(gen) & (dist < 18)
        left = cv2.morphologyEx(left.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        info["leftover_px"] = int(left.sum())
        if left.sum() > 0.04 * keep.sum():
            info["leftover_skipped"] = "too much to be leftovers"  # something else matched; leave it
        elif left.sum() > 30:
            sys.path.insert(0, str(ROOT / "spikes" / "overlay"))
            import lama
            out, _ = lama.inpaint(out, cv2.dilate(left, np.ones((5, 5), np.uint8)))
    info.update(applied=True, covered_px=int(keep.sum()))
    return out, info


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out, info = overlay(cv2.imread(src))
    cv2.imwrite(dst, out)
    print(info)

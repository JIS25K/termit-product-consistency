"""Re-render the seller's shelf photo at the scene's angle (projective texturing on the 3D proxy).

compose_lama pastes the frontal cutout flat, so a scene seen 20-35 degrees from the side shows a
frontal shelf in an oblique room. Here the real photo is mapped onto the proxy and drawn again
from the scene's camera:
  1. fit the real photo's camera: proxy silhouette vs the real cutout alpha (pinhole camera with
     its own focal length and image centre, so scale and position are part of the fit)
  2. fit the scene camera to the generated shelf the same way (props on the shelf ignored)
  3. for every target pixel: hit point on the proxy -> pixel in the real photo -> colour, if the
     real camera saw that point (same part, same face, same depth). Boards are textured in place
     (wood grain stays on the wood). Chrome legs are sampled view-relative (a reflective
     cylinder looks the same from any side), the clear acrylic top shows the room through it
     with its edges drawn in. Unseen parts (board ends) are filled from the nearest seen pixels
     of the same board.
  4. composite on the LaMa plate of compose_lama, with the same tint, and floor shadows cast
     from the proxy (footprint under the lowest board, contact at each foot).

  python spikes/overlay/reproject.py 0 2 9
"""
import os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import proxy3d as X, place as P, erase_lama as E, compose as C

ROOT = P.ROOT
OUT = f"{ROOT}/out/exp13"
SS = 2  # supersampling for the target render
# how strongly the clear acrylic top shows: veil over its top face, its edge faces, the outline
# of its top face (alpha) and that outline's width in output pixels. Product-first v2 raises them:
# the blind judge read the faint default as "the top is missing".
ACR = dict(top=0.07, edge=0.75, line=0.45, line_w=1)


# ---------------------------------------------------------------- shelf proportions
# proxy3d's middle board sits too high for this shelf (0.588 of the height; the seller's frontal
# photo puts it near 0.49, the two gaps are almost equal). The proportions here start from
# measurements on that photo (front edges, in units where the width is 0.98) and are refined
# together with its camera (fit_real). Middle legs stand at exactly half the width; the acrylic
# top starts a little left of them and runs just past the right legs.
DIMS = dict(FEET=0.06, T=0.025, MID=0.487, TOP=0.96, D=0.45)
W_ = X.W_
R_LEG = 0.014  # legs are 17 px wide on the frontal photo at 607 px per unit (proxy3d had 0.012)


def shelf(dims=None):
    g = {**DIMS, **(dims or {})}
    FEET, T, MID, TOP, D = g["FEET"], g["T"], g["MID"], g["TOP"], g["D"]
    ta = g.get("TA", 0.012)
    boxes = [
        (0, FEET, 0, W_, FEET + T, D, "board"),
        (0, MID, 0, W_, MID + T, D, "board"),
        (0.46 * W_, TOP, -0.01, W_ + 0.012, TOP + ta, g.get("AZ1", D + 0.01), "acrylic"),
    ]
    xi, zi = g.get("XI", 0.03 * W_), g.get("ZI", 0.06 * D)
    cyl = []
    for z in (zi, D - zi):
        cyl.append((xi, z, R_LEG, 0.0, MID + T + 0.008, "leg"))       # left legs stop at the middle board
        cyl.append((0.5 * W_, z, R_LEG, 0.0, TOP + 0.02, "leg"))      # middle legs run to the acrylic top
        cyl.append((W_ - xi, z, R_LEG, 0.0, TOP + 0.02, "leg"))
    return boxes, cyl


# ---------------------------------------------------------------- camera and ray casting
def prims(acrylic=True, dims=None):
    boxes, cyl = shelf(dims)
    if not acrylic:
        boxes = [b for b in boxes if b[6] != "acrylic"]
    return boxes, cyl


def cam(p, level=None):
    """p = (az, el, dist, roll, fpx, cx, cy[, pitch]): eye on proxy3d's sphere around the shelf
    (azimuth, elevation, distance), pinhole intrinsics, and how far the camera looks down
    (pitch, degrees; 0 keeps verticals vertical, as in interior photos; without it the camera
    looks at the shelf's middle). The image centre (cx, cy) absorbs the rest (a shift lens)."""
    az, el, dist, roll, fpx, cx, cy = p[:7]
    c = X.camera(az, el, dist=dist)
    pitch = p[7] if len(p) > 7 else None
    if pitch is not None:
        h = c["f"].copy(); h[1] = 0.0; h /= np.linalg.norm(h)
        pr = np.radians(pitch)
        f = np.cos(pr) * h + np.array([0.0, -np.sin(pr), 0.0])
        r = np.cross(np.array([0.0, 1.0, 0.0]), f); r /= np.linalg.norm(r)
        u = np.cross(f, r)
        c.update(f=f, r=r, u=u)
    if roll:
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        r, u = c["r"], c["u"]
        c.update(r=cr * r + sr * u, u=-sr * r + cr * u)
    c.update(fpx=float(fpx), cx=float(cx), cy=float(cy))
    return c


def rays(c, h, w, ss=1):
    ys = (np.arange(h * ss) + 0.5) / ss
    xs = (np.arange(w * ss) + 0.5) / ss
    Xg, Yg = np.meshgrid((xs - c["cx"]) / c["fpx"], -(ys - c["cy"]) / c["fpx"])
    d = c["f"][None, None] + Xg[..., None] * c["r"][None, None] + Yg[..., None] * c["u"][None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(c, pts):
    v = np.asarray(pts, np.float64) - c["eye"]
    zc = v @ c["f"]
    return np.stack([c["cx"] + c["fpx"] * (v @ c["r"]) / zc, c["cy"] - c["fpx"] * (v @ c["u"]) / zc], -1)


def trace(c, h, w, pr, ss=1):
    """Nearest hit per pixel: distance t, primitive index (boxes first, then cylinders; -1 none),
    face (boxes: 0/1 x min/max, 2/3 y min/max, 4/5 z min/max; cylinders: 6), ray directions."""
    boxes, cyl = pr
    o = c["eye"]
    d = rays(c, h, w, ss)
    H, W = d.shape[:2]
    tb = np.full((H, W), np.inf)
    pid = np.full((H, W), -1, np.int16)
    face = np.full((H, W), -1, np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for i, (x0, y0, z0, x1, y1, z1, _) in enumerate(boxes):
            lo = (np.array([x0, y0, z0]) - o) * inv
            hi = (np.array([x1, y1, z1]) - o) * inv
            mn = np.minimum(lo, hi)
            tmin = np.nanmax(mn, axis=-1)
            tmax = np.nanmin(np.maximum(lo, hi), axis=-1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
            t = np.where(hit, tmin, np.inf)
            better = t < tb
            ax = np.nanargmax(mn, axis=-1)
            side = np.take_along_axis(d, ax[..., None], -1)[..., 0] < 0  # entering through the max face
            tb = np.where(better, t, tb); pid[better] = i
            face[better] = (2 * ax + side)[better]
        for j, (cx, cz, r, y0, y1, _) in enumerate(cyl):
            ox, oz = o[0] - cx, o[2] - cz
            a = d[..., 0] ** 2 + d[..., 2] ** 2
            b = 2 * (ox * d[..., 0] + oz * d[..., 2])
            cc = ox * ox + oz * oz - r * r
            disc = b * b - 4 * a * cc
            t = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
            y = o[1] + t * d[..., 1]
            hit = (disc >= 0) & (t > 0) & (y >= y0) & (y <= y1)
            t = np.where(hit, t, np.inf)
            better = t < tb
            tb = np.where(better, t, tb); pid[better] = len(boxes) + j; face[better] = 6
    return tb, pid, face, d


def silhouette(p, h, w, acrylic=False, dims=None):
    _, pid, _, _ = trace(cam(p), h, w, prims(acrylic, dims))
    return pid >= 0


# ---------------------------------------------------------------- camera fitting
def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def centred(az, el, dist, roll, fpx, w, h, pitch=0.0):
    """Camera whose image centre is shifted so the shelf's middle lands in the middle of the
    image (a level camera high above the shelf would otherwise miss it)."""
    c = cam((az, el, dist, roll, fpx, 0.0, 0.0, pitch))
    t = np.array([W_ / 2, 0.45, 0.22])
    x, y = project(c, t[None])[0]
    return (az, el, dist, roll, fpx, w / 2 - x, h / 2 - y, pitch)


def align(p, target_box, h, w, fit="width", dims=None):
    """Scale and move the camera's image (focal length, image centre) so the proxy silhouette's
    width (or height) and bottom match target_box; horizontally centred."""
    sx0, sy0, sx1, sy1 = bbox(silhouette(p, h, w, dims=dims))
    gx0, gy0, gx1, gy1 = target_box
    s = (gx1 - gx0) / (sx1 - sx0) if fit == "width" else (gy1 - gy0) / (sy1 - sy0)
    tx = (gx0 + gx1) / 2 - s * (sx0 + sx1) / 2
    ty = gy1 - s * sy1
    az, el, dist, roll, fpx, cx, cy = p[:7]
    return (az, el, dist, roll, s * fpx, s * cx + tx, s * cy + ty) + tuple(p[7:])


def iou(a, b, ignore=None, r=1):
    if r > 1:
        k = np.ones((r, r), np.uint8)
        a = cv2.dilate(a.astype(np.uint8), k) > 0
        b = cv2.dilate(b.astype(np.uint8), k) > 0
    if ignore is not None:
        a, b = a & ~ignore, b & ~ignore
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def nelder_mead(loss, n, iters):
    """Plain Nelder-Mead from the origin with unit steps (callers scale their own variables)."""
    simplex = [np.zeros(n)] + [np.eye(n)[i] for i in range(n)]
    fz = [loss(s) for s in simplex]
    for _ in range(iters):
        o = np.argsort(fz); simplex = [simplex[i] for i in o]; fz = [fz[i] for i in o]
        cen = np.mean(simplex[:-1], 0)
        xr = cen + (cen - simplex[-1]); fr = loss(xr)
        if fr < fz[0]:
            xe = cen + 2 * (cen - simplex[-1]); fe = loss(xe)
            simplex[-1], fz[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fz[-2]:
            simplex[-1], fz[-1] = xr, fr
        else:
            xc = cen + 0.5 * (simplex[-1] - cen); fc = loss(xc)
            if fc < fz[-1]:
                simplex[-1], fz[-1] = xc, fc
            else:
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (s - simplex[0]) for s in simplex[1:]]
                fz = [fz[0]] + [loss(s) for s in simplex[1:]]
    i = int(np.argmin(fz))
    return simplex[i], fz[i]


def _small(m, h, w):
    return None if m is None else cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def fit_camera(target, azs, els, dists=(2.0, 2.6, 3.6, 5.0), ignore=None, r=1, res=2, iters=160,
               dims=None, rolls=(0.0,)):
    """Grid over (azimuth, elevation, distance) with the image aligned to the target extents, then
    Nelder-Mead over all seven camera numbers on the silhouette IoU. Works at 1/res resolution."""
    h, w = target.shape[0] // res, target.shape[1] // res
    tg, ig = _small(target, h, w), _small(ignore, h, w)
    tbox = bbox(tg & (~ig if ig is not None else True))
    f0 = (h / 2) / np.tan(np.radians(35) / 2)
    best = (-1, None)
    for az in azs:
        for el in els:
            for dist in dists:
                for roll in rolls:
                    for fit in ("width", "height"):
                        p = align(centred(az, el, dist, roll, f0, w, h), tbox, h, w, fit, dims)
                        sc = iou(silhouette(p, h, w, dims=dims), tg, ig, r)
                        if sc > best[0]:
                            best = (sc, p)
    p0 = np.array(best[1], np.float64)
    scale = np.array([4, 3, 0.4 * p0[2], 2, 0.04 * p0[4], 0.02 * w, 0.02 * h, 3])

    def loss(z):
        q = p0 + z * scale
        if not (-89 < q[0] < 89 and -5 < q[1] < 60 and 1.2 < q[2] < 15 and -5 < q[7] < 30):
            return 1.0
        return -iou(silhouette(tuple(q), h, w, dims=dims), tg, ig, r)

    z, f = nelder_mead(loss, 8, iters)
    q = p0 + z * scale
    q[4:7] *= res  # back to full resolution
    return tuple(float(x) for x in q), dict(grid_iou=round(best[0], 3), iou=round(-f, 3))


def fit_real(alpha, p_init, dims_init=None, res=2, iters=400):
    """Camera and shelf proportions together on the seller's frontal photo (silhouette IoU)."""
    h, w = alpha.shape[0] // res, alpha.shape[1] // res
    tg = _small(alpha > 0.5, h, w)
    keys = ["FEET", "T", "MID", "TOP", "D"]
    d0 = {**DIMS, **(dims_init or {})}
    p0 = np.array(tuple(p_init[:7]) + (p_init[7] if len(p_init) > 7 else 0.0,), np.float64); p0[4:7] /= res
    v0 = np.concatenate([p0, [d0[k] for k in keys]])
    scale = np.array([2, 2, 0.15 * p0[2], 1, 0.03 * p0[4], 0.01 * w, 0.01 * h, 2, 0.01, 0.005, 0.02, 0.02, 0.04])

    def unpack(z):
        v = v0 + z * scale
        return tuple(v[:8]), dict(zip(keys, v[8:]))

    def loss(z):
        p, d = unpack(z)
        if not (1.2 < p[2] < 15 and 0.01 < d["T"] < 0.05 and d["FEET"] + 0.2 < d["MID"] < d["TOP"] - 0.2 and 0.25 < d["D"] < 0.7):
            return 1.0
        return -iou(silhouette(p, h, w, dims=d), tg)

    z, f = nelder_mead(loss, len(v0), iters)
    p, d = unpack(z)
    p = tuple(float(x) * (res if 4 <= i < 7 else 1) for i, x in enumerate(p))
    return p, {k: round(float(v), 4) for k, v in d.items()}, round(-f, 3)


# Edges read off the seller's frontal photo (u0, 768 x 1024). Rows from light profiles at
# x = 300 and 520 (same values): the clear top's back edge, front face top and bottom; then
# the middle and lower boards the same way. Columns: the boards' front edge ends (alpha), leg
# centres at three rows (front-left, back-left, middle, back-right, front-right), feet bottom.
EDGES = dict(
    rows=[("acr", "back_top", 227.5), ("acr", "front_top", 252.0), ("acr", "front_bottom", 257.5),
          ("mid", "back_top", 477.5), ("mid", "front_top", 533.5), ("mid", "front_bottom", 540.5),
          ("low", "back_top", 714.5), ("low", "front_top", 800.0), ("low", "front_bottom", 807.5)],
    board_ends=[("mid", 86.5, 682.5), ("low", 94.5, 676.5)],
    legs={620: [("FL", 116.0), ("BL", 148.0), ("M", 385.0), ("BR", 621.0), ("FR", 653.0)],
          830: [("FL", 120.0), ("BL", 149.0), ("M", 385.0), ("BR", 620.0), ("FR", 652.0)],
          300: [("M", 383.5), ("BR", 629.5), ("FR", 660.0)]},
    feet=843.5,
)


# Depth from the seller's side-view panel (out/exp13/views/v_side): at the near end the board
# spans 128 px, the near legs' centres 95 px apart, feet to the clear top 286 px; the clear top
# is 132 px deep. Height (feet to the top) is 0.975 in width units on the frontal photo.
SIDE = dict(D=round(128 / 286 * 0.975, 4), ZI=round((128 - 95) / 2 / 128 * 128 / 286 * 0.975, 4),
            AZ1=round(132 / 286 * 0.975 - 0.01, 4))


def measure_real(p0, dims0, fixed=SIDE):
    """Camera (with its look-down angle) and shelf dimensions from the measured edges, by least
    squares. The silhouette alone cannot tell a board's top from its front edge, so the depth,
    thickness and camera height drift together; the edges separate them."""
    from scipy.optimize import least_squares
    keys = [k for k in ["FEET", "MID", "TOP", "T", "TA", "D", "AZ1", "XI", "ZI"] if k not in fixed]
    d0 = {"T": 0.013, "TA": 0.009, "XI": 0.03 * W_, **dims0}
    d0.setdefault("AZ1", d0["D"] + 0.01); d0.setdefault("ZI", 0.06 * d0["D"])
    v0 = np.concatenate([np.array(p0[:7], float), [p0[7] if len(p0) > 7 else p0[1]], [d0[k] for k in keys]])
    big = 1e9
    lo = np.array([-30, 0, 2.0, -5, 100, -big, -big, -5] + [0.0] * len(keys))
    hi = np.array([30, 35, 15.0, 5, 50000, big, big, 30] + [2.0] * len(keys))
    v0 = np.clip(v0, lo + 1e-6, hi - 1e-6)

    def unpack(v):
        return tuple(v[:8]), {**dict(zip(keys, v[8:])), **fixed}

    def res(v):
        p, d = unpack(v)
        c = cam(p)
        px = lambda pts: project(c, np.array(pts, float))
        y_of = {"acr": (d["TOP"], d["TOP"] + d["TA"], -0.01, d["AZ1"]), "mid": (d["MID"], d["MID"] + d["T"], 0.0, d["D"]),
                "low": (d["FEET"], d["FEET"] + d["T"], 0.0, d["D"])}
        out = []
        for b, which, r in EDGES["rows"]:
            y0, y1, zf, zb = y_of[b]
            pt = {"back_top": (0.6, y1, zb), "front_top": (0.6, y1, zf), "front_bottom": (0.6, y0, zf)}[which]
            out.append(px([pt])[0][1] - r)
        for b, xl, xr in EDGES["board_ends"]:
            y0, y1, _, _ = y_of[b]
            q = px([(0.0, (y0 + y1) / 2, 0.0), (W_, (y0 + y1) / 2, 0.0)])
            out += [q[0][0] - xl, q[1][0] - xr]
        legs = {"FL": (d["XI"], d["ZI"]), "BL": (d["XI"], d["D"] - d["ZI"]), "M": (0.5 * W_, d["ZI"]),
                "BR": (W_ - d["XI"], d["D"] - d["ZI"]), "FR": (W_ - d["XI"], d["ZI"])}
        for row, obs in EDGES["legs"].items():
            for name, col in obs:
                lx, lz = legs[name]
                a, b = px([(lx, 0.1, lz), (lx, 0.9, lz)])
                t = (row - a[1]) / (b[1] - a[1])
                out.append(a[0] + t * (b[0] - a[0]) - col)
        for name in ("FL", "M", "FR"):
            lx, lz = legs[name]
            out.append(px([(lx, 0.0, lz - R_LEG)])[0][1] - EDGES["feet"])
        return np.array(out)

    sol = least_squares(res, v0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
    p, d = unpack(sol.x)
    r = res(sol.x)
    return tuple(float(x) for x in p), {k: round(float(v), 4) for k, v in d.items()}, dict(rms_px=round(float(np.sqrt((r ** 2).mean())), 2), max_px=round(float(np.abs(r).max()), 2))


def outline(img, p, color=(0, 0, 255), acrylic=True, dims=None):
    h, w = img.shape[:2]
    _, pid, _, _ = trace(cam(p), h, w, prims(acrylic, dims))
    m = (pid >= 0).astype(np.uint8)
    e = (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    out = img.copy(); out[e] = color
    return out


# ---------------------------------------------------------------- texturing
def fill_from(img, valid, region, steps=12):
    """Fill region pixels that are not valid from valid pixels of the same region (normalized
    convolution at growing scales; never borrows from outside the region)."""
    out = img.astype(np.float32).copy()
    v = (valid & region).astype(np.float32)
    acc = out * v[..., None]
    todo = region & ~valid
    s = 1.0
    for _ in range(steps):
        if not todo.any():
            break
        num = cv2.GaussianBlur(acc, (0, 0), s)
        den = cv2.GaussianBlur(v, (0, 0), s)
        ok = todo & (den > 1e-3)
        out[ok] = num[ok] / den[ok][:, None]
        acc[ok] = out[ok]; v[ok] = 1.0
        todo &= ~ok
        s *= 1.8
    return out


def bilinear(img, xy):
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0] - 0.5, 0, w - 1.001); y = np.clip(xy[:, 1] - 0.5, 0, h - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    im = img.astype(np.float32)
    return ((im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx) * (1 - fy)
            + (im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx) * fy)


class Real:
    """The seller's photo with its fitted camera and proportions, and its own view of the proxy."""

    def __init__(self, i=0, fit=None):
        self.im, self.a, _ = P.real(i)
        fit = fit or json.load(open(f"{OUT}/reproject_u{i}_cam.json"))
        self.p, self.dims = tuple(fit["p"]), fit["dims"]
        self.c = cam(self.p)
        h, w = self.a.shape
        self.t, self.pid, self.face, _ = trace(self.c, h, w, prims(False, self.dims))
        self.nb = len(prims(False, self.dims)[0])
        # colour of the acrylic's front edge (the only part of the clear top the cutout keeps)
        acr = prims(True, self.dims)[0][-1]
        xy = project(self.c, [[acr[0], acr[4], acr[2]], [acr[3], acr[1], acr[2]]])
        x0, y1 = xy[0]; x1, y0 = xy[1]
        band = np.zeros_like(self.a, bool)
        band[int(min(y0, y1)) - 4:int(max(y0, y1)) + 4, int(x0):int(x1)] = True
        sel = band & (self.a > 0.5) & (self.pid < 0)
        if sel.sum() > 20:
            px = self.im[sel].astype(np.float32); lum = px.mean(1)
            self.acr_edge = px[lum >= np.percentile(lum, 60)].mean(0)
        else:
            self.acr_edge = np.array([235, 235, 232.0])

    def sample(self, pts, want_pid, want_face=None, tol=0.012):
        """Colour of 3D points in the photo, and whether the photo really shows them there."""
        h, w = self.a.shape
        xy = project(self.c, pts)
        xi = np.round(xy[:, 0]).astype(int); yi = np.round(xy[:, 1]).astype(int)
        ins = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi_c, yi_c = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        ok = ins & (self.pid[yi_c, xi_c] == want_pid) & (self.a[yi_c, xi_c] > 0.5)
        if want_face is not None:
            ok &= self.face[yi_c, xi_c] == want_face
            dist = np.linalg.norm(pts - self.c["eye"], axis=1)
            ok &= np.abs(self.t[yi_c, xi_c] - dist) < tol
        return bilinear(self.im, xy), ok


def fill_runs(tex, ok):
    """Fill invalid runs along each row with the texture just left and right of the run,
    crossfaded (keeps the wood grain where a leg hid the board); leftovers by fill_from."""
    out = tex.copy(); ok = ok.copy()
    n = tex.shape[1]
    for v in range(tex.shape[0]):
        bad = np.nonzero(~ok[v])[0]
        if not len(bad) or len(bad) == n:
            continue
        for run in np.split(bad, np.nonzero(np.diff(bad) > 1)[0] + 1):
            a, b = run[0], run[-1] + 1; L = b - a
            left = a - L >= 0 and ok[v, a - L:a].all()
            right = b + L <= n and ok[v, b:b + L].all()
            if left and right:
                wgt = ((np.arange(L) + 0.5) / L)[:, None]
                out[v, a:b] = tex[v, a - L:a] * (1 - wgt) + tex[v, b:b + L] * wgt
            elif right:   # the run touches the board's end: copy the grain from the inner side
                out[v, a:b] = tex[v, b:b + L]
            elif left:
                out[v, a:b] = tex[v, a - L:a]
            else:
                continue
            ok[v, a:b] = True
    if not ok.all():
        out = fill_from(out, ok, np.ones(ok.shape, bool))
    return out


def board_textures(real, ppu=1200, guard=12):
    """Top and front face of every board as images in the board's own coordinates, read from the
    photo where it shows them. Texels next to a leg (its contact shadow and reflection) count as
    unseen too, and unseen texels are filled along the grain."""
    boxes = prims(False, real.dims)[0]
    out = []
    for b, (x0, y0, z0, x1, y1, z1, _) in enumerate(boxes):
        us = x0 + (np.arange(int((x1 - x0) * ppu)) + 0.5) / ppu
        zs = z0 + (np.arange(int((z1 - z0) * ppu)) + 0.5) / ppu
        ys = y1 - (np.arange(max(int((y1 - y0) * ppu), 2)) + 0.5) / ppu
        U, Z = np.meshgrid(us, zs)
        top, ok_t = real.sample(np.stack([U, np.full_like(U, y1), Z], -1).reshape(-1, 3), b, 3)
        U2, Y2 = np.meshgrid(us, ys)
        fr, ok_f = real.sample(np.stack([U2, Y2, np.full_like(U2, z0)], -1).reshape(-1, 3), b, 4)
        top, ok_t = top.reshape(U.shape + (3,)), ok_t.reshape(U.shape)
        fr, ok_f = fr.reshape(U2.shape + (3,)), ok_f.reshape(U2.shape)
        ok_t = ~(cv2.dilate((~ok_t).astype(np.uint8), np.ones((1, 2 * guard + 1), np.uint8)) > 0)
        # the photo's white background just past the board's back edge reads as a bright line
        # along the back of every re-rendered board (blind judge: "halo lines"); those texels
        # are treated as unseen and filled from the wood in front of them
        back = (np.arange(top.shape[0]) >= 0.93 * top.shape[0])[:, None]
        ok_t &= ~(back & (top.mean(axis=2) > 225))
        out.append(dict(box=(x0, y0, z0, x1, y1, z1), ppu=ppu, top=fill_runs(top, ok_t), front=fill_runs(fr, ok_f),
                        seen=(round(float(ok_t.mean()), 3), round(float(ok_f.mean()), 3))))
    return out


def board_colour(tx, pts, face):
    """Colour of board hit points from its textures: top/bottom from the top image, the rest
    from the front edge (the plywood edge looks the same all round); ends a little darker."""
    x0, y0, z0, x1, y1, z1 = tx["box"]; ppu = tx["ppu"]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    col = np.zeros((len(pts), 3), np.float32)
    tb = (face == 2) | (face == 3)
    if tb.any():
        col[tb] = bilinear(tx["top"], np.stack([(x[tb] - x0) * ppu, (z[tb] - z0) * ppu], -1))
    ed = ~tb
    if ed.any():
        u = np.where(face[ed] == 0, x0 + 0.03 + (z[ed] - z0), np.where(face[ed] == 1, x1 - 0.03 - (z[ed] - z0), x[ed]))
        col[ed] = bilinear(tx["front"], np.stack([(u - x0) * ppu, (y1 - y[ed]) * ppu], -1))
    ends = np.isin(face, [0, 1])
    col[ends] *= 0.9
    return col


def render(real, p, H, W, ss=SS):
    """The real shelf seen from camera p: colour, opaque coverage, and the acrylic top's layer
    (veil alpha and edge colour), all at ss times the output size."""
    c = cam(p)
    pr = prims(False, real.dims)
    t, pid, face, d = trace(c, H, W, pr, ss)
    hit = pid >= 0
    P3 = c["eye"][None, None] + t[..., None] * d
    col = np.zeros(t.shape + (3,), np.float32)
    valid = np.zeros(t.shape, bool)
    boxes, cyl = pr
    nb = len(boxes)
    # boards: texture stays on the wood
    if not hasattr(real, "tex"):
        real.tex = board_textures(real)
    for b in range(nb):
        m = pid == b
        if m.any():
            col[m] = board_colour(real.tex[b], P3[m], face[m]); valid[m] = True
    # chrome legs: view-relative (u across the leg as seen, same height), from this leg or any leg
    # the photo shows at that height
    def horiz(eye, x, z):
        v = np.array([x - eye[0], 0.0, z - eye[2]]); v /= np.linalg.norm(v)
        return v, np.array([v[2], 0.0, -v[0]])
    order = lambda j: [j] + sorted([k for k in range(len(cyl)) if k != j], key=lambda k: (abs(cyl[k][0] - cyl[j][0]), abs(cyl[k][1] - cyl[j][1])))
    for j, (lx, lz, r, y0, y1, _) in enumerate(cyl):
        m = pid == nb + j
        if not m.any():
            continue
        pts = P3[m]
        vt, rt = horiz(c["eye"], lx, lz)
        u = np.clip(((pts - np.array([lx, 0, lz])) @ rt) / r, -0.92, 0.92)
        y = pts[:, 1]
        got = np.zeros(len(pts), bool); cj = np.zeros((len(pts), 3), np.float32)
        for k in order(j):
            kx, kz, kr, ky0, ky1, _ = cyl[k]
            vr, rr = horiz(real.c["eye"], kx, kz)
            src = np.array([kx, 0, kz])[None] + kr * (u[:, None] * rr[None] - np.sqrt(1 - u ** 2)[:, None] * vr[None])
            src[:, 1] = y
            cc, ok = real.sample(src, nb + k)
            ok &= (y >= ky0) & (y <= ky1) & ~got
            cj[ok] = cc[ok]; got |= ok
        col[m] = cj; valid[m] = got
        feet = m.copy(); feet[m] = y < boxes[0][1] + 0.5 * (boxes[0][4] - boxes[0][1])
        if feet.any():
            real.feet_masks = getattr(real, "feet_masks", []) + [feet]
    # parts the photo never shows (board ends, a leg segment hidden in every leg): fill from
    # the nearest shown pixels of the same part
    for q in np.unique(pid[hit]):
        reg = pid == q
        col = np.where(reg[..., None], fill_from(col, valid, reg), col)
    # the clear acrylic top, drawn over whatever is behind it
    acr = prims(True, real.dims)[0][-1:]
    ta, pa, fa, _ = trace(c, H, W, (acr, []), ss)
    front = (pa >= 0) & (ta < t)
    edge_face = front & (fa != 2) & (fa != 3)
    veil = np.where(front, ACR["top"], 0.0).astype(np.float32)
    veil[edge_face] = ACR["edge"]
    # outline of the top face (back and side edges read through the clear plate)
    x0, y0, z0, x1, y1, z1, _ = acr[0]
    corners = np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]])
    xy = project(c, corners) * ss
    lines = np.zeros(t.shape, np.uint8)
    cv2.polylines(lines, [np.round(xy * 8).astype(np.int32)], True, 255, max(1, ss * ACR["line_w"]), cv2.LINE_AA, shift=3)
    veil = np.maximum(veil, (lines / 255.0 * ACR["line"]) * (front | (lines > 0)))
    feet = np.zeros(t.shape, bool)
    for fm in getattr(real, "feet_masks", []):
        feet |= fm
    real.feet_masks = []
    return dict(col=col, op=hit.astype(np.float32), veil=veil, acr_col=real.acr_edge.astype(np.float32), pid=pid, face=face, ss=ss, feet=feet)


def floor_shadow(real, p, H, W, strength=0.42, foot=0.55):
    """Shadow on the floor (y = 0) from the proxy: the lowest board's footprint, softened by its
    height above the floor, and a contact spot under every foot."""
    c = cam(p)
    boxes, cyl = prims(False, real.dims)
    b = boxes[0]
    sh = np.zeros((H, W), np.float32)
    quad = project(c, [[b[0], 0, b[2]], [b[3], 0, b[2]], [b[3], 0, b[5]], [b[0], 0, b[5]]])
    cv2.fillPoly(sh, [np.round(quad * 8).astype(np.int32)], strength, cv2.LINE_AA, shift=3)
    px_per_unit = np.linalg.norm(quad[1] - quad[0]) / (b[3] - b[0])
    sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, 0.06 * px_per_unit))
    spots = np.zeros((H, W), np.float32)
    for (lx, lz, r, _, _, _) in cyl:
        ring = project(c, [[lx + 2.2 * r * np.cos(a), 0, lz + 2.2 * r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24)])
        cv2.fillPoly(spots, [np.round(ring * 8).astype(np.int32)], foot, cv2.LINE_AA, shift=3)
    spots = cv2.GaussianBlur(spots, (0, 0), max(1.5, 0.012 * px_per_unit))
    return np.clip(np.maximum(sh, spots), 0, 0.85)


def generated_materials(scene, part, ignore):
    """Pixels of the generated shelf by material: chrome legs (tall thin parts) and wood boards
    (wood-coloured, not leg). The generator drew them in this room's light."""
    legs = thin_vertical(part) & ~ignore
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    wood = part & ~cv2.dilate(legs.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~ignore
    wood &= (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 30)
    return legs, wood


def match_material(lab, ours, ref_lab, ref, s_L=0.7, s_ab=0.35, sd_clip=(0.7, 1.4), lit=None):
    """Move our pixels' Lab statistics toward the generated material's: brightness and contrast by
    s_L, colour by s_ab (the real colour stays mostly; the room's light cast comes in).
    lit: compare only each side's brighter pixels (above that brightness percentile), so a
    shadow caught in either mask does not drag the level down."""
    if ours.sum() < 50 or ref.sum() < 50:
        return lab, None
    o, g = lab[ours], ref_lab[ref]
    if lit is not None:
        o = o[o[:, 0] >= np.percentile(o[:, 0], lit)]
        g = g[g[:, 0] >= np.percentile(g[:, 0], lit)]
    mo, so, mg, sg = o.mean(0), o.std(0) + 1e-3, g.mean(0), g.std(0) + 1e-3
    k = np.clip(1 + s_L * (sg[0] / so[0] - 1), *sd_clip)
    out = lab.copy()
    x = out[ours]
    x[:, 0] = mo[0] + (x[:, 0] - mo[0]) * k + s_L * (mg[0] - mo[0])
    x[:, 1:] += s_ab * (mg[1:] - mo[1:])
    out[ours] = x
    return out, dict(L=[round(float(mo[0]), 1), round(float(mg[0]), 1)], contrast=round(float(k), 2),
                     ab_shift=[round(float(v), 1) for v in s_ab * (mg[1:] - mo[1:])])


def harmonize(fg, L, nb, scene, part, ignore):
    """Chrome legs take the generated legs' brightness and contrast (chrome mirrors the room);
    wood keeps its own colour and grain and takes the room's brightness and a little of its cast."""
    ss = L["ss"]
    H, W = scene.shape[:2]
    gl, gw = generated_materials(scene, part, ignore)
    # only where our own boards and legs land (the generated mask also holds the shelf's floor
    # shadow, which is wood-coloured but dark)
    near = lambda m, k: cv2.dilate(cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), np.ones((k, k), np.uint8)) > 0
    gw &= near((L["pid"] >= 0) & (L["pid"] < nb), 9)
    gl &= near(L["pid"] >= nb, 15)
    ref_lab = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.cvtColor(np.clip(fg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    pid = L["pid"]
    legs, wood = pid >= nb, (pid >= 0) & (pid < nb)
    lab, il = match_material(lab, legs, ref_lab, gl, s_L=0.8, s_ab=0.6)
    lab, iw = match_material(lab, wood, ref_lab, gw, s_L=0.6, s_ab=0.35, sd_clip=(0.85, 1.2), lit=40)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return out, dict(legs=il, wood=iw)


def composite(plate, real, p, tint=True, gen=None):
    """Plate (generated shelf erased) + floor shadow + re-rendered real shelf (+ acrylic layer).
    gen = (scene, part, ignore): match materials to the generated shelf instead of the plain tint."""
    H, W = plate.shape[:2]
    L = render(real, p, H, W)
    ss = L["ss"]
    base = plate.astype(np.float32) * (1 - floor_shadow(real, p, H, W)[..., None])
    base_s = cv2.resize(base, (W * ss, H * ss), interpolation=cv2.INTER_CUBIC)
    fg = L["col"]
    L["harmonize"] = None
    if gen is not None:
        fg, L["harmonize"] = harmonize(fg, L, real.nb, *gen)
    elif tint:
        fg = C.tint(fg, L["op"], np.clip(base_s, 0, 255).astype(np.uint8))  # Lab of a float image is 0-1 based
    if L["feet"].any():  # feet mirror the floor around them: half the floor's colour, from just beside each foot
        floor = cv2.blur(base_s, (41, 9))
        fm = L["feet"]
        fg[fm] = 0.5 * fg[fm] + 0.5 * floor[fm]
    op = L["op"][..., None]
    out = base_s * (1 - op) + fg * op
    v = L["veil"][..., None]
    out = out * (1 - v) + L["acr_col"][None, None] * v
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    al = cv2.resize(np.maximum(L["op"], L["veil"]), (W, H), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 255).astype(np.uint8), al, L


def thin_vertical(m, k=25):
    """Leg-like parts of a mask: tall thin structures (vertical opening) minus wide ones."""
    m = m.astype(np.uint8)
    v = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    h = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    return (v > 0) & ~(cv2.dilate(h, np.ones((3, 3), np.uint8)) > 0)


def fit_scene(part, ignore, dims, azs=range(-60, 41, 10), els=range(2, 33, 6), dists=(2.2, 3.2, 4.5, 6.8),
              res=2, iters=260, w_legs=0.5, pitch=0.0):
    """Scene camera from the generated shelf: silhouette IoU (tolerant) plus IoU of the legs alone.
    The legs' feet and tops pin the angle even where the boards blend into a wooden floor or the
    shelf's own floor shadow joins the silhouette."""
    h, w = part.shape[0] // res, part.shape[1] // res
    tg, ig = _small(part, h, w), _small(ignore, h, w)
    tl = thin_vertical(tg, max(9, 25 // res))
    tbox = bbox(tg & ~ig)
    f0 = (h / 2) / np.tan(np.radians(35) / 2)
    pr = prims(False, dims)
    nb = len(pr[0])

    def score(p):
        _, pid, _, _ = trace(cam(p), h, w, pr)
        return (1 - w_legs) * iou(pid >= 0, tg, ig, 5) + w_legs * iou(pid >= nb, tl, ig, 5)

    best = (-1, None)
    for az in azs:
        for el in els:
            for dist in dists:
                for fit in ("width", "height"):
                    p = align(centred(az, el, dist, 0.0, f0, w, h, pitch or 0.0), tbox, h, w, fit, dims)
                    sc = score(p)
                    if sc > best[0]:
                        best = (sc, p)
    p0 = np.array(best[1], np.float64)
    scale = np.array([4, 3, 0.4 * p0[2], 2, 0.04 * p0[4], 0.02 * w, 0.02 * h, 3])
    n = 7 if pitch is not None else 8  # scenes keep verticals vertical: pitch fixed

    def loss(z):
        q = p0.copy(); q[:n] += z * scale[:n]
        if not (-89 < q[0] < 89 and -5 < q[1] < 45 and 1.2 < q[2] < 15 and -5 < q[7] < 30):
            return 1.0
        return -score(tuple(q))

    z, f = nelder_mead(loss, n, iters)
    q = p0.copy(); q[:n] += z * scale[:n]
    q[4:7] *= res
    return tuple(float(x) for x in q), dict(grid_score=round(best[0], 3), score=round(-f, 3))


def anchor_lower(p, part, ignore, dims, rounds=3, res=2, iters=60):
    """Keep the fitted angle; scale and move the image (focal length, image centre) so the real
    shelf's lower board and feet sit on the generated shelf's lower board and feet. The generated
    shelf's proportions differ from the real one's, so a whole-silhouette match floats or sinks
    the real shelf; the lower board is where it meets the floor. Each round takes the generated
    pixels near the proxy's lower part and refits three numbers to them."""
    H, W = part.shape
    h, w = H // res, W // res
    tg, ig = _small(part, h, w), _small(ignore, h, w)
    boxes, cyl = prims(False, dims)
    lo_top = boxes[0][4] + 0.05  # a little above the lower board's top

    def lower(q):
        c = cam(q)
        t, pid, _, d = trace(c, h, w, (boxes, cyl))
        y = c["eye"][1] + t * d[..., 1]
        return (pid >= 0) & (y <= lo_top)

    q = np.array(p, np.float64); q[4:7] /= res
    for _ in range(rounds):
        near = cv2.dilate(lower(tuple(q)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        T = tg & near & ~ig
        scale = np.array([0.02 * q[4], 0.01 * w, 0.01 * h])

        def loss(z):
            qq = q.copy(); qq[4:7] += z * scale
            return -iou(lower(tuple(qq)), T, ig, 3)

        z, f = nelder_mead(loss, 3, iters)
        q[4:7] += z * scale
    q[4:7] *= res
    return tuple(float(x) for x in q), dict(lower_iou=round(-f, 3))


def run(k, real=None):
    """One CS 13 output: the generated shelf erased (compose_lama's plate), the real shelf drawn
    again at the scene camera, materials matched. Writes out/exp13/ours5_O{k}_u0.png and .json."""
    real = real or Real(0)
    cams = json.load(open(f"{OUT}/reproject_scene_cams.json"))
    z = np.load(f"{OUT}/reproject_gen_O{k}.npz")
    scene = cv2.imread(f"{ROOT}/data/cs/13/outputs/{P.OUTS[k]}")
    plate = cv2.imread(f"{OUT}/lama_plate_O{k}.png")
    p = tuple(cams[str(k)]["p"])
    t0 = time.time()
    out, al, L = composite(plate, real, p, gen=(scene, z["part"] > 0, z["ignore"]))
    cv2.imwrite(f"{OUT}/ours5_O{k}_u0.png", out)
    meta = dict(k=k, file=P.OUTS[k], real_camera=dict(az=round(real.p[0], 1), el=round(real.p[1], 1)),
                scene_camera=dict(az=round(p[0], 1), el=round(p[1], 1), dist=round(p[2], 2), fit=cams[str(k)].get("score")),
                turn_deg=round(p[0] - real.p[0], 1), dims=real.dims, harmonize=L["harmonize"], sec=round(time.time() - t0, 1))
    json.dump(meta, open(f"{OUT}/ours5_O{k}_u0.json", "w"), indent=1)
    return out, meta


if __name__ == "__main__":
    r = Real(0)
    for k in [int(x) for x in sys.argv[1:]] or [0, 2, 9]:
        print(json.dumps(run(k, r)[1]), flush=True)

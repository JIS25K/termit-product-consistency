"""Primitive 3D proxy of the CS 13 shelf (boxes for boards and the acrylic top, vertical
cylinders for legs) and a numpy ray caster that renders its depth and silhouette from any
camera. Proportions are measured from the seller's photos (no dimension table was given):
height 1, width 0.98, depth 0.45, the acrylic top over the right half."""
import numpy as np

H_, W_, D_ = 1.0, 0.98, 0.45
T = 0.022            # board thickness
R = 0.012            # leg radius
FEET = 0.065         # bottom board's underside height
MID = 0.588          # middle board's underside height
TOP = 0.985          # acrylic top's underside height


def shelf():
    """Primitives in shelf space: x 0..W (left to right), y up (floor 0), z 0..D (front to back).
    Returns (boxes, cylinders); boxes as (x0, y0, z0, x1, y1, z1, kind), cylinders as
    (x, z, r, y0, y1, kind)."""
    boxes = [
        (0, FEET, 0, W_, FEET + T, D_, "board"),
        (0, MID, 0, W_, MID + T, D_, "board"),
        (0.5 * W_, TOP, 0, W_, TOP + 0.012, D_, "acrylic"),
    ]
    xi, zi = 0.03 * W_, 0.06 * D_
    xs = {"l": xi, "m": 0.5 * W_ + 0.0, "r": W_ - xi}
    zs = (zi, D_ - zi)
    cyl = []
    for z in zs:
        cyl.append((xs["l"], z, R, 0.0, MID + T + 0.008, "leg"))        # left legs stop at the middle board
        cyl.append((xs["m"] + 0.015, z, R, 0.0, TOP + 0.02, "leg"))     # middle legs run to the acrylic top
        cyl.append((xs["r"], z, R, 0.0, TOP + 0.02, "leg"))
    return boxes, cyl


def camera(az_deg, el_deg, dist=2.6, fov_deg=35.0, target=(W_ / 2, 0.45, D_ / 2), roll_deg=0.0):
    """Look-at camera on a sphere around the shelf. az 0 = straight from the front (from -z),
    positive az = camera moves to the shelf's right; el = degrees above the horizon."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    t = np.array(target, np.float64)
    eye = t + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    f = t - eye; f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f); r /= np.linalg.norm(r)   # image right
    u = np.cross(f, r)                             # image up
    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        r, u = c * r + s * u, -s * r + c * u
    return dict(eye=eye, f=f, r=r, u=u, fov=np.radians(fov_deg))


def rays(cam, h, w):
    """Ray directions for an h x w image (square pixels, fov is the vertical field of view)."""
    th = np.tan(cam["fov"] / 2)
    ys = (1 - (np.arange(h) + 0.5) / h * 2) * th
    xs = ((np.arange(w) + 0.5) / w * 2 - 1) * th * w / h
    X, Y = np.meshgrid(xs, ys)
    d = cam["f"][None, None] + X[..., None] * cam["r"][None, None] + Y[..., None] * cam["u"][None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def render(cam, h, w, prims=None):
    """Distance along each ray to the nearest primitive (inf where none) and a kind map
    (0 none, 1 board, 2 acrylic, 3 leg)."""
    boxes, cyl = prims or shelf()
    o = cam["eye"]
    d = rays(cam, h, w)
    tbest = np.full((h, w), np.inf)
    kind = np.zeros((h, w), np.uint8)
    codes = {"board": 1, "acrylic": 2, "leg": 3}
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for (x0, y0, z0, x1, y1, z1, k) in boxes:
            lo = (np.array([x0, y0, z0]) - o) * inv
            hi = (np.array([x1, y1, z1]) - o) * inv
            tmin = np.nanmax(np.minimum(lo, hi), axis=-1)
            tmax = np.nanmin(np.maximum(lo, hi), axis=-1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
            t = np.where(hit, tmin, np.inf)
            better = t < tbest
            tbest = np.where(better, t, tbest); kind[better] = codes[k]
        for (cx, cz, r, y0, y1, k) in cyl:
            ox, oz = o[0] - cx, o[2] - cz
            a = d[..., 0] ** 2 + d[..., 2] ** 2
            b = 2 * (ox * d[..., 0] + oz * d[..., 2])
            c = ox * ox + oz * oz - r * r
            disc = b * b - 4 * a * c
            t = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
            y = o[1] + t * d[..., 1]
            hit = (disc >= 0) & (t > 0) & (y >= y0) & (y <= y1)
            t = np.where(hit, t, np.inf)
            better = t < tbest
            tbest = np.where(better, t, tbest); kind[better] = codes[k]
    return tbest, kind


def project(cam, h, w, pts):
    """Image coordinates (x, y) of 3D points."""
    th = np.tan(cam["fov"] / 2)
    v = np.asarray(pts, np.float64) - cam["eye"]
    zc = v @ cam["f"]
    xc, yc = v @ cam["r"], v @ cam["u"]
    px = (xc / zc / (th * w / h) + 1) / 2 * w
    py = (1 - yc / zc / th) / 2 * h
    return np.stack([px, py], -1)

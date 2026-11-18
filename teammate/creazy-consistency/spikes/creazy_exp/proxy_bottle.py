"""3D proxy of a round pump bottle from ONE front photo with a transparent background, and guide
renders for views the photo does not show.

Model (all from the photo, no hand tuning):
  - body and pump stem: a surface of revolution. The radius at each row is half the width of the
    photo's alpha at that row, around the body's centre line.
  - spout: the rows at the top where the silhouette sticks out to one side of the stem; a box from
    the axis to the tip, on that side.
  - texture: the front photo projected onto the front of the model (angles within 80 degrees of
    the camera). The rest of the surface takes the body's own colour (median of the body outside
    the label). That colour for the unseen back is an assumption and is reported as one.
Camera from the scene sentence by fixed rules (VIEW_RULES), not chosen by a person.
Rendering: dense coloured points, rotated, projected orthographically with a z-buffer.

  python spikes/creazy_exp/proxy_bottle.py      # writes guides for the scenes that need one
"""
import json, re, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "out" / "creazy_exp2"
SRC = EXP / "inputs" / "P3" / "original.png"
OUT = EXP / "B" / "guides"

# sentence keyword -> (azimuth deg, elevation deg). First match wins. None: the photo's own view.
VIEW_RULES = [
    (r"뒷면|뒤쪽|뒤에서", (180, 8)),
    (r"비스듬", (40, 12)),
    (r"옆에서|측면|옆모습", (90, 5)),
    (r"위에서|내려다", (0, 45)),
]


def view_for(sentence):
    m = re.search(r"([0-9]{1,3})\s*도", sentence)  # an explicit angle: "30도 비스듬히"
    if m and not re.search(r"뒷면|뒤쪽|뒤에서", sentence):
        return (int(m.group(1)), 12)
    for pat, v in VIEW_RULES:
        if re.search(pat, sentence):
            return v
    return None


def load():
    rgba = np.asarray(Image.open(SRC).convert("RGBA")).astype(np.float32)
    a = rgba[..., 3] / 255
    rgb = rgba[..., :3]
    rows = np.nonzero((a > 0.5).any(1))[0]
    widths = {y: np.nonzero(a[y] > 0.5)[0] for y in rows}
    body_rows = [y for y in rows if len(widths[y]) and (widths[y][-1] - widths[y][0]) > 0.8 * max(w[-1] - w[0] for w in widths.values())]
    yb0, yb1 = min(body_rows), max(body_rows)
    cx = float(np.median([(widths[y][0] + widths[y][-1]) / 2 for y in body_rows]))
    return rgb, a, rows, widths, cx, yb0, yb1


def body_colour(rgb, a, yb0, yb1, cx):
    """The body's own colour: median of the lowest part of the body (below a front label), in
    the middle half of its width. The centre rows hold the label and would give its colours."""
    h = yb1 - yb0
    r = np.nonzero(a[yb1 - 10] > 0.5)[0]
    half = (r[-1] - r[0]) / 4 if len(r) else 20
    reg = rgb[int(yb1 - 0.12 * h):yb1 - 4, int(cx - half):int(cx + half)]
    m = a[int(yb1 - 0.12 * h):yb1 - 4, int(cx - half):int(cx + half)] > 0.9
    return np.median(reg[m], 0)


def build_points(step_deg=0.5):
    rgb, a, rows, widths, cx, yb0, yb1 = load()
    base = body_colour(rgb, a, yb0, yb1, cx)
    pts, cols = [], []
    th = np.deg2rad(np.arange(-180, 180, step_deg))
    # stem: rows above the body whose silhouette is centred on the axis; spout rows: one side only
    spout = []
    for y in rows:
        xs = widths[y]
        if not len(xs):
            continue
        left, right = cx - xs[0], xs[-1] - cx
        if y < yb0 and max(left, right) > 1.6 * min(left, right) + 4:
            spout.append((y, xs[0], xs[-1]))
            r = min(left, right)
        else:
            r = (left + right) / 2
        if r < 1:
            continue
        X, Z = r * np.sin(th), r * np.cos(th)
        u = np.clip(np.round(cx + X).astype(int), 0, rgb.shape[1] - 1)
        front = np.cos(th) > np.cos(np.deg2rad(80))
        c = np.where(front[:, None], rgb[y, u], base[None, :])
        # the photo's own pixel may be background at the silhouette edge: use the base colour there
        c = np.where((a[y, u] > 0.5)[:, None] | ~front[:, None], c, base[None, :])
        pts.append(np.stack([X, np.full_like(X, -y), Z], 1)); cols.append(c)
    # bottom disk and top of the stem
    for y, sign in ((yb1, -1),):
        r = (widths[y][-1] - widths[y][0]) / 2
        for rr in np.arange(0, r, 1.0):
            X, Z = rr * np.sin(th), rr * np.cos(th)
            pts.append(np.stack([X, np.full_like(X, -y), Z], 1)); cols.append(np.repeat(base[None] * 0.8, len(th), 0))
    # spout: box from the axis to the tip, on the side it sticks out, depth = its height
    if spout:
        ys = [s[0] for s in spout]
        h = max(ys) - min(ys) + 1
        side = 1 if np.mean([s[2] - cx for s in spout]) > np.mean([cx - s[1] for s in spout]) else -1
        tip = max(s[2] - cx for s in spout) if side > 0 else max(cx - s[1] for s in spout)
        sc = np.median(np.concatenate([rgb[y, int(cx):int(cx + side * tip):side] for y in ys]), 0)
        for y in ys:
            for x in np.arange(0, tip, 0.7):
                for z in np.arange(-h / 2, h / 2, 0.7):
                    pts.append(np.array([[side * x, -y, z]])); cols.append(sc[None])
    P = np.concatenate(pts).astype(np.float32)
    C = np.concatenate(cols).astype(np.float32)
    return P, C, dict(axis_x=cx, body_rows=(int(yb0), int(yb1)), body_colour=[round(float(v)) for v in base],
                      spout=bool(spout))


def render(P, C, az, el, size=1024, fill=0.78):
    a, e = np.deg2rad(az), np.deg2rad(el)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Q = P @ (Rx @ Ry).T
    # normals for light: radial direction for points of revolution (good enough for a guide)
    n = P.copy(); n[:, 1] = 0
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
    n = n @ (Rx @ Ry).T
    light = np.array([-0.4, 0.5, 0.75]); light /= np.linalg.norm(light)
    shade = 0.78 + 0.22 * np.clip(n @ light, 0, 1)
    xs, ys = Q[:, 0], -Q[:, 1]
    s = fill * size / max(xs.max() - xs.min(), ys.max() - ys.min())
    u = ((xs - (xs.max() + xs.min()) / 2) * s + size / 2).astype(int)
    v = ((ys - (ys.max() + ys.min()) / 2) * s + size / 2).astype(int)
    img = np.full((size, size, 3), 255, np.float32)
    zbuf = np.full((size, size), -np.inf, np.float32)
    rad = max(1, int(np.ceil(s * 0.7)))
    col = C * shade[:, None]
    du, dv = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1))
    du, dv = du.ravel(), dv.ravel()
    uu = np.clip(u[:, None] + du[None], 0, size - 1).ravel()
    vv = np.clip(v[:, None] + dv[None], 0, size - 1).ravel()
    zz = np.repeat(Q[:, 2], len(du))
    idx = np.repeat(np.arange(len(Q)), len(du))
    order = np.argsort(zz, kind="stable")  # far first; the nearest point at a pixel is written last
    img[vv[order], uu[order]] = col[idx[order]]
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    global EXP, OUT
    if len(sys.argv) > 1:
        EXP = Path(sys.argv[1]).resolve(); OUT = EXP / "B" / "guides"
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    P, C, info = build_points()
    OUT.mkdir(parents=True, exist_ok=True)
    made = {}
    for t in cfg["products"][0]["targets"]:
        v = view_for(t["scene"])
        if v is None:
            continue
        img = render(P, C, *v)
        Image.fromarray(img).save(OUT / f"{t['id']}.png")
        made[t["id"]] = {"azimuth": v[0], "elevation": v[1], "rule": next((p for p, vv in VIEW_RULES if vv == v), "explicit angle in the sentence")}
    # also the photo's own view, to check the model against the photo
    Image.fromarray(render(P, C, 0, 0)).save(OUT.parent / "proxy_front_check.png")
    info["guides"] = made
    info["assumption"] = "사진에 없는 뒷면과 옆면 뒤쪽은 병 몸통 색 한 가지로 칠했다(가정)."
    (OUT.parent / "proxy_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Showcase of a shape fix on CS case 39 (long-tip marker), for an internal presentation.

The product's point is its long metal tip. In the generated cut of a hand drawing on paper,
the pen ends in a long black cone with a short broad nib and draws a thick stroke, like an
ordinary marker. Here the generated nib is replaced with the real tip taken from the seller's
own close-up photo:

1. the real tip (metal needle, its point and the ferrule of a black pen) is located in the
   seller's close-up: its centre line is fitted on the dark edges of the needle;
2. the pen axis, the cone's width and the point where the broad nib meets the paper are
   measured from the dark pixels of the cone, inside a region set by hand around the pen tip;
3. the cone below the ferrule, the broad nib and the stroke are erased with LaMa (paper only
   around them); the rest of the generated cone is narrowed toward the ferrule (its own
   pixels, squeezed across the axis); the stroke comes back as the real tip draws it (its own
   centre line, thin); a soft shadow meeting the point is added; the real tip is warped in
   along the generated axis, its point where the nib met the paper.

Everything outside the pen tip and the stroke stays as generated. Not part of the evaluation.
Output goes to out/showcase/c39/ (git-ignored): customer images stay on this machine.

    HF_HUB_OFFLINE=1 .venv/bin/python spikes/validation/showcase_shape.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "spikes")]

from overlay.lama import inpaint  # noqa: E402

GEN = ROOT / "data/cs/39/outputs/AI-Generated-ee8fe9bc-8eb2-41cf-91f6-c7e0638d219f.png"
SELLER = ROOT / "data/cs/39/uploads/982d21a5-1b9a-4474-bc80-c1ec1b22b0b2.webp"
OUT = ROOT / "out/showcase/c39"

# Seller close-up: a black pen's needle, rough tip and the point where it enters the ferrule.
# Refined by fit_needle(). The paper patch is plain paper next to the needles.
SRC_TIP, SRC_BACK, SRC_PAPER = (500.0, 330.0), (637.0, 219.0), (600, 300, 700, 360)
# Generated cut: region around the cone and nib (set by hand); the axis and the nib are
# measured inside it. The stroke lies left of the nib.
ROI = (340, 430, 620, 680)
STROKE_ROI = (0, 560, 400, 720)
TIP_RATIO = 3.6       # needle length / barrel width (needle about 45 mm, barrel about 12 mm)
COLLAR = 9            # source pixels above the needle: the metal ferrule
BALL = 14             # source pixels of the dark ball at the needle's point
LINE_HALF = 3.0       # half width of the thinned stroke, in pixels of the generated cut
TAPER = 70            # length over which the generated cone narrows to the ferrule
FERRULE_HALF = 6.0    # half width of the cone where it meets the ferrule
SHADOW = 0.16         # darkest point of the needle's shadow


# ---------------------------------------------------------------- source tip
def fit_needle(src, tip, back, paper_box, thr=45):
    """Centre line of the needle: the two dark edges, step by step along the rough axis."""
    L = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    x0, y0, x1, y1 = paper_box
    paper = float(np.median(L[y0:y1, x0:x1]))
    tip, back = np.array(tip), np.array(back)
    u = (back - tip) / np.linalg.norm(back - tip)
    p = np.array([-u[1], u[0]])
    ts, cs, ws = [], [], []
    for t in np.arange(25, np.linalg.norm(back - tip) - 8, 3):
        c = tip + u * t
        rs = np.arange(-7, 7.25, 0.25)
        q = c[None] + rs[:, None] * p[None]
        v = cv2.remap(L, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        dark = np.abs(v - paper) > thr
        if dark.sum() < 2:
            continue
        lo, hi = rs[dark].min(), rs[dark].max()
        ts.append(t); cs.append((lo + hi) / 2); ws.append(hi - lo)
    a, b = np.polyfit(ts, cs, 1)
    half = float(np.median(ws)) / 2
    t_back = np.linalg.norm(back - tip)
    return tip + p * b, back + p * (a * t_back + b), half, paper


def tip_patch(src, tip, back, half, paper, dst_len):
    """Sampling frame of the real tip: a function mapping (s, r) in destination pixels (s along
    the axis from the point toward the ferrule, r across) to source coordinates, the alpha of
    the tip in the same frame, and the scale (source px per destination px)."""
    n = np.linalg.norm(back - tip)
    u = (back - tip) / n
    p = np.array([-u[1], u[0]])
    k = n / dst_len
    L = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    x, y = int(tip[0]), int(tip[1])
    paper_here = float(np.percentile(L[y - 25:y + 25, x - 25:x + 25], 90))   # the paper by the point

    def to_src(s, r):
        return tip[None] + (s * k)[:, None] * u[None] + (r * k)[:, None] * p[None]

    def alpha(s, r):
        q = to_src(s, r)
        v = cv2.remap(L, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        dark = np.clip((paper_here - v - 15) / 25, 0, 1)          # darker than the paper
        sk, rs = s * k, r * k
        needle = (sk >= BALL) & (sk <= n) & (np.abs(rs) <= half + 0.8)
        ferrule = (sk > n) & (sk <= n + COLLAR) & (np.abs(rs) <= half + 2.5)
        ball = (sk >= -2) & (sk < BALL) & (rs >= -4) & (rs <= 7)   # the dark ball point, off centre
        a = np.where(needle | ferrule, np.maximum(dark, 0.92), 0.0)  # metal highlights are paper-bright
        return np.where(ball, dark, a)
    return to_src, alpha, k


# ---------------------------------------------------------------- generated pen
def dark_profile(L, centre, ax, ts, half=35, dark_l=70):
    """Width and centre offset of the dark run on the axis (the contiguous run nearest to it,
    so a separate dark spot beside the pen does not count), at each step along it."""
    perp = np.array([-ax[1], ax[0]])
    rs = np.arange(-half, half + 0.5, 1.0)
    out = []
    for t in ts:
        q = (centre + ax * t)[None] + rs[:, None] * perp[None]
        v = cv2.remap(L, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        dark = v < dark_l
        if not dark.any():
            out.append((float(t), 0.0, 0.0))
            continue
        i = int(np.argmin(np.where(dark, np.abs(rs), np.inf)))
        lo = hi = i
        while lo > 0 and dark[lo - 1]:
            lo -= 1
        while hi < len(rs) - 1 and dark[hi + 1]:
            hi += 1
        out.append((float(t), float(rs[hi] - rs[lo]), float((rs[hi] + rs[lo]) / 2)))
    return out


def pen_axis(gen, roi, dark_l=70):
    """Pen axis (unit vector toward the nib), the point where the nib meets the paper, the
    cone's half width, the barrel width just above the cone, and the cone's centre."""
    L = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    x0, y0, x1, y1 = roi
    ys, xs = np.nonzero(L[y0:y1, x0:x1] < dark_l)
    pts = np.c_[xs + x0, ys + y0].astype(float)
    cone = pts[(pts[:, 1] < y1 - 45) & (pts[:, 0] > x0 + 45)]     # stay clear of the stroke
    m = cone.mean(0)
    ax = np.linalg.svd(cone - m)[2][0]
    ax = -ax if ax[0] > 0 else ax                                 # the nib is down-left
    perp = np.array([-ax[1], ax[0]])
    prof = dark_profile(L, m, ax, np.arange(-60, 260, 2.0))
    w0 = float(np.median([w for t, w, c in prof if 0 <= t <= 100]))
    # brightness on the run's centre: the cone is black plastic, the broad nib a lighter felt
    # grey, the stroke the darkest
    lum = []
    for t, w, c in prof:
        q = (m + ax * t + perp * c)[None] + np.arange(-6, 7)[:, None] * perp[None]
        v = cv2.remap(L, q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        lum.append((t, float(v.mean()), c))
    cone_l = float(np.median([v for t, v, c in lum if 0 <= t <= 100]))
    nib_start = next(t for t, v, c in lum if t > 100 and v > cone_l + 6)
    t_pt, _, c_pt = next((t, v, c) for t, v, c in lum if t > nib_start and v < cone_l - 3)
    point = m + ax * t_pt + perp * c_pt
    t = (pts - m) @ ax
    w = np.abs((pts - m) @ perp)
    widths = [np.ptp(((pts - m) @ perp)[(t > tt) & (t < tt + 6) & (w < 40)]) for tt in range(-70, -40, 6)]
    return ax, point, w0 / 2, float(np.median(widths)), m, nib_start


def stroke_mask(gen, stroke_roi, point, ax):
    """The drawn stroke: the largest dark component on the far side of the nib's point."""
    L = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB)[..., 0]
    x0, y0, x1, y1 = stroke_roi
    m = np.zeros(L.shape, bool)
    m[y0:y1, x0:x1] = L[y0:y1, x0:x1] < 90
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    s = (xx - point[0]) * -ax[0] + (yy - point[1]) * -ax[1]       # toward the pen
    m &= s < 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return lab == 1 + int(np.argmax(st[1:, 4]))


def thin(mask, half, point, ax, end_skip=12):
    """Soft alpha of a band of the given half width around the stroke's centre line. The
    centre line near the broad nib branches (the nib's square end), so it is cut back by
    end_skip pixels and joined to the point in a straight line."""
    sk = skeletonize(mask)
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    s = (xx - point[0]) * -ax[0] + (yy - point[1]) * -ax[1]
    sk &= s < -end_skip
    ys, xs = np.nonzero(sk)
    j = int(np.argmax(s[ys, xs]))                                  # the end nearest the nib
    cv2.line(sk.view(np.uint8), (int(xs[j]), int(ys[j])), (int(round(point[0])), int(round(point[1]))), 1, 1)
    d = cv2.distanceTransform((sk == 0).astype(np.uint8), cv2.DIST_L2, 5)
    a = np.clip(half + 0.5 - d, 0, 1)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)
    return np.clip((a - 0.25) / 0.5, 0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen, src = cv2.imread(str(GEN)), cv2.imread(str(SELLER))
    t0 = time.time()
    tip, back, half, paper = fit_needle(src, SRC_TIP, SRC_BACK, SRC_PAPER)
    ax, point, cone_half, barrel, centre, nib_start = pen_axis(gen, ROI)
    dst_len = TIP_RATIO * barrel
    to_src, alpha_fn, k = tip_patch(src, tip, back, half, paper, dst_len)

    Hh, Ww = gen.shape[:2]
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    perp = np.array([-ax[1], ax[0]])
    up = -ax                                                       # from the point toward the hand
    s = (xx - point[0]) * up[0] + (yy - point[1]) * up[1]
    r = (xx - point[0]) * perp[0] + (yy - point[1]) * perp[1]

    # the real tip (point, needle, ferrule) sampled from the seller's close-up
    band = (s > -14) & (s < dst_len + COLLAR / k + 2) & (np.abs(r) < 14 / k + 6)
    sv, rv = s[band], r[band]
    q = to_src(sv, rv)
    a = np.zeros((Hh, Ww), np.float32)
    a[band] = alpha_fn(sv, rv)
    patch = np.zeros_like(gen)
    for ch in range(3):
        patch[..., ch][band] = cv2.remap(src[..., ch], q[:, 0].astype(np.float32)[None], q[:, 1].astype(np.float32)[None], cv2.INTER_CUBIC)[0]

    # 1. paper: erase the cone below the ferrule, the nib, their shadow, and the whole stroke
    stroke = stroke_mask(gen, STROKE_ROI, point, ax)
    s_top = dst_len + TAPER
    lower = (s > -16) & (s < dst_len) & (np.abs(r) < 30)
    lower |= (s > -16) & (s < 60) & (np.abs(r) < 42)              # the nib's shadow on the paper
    tapering = (s >= dst_len) & (s < s_top) & (np.abs(r) < 30)
    erase = lower | tapering | (cv2.dilate(stroke.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    filled, dev = inpaint(gen, erase.astype(np.uint8) * 255)
    out = filled.astype(np.float32)

    # 2. the cone above the ferrule, narrowed toward it like the real nose: its own pixels,
    # squeezed across the axis so its shading and rim stay
    nat = cone_half + 5.0                                          # dark core and its soft rim
    tw = np.interp(s, [dst_len, s_top], [FERRULE_HALF, nat])
    rr = r * (nat / np.maximum(tw, 1e-3))
    sx = point[0] + s * up[0] + rr * perp[0]
    sy = point[1] + s * up[1] + rr * perp[1]
    squeezed = cv2.remap(gen, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR)
    edge = np.clip(tw - np.abs(r) + 0.5, 0, 1) * tapering
    edge = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 0.5)[..., None]
    out = out * (1 - edge) + squeezed.astype(np.float32) * edge

    # 3. the stroke as the real tip draws it: along its own centre line, thin, in its own ink
    # colour (under the broad nib the line is joined straight to the point)
    ka = thin(stroke, LINE_HALF, point, ax)[..., None]
    core = cv2.erode(stroke.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ink = np.median(gen[core], axis=0).astype(np.float32)
    out = out * (1 - ka) + ink[None, None] * ka

    # 4. soft shadow of the needle on the paper, meeting it at the point (down-right, away
    # from the light)
    lift = np.clip(s / dst_len, 0, 1)
    sh = np.zeros((Hh, Ww), np.float32)
    for frac in np.linspace(0, 1, 14):
        dx, dy = (-perp) * 7 * frac
        layer = ((a > 0.3) & (np.abs(lift - frac) < 0.08)).astype(np.float32)
        sh = np.maximum(sh, cv2.warpAffine(layer, np.float32([[1, 0, dx], [0, 1, dy]]), (Ww, Hh)))
    sh = cv2.GaussianBlur(sh, (0, 0), 1.8) * SHADOW
    out *= (1 - sh[..., None])

    # 5. the real tip on top
    aa = cv2.GaussianBlur(a, (0, 0), 0.5)[..., None]
    out = out * (1 - aa) + patch.astype(np.float32) * aa
    out = np.clip(out, 0, 255).astype(np.uint8)
    seconds = round(time.time() - t0, 2)
    cv2.imwrite(str(OUT / "after.png"), out)

    changed = np.abs(out.astype(np.int16) - gen.astype(np.int16)).max(2) > 2
    work = cv2.dilate((erase | (a > 0) | (sh > 0.005)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.nonzero(work)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    report = dict(case=39, gen=GEN.name, seller=SELLER.name, lama_device=dev, seconds=seconds,
                  source_tip=[round(float(v), 1) for v in tip], source_back=[round(float(v), 1) for v in back],
                  source_half_width=round(half, 2), axis=[round(float(v), 3) for v in ax],
                  point=[round(float(v), 1) for v in point], nib_start=nib_start, cone_half=round(cone_half, 1),
                  barrel_px=round(barrel, 1), tip_len_px=round(dst_len, 1), scale_src_per_dst=round(k, 3),
                  changed_px=int(changed.sum()), changed_outside_work=int((changed & ~work).sum()), work_box=box)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    slides(gen, src, out, report)
    return gen, src, out, report


# ---------------------------------------------------------------- slides
SELLER_VIEW = (0, 0, 780, 484)        # the seller's photo of the pens and the close-up of the tips
SELLER_TIPS = (430, 120, 770, 420)    # the close-up of the tips
TIP_VIEW = (290, 410, 630, 705)       # around the pen tip in the generated cut


def slides(gen, src, out, report):
    from PIL import Image, ImageDraw
    from validation.showcase import AFTER, BEFORE, BG, H, INK, LINE, SUB, W, fit, font, header, panel, pil

    def crop(img, b):
        return img[b[1]:b[3], b[0]:b[2]]

    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    header(d, "AI가 바꾼 펜 끝 모양을 셀러 사진의 실물로 되돌립니다",
           "이 상품의 핵심인 길고 가는 금속 심이, AI 생성에서는 일반 마커의 굵은 심으로 바뀌었습니다. 그 부분만 실물로 바꿉니다.")
    cols = [(80, 250, 560, 600), (680, 250, 560, 600), (1280, 250, 560, 600)]
    panel(c, d, pil(crop(src, SELLER_VIEW)), cols[0], "셀러 원본 사진")
    panel(c, d, pil(gen), cols[1], "AI 생성 결과 (처리 전)", BEFORE)
    panel(c, d, pil(out), cols[2], "처리 후", AFTER)
    d.line([(80, 900), (1840, 900)], fill=LINE, width=1)
    note = ["생성 모델을 다시 돌리지 않습니다. 셀러 사진에서 실물 심을 가져와 생성된 펜의 축에 맞춰 넣고, 굵은 심과 굵은 선은 지웁니다.",
            f"처리에 {report['seconds']:.1f}초, 펜 끝과 선 주변 바깥 픽셀은 하나도 바뀌지 않습니다. 선은 생성된 선의 경로를 따라 실물 심 굵기로 남깁니다.",
            "평가 뒤에 만든 시연본으로, 블라인드 평가는 거치지 않았습니다. 펜 끝 주변 영역은 사람이 지정했고, 축과 심 끝 위치는 자동으로 찾았습니다. 라벨 글씨는 그대로입니다."]
    for i, t in enumerate(note):
        d.text((80, 922 + i * 40), t, font=font(25, 0), fill=SUB if i else INK)
    c.save(OUT / "slide1_overview.jpg", quality=95)

    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    header(d, "가까이 보면: 굵은 심이 실물의 긴 금속 심으로 바뀝니다",
           "왼쪽은 셀러 사진의 실물 심, 가운데는 AI 생성 그대로, 오른쪽은 처리 후입니다.")
    cols = [(80, 270, 560, 700), (680, 270, 560, 700), (1280, 270, 560, 700)]
    panel(c, d, pil(crop(src, SELLER_TIPS)), cols[0], "셀러 원본")
    panel(c, d, pil(crop(gen, TIP_VIEW)), cols[1], "처리 전", BEFORE)
    panel(c, d, pil(crop(out, TIP_VIEW)), cols[2], "처리 후", AFTER)
    c.save(OUT / "slide2_closeup.jpg", quality=95)

    pages = [Image.open(OUT / f) for f in ("slide1_overview.jpg", "slide2_closeup.jpg")]
    pages[0].save(OUT / "slides.pdf", save_all=True, append_images=pages[1:], resolution=144)
    both = OUT.parent / "c19"
    if (both / "slides.pdf").exists():
        cans = [Image.open(both / f) for f in ("slide1_overview.jpg", "slide2_closeups.jpg", "slide3_zoom.jpg")]
        cans[0].save(OUT.parent / "slides_print_and_shape.pdf", save_all=True, append_images=cans[1:] + pages, resolution=144)


if __name__ == "__main__":
    main()

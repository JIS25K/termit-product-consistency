"""Match the pasted real product's crispness and grain to the generated scene.

Seller photos are retouched studio shots: smooth surfaces, no sensor grain, edges softened by the
cutout's feathering and by resizing. Pasted into a generated image that has crisp edges and fine
grain, the real product reads as blurry even when it is not blurred. Two numbers decide:
  sharpness  on the strongest edges, mean |Laplacian| / mean |gradient|. A soft edge has a weak
             second derivative for its contrast, so the ratio falls as edges widen; it does not
             depend on how strong the edges are or what they belong to.
  grain      median absolute deviation from a 3x3 median filter (robust noise level).
The product's pixels are unsharp-masked until their edge ratio reaches the generated product's,
the cutout's edge is tightened the same way, and luma grain is added up to the scene's level.
Never blurs: a pasted product sharper than the scene is left alone here.
"""
import numpy as np
import cv2


def gray(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def sharpness(img, m, top=95):
    g = gray(img)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    v = gm[m]
    if v.size < 50:
        return None
    sel = m & (gm >= np.percentile(v, top))
    return float(lap[sel].mean() / max(gm[sel].mean(), 1e-6))


def grain(img, m):
    g = gray(img)
    med = cv2.medianBlur(np.clip(g, 0, 255).astype(np.uint8), 3).astype(np.float32)
    return float(np.median(np.abs(g - med)[m]) * 1.48)


def match_detail(fg, al, base, scene, gen_mask, amounts=(0.0, 0.25, 0.5, 0.8, 1.2, 1.7, 2.3), sigma=1.0, seed=0):
    """fg/al: the real product layer and its alpha on the scene's canvas; base: what it is
    composited over; scene + gen_mask: the generated image and its product. Returns the new
    layer, the new alpha and what was done."""
    band = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m_gen, m_ours = band(gen_mask > 0), band(al > 0.5)
    target = sharpness(scene, m_gen)
    comp = lambda f, a: base * (1 - a[..., None]) + f * a[..., None]
    before = sharpness(comp(fg, al), m_ours)
    info = dict(target=None if target is None else round(target, 3), before=None if before is None else round(before, 3))
    if target is None or before is None:
        return fg, al, {**info, "amount": 0.0, "grain": 0.0}
    blur_f = cv2.GaussianBlur(fg, (0, 0), sigma)
    blur_a = cv2.GaussianBlur(al, (0, 0), sigma)
    best = (abs(before - target), 0.0, fg, al, before)
    if before < 0.97 * target:
        for amt in amounts[1:]:
            f2 = np.clip(fg + amt * (fg - blur_f), 0, 255)
            a2 = np.clip(al + amt * (al - blur_a), 0, 1)
            s2 = sharpness(comp(f2, a2), m_ours)
            if abs(s2 - target) < best[0]:
                best = (abs(s2 - target), amt, f2, a2, s2)
            if s2 >= target:
                break
    _, amt, fg2, al2, after = best
    # grain: the scene's noise level over its product, added to ours as luma noise
    inner = lambda m: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ng, nt = grain(scene, inner(gen_mask > 0)), grain(fg2, inner(al2 > 0.5))
    add = float(np.sqrt(max(0.0, ng ** 2 - nt ** 2)))
    if add > 0.3:
        n = np.random.default_rng(seed).normal(0, add, fg2.shape[:2]).astype(np.float32)
        fg2 = np.clip(fg2 + n[..., None], 0, 255)
    return fg2, al2, {**info, "amount": amt, "after": round(after, 3), "grain_scene": round(ng, 2), "grain_ours": round(nt, 2), "grain": round(add, 2)}

"""Spike: same scene, real product. Erase the generated shelf with SDXL inpainting, paste the
real shelf cutout at the same place, tint it to the room light, add contact shadows."""
import os, sys, json, numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import place as P

ROOT = P.ROOT
_pipe = None

def pipe():
    global _pipe
    if _pipe is None:
        from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        _pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", vae=vae, torch_dtype=torch.float16, variant="fp16")
        _pipe.to("mps")
    return _pipe

def erase(scene_bgr, mask, prompt, negative, seed=0, steps=30):
    H, W = scene_bgr.shape[:2]
    s = 1024 / max(H, W)
    w8, h8 = int(W * s) // 8 * 8, int(H * s) // 8 * 8
    img = Image.fromarray(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)).resize((w8, h8))
    m = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((w8, h8))
    g = torch.Generator("cpu").manual_seed(seed)
    out = pipe()(prompt=prompt, negative_prompt=negative, image=img, mask_image=m, width=w8, height=h8,
                 num_inference_steps=steps, guidance_scale=7.0, strength=0.99, generator=g).images[0]
    out = cv2.cvtColor(np.array(out.resize((W, H), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    # keep original pixels outside the mask exactly (hard re-composite with a soft edge)
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 3)[..., None]
    return (out * soft + scene_bgr * (1 - soft)).astype(np.uint8)

def tint(fg, al, scene, strength=0.6):
    """Shift the cutout's Lab mean toward the room light: compare the scene ring around the
    product with the plain studio background the photo was taken on (near-neutral)."""
    lab_s = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = (cv2.dilate((al > 0.1).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & (al < 0.05)
    a_s, b_s = lab_s[..., 1][ring].mean() - 128, lab_s[..., 2][ring].mean() - 128
    L_s = lab_s[..., 0][ring].mean()
    lab_f = cv2.cvtColor(np.clip(fg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    msk = al > 0.5
    lab_f[..., 1][msk] += strength * a_s
    lab_f[..., 2][msk] += strength * b_s
    # brightness: studio walls are ~235 L; scale toward the room's wall brightness
    lab_f[..., 0][msk] *= 1 + strength * 0.5 * (L_s / 235.0 - 1)
    return cv2.cvtColor(np.clip(lab_f, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

def shadows(scene, al, light_dx=-0.15, dark=0.55):
    """Contact shadows: soft dark ellipses under each leg foot and a soft band under the
    lowest board, offset a little away from the light."""
    H, W = al.shape
    op = (al > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(op)
    floor_y = ys.max()
    sh = np.zeros((H, W), np.float32)
    # leg feet: columns with opaque pixels in the last 6 rows
    feet = np.nonzero(op[floor_y - 6:floor_y + 1].any(0))[0]
    if len(feet):
        groups = np.split(feet, np.nonzero(np.diff(feet) > 3)[0] + 1)
        for gcols in groups:
            if len(gcols) > 0.06 * (xs.max() - xs.min() + 1):  # a wide run is not a foot
                continue
            cx, wleg = int(gcols.mean()), len(gcols)
            cv2.ellipse(sh, (cx + int(light_dx * 40), floor_y), (int(wleg * 2.2), max(3, wleg // 2)), 0, 0, 360, 1.0, -1)
    # band under the lowest board: from the board's bottom to the floor
    # the floor seen under the lowest board, following the board's own bottom edge (a slanted
    # board in an oblique view gets a slanted band): per column, from the board's lowest pixel
    # down by the leg gap (feet line minus board bottom, measured at the legs)
    wide = cv2.morphologyEx(op, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8))  # boards, not legs
    cols = np.nonzero(wide.any(0))[0]
    if len(cols):
        bot = np.array([np.nonzero(wide[:, c])[0].max() for c in cols])
        legcols = [g for g in (np.split(feet, np.nonzero(np.diff(feet) > 3)[0] + 1) if len(feet) else [])
                   if len(g) <= 0.06 * (xs.max() - xs.min() + 1)]
        gaps = []
        for g in legcols:
            c = int(g.mean())
            if wide[:, c].any():
                gaps.append(floor_y - np.nonzero(wide[:, c])[0].max())
        gap = int(np.median(gaps)) if gaps else int(0.035 * (ys.max() - ys.min()))
        gap = max(gap, 4)
        for c, b in zip(cols[5:-5], bot[5:-5]):
            sh[b + 1:b + 1 + gap, c] = np.maximum(sh[b + 1:b + 1 + gap, c], 0.55)
    sh = cv2.GaussianBlur(sh, (0, 0), 9)
    sh *= (1 - al)  # never darken the product itself
    return (scene.astype(np.float32) * (1 - dark * sh[..., None])).astype(np.uint8)

def run(k, ri, prompt, negative, extra_boxes=(), seed=0, pad=24):
    scene = cv2.imread(f"{ROOT}/data/cs/13/outputs/{P.OUTS[k]}")
    g, dets = P.gen_box(scene)
    mask = np.zeros(scene.shape[:2], np.uint8)
    for d in dets:
        if d["query"] in ("a shelf", "a bookshelf") and d["score"] >= 0.15:
            x0, y0, x1, y1 = d["box"]
            mask[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad] = 1
    for (x0, y0, x1, y1) in extra_boxes:
        mask[y0:y1, x0:x1] = 1
    plate = erase(scene, mask, prompt, negative, seed=seed)
    im, a, rb = P.real(ri)
    fg, al, pb = P.place(plate, im, a, rb, g)
    fg = tint(fg, al, plate)
    base = shadows(plate, al)
    comp = (base.astype(np.float32) * (1 - al[..., None]) + fg * al[..., None]).astype(np.uint8)
    tag = f"O{k}_u{ri}_s{seed}"
    cv2.imwrite(f"{ROOT}/out/exp13/plate_{tag}.jpg", plate)
    cv2.imwrite(f"{ROOT}/out/exp13/ours_{tag}.png", comp)
    return comp

if __name__ == "__main__":
    import time
    t = time.time()
    NEG = "shelf, shelving, bookcase, furniture, table, cabinet, metal rods, poles, text, watermark, logo"
    run(0, 0, "a bright minimalist living room, white wall, light oak wooden floor, beige rug, natural daylight from a window, empty space against the wall, photorealistic interior photo", NEG)
    print("O0 done", round(time.time() - t), "s", flush=True)
    run(13, 0, "an empty seamless light grey studio backdrop and floor, soft even studio lighting, photorealistic product photo background", NEG)
    print("O13 done", round(time.time() - t), "s", flush=True)

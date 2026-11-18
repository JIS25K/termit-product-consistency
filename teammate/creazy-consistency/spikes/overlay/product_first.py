"""Product first: draw the real product before anything is generated, let the generator draw only
the room around it, then put the product's own pixels back unchanged.

The other pipelines generate the whole picture and repair the product afterwards; the hardest
step there is reading the angle and place of a product the generator has already deformed.
Here the angle and place are chosen first, so nothing has to be read back:
  1. camera and placement: chosen (for this comparison, the cameras reproject.py found for the
     CS 13 scenes, so the framing matches the CS outputs)
  2. the real shelf at that camera: reproject.render (seller photo pixels on the 3D proxy)
  3. the room's geometry from the same camera: floor plane and back wall, plus the shelf, as a
     depth map; the generator (SDXL inpainting + depth ControlNet, local) fills everything
     except the shelf, so floor and wall meet the shelf in its own perspective
  4. the shelf's pixels go back on top as they were rendered; floor shadow from the proxy; the
     room light's tint; crispness and grain of the generated room (detail.py); chrome legs take
     a little of the room's colour (they mirror it)

  python spikes/overlay/product_first.py            # all scenes, two seeds
  python spikes/overlay/product_first.py room_left  # one scene
Outputs: out/product_first/<scene>_s<seed>_{canvas,depth,mask,gen,final}.png and <scene>.json
"""
import os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import reproject as R, compose as C, detail as D

ROOT = R.ROOT
OUT = f"{ROOT}/out/product_first"
NEG = ("shelf, bookcase, furniture in the middle, extra legs, poles, table, cabinet, people, text, watermark, logo, "
       "cartoon, blurry, distorted perspective, floating objects")
SCENES = {
    # framing and angle of CS 13 output 0: living room, seen a little from the left
    "room_left": dict(ref=0, wall=0.30, prompt=(
        "a bright modern living room interior, white wall with a framed abstract art print, light oak wooden floor, "
        "large window with soft daylight on the right, a dark grey armchair at the right edge, a soft beige rug, "
        "photorealistic interior photography, natural light")),
    # output 2: sunny living room, seen from the right
    "sunny_right": dict(ref=2, wall=0.30, prompt=(
        "a bright scandinavian living room, white walls, light wooden floor with warm sunlight streaming through a large "
        "window on the left, a potted olive tree in a white planter, a white built-in bookcase on the right, "
        "photorealistic interior photo")),
    # output 9: white studio, seen from the left
    "studio_left": dict(ref=9, wall=1.2, prompt=(
        "a seamless pure white photo studio background, soft even light, clean white floor, product photography, "
        "high key")),
}


def room_depth(c, H, W, dims, wall_gap):
    """Distance to the floor (y = 0) or the back wall (z = shelf depth + gap), whichever comes first."""
    d = R.rays(c, H, W)
    o = c["eye"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(d[..., 1] < -1e-6, -o[1] / d[..., 1], np.inf)
        zw = dims["D"] + wall_gap
        tw = np.where(d[..., 2] > 1e-6, (zw - o[2]) / d[..., 2], np.inf)
    return np.minimum(tf, tw)


def blockout(c, H, W, dims, wall_gap, floor_bgr=(126, 167, 200), wall_bgr=(228, 233, 236), plank=0.16, room=True):
    """A plain painted room from the same camera: floor (wood tone with plank seams in true
    perspective), back wall, baseboard. Given as the starting picture, it tells an image
    generator where the floor, the wall and the horizon are, so it keeps the layout and only
    adds materials, light and furniture."""
    d = R.rays(c, H, W)
    o = c["eye"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(d[..., 1] < -1e-6, -o[1] / d[..., 1], np.inf)
        zw = dims["D"] + wall_gap
        tw = np.where(d[..., 2] > 1e-6, (zw - o[2]) / d[..., 2], np.inf)
    floor = tf < tw
    img = np.empty((H, W, 3), np.float32)
    img[:] = wall_bgr
    P3 = o[None, None] + np.where(floor, tf, tw)[..., None] * d
    seam = (np.abs(((P3[..., 0] / plank) % 1.0) - 0.5) > 0.47) & room
    img[floor] = floor_bgr
    img[floor & seam] = np.array(floor_bgr, np.float32) * 0.85
    base = ~floor & (P3[..., 1] < 0.07) & room
    img[base] = np.array(wall_bgr, np.float32) * 1.02
    edge = ~floor & (np.abs(P3[..., 1] - 0.07) < 0.004) & room
    img[edge] = np.array(wall_bgr, np.float32) * 0.85
    shade = np.clip(1.0 - 0.15 * (np.arange(H) / H)[:, None], 0.8, 1.0)[..., None]
    return np.clip(img * shade, 0, 255).astype(np.uint8)


def depth_image(t):
    """Near = bright, as depth ControlNets expect (inverse distance, stretched over the frame)."""
    inv = np.where(np.isfinite(t), 1.0 / np.maximum(t, 1e-6), 0.0)
    lo, hi = np.percentile(inv[inv > 0], [1, 99.5])
    return (np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def layers(real, p, H, W):
    """The shelf as layers on the scene canvas: colour, opaque alpha, acrylic veil (all full size)."""
    L = R.render(real, p, H, W)
    ss = L["ss"]
    dn = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)
    return L, dn(L["col"]), dn(L["op"]), dn(L["veil"])


def run(name, seeds=(0, 1), real=None, cscale=0.7, dry=False):
    cfg = SCENES[name]
    real = real or R.Real(0)
    cams = json.load(open(f"{R.OUT}/reproject_scene_cams.json"))
    ref = cv2.imread(f"{ROOT}/data/cs/13/outputs/{R.P.OUTS[cfg['ref']]}")
    H, W = ref.shape[:2]
    p = tuple(cams[str(cfg["ref"])]["p"])
    c = R.cam(p)
    os.makedirs(OUT, exist_ok=True)
    L, col, op, veil = layers(real, p, H, W)
    # depth: room planes and the shelf, nearest wins
    t_shelf, pid, _, _ = R.trace(c, H, W, R.prims(False, real.dims))
    t = np.minimum(room_depth(c, H, W, real.dims, cfg["wall"]), t_shelf)
    depth = depth_image(t)
    # canvas: the shelf on mid grey; the generator draws everything that is not opaque shelf
    canvas = (col * op[..., None] + 128 * (1 - op[..., None])).astype(np.uint8)
    mask = (cv2.dilate((op < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.uint8)
    cv2.imwrite(f"{OUT}/{name}_canvas.png", canvas)
    cv2.imwrite(f"{OUT}/{name}_depth.png", depth)
    cv2.imwrite(f"{OUT}/{name}_mask.png", mask * 255)
    if dry:
        return dict(scene=name)
    import stage4 as S4  # loads SDXL inpainting + depth ControlNet on first use
    meta = dict(scene=name, camera=dict(az=round(p[0], 1), el=round(p[1], 1), dist=round(p[2], 2)), ref_output=R.P.OUTS[cfg["ref"]],
                prompt=cfg["prompt"], cscale=cscale, runs=[])
    for seed in seeds:
        t0 = time.time()
        gen = S4.generate(canvas, mask, depth, cfg["prompt"], NEG, seed, cscale, steps=30, strength=0.99)
        cv2.imwrite(f"{OUT}/{name}_s{seed}_gen.png", gen)
        final, info = put_back(gen, real, p, L, col, op, veil)
        cv2.imwrite(f"{OUT}/{name}_s{seed}_final.png", final)
        meta["runs"].append(dict(seed=seed, sec=round(time.time() - t0), **info))
        print(json.dumps(meta["runs"][-1]), flush=True)
        json.dump(meta, open(f"{OUT}/{name}.json", "w"), indent=1)
    return meta


def generate(canvas, mask, depth, prompt, seed, cscale=0.5, cn_end=0.8, steps=30, full=False, strength=1.0):
    """SDXL inpainting + depth ControlNet (stage4's pipeline). strength 1.0 starts the masked part
    from pure noise (at 0.99 the grey canvas shows through). full=True: the whole frame is
    generated (shelf included, following the depth), and the real shelf is laid over it later."""
    import torch
    from PIL import Image
    import stage4 as S4
    H, W = canvas.shape[:2]
    s = 1024 / max(H, W); w8, h8 = int(W * s) // 8 * 8, int(H * s) // 8 * 8
    m = np.ones_like(mask) if full else mask
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize((w8, h8))
    mm = Image.fromarray((m > 0).astype(np.uint8) * 255).resize((w8, h8))
    ctrl = Image.fromarray(depth).convert("RGB").resize((w8, h8))
    g = torch.Generator("cpu").manual_seed(seed)
    out = S4.cn_pipe()(prompt=prompt, negative_prompt=NEG, image=img, mask_image=mm, control_image=ctrl,
                       controlnet_conditioning_scale=cscale, control_guidance_end=cn_end, width=w8, height=h8,
                       num_inference_steps=steps, guidance_scale=7.5, strength=strength, generator=g).images[0]
    return cv2.cvtColor(np.array(out.resize((W, H), Image.LANCZOS)), cv2.COLOR_RGB2BGR)


def relight(fg, op, gen, sigma, clip_L=(0.75, 1.3), clip_ab=10.0):
    """Low frequencies (brightness, light colour, shading across the product) from the generated
    image at the product's place; high frequencies (grain, print, edges) stay the real photo's.
    Normalized blur inside the product only, so the room around does not bleed in."""
    m = (op > 0.5).astype(np.float32)
    lf = cv2.cvtColor(np.clip(fg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lg = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    low = lambda x: cv2.GaussianBlur(x * m, (0, 0), sigma) / den
    out = lf.copy()
    ratio = np.clip((low(lg[..., 0]) + 1) / (low(lf[..., 0]) + 1), *clip_L)
    out[..., 0] = lf[..., 0] * ratio
    for c in (1, 2):
        out[..., c] = lf[..., c] + np.clip(low(lg[..., c]) - low(lf[..., c]), -clip_ab, clip_ab)
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return np.where(m[..., None] > 0, res, fg), dict(L_ratio_median=round(float(np.median(ratio[m > 0])), 3))


def ring_regenerate(image, op, ring, depth, prompt, seed=0, cscale=0.5, strength=1.0):
    """The product stays locked; only the ring around it (floor contact, the floor under it,
    the edge against the room) is drawn again by SDXL inpainting, which keeps unmasked pixels.
    The generator then draws shadows and contact for the real shape. The ring's soft edge is
    zero on the product, so its pixels come back unchanged."""
    gen = generate(image, ring.astype(np.uint8), depth, prompt, seed, cscale=cscale, cn_end=0.8, strength=strength)
    prod = cv2.dilate((op > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    soft = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 2.0) * ~prod
    out = image.astype(np.float32) * (1 - soft[..., None]) + gen.astype(np.float32) * soft[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8), gen


def put_back(gen, real, p, L, col, op, veil, legs_room=0.2, lowfreq=False, light_src=None):
    """Shelf pixels back on the generated room: floor shadow from the proxy, then either the
    generated image's low frequencies at the shelf's place (lowfreq: the generator drew the shelf
    in the room's light there) or a plain room-light tint, chrome legs mirroring a little of the
    room, a one-pixel feather on the edge, then crispness and grain of the room."""
    H, W = gen.shape[:2]
    base = gen.astype(np.float32) * (1 - 0.8 * R.floor_shadow(real, p, H, W)[..., None])
    rinfo = None
    src = gen if light_src is None else light_src  # where the generator drew its own shelf, if erased from gen
    pid = cv2.resize(L["pid"].astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
    legs = pid >= real.nb
    if lowfreq:
        # boards only: the generator may redraw the legs in another material (clear acrylic in
        # the first runs), and their light would turn chrome into glass; legs get the plain tint
        boards = (pid >= 0) & (pid < real.nb)
        fg, rinfo = relight(col.copy(), op * boards, src, sigma=max(6.0, 0.012 * max(H, W)))
        fg = np.where(legs[..., None], C.tint(col.copy(), op, gen), fg)
    else:
        fg = C.tint(col.copy(), op, gen)
    room = cv2.blur(gen.astype(np.float32), (61, 61))
    fg[legs] = (1 - legs_room) * fg[legs] + legs_room * room[legs]
    if light_src is not None:  # crispness of the generator's own shelf at the same place
        fg, al, dinfo = D.match_detail(fg, op, base, light_src.astype(np.float32), op > 0.5, amounts=(0.0, 0.25, 0.5, 0.8))
    else:  # no shelf drawn by the generator: the room right around the shelf
        ring = (cv2.dilate((op > 0.5).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & (op < 0.05)
        fg, al, dinfo = D.match_detail(fg, op, base, gen.astype(np.float32), ring, amounts=(0.0, 0.25, 0.5, 0.8))
    al = np.maximum(cv2.GaussianBlur(al, (0, 0), 0.7), al * (al > 0.98))  # soft one-pixel edge, solid inside
    out = base * (1 - al[..., None]) + fg * al[..., None]
    v = veil[..., None]
    out = out * (1 - v) + L["acr_col"][None, None] * v
    return np.clip(out, 0, 255).astype(np.uint8), dict(detail=dinfo, relight=rinfo)


if __name__ == "__main__":
    r = R.Real(0)
    for n in sys.argv[1:] or list(SCENES):
        print(n, json.dumps({k: v for k, v in run(n, real=r).items() if k != "runs"}), flush=True)

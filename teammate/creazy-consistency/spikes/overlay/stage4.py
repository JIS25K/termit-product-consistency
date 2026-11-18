"""Stage 4 spike: a view the seller never photographed. Erase the generated shelf with LaMa,
fit the 3D proxy's camera to the generated shelf (silhouette IoU over azimuth/elevation), build
a depth map (empty-room planes + proxy), and let SDXL inpainting + depth ControlNet draw the shelf
in that region. Structure comes from the proxy, materials and light from the model.

Output: out/exp13/stage4_O{k}_*.png and stage4_O{k}.json (camera, structure scores)."""
import os, sys, json, time, numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, place as P, select_view as V, proxy3d as X

ROOT = P.ROOT
_cn = None


def cn_pipe():
    global _cn
    if _cn is None:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=torch.float16, variant="fp16")
        _cn = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", controlnet=cn, vae=vae, torch_dtype=torch.float16, variant="fp16")
        _cn.to("mps")
    return _cn


def floor_line(scene, box):
    g = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=5))
    H, W = g.shape
    cols = np.ones(W, bool); cols[max(box[0] - 10, 0):box[2] + 10] = False
    lo, hi = int(H * 0.3), int(H * 0.85)
    return lo + int(np.argmax(gy[lo:hi][:, cols].mean(1)))


def planar_depth(H, W, y_floor, wall=0.22):
    rows = np.arange(H)[:, None].astype(np.float32)
    floor = wall + (1 - wall) * np.clip((rows - y_floor) / max(H - y_floor, 1), 0, 1) ** 0.8
    return np.where(rows >= y_floor, floor, wall) * np.ones((1, W), np.float32)


def fit_camera(part, shape, azs=range(-70, 71, 10), els=range(5, 31, 5), res=4):
    """Grid search: render the proxy silhouette, scale it to the generated shelf's extents (width
    fit, bottom aligned) and keep the camera with the best tolerant IoU. Then refine."""
    ys, xs = np.nonzero(part)
    ext = (int(np.percentile(xs, 0.5)), int(ys.min()), int(np.percentile(xs, 99.5)) + 1, int(np.percentile(ys, 99.5)) + 1)
    h, w = shape[0] // res, shape[1] // res
    r = max(5, int(0.02 * (ext[2] - ext[0]))) | 1
    def score(az, el):
        _, kind = X.render(X.camera(az, el), h, w)
        sil = cv2.resize((kind > 0).astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        best = max((V.tol_iou(V.fit_mask(sil.astype(np.float32), ext, shape, f), part.astype(np.uint8), r), f) for f in ("width", "height"))
        return best
    grid = {(az, el): score(az, el) for az in azs for el in els}
    (az, el), (s, f) = max(grid.items(), key=lambda kv: kv[1][0])
    for daz in (-5, 0, 5):
        for de in (-2.5, 0, 2.5):
            if not (min(els) <= el + de <= max(els)):  # interior shots: keep the camera height plausible
                continue
            sc = score(az + daz, el + de)
            if sc[0] > s:
                s, f, best = sc[0], sc[1], (az + daz, el + de)
                grid[best] = sc
    (az, el) = max(grid, key=lambda k: grid[k][0])
    return dict(az=float(az), el=float(el), iou=round(grid[(az, el)][0], 3), fit=grid[(az, el)][1], ext=ext)


def place_render(cam_fit, shape):
    """Full-resolution proxy depth and kind maps, scaled and moved onto the generated extents."""
    H, W = shape[:2]
    t, kind = X.render(X.camera(cam_fit["az"], cam_fit["el"]), H, W)
    sil = kind > 0
    ys, xs = np.nonzero(sil)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    gx0, gy0, gx1, gy1 = cam_fit["ext"]
    s = (gx1 - gx0) / (x1 - x0) if cam_fit["fit"] == "width" else (gy1 - gy0) / (y1 - y0)
    tx = (gx0 + gx1) / 2 - s * (x0 + x1) / 2
    ty = gy1 - s * y1
    A = np.float32([[s, 0, tx], [0, s, ty]])
    inv = np.where(np.isfinite(t), 1.0 / np.maximum(t, 1e-6), 0).astype(np.float32)
    inv_w = cv2.warpAffine(inv, A, (W, H), flags=cv2.INTER_NEAREST)
    kind_w = cv2.warpAffine(kind, A, (W, H), flags=cv2.INTER_NEAREST)
    return inv_w, kind_w


def depth_map(scene, gbox, inv, kind):
    """Empty-room planes plus the proxy, the proxy scaled so its lowest visible point sits on the
    floor plane's value at that row (near = bright, as depth ControlNets expect)."""
    H, W = scene.shape[:2]
    yf = floor_line(scene, gbox)
    room = planar_depth(H, W, yf)
    m = kind > 0
    if m.any():
        ys, xs = np.nonzero(m)
        yb = ys.max()
        ref = inv[yb][m[yb]].max()
        s = room[min(yb + 2, H - 1), 0] / max(ref, 1e-6)
        prox = np.clip(inv * s, 0, 1)
        room = np.where(m, np.maximum(prox, room), room)
    return (room * 255).astype(np.uint8), yf


def edge_follow(img, kind, tol=4):
    """Share of the proxy's outline pixels that have a strong image edge within `tol` px: how
    closely the generated shelf follows the proxy's structure."""
    k = (kind > 0).astype(np.uint8)
    outline = (k - cv2.erode(k, np.ones((3, 3), np.uint8))) > 0
    g = E.grad(img) > 30
    near = cv2.dilate(g.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    return round(float((outline & near).sum()) / max(int(outline.sum()), 1), 3)


def generate(plate, mask, depth, prompt, negative, seed, cscale, steps=30, strength=0.99):
    H, W = plate.shape[:2]
    s = 1024 / max(H, W); w8, h8 = int(W * s) // 8 * 8, int(H * s) // 8 * 8
    img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)).resize((w8, h8))
    m = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((w8, h8))
    ctrl = Image.fromarray(depth).convert("RGB").resize((w8, h8))
    g = torch.Generator("cpu").manual_seed(seed)
    out = cn_pipe()(prompt=prompt, negative_prompt=negative, image=img, mask_image=m, control_image=ctrl,
                    controlnet_conditioning_scale=cscale, width=w8, height=h8, num_inference_steps=steps,
                    guidance_scale=7.0, strength=strength, generator=g).images[0]
    out = cv2.cvtColor(np.array(out.resize((W, H), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 3)[..., None]
    return (out * soft + plate * (1 - soft)).astype(np.uint8)


def paint_proxy(plate, kind, inv):
    """Colour the proxy with the real product's materials before diffusion: a wood patch cut from
    the seller's photo for boards, light chrome grey for legs, a faint white veil for the clear
    acrylic. Diffusion at moderate strength then keeps this layout and only relights it."""
    im, a, _ = P.real(0)
    ys, xs = np.nonzero(a > 0.5)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    wood = (a > 0.5) & (hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 45)
    wy, wx = np.nonzero(wood)
    cy, cx = int(np.median(wy)), int(np.median(wx))
    patch = im[max(cy - 20, 0):cy + 20, max(cx - 120, 0):cx + 120]
    H, W = plate.shape[:2]
    tex = cv2.resize(patch, (W, H), interpolation=cv2.INTER_CUBIC)
    out = plate.astype(np.float32)
    shade = np.clip(inv / max(inv[kind > 0].max(), 1e-6), 0.6, 1.0)[..., None] if (kind > 0).any() else 1.0
    out = np.where((kind == 1)[..., None], tex * (0.85 + 0.15 * shade), out)
    leg = np.array([200, 200, 205], np.float32)
    out = np.where((kind == 3)[..., None], leg * shade, out)
    out = np.where((kind == 2)[..., None], 0.8 * out + 0.2 * 255, out)
    return np.clip(out, 0, 255).astype(np.uint8)


PROMPT = ("a minimalist shelf made of two light birch plywood boards and slim polished chrome round legs, "
          "a small clear acrylic top shelf over the right half, a few books on the lower board, "
          "in a cozy living room, warm natural light, photorealistic interior photo")
NEG = "deformed, extra legs, extra shelves, cabinet, drawers, doors, solid sides, text, watermark, logo, blurry, cartoon"


# camera chosen by eye from each scene's perspective (silhouette fitting follows the generated
# shelf's deformation, see fit_camera); position and size still come from the generated footprint
MANUAL = {7: (-30.0, 20.0), 1: (-10.0, 12.0)}


def run(k, seeds=(0, 1), scales=(0.9, 0.0), manual=True):
    scene = cv2.imread(f"{ROOT}/data/cs/13/outputs/{P.OUTS[k]}")
    part, M, core, (d, bg, dets) = V.gen_part(scene)
    t0 = time.time()
    if manual and k in MANUAL:
        ys, xs = np.nonzero(part)
        ext = (int(np.percentile(xs, 0.5)), int(ys.min()), int(np.percentile(xs, 99.5)) + 1, int(np.percentile(ys, 99.5)) + 1)
        cam = dict(az=MANUAL[k][0], el=MANUAL[k][1], iou=None, fit="width", ext=ext)
    else:
        cam = fit_camera(part, scene.shape)
    plate = E.flat_match(scene, E.lama_lowres(scene, M, 512), M)[0]
    inv, kind = place_render(cam, scene.shape)
    gbox = dets[0]["box"]
    depth, yf = depth_map(plate, gbox, inv, kind)
    mask = cv2.dilate(((M > 0) | (kind > 0)).astype(np.uint8), np.ones((15, 15), np.uint8))
    cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_plate.png", plate)
    cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_depth.png", depth)
    vis = plate.copy(); vis[kind == 1] = (120, 170, 210); vis[kind == 2] = (230, 200, 160); vis[kind == 3] = (60, 60, 60)
    cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_proxy.png", vis)
    meta = dict(k=k, camera={x: cam[x] for x in ("az", "el", "iou", "fit")}, floor_y=yf,
                generated_follow=edge_follow(scene, kind), runs=[])
    print(json.dumps(meta), round(time.time() - t0), "s", flush=True)
    for cs in scales:
        for seed in seeds:
            t = time.time()
            out = generate(plate, mask, depth, PROMPT, NEG, seed, cs)
            tag = f"c{int(cs * 10)}_s{seed}"
            cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_{tag}.png", out)
            meta["runs"].append(dict(cscale=cs, seed=seed, follow=edge_follow(out, kind), sec=round(time.time() - t)))
            print(json.dumps(meta["runs"][-1]), flush=True)
            json.dump(meta, open(f"{ROOT}/out/exp13/stage4_O{k}.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    for k in [int(x) for x in sys.argv[1:]] or [7]:
        run(k, seeds=(0, 1), scales=(0.9,)) if False else run(k, seeds=(0, 1), scales=(0.9, 0.0))


def setup(k):
    scene = cv2.imread(f"{ROOT}/data/cs/13/outputs/{P.OUTS[k]}")
    part, M, core, (d, bg, dets) = V.gen_part(scene)
    ys, xs = np.nonzero(part)
    ext = (int(np.percentile(xs, 0.5)), int(ys.min()), int(np.percentile(xs, 99.5)) + 1, int(np.percentile(ys, 99.5)) + 1)
    cam = dict(az=MANUAL[k][0], el=MANUAL[k][1], iou=None, fit="width", ext=ext)
    plate = cv2.imread(f"{ROOT}/out/exp13/stage4_O{k}_plate.png")
    inv, kind = place_render(cam, scene.shape)
    depth, yf = depth_map(plate, dets[0]["box"], inv, kind)
    mask = cv2.dilate(((M > 0) | (kind > 0)).astype(np.uint8), np.ones((15, 15), np.uint8))
    return scene, plate, inv, kind, depth, mask


def run_paint(k, seeds=(0, 1), strength=0.6, cscale=0.9):
    scene, plate, inv, kind, depth, mask = setup(k)
    painted = paint_proxy(plate, kind, inv)
    cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_painted.png", painted)
    meta = json.load(open(f"{ROOT}/out/exp13/stage4_O{k}.json"))
    for seed in seeds:
        t = time.time()
        out = generate(painted, mask, depth, PROMPT, NEG, seed, cscale, strength=strength)
        cv2.imwrite(f"{ROOT}/out/exp13/stage4_O{k}_paint_s{seed}.png", out)
        meta["runs"].append(dict(mode="painted", strength=strength, cscale=cscale, seed=seed, follow=edge_follow(out, kind), sec=round(time.time() - t)))
        print(json.dumps(meta["runs"][-1]), flush=True)
        json.dump(meta, open(f"{ROOT}/out/exp13/stage4_O{k}.json", "w"), indent=1)

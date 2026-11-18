"""Product first vs plain generation with the same generator (gpt-image-2), CS 13 shelf.

  A  plain: the seller's frontal photo + the scene sentence (asks to keep the shelf as it is)
  B  product first: the real shelf already drawn at the scene's camera (reproject.render) on a
     grey canvas + a mask that leaves only the shelf fixed + the same scene sentence; the shelf's
     own pixels are put back afterwards (product_first.put_back), after aligning to where the
     generator kept it

Three scenes in the framing of CS 13 outputs 0, 2 and 9, two images per condition: 12 calls,
about $0.06 each (medium, 1024x1024); stops at CAP_USD.

The seller's photo is customer data. The user allowed sending it (and the canvas drawn from it)
to the OpenAI API for this experiment only (2026-11-18).

  python spikes/creazy_exp/run_pf.py --dry      # inputs and prompts only
  python spikes/creazy_exp/run_pf.py --pilot    # one call
  python spikes/creazy_exp/run_pf.py           # the rest (skips files already made)
  python spikes/creazy_exp/run_pf.py --finish   # put the real shelf back on the B images
"""
import argparse, csv, json, os, sys, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "spikes" / "overlay"))
from run_b import cost_of, load_env  # noqa: E402

EXP = ROOT / "out" / "pf_api"
CAP_USD = 1.5  # approved: 12 calls, about $0.75
N = 2
CFG = dict(model="gpt-image-2", quality="medium", size="1024x1024",
           price_per_million={"text_in": 5, "image_in": 8, "image_out": 30})
LOG_COLS = ["time", "scene", "cond", "k", "file", "text_in_tokens", "image_in_tokens", "out_tokens", "cost_usd", "total_spent_usd", "error"]

# Light and camera first, so the room is made to fit the product (the product is then changed
# as little as possible). Light read off the seller's photo: neutral background (a*, b* about 0),
# brighter to the upper left, chrome highlights on the legs' left sides, the board's shadow
# shifted right: soft neutral daylight from the front left and a little above.
LIGHT = ("Lighting: soft, diffuse, neutral white daylight (about 5500 K) coming from the front left and a little "
         "above the shelf, with soft shadows falling slightly to the right; no hard sunlight.")
SCENES = {
    "room_left": dict(ref=0, pad=0,
                      camera="Camera: level (vertical lines straight), about 10 degrees to the left of the shelf's "
                             "front, eye a little above the middle board.",
                      scene="a bright modern living room with a white wall and a framed abstract art print, a light oak "
                            "wooden floor, a large window on the right, a dark grey armchair at the right edge and a "
                            "soft beige rug"),
    "sunny_right": dict(ref=2, pad=128,
                        camera="Camera: level (vertical lines straight), about 30 degrees to the right of the shelf's "
                               "front, eye a little above the middle board.",
                        scene="a bright scandinavian living room with white walls, a light wooden floor, soft daylight "
                              "from a large window on the left, a potted olive tree in a white planter and a white "
                              "built-in bookcase on the right"),
    "studio_left": dict(ref=9, pad=0,
                        camera="Camera: level (vertical lines straight), about 25 degrees to the left of the shelf's "
                               "front, a little above the shelf.",
                        scene="a seamless pure white photo studio with a clean white floor"),
}


def prompt_a(s):
    return (f"{s['camera']} {LIGHT} Photorealistic photo of {s['scene']}. The shelf from the attached product photo "
            f"stands on the floor. Keep the shelf exactly as in the product photo: same parts, proportions, materials "
            f"and layout. No other shelf, no text.")


def prompt_b(s):
    # pilots 1 and 2 (grey canvas, then a painted room) asked for the whole room; the generator
    # zoomed out to fit it, shrank and moved the shelf and dropped its clear top. Here the
    # picture is the layout: only materials and light are asked for, furniture only at the edges.
    return (f"{s['camera']} {LIGHT} Turn this rough render into a photorealistic photo of {s['scene']}. Keep the "
            f"exact framing and composition of this image: the shelf stays exactly where it is, at the same size and "
            f"angle, and keeps its clear acrylic top. Do not zoom out or move the camera. Other furniture may appear "
            f"only partly at the edges of the frame. Keep the shelf unchanged and put nothing on it. Make the floor "
            f"and wall look real, with soft natural contact shadows under the shelf's feet. No other shelf, no text.")


def build_inputs():
    """Product photo (A), and per scene the canvas and mask (B), all written under out/pf_api/inputs."""
    import reproject as R
    import product_first as PF
    inp = EXP / "inputs"; inp.mkdir(parents=True, exist_ok=True)
    real = R.Real(0)
    cv2.imwrite(str(inp / "product.png"), real.im)
    cams = json.load(open(f"{R.OUT}/reproject_scene_cams.json"))
    meta = {}
    for name, s in SCENES.items():
        p = list(cams[str(s["ref"])]["p"])
        p[6] += s["pad"]  # 1024 x 768 scene on a 1024 x 1024 canvas: same camera, image centre moved down
        p = tuple(p)
        H = W = 1024
        L, col, op, veil = PF.layers(real, p, H, W)
        # the starting picture: a plain painted room from the same camera (floor, wall, baseboard),
        # the floor shadow, then the real shelf; on an empty grey canvas the generator re-composed
        # the whole picture (pilot: shelf shrunk to half and moved, acrylic top dropped)
        room = PF.blockout(R.cam(p), H, W, real.dims, s.get("wall", 0.30), **({"floor_bgr": (238, 240, 242), "wall_bgr": (245, 246, 247), "room": False} if name == "studio_left" else {}))
        room = (room.astype(np.float32) * (1 - 0.7 * R.floor_shadow(real, p, H, W)[..., None]))
        canvas = room * (1 - op[..., None]) + col * op[..., None]
        canvas = canvas * (1 - veil[..., None]) + real.acr_edge[None, None] * veil[..., None]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        keep = (cv2.erode((op > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        rgba = np.dstack([canvas, np.where(keep, 255, 0).astype(np.uint8)])  # alpha 0 = the generator may draw
        cv2.imwrite(str(inp / f"{name}_canvas.png"), canvas)
        cv2.imwrite(str(inp / f"{name}_mask.png"), rgba)
        np.savez_compressed(inp / f"{name}_layers.npz", col=col, op=op, veil=veil, pid=L["pid"], p=np.array(p))
        meta[name] = dict(camera=[round(x, 3) for x in p], prompt_a=prompt_a(s), prompt_b=prompt_b(s))
    (EXP / "inputs" / "inputs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def call(paths, prompt, mask=None):
    files = [("image[]", (p.name, open(p, "rb"), "image/png")) for p in paths]
    if mask is not None:
        files.append(("mask", (mask.name, open(mask, "rb"), "image/png")))
    try:
        r = requests.post("https://api.openai.com/v1/images/edits",
                          headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                          data={"model": CFG["model"], "prompt": prompt, "size": CFG["size"], "quality": CFG["quality"], "n": "1"},
                          files=files, timeout=300)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def spent():
    f = EXP / "log.csv"
    if not f.exists():
        return 0.0
    with open(f) as fh:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(fh))


def log(row):
    f = EXP / "log.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def generate(pilot=False):
    import base64
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없어요.")
    inp = EXP / "inputs"
    out = EXP / "outputs"; out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, c, k) for n in SCENES for c in ("B", "A") for k in range(1, N + 1)]
    if pilot:
        jobs = jobs[:1]
    total = spent()
    print(f"spent so far ${total:.3f} / cap ${CAP_USD}", flush=True)
    for n, c, k in jobs:
        f = out / f"{n}_{c}_{k}.png"
        if f.exists():
            continue
        if total >= CAP_USD:
            print("cap reached, stopping", flush=True)
            break
        s = SCENES[n]
        row = dict(time=datetime.now().isoformat(timespec="seconds"), scene=n, cond=c, k=k, file=f.name)
        try:
            if c == "A":
                res = call([inp / "product.png"], prompt_a(s))
            else:
                res = call([inp / f"{n}_mask.png"], prompt_b(s), mask=inp / f"{n}_mask.png")
            f.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
            ti, ii, oo, usd = cost_of(res.get("usage"), CFG["price_per_million"])
            total += usd
            row.update(text_in_tokens=ti, image_in_tokens=ii, out_tokens=oo, cost_usd=round(usd, 4), total_spent_usd=round(total, 4))
            print(f"[done] {f.name}  ${usd:.3f}  (total ${total:.3f})", flush=True)
        except Exception as e:  # noqa: BLE001
            row.update(cost_usd=0, total_spent_usd=round(total, 4), error=str(e)[:300])
            print(f"[error] {f.name}: {str(e)[:200]}", flush=True)
        log(row)


def align(gen, canvas, op, pid):
    """Where the generator kept the shelf. It kept the angle (it was shown the shelf at that angle)
    but may shift and rescale it slightly, so a similarity is enough: the shelf's outline and its
    inner part boundaries (from the render) against the generated image's edges, chamfer distance,
    coarse grid then Nelder-Mead. Returns a 2x3 warp (canvas -> generated) and the mean distance."""
    import reproject as R
    H, W = gen.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(g, 30, 90)
    dt = np.minimum(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), 20.0)
    k = (op > 0.5).astype(np.uint8)
    pidr = cv2.resize(pid.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
    bnd = (k - cv2.erode(k, np.ones((3, 3), np.uint8))) > 0
    inner = (np.abs(cv2.Sobel(pidr.astype(np.float32), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(pidr.astype(np.float32), cv2.CV_32F, 0, 1))) > 0
    ys, xs = np.nonzero(bnd | (inner & (k > 0)))
    sel = np.random.default_rng(0).choice(len(xs), min(4000, len(xs)), replace=False)
    pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    c0 = pts.mean(0)

    def cost(s, tx, ty):
        q = c0 + s * (pts - c0) + [tx, ty]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        out = (q[:, 0] < 0) | (q[:, 0] >= W) | (q[:, 1] < 0) | (q[:, 1] >= H)
        return float(np.where(out, 20.0, dt[yi, xi]).mean())

    best = (1e9, 1.0, 0.0, 0.0)
    for s in np.arange(0.80, 1.16, 0.025):
        for tx in range(-160, 161, 8):
            for ty in range(-160, 161, 8):
                c = cost(s, tx, ty)
                if c < best[0]:
                    best = (c, s, tx, ty)
    v0 = np.array(best[1:], float); sc = np.array([0.01, 3.0, 3.0])
    z, f = R.nelder_mead(lambda z: cost(*(v0 + z * sc)), 3, 120)
    s, tx, ty = v0 + z * sc
    warp = np.float32([[s, 0, c0[0] * (1 - s) + tx], [0, s, c0[1] * (1 - s) + ty]])
    return warp, round(float(f), 2)


def detail_score(a, b, m):
    """Fine detail of b against a inside m: correlation of their high-pass layers (1.0 = the same
    fine detail), and SIFT matches that agree on one similarity (RANSAC) per 100 keypoints of a."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ha = ga - cv2.GaussianBlur(ga, (0, 0), 2.0); hb = gb - cv2.GaussianBlur(gb, (0, 0), 2.0)
    mm = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    corr = float(np.corrcoef(ha[mm], hb[mm])[0, 1]) if mm.sum() > 100 else None
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(ga.astype(np.uint8), m.astype(np.uint8) * 255)
    kb, db = sift.detectAndCompute(gb.astype(np.uint8), (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) * 255))
    inl = 0
    if da is not None and db is not None and len(ka) >= 4 and len(kb) >= 4:
        mt = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
        if len(mt) >= 4:
            A = np.float32([ka[x.queryIdx].pt for x in mt]); B = np.float32([kb[x.trainIdx].pt for x in mt])
            _, inl_m = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            inl = int(inl_m.sum()) if inl_m is not None else 0
    return dict(highpass_corr=None if corr is None else round(corr, 3), sift_inliers=inl, sift_keypoints=len(ka),
                sift_per_100=round(100 * inl / max(len(ka), 1), 1))


def finish():
    """B images: find where the generator kept the shelf, put the real shelf's own pixels there
    (low frequencies from the generated image, product_first.put_back), and score the fine detail
    of the shelf against the render (B) and of the plain outputs against the seller photo (A)."""
    import reproject as R
    import product_first as PF
    real = R.Real(0)
    inp, out = EXP / "inputs", EXP / "outputs"
    fin = EXP / "final"; fin.mkdir(exist_ok=True)
    info = {}
    for n in SCENES:
        z = np.load(inp / f"{n}_layers.npz")
        canvas = cv2.imread(str(inp / f"{n}_canvas.png"))
        for k in range(1, N + 1):
            f = out / f"{n}_B_{k}.png"
            if not f.exists():
                continue
            gen = cv2.imread(str(f))
            if gen.shape[:2] != canvas.shape[:2]:
                gen = cv2.resize(gen, canvas.shape[1::-1], interpolation=cv2.INTER_AREA)
            warp, chamfer = align(gen, canvas, z["op"], z["pid"])
            H, W = gen.shape[:2]
            # the same camera with its image scaled and moved by the warp; the shelf is drawn again
            # there (warping the drawn layers would resample them and soften the detail)
            q = list(z["p"]); s_ = float(warp[0, 0]); q[4] *= s_; q[5] = s_ * q[5] + float(warp[0, 2]); q[6] = s_ * q[6] + float(warp[1, 2])
            L, col, op, veil = PF.layers(real, tuple(q), H, W)
            # the generator redraws the shelf its own way (pilots: clear legs, no top); erase its
            # shelf first (LaMa, the CS pipeline's product-shaped mask), keep it as the light source
            import erase_lama as E
            ys, xs = np.nonzero(op > 0.5)
            box = (max(int(xs.min()) - 30, 0), max(int(ys.min()) - 30, 0), min(int(xs.max()) + 30, W), min(int(ys.max()) + 30, H))
            M, core, d, bg = E.shape_mask(gen, E.box_mask(gen.shape, [box]))
            near = cv2.dilate((op > 0.05).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0  # its shelf lies within ~15 px of ours
            M = ((M > 0) & near) | (cv2.dilate((op > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            plate = E.flat_match(gen, E.lama_lowres(gen, M.astype(np.uint8), 512), M.astype(np.uint8))[0]
            cv2.imwrite(str(fin / f"{n}_B_{k}_plate.png"), plate)
            final, pinfo = PF.put_back(plate, real, tuple(q), L, col, op, veil, lowfreq=True, light_src=gen)
            cv2.imwrite(str(fin / f"{n}_B_{k}.png"), final)
            m = op > 0.5
            info[f"{n}_B_{k}"] = dict(scale=round(s_, 3), shift_px=[round(float(warp[0, 2]), 1), round(float(warp[1, 2]), 1)],
                                      chamfer_px=chamfer, detail_vs_render=detail_score(np.clip(col, 0, 255).astype(np.uint8), final, m),
                                      generated_vs_render=detail_score(np.clip(col, 0, 255).astype(np.uint8), gen, m), **pinfo)
            print(f"{n}_B_{k}", json.dumps(info[f"{n}_B_{k}"]), flush=True)
    # A: the plain outputs against the seller's photo (any angle; SIFT only)
    prod = cv2.imread(str(inp / "product.png"))
    pm = (real.a > 0.5)
    for n in SCENES:
        for k in range(1, N + 1):
            f = out / f"{n}_A_{k}.png"
            if f.exists():
                g = cv2.imread(str(f))
                sift = cv2.SIFT_create()
                ka, da = sift.detectAndCompute(cv2.cvtColor(prod, cv2.COLOR_BGR2GRAY), pm.astype(np.uint8) * 255)
                kb, db = sift.detectAndCompute(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), None)
                mt = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance] if db is not None else []
                inl = 0
                if len(mt) >= 4:
                    A = np.float32([ka[x.queryIdx].pt for x in mt]); B = np.float32([kb[x.trainIdx].pt for x in mt])
                    _, im_ = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    inl = int(im_.sum()) if im_ is not None else 0
                info[f"{n}_A_{k}"] = dict(sift_inliers_vs_seller_photo=inl, sift_keypoints=len(ka))
                print(f"{n}_A_{k}", info[f"{n}_A_{k}"], flush=True)
    (EXP / "finish.json").write_text(json.dumps(info, indent=1))


def finish2(seed=0, ring_mode="floor", strength=0.5, cscale=0.0, only=None, tag="final_v2"):
    """v2 of the B finish, aimed at what the blind judge saw in v1 (halo lines, doubled legs,
    no contact shadow, the clear top read as missing):
      1. erase everything the generator drew near the shelf (wider than v1)
      2. put the real shelf back (light layer from the generated image, detail from the photo),
         the acrylic top drawn stronger
      3. lock the shelf and let SDXL inpainting redraw only a ring around it and the floor under
         its lowest board (contact, shadow), with the room's depth from the same camera
      4. check the lock: the shelf's pixels must equal the put-back ones exactly"""
    import reproject as R
    import product_first as PF
    import erase_lama as E
    R.ACR.update(top=0.16, edge=0.9, line=0.75, line_w=2)
    real = R.Real(0)
    inp, out = EXP / "inputs", EXP / "outputs"
    fin = EXP / tag; fin.mkdir(exist_ok=True)
    info = {}
    for n, sc in SCENES.items():
        z = np.load(inp / f"{n}_layers.npz")
        canvas = cv2.imread(str(inp / f"{n}_canvas.png"))
        for k in range(1, N + 1):
            f = out / f"{n}_B_{k}.png"
            if not f.exists() or (only and f"{n}_B_{k}" not in only):
                continue
            gen = cv2.imread(str(f))
            warp, chamfer = align(gen, canvas, z["op"], z["pid"])
            H, W = gen.shape[:2]
            q = list(z["p"]); s_ = float(warp[0, 0]); q[4] *= s_; q[5] = s_ * q[5] + float(warp[0, 2]); q[6] = s_ * q[6] + float(warp[1, 2])
            q = tuple(q)
            L, col, op, veil = PF.layers(real, q, H, W)
            shelf = (op > 0.05) | (veil > 0.05)
            # 1. erase: the generator's own shelf lies within a few tens of pixels of ours
            ys, xs = np.nonzero(shelf)
            box = (max(int(xs.min()) - 40, 0), max(int(ys.min()) - 40, 0), min(int(xs.max()) + 40, W), min(int(ys.max()) + 40, H))
            M, _, _, _ = E.shape_mask(gen, E.box_mask(gen.shape, [box]))
            near = cv2.dilate(shelf.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
            M = ((M > 0) & near) | (cv2.dilate(shelf.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
            plate = E.flat_match(gen, E.lama_lowres(gen, M.astype(np.uint8), 512), M.astype(np.uint8))[0]
            # 2. the real shelf back on the plate
            put, pinfo = PF.put_back(plate, real, q, L, col, op, veil, lowfreq=True, light_src=gen)
            # 3. ring around the shelf + the floor under the lowest board, shelf locked
            c = R.cam(q)
            b = R.prims(False, real.dims)[0][0]
            foot = R.project(c, [[b[0], 0, b[2]], [b[3], 0, b[2]], [b[3], 0, b[5]], [b[0], 0, b[5]]])
            floor = np.zeros((H, W), np.uint8); cv2.fillPoly(floor, [np.round(foot).astype(np.int32)], 1)
            floor = cv2.dilate(floor, np.ones((25, 25), np.uint8))
            solid = op > 0.05
            if ring_mode == "floor":  # v2 run 1 redrew a ring around every leg: SDXL filled it with dark outlines
                feet = np.zeros((H, W), np.uint8)
                for (lx, lz, r_, *_) in R.prims(False, real.dims)[1]:
                    fx, fy = R.project(c, [[lx, 0.0, lz]])[0]
                    cv2.circle(feet, (int(fx), int(fy)), max(6, int(0.012 * W)), 1, -1)
                ring = ((floor > 0) | (feet > 0)) & ~solid
            else:
                ring = ((cv2.dilate(solid.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) | (floor > 0)) & ~solid
            cv2.imwrite(str(fin / f"{n}_B_{k}_put.png"), put)
            t_shelf, _, _, _ = R.trace(c, H, W, R.prims(False, real.dims))
            depth = PF.depth_image(np.minimum(PF.room_depth(c, H, W, real.dims, 0.30 if n != "studio_left" else 1.2), t_shelf))
            prompt = f"photorealistic photo of {sc['scene']}, the floor under and around a shelf with soft natural contact shadows"
            final, _ = PF.ring_regenerate(put, op, ring, depth, prompt, seed=seed, cscale=cscale, strength=strength)
            # 4. the lock: shelf pixels unchanged by step 3
            lock = op > 0.98
            lock_diff = int(np.abs(final.astype(np.int16) - put.astype(np.int16))[lock].max(initial=0))
            cv2.imwrite(str(fin / f"{n}_B_{k}.png"), final)
            info[f"{n}_B_{k}"] = dict(lock_max_diff=lock_diff, lock_px=int(lock.sum()), ring_px=int(ring.sum()), erased_px=int(M.sum()),
                                      detail_vs_render=detail_score(np.clip(col, 0, 255).astype(np.uint8), final, op > 0.5), **pinfo)
            print(f"{n}_B_{k}", json.dumps({k2: info[f"{n}_B_{k}"][k2] for k2 in ("lock_max_diff", "ring_px", "erased_px", "detail_vs_render")}), flush=True)
    (EXP / "finish_v2.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--finish2", action="store_true", help="v2: wider erase, stronger clear top, ring regenerated with the shelf locked")
    a = ap.parse_args()
    if a.finish2:
        finish2()
    elif a.finish:
        finish()
    elif a.dry:
        m = build_inputs()
        for n, v in m.items():
            print(f"--- {n}\nA: {v['prompt_a']}\nB: {v['prompt_b']}\n")
        print(f"{len(SCENES) * 2 * N} calls planned, cap ${CAP_USD}")
    else:
        generate(pilot=a.pilot)

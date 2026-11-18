"""Fair positive control for the erase check: SDXL inpainting with the same product-shaped mask."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, place as P, compose as C
R = P.ROOT
scene = cv2.imread(f"{R}/data/cs/13/outputs/{P.OUTS[0]}")
info = json.load(open(f"{R}/out/exp13/place_info.json"))
box = E.box_mask(scene.shape, [d[2] for d in info["0"]["dets"]])
M, core, d, bg = E.shape_mask(scene, box)
NEG = "shelf, shelving, bookcase, furniture, table, cabinet, metal rods, poles, text, watermark, logo"
PR = "a bright minimalist living room, white wall, light oak wooden floor, natural daylight, empty space against the wall, photorealistic interior photo"
res = {}
lam = cv2.imread(f"{R}/out/exp13/lama_plate_O0.png")
r = E.residuals(lam, M); res["lama"] = dict(hot_frac=r["hot_frac"], blobs=len(r["blobs"]), leftover=E.leftover(scene, lam, M))
for seed in (0, 1, 2):
    pl = C.erase(scene, M, PR, NEG, seed=seed)
    cv2.imwrite(f"{R}/out/exp13/sdxl_samemask_O0_s{seed}.jpg", pl)
    r = E.residuals(pl, M)
    res[f"sdxl_s{seed}"] = dict(hot_frac=r["hot_frac"], blobs=len(r["blobs"]), leftover=E.leftover(scene, pl, M))
    print(f"sdxl_s{seed}", res[f"sdxl_s{seed}"], flush=True)
print(json.dumps(res))
json.dump(res, open(f"{R}/out/exp13/erase_check_O0.json", "w"), indent=1)

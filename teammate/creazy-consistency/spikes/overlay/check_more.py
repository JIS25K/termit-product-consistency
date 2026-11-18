"""Erase check on more cuts: product-shaped mask + LaMa, then new-structure ratio (hot_frac) and
leftover ratio. SDXL with the same mask on a few cuts as the positive control."""
import os, sys, json, numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import erase_lama as E, place as P, compose as C
from refset import models

ROOT = P.ROOT
CUTS = [("13", k, ["a shelf", "a bookshelf"]) for k in (1, 2, 7, 9, 11, 12)] + \
       [("1", k, ["a tumbler", "a cup", "a mug"]) for k in (7, 30)]
SDXL = {("13", 2): "a bright living room interior, wall and wooden floor, natural light, photorealistic interior photo",
        ("1", 7): "a cafe table top by a window, warm light, photorealistic photo"}
NEG = "shelf, shelving, bookcase, furniture, table, cabinet, cup, tumbler, mug, bottle, poles, text, watermark, logo"

def outs(case):
    d = f"{ROOT}/data/cs/{case}/outputs"
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))

def main(only=None):
    path = f"{ROOT}/out/erase_check/results.json"
    res = json.load(open(path)) if only and os.path.exists(path) else {}
    os.makedirs(f"{ROOT}/out/erase_check", exist_ok=True)
    for case, k, q in CUTS:
        if only and f"{case}/O{k}" not in only:
            continue
        scene = cv2.imread(f"{ROOT}/data/cs/{case}/outputs/{outs(case)[k]}")
        dets = models.detect(Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)), q, threshold=0.15)[:3]
        if not dets:
            res[f"{case}/O{k}"] = {"error": "no product found"}; continue
        box = E.box_mask(scene.shape, [d["box"] for d in dets])
        M, core, d, bg = E.shape_mask(scene, box, hull=(case == "1"))
        plate = E.flat_match(scene, E.lama_lowres(scene, M, 512), M)[0]
        r = E.residuals(plate, M)
        row = {"mask_px": int(M.sum()), "lama": {"hot_frac": r["hot_frac"], "blobs": len(r["blobs"]), "leftover": E.leftover(scene, plate, M)}}
        cv2.imwrite(f"{ROOT}/out/erase_check/cs{case}_O{k}_lama.jpg", plate)
        cv2.imwrite(f"{ROOT}/out/erase_check/cs{case}_O{k}_mask.png", M * 255)
        if (case, k) in SDXL:
            row["sdxl"] = []
            for seed in (0, 1):
                pl = C.erase(scene, M, SDXL[(case, k)], NEG, seed=seed)
                rr = E.residuals(pl, M)
                row["sdxl"].append({"seed": seed, "hot_frac": rr["hot_frac"], "blobs": len(rr["blobs"]), "leftover": E.leftover(scene, pl, M)})
                cv2.imwrite(f"{ROOT}/out/erase_check/cs{case}_O{k}_sdxl_s{seed}.jpg", pl)
        res[f"{case}/O{k}"] = row
        print(f"{case}/O{k}", json.dumps(row), flush=True)
        json.dump(res, open(f"{ROOT}/out/erase_check/results.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1:] or None)

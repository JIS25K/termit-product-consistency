"""CS validation runner (docs/validation_cs_plan.md). Every step reads and writes under
out/validation/; customer images stay on this machine; no generation API is called.

  python spikes/validation/run.py merge --who A B          # two labelers -> labels/final.json (+ disagreements)
  python spikes/validation/run.py rehearsal-labels           # sub-agent labels -> labels/rehearsal.json (rehearsal only)
  python spikes/validation/run.py route --labels out/validation/labels/final.json
  python spikes/validation/run.py process [--c-version 3]
  python spikes/validation/run.py evalset [--seed N]
  python spikes/validation/run.py summary

Routing (fixed, plan v0.2 section 3): same -> pass; not visible -> pass (counted apart); different
in print only -> C if the real print is located (score >= 12) else drop; different in shape (or
both) -> A only in an A case, when the reference has this angle, no hand grips it, it is a real
photo, and the product is alone with nothing on it; otherwise drop.
Processing (fixed, plan section 4): C = check/print_restore.restore_best; A for the shelf (13) =
3D re-render of the seller photo at the scene camera; A for the tumbler (1) = erase and replace
with the seller cutout (exp1_lama). A processed cut is delivered only if its step succeeded.
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spikes" / "overlay"))
VAL = ROOT / "out" / "validation"
ROUNDS = {"c": [37, 32, 39, 19, 10, 28], "a": [13, 1]}  # docs/validation_cs_plan.md section 1
A_CASES = {13, 1}
EXT = (".png", ".jpg", ".jpeg", ".webp")
SCRATCH = Path("/private/tmp/claude-501/-Users-yooseok-NEXT------camera-tool--claude-worktrees-annyeong-ef1f80/277c7a92-b915-4e81-a250-cf628f920d54/scratchpad/judge_cs")
MAX_TURN = 30.0  # degrees the 3D re-render may turn the seller's photo (plan section 3)


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def jsave(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=1))


def files(case):
    return sorted(f.name for f in (ROOT / "data" / "cs" / str(case) / "outputs").iterdir() if f.suffix.lower() in EXT)


# ---------------------------------------------------------------- labels
def merge(who):
    """Two labelers' answers -> final labels where they agree; disagreements listed to settle
    together (plan section 2). Agreement rate on the main label is reported."""
    a, b = (jload(VAL / "labels" / f"{w}.json") for w in who)
    final, disagree, same_main = {}, {}, 0
    keys = sorted(set(a) & set(b))
    for k in keys:
        if a[k]["label"] == b[k]["label"]:
            same_main += 1
        if a[k] == b[k]:
            final[k] = a[k]
        else:
            disagree[k] = {who[0]: a[k], who[1]: b[k]}
    jsave(VAL / "labels" / "final.json", final)
    jsave(VAL / "labels" / "disagreements.json", disagree)
    print(f"both labelled {len(keys)}, main label agrees {same_main} ({100 * same_main / max(len(keys), 1):.0f}%), fully equal {len(final)}, to settle {len(disagree)}")


def rehearsal_labels():
    """Sub-agent labels in the final-label format. Rehearsal only: never the ground truth."""
    out = {}
    for c in (37, 32):
        for f, v in jload(SCRATCH / f"labels_{c}.json").items():
            out[f"{c}/{f}"] = dict(label=v["label"], error=v.get("error") or None,
                                   angle="yes" if v.get("angle_available") else "no", hand="yes" if v.get("hand") else "no")
    for c, parts in ((13, ["13"]), (1, ["1a", "1b"])):
        for part in parts:
            for f, v in jload(SCRATCH / f"labels_{part}.json").items():
                # the earlier sub-agent pass had no error type; its differences on these two
                # products were all form or part changes (handle, lid, top, legs)
                angle = "yes" if c == 13 or v.get("view") in ("front", "angle") else "no"
                out[f"{c}/{f}"] = dict(label=v["label"], error="shape" if v["label"] == "different" else None,
                                       angle=angle, hand="yes" if v.get("hand") else "no")
    jsave(VAL / "labels" / "rehearsal.json", out)
    print(len(out), "rehearsal labels")


# ---------------------------------------------------------------- routing
def refset(case):
    return ROOT / "out" / "cs_all" / str(case) / "refset" / "refset.json"


def route(labels_path):
    from check.print_restore import prints_for, MIN_INLIERS
    labels = jload(labels_path)
    routes = {}
    multi = {}
    for k, v in sorted(labels.items()):
        case, f = k.split("/", 1)
        r = dict(label=v["label"], error=v.get("error"))
        if v["label"] == "match":
            r["route"] = "pass"
        elif v["label"] == "not_visible":
            r["route"] = "pass_not_visible"
        elif v.get("error") == "print":
            if case not in multi:
                multi[case] = prints_for(refset(int(case)))
            img = cv2.imread(str(ROOT / "data" / "cs" / case / "outputs" / f))
            _, _, m = multi[case][0]._best(img)
            r["print_score"] = int(m["inliers"])
            r["route"] = "C" if m["inliers"] >= MIN_INLIERS else "drop_print"
        else:
            a_ok = (int(case) in A_CASES and v.get("angle") == "yes" and v.get("hand") == "no"
                    and v.get("scene") == "photo" and v.get("single") == "yes")
            r["route"] = "A" if a_ok else "drop_shape"
        routes[k] = r
    jsave(VAL / "routes.json", dict(labels=str(labels_path), routes=routes))
    counts = {}
    for r in routes.values():
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    print(counts)


# ---------------------------------------------------------------- processing
def file_hashes():
    names = ["check/print_restore.py", "check/print_set.py", "check/sift.py", "spikes/overlay/reproject.py", "spikes/overlay/detail.py",
             "spikes/overlay/exp1_lama.py", "spikes/overlay/erase_lama.py", "spikes/validation/run.py"]
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()[:12] for n in names}


def product_region(img, case):
    """The product in the generated image: detector boxes for the reference set's query (for
    example "a bottle"), widened a little. One fix after the first C run (plan section 4: a
    repeated failure gets one low-cost fix, both runs reported): print crops that also caught
    the box behind the product wrote its label onto the background in case 10."""
    from PIL import Image
    from refset import models
    q = json.loads(refset(int(case)).read_text()).get("query") or "a product"
    dets = models.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), [q], threshold=0.15)
    m = np.zeros(img.shape[:2], np.uint8)
    for d in dets:
        x0, y0, x1, y1 = d["box"]; px, py = int(0.03 * (x1 - x0)), int(0.03 * (y1 - y0))
        m[max(y0 - py, 0):y1 + py, max(x0 - px, 0):x1 + px] = 1
    return m if m.any() else None


def do_c(case, f, cache, product_box=True):
    from check.print_restore import prints_for, restore_best
    if case not in cache:
        cache[case] = prints_for(refset(int(case)))
    img = cv2.imread(str(ROOT / "data" / "cs" / case / "outputs" / f))
    region = product_region(img, case) if product_box else None
    out, info = restore_best(img, *cache[case], region=region)
    ok = info["status"] == "restored"
    return (out if ok else None), dict(step="C", status=info["status"], before=info["inliers_before"],
                                       after=info.get("inliers_after"), outside_max_diff=info.get("outside_max_diff"), print=info.get("print"))


# C version 3 (after the first blind evaluation, plan section 12): prints cut from the product
# in every product photo (check/print_set.py), the variant whose detail agrees best, and
# restoration kept on the surfaces the match vouches for, without the light rim or smeared
# edges. Two cut-level gates, set on this round's cuts (so the second evaluation checks them on
# the same cuts, not on new ones): the generated label must line up with the real one
# (AGREE_MIN; below it the layout differs and C cannot help), and restoration must clearly
# raise the match (by GAIN_ABS inliers and GAIN_REL of the score before; a rise of two or three
# means the print did not take hold).
AGREE_MIN = 0.25
GAIN_ABS, GAIN_REL = 10, 0.10
C3 = dict(clean_print=True, refine=True, support=True, edge_safe=True)


def do_c3(case, f, cache):
    from check import print_set as PS
    from check.print_restore import restore, MIN_INLIERS
    if case not in cache:
        d = VAL / "prints_v3" / case
        if not (d / "prints.json").exists():
            PS.build(refset(int(case)), d)
        cache[case] = PS.load(d)
    multi, images, parts = cache[case]
    img = cv2.imread(str(ROOT / "data" / "cs" / case / "outputs" / f))
    name, m, r, cands = PS.best_print(img, multi, parts, images)
    agree = next(c["agreement"] for c in cands if c["print"] == name)
    meta = dict(step="C", version=3, print=name, candidates=cands, agreement=agree, before=int(r["inliers"]))
    if r["inliers"] < MIN_INLIERS:
        return None, {**meta, "status": "no_print"}
    if agree < AGREE_MIN:
        return None, {**meta, "status": "layout_differs"}
    out, info = restore(img, images[name], m, **C3)
    meta.update(after=info.get("inliers_after"), outside_max_diff=info.get("outside_max_diff"), supported_share=info.get("supported_share"))
    if info["status"] != "restored":
        return None, {**meta, "status": info["status"]}
    if info["inliers_after"] - info["inliers_before"] < max(GAIN_ABS, GAIN_REL * info["inliers_before"]):
        return None, {**meta, "status": "small_gain"}
    return out, {**meta, "status": "restored"}


def do_a_shelf(f, real):
    """Shelf (13): find the generated shelf, fit the scene camera (level, silhouette + legs, then
    the lower board), refuse turns over MAX_TURN, erase with LaMa, draw the seller's photo on
    the 3D proxy at that camera, match materials, then crispness and grain."""
    import reproject as R, select_view as V, erase_lama as E, detail as D
    from PIL import Image
    from refset import models
    scene = cv2.imread(str(ROOT / "data" / "cs" / "13" / "outputs" / f))
    part, M, core, extra = V.gen_part(scene)
    if part is None:
        return None, dict(step="A", status="no_shelf_found")
    pil = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    ig = np.zeros(part.shape, bool)
    for x in models.detect(pil, ["a speaker", "a potted plant", "a vase", "a book", "a decorative object", "a chair", "a person"], threshold=0.2):
        x0, y0, x1, y1 = x["box"]
        if (x1 - x0) * (y1 - y0) < 0.25 * part.size:
            ig[y0:y1, x0:x1] = True
    tgt = cv2.morphologyEx((part > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    p, finfo = R.fit_scene(tgt, ig, real.dims)
    p, ainfo = R.anchor_lower(p, tgt, ig, real.dims)
    turn = float(p[0] - real.p[0])
    meta = dict(step="A", camera=dict(az=round(p[0], 1), el=round(p[1], 1), dist=round(p[2], 2)), turn=round(turn, 1), fit=finfo, anchor=ainfo)
    if abs(turn) > MAX_TURN:
        return None, {**meta, "status": "angle_out_of_range"}
    plate = E.flat_match(scene, E.lama_lowres(scene, M, 512), M)[0]
    out, al, L = R.composite(plate, real, p, gen=(scene, part > 0, ig))
    o2, a2, dinfo = D.match_detail(out.astype(np.float32), al, out.astype(np.float32), scene.astype(np.float32), part > 0)
    final = np.clip(out.astype(np.float32) * (1 - a2[..., None]) + o2 * a2[..., None], 0, 255).astype(np.uint8)
    return final, {**meta, "status": "done", "detail": dinfo}


def do_a_tumbler(f):
    """Tumbler (1): erase the generated tumbler and put the seller's cutout in at the same height
    (exp1_lama), plain studio backgrounds with the plain fill, then crispness and grain."""
    import exp1 as X, exp1_lama as L1
    k = X.OUTS.index(f)
    scene = cv2.imread(f"{X.CASE}/outputs/{f}")
    border = np.concatenate([scene[:20].reshape(-1, 3), scene[:, :20].reshape(-1, 3), scene[:, -20:].reshape(-1, 3)]).astype(np.float32)
    plain = float(border.std(axis=0).mean()) < 12.0
    if plain:
        _, _, comp, meta = L1.run_plain(k, 16, tag="_val", detail=True)
    else:
        _, _, comp, _, meta = L1.run(k, 16, tag="_val", detail=True)
    return comp, dict(step="A", status="done", background="plain" if plain else "complex", detail=meta.get("detail"))


def process(c_version=2):
    routes = jload(VAL / "routes.json")["routes"]
    outdir = VAL / "processed"
    report = jload(VAL / "process.json")
    report["_hashes"] = file_hashes()
    cache, real = {}, None
    for k, r in routes.items():
        if r["route"] not in ("C", "A") or (k in report and report[k].get("status") in ("restored", "done")):
            continue
        case, f = k.split("/", 1)
        t0 = time.time()
        try:
            if r["route"] == "C":
                out, info = do_c3(case, f, cache) if c_version == 3 else do_c(case, f, cache)
            elif case == "13":
                if real is None:
                    import reproject as R
                    real = R.Real(0)
                out, info = do_a_shelf(f, real)
            elif case == "1":
                out, info = do_a_tumbler(f)
            else:
                out, info = None, dict(step="A", status="no_A_method_for_this_case")
        except Exception as e:  # noqa: BLE001  (recorded, the cut is not delivered)
            out, info = None, dict(step=r["route"], status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        info["sec"] = round(time.time() - t0, 1)
        if out is not None:
            p = outdir / case / (Path(f).stem + ".png")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), out)
            info["file"] = str(p.relative_to(ROOT))
        report[k] = info
        jsave(VAL / "process.json", report)
        print(k, info["status"], info.get("before", ""), info.get("after", ""), info.get("turn", ""), info["sec"], "s", flush=True)


# ---------------------------------------------------------------- evaluation set
def evalset(seed=0):
    """Processed cuts that were delivered, plus half as many untouched 'same' cuts as controls,
    under neutral ids (plan section 5)."""
    routes = jload(VAL / "routes.json")["routes"]
    proc = jload(VAL / "process.json")
    items, key = [], {}
    processed = [(k, proc[k]) for k in routes if k in proc and proc[k].get("file")]
    passes = [k for k, r in routes.items() if r["route"] == "pass"]
    rng = random.Random(seed)
    controls = rng.sample(passes, min(len(passes), math.ceil(len(processed) / 2))) if processed else []
    entries = [(k, ROOT / v["file"], "processed_" + v["step"]) for k, v in processed]
    entries += [(k, ROOT / "data" / "cs" / k.split("/")[0] / "outputs" / k.split("/", 1)[1], "control") for k in controls]
    rng.shuffle(entries)
    img_dir = VAL / "eval" / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    for i, (k, src, kind) in enumerate(entries):
        eid = f"e{i:03d}{rng.randrange(16 ** 4):04x}"
        cv2.imwrite(str(img_dir / f"{eid}.jpg"), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        case = int(k.split("/")[0])
        items.append(dict(id=eid, case=case))
        key[eid] = dict(item=k, kind=kind, src=str(src.relative_to(ROOT)))
    jsave(VAL / "eval" / "items.json", items)
    jsave(VAL / "eval" / "key.json", key)
    print(len(items), "items:", sum(1 for v in key.values() if v["kind"] != "control"), "processed,", sum(1 for v in key.values() if v["kind"] == "control"), "controls")


# ---------------------------------------------------------------- summary
def summary():
    """Plan section 6 metrics, per path and overall. Several evaluators: majority for same /
    different, mean for the pasted score."""
    routes = jload(VAL / "routes.json")["routes"]
    proc = jload(VAL / "process.json")
    key = jload(VAL / "eval" / "key.json")
    answers = [jload(p) for p in sorted((VAL / "eval").glob("answers_*.json"))]
    per = {}
    for eid in key:
        got = [a[eid] for a in answers if eid in a]
        if not got:
            continue
        same = sum(1 for g in got if g["same"] == "same"); diff = sum(1 for g in got if g["same"] == "different")
        per[eid] = dict(judged="same" if same > diff else ("different" if diff > same else "unsure"),
                        pasted=float(np.mean([g["pasted"] for g in got])), n=len(got))
    visible = [k for k, r in routes.items() if r["label"] != "not_visible"]
    before = sum(1 for k in visible if routes[k]["label"] == "different")
    rows = {}
    for path in ("C", "A"):
        ids = [e for e, v in key.items() if v["kind"] == f"processed_{path}" and e in per]
        routed = [k for k, r in routes.items() if r["route"] == path]
        delivered = [k for k in routed if proc.get(k, {}).get("file")]
        fixed = sum(1 for e in ids if per[e]["judged"] == "same")
        rows[path] = dict(routed=len(routed), delivered=len(delivered), evaluated=len(ids),
                          fixed_rate=round(fixed / len(ids), 3) if ids else None,
                          pasted_mean=round(float(np.mean([per[e]["pasted"] for e in ids])), 2) if ids else None,
                          pasted_4_5=round(sum(1 for e in ids if per[e]["pasted"] >= 4) / len(ids), 3) if ids else None)
    ctrl = [e for e, v in key.items() if v["kind"] == "control" and e in per]
    control = dict(n=len(ctrl), pasted_mean=round(float(np.mean([per[e]["pasted"] for e in ctrl])), 2) if ctrl else None,
                   judged_different=sum(1 for e in ctrl if per[e]["judged"] == "different"))
    by_item = {v["item"]: e for e, v in key.items()}
    processed_items = [k for k in visible if proc.get(k, {}).get("file")]
    passes = [k for k in visible if routes[k]["route"] == "pass"]
    # without a gate every processed cut goes out; with the gate (plan v0.2) a processed cut goes
    # out only if the blind evaluation says same and the pasted score is under 4
    shipped_ungated = passes + processed_items
    bad_after = sum(1 for k in processed_items if by_item.get(k) in per and per[by_item[k]]["judged"] == "different")
    gated_ok = [k for k in processed_items if by_item.get(k) in per and per[by_item[k]]["judged"] == "same" and per[by_item[k]]["pasted"] < 4]
    delivered_visible = passes + gated_ok
    outside = [v.get("outside_max_diff") for v in proc.values() if isinstance(v, dict) and v.get("step") == "C" and v.get("outside_max_diff") is not None]
    s = dict(visible=len(visible), different_before=before, deformation_before=round(before / max(len(visible), 1), 3),
             deformation_after_ungated=round(bad_after / max(len(shipped_ungated), 1), 3),
             delivered_gated=len(delivered_visible), yield_rate=round(len(delivered_visible) / max(len(visible), 1), 3),
             paths=rows, control=control,
             c_outside_max_diff=max(outside) if outside else None, evaluators=len(answers))
    jsave(VAL / "summary.json", s)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--who", nargs=2, required=True)
    sub.add_parser("rehearsal-labels")
    r = sub.add_parser("route"); r.add_argument("--labels", required=True)
    pr = sub.add_parser("process"); pr.add_argument("--c-version", type=int, choices=(2, 3), default=2)
    ev = sub.add_parser("evalset"); ev.add_argument("--seed", type=int, default=0)
    sub.add_parser("summary")
    a = ap.parse_args()
    {"merge": lambda: merge(a.who), "rehearsal-labels": rehearsal_labels, "route": lambda: route(a.labels),
     "process": lambda: process(a.c_version), "evalset": lambda: evalset(a.seed), "summary": summary}[a.cmd]()

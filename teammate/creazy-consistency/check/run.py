"""Check generated images and videos against a reference set.

  python -m check.run --refset out/mug/refset.json --outputs data/mug/outputs/run9 --out out/mug/check_run9

Images get one score each. Videos are sampled every 0.5 s (--fps 2) and every frame is scored,
so a print that breaks halfway through a clip shows up as a dip.

Verdicts from the first pass (key "verdict", SIFT only)
  match       inliers >= 20: the real print is there
  unverified  below 20: either the print was redrawn/invented, or it is simply not in view
              (back side, handle side). This pass cannot tell those apart.

With --spec (spec/make.py), the confirmed keep conditions of the production spec go to the
VLM with every second-pass call, so "one unit only" or "the lettering stays" is checked too.

Second pass (key "vlm", needs an API key, see refset/vlm.py): every unverified image, and the
lowest frame of every dip in a video, goes to the VLM together with the reference crops. It
answers match / not_visible / different. With --vlm-all, images SIFT matched go to the VLM
too, for a strict look at what SIFT cannot see: small text lost next to a kept logo, parts,
shape (CS 10, 17, 19, 24 kept the big logo and lost the small text). Key "final" combines both passes:
  match        print verified (SIFT, or the VLM saw it in a view SIFT missed)
  not_visible  the print side is not in view and nothing visible contradicts the product
  different    something visible contradicts the real product (invented print, lost text,
               changed shape...). This is the one a person should look at.
  unverified   first pass failed and no second pass ran (no API key)
For a video, final is different when any dip is judged different; dips judged not_visible are
listed in "hidden_at" and do not fail the video.
"""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from refset import vlm

from . import expect as expect_mod
from . import strong as strong_mod
from .sift import PASS, MultiPrintMatcher, PrintMatcher

IMG = {".png", ".jpg", ".jpeg", ".webp"}
VID = {".mp4", ".mov", ".webm"}


def frames(path, fps=2.0):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(src_fps / fps)))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / src_fps, f
        i += 1
    cap.release()


def _pil(bgr, side=1280):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    s = side / max(im.size)
    return im.resize((int(im.width * s), int(im.height * s))) if s < 1 else im


def vlm_refs(ref, base, limit=6):
    """Reference crops for the VLM, representative first, then the print region."""
    items = sorted(ref["items"], key=lambda i: i["crop"] != ref["representative"])[:limit]
    refs = []
    for it in items:
        label = it.get("face_ko") or ("print side" if it["crop"] == ref["representative"] else "another view")
        refs.append((label, Image.open(base / it["crop"]).convert("RGB")))
    for p in ref.get("prints") or ([{"path": ref["print"]["path"]}] if ref["print"]["path"] else []):
        refs.append(("print region (close-up of a printed side)", Image.open(base / p["path"]).convert("RGB")))
    return refs


def second_pass(refs, product, im, found=False, plain=False, keep=None, zoom=None, checks=None):
    try:
        v = vlm.judge(refs, im, product, found=found, plain=plain, keep=keep, zoom=zoom, checks=checks)
        return v.model_dump()
    except Exception as e:  # keep the first-pass result when the VLM call fails
        return {"verdict": None, "error": f"{type(e).__name__}: {e}"}


def dip_segments(rows):
    """Runs of consecutive frames below PASS, as lists of row indices."""
    segs, cur = [], []
    for i, r in enumerate(rows):
        if r["inliers"] < PASS:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def spread(n, k):
    """k frame indices spread over n frames (first, last and evenly between)."""
    if n <= 0:
        return []
    return sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})


def make_matcher(ref, base):
    prints = ref.get("prints") or ([{"path": ref["print"]["path"]}] if ref["print"]["path"] else [])
    if not prints:
        return None
    if len(prints) == 1:
        return PrintMatcher(base / prints[0]["path"])
    return MultiPrintMatcher([(p["path"], base / p["path"]) for p in prints])


def load_spec(spec):
    """The production spec (spec/make.py): a spec.json file or the folder holding it."""
    if not spec:
        return None
    p = Path(spec)
    p = p / "spec.json" if p.is_dir() else p
    return json.loads(p.read_text()) if p.exists() else None


def run(refset_path, outputs, out_dir, fps=2.0, draw=True, use_vlm=None, vlm_all=False, spec=None,
        strong=False, ocr=None, detect=None):
    """spec: the production spec. Its keep conditions go to the VLM as checks.

    strong: also read the product's text and count its instances (check/strong.py). ocr/detect
    replace the real models (tests)."""
    refset_path = Path(refset_path)
    spec_doc = load_spec(spec)
    keep = spec_doc["spec"]["keep"] if spec_doc else None
    ref = json.loads(refset_path.read_text())
    matcher = make_matcher(ref, refset_path.parent)
    out_dir = Path(out_dir)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    use_vlm = vlm.available() if use_vlm is None else (use_vlm and vlm.available())
    expect = expect_mod.build_expect(refset_path, ocr=ocr, detect=detect) if strong else None
    if matcher is None and not use_vlm and not (expect and expect.get("query")):
        raise SystemExit("reference set has no print region (plain product): only the VLM pass can check it, and no API key is set")
    refs = vlm_refs(ref, refset_path.parent) if use_vlm else None
    product = ref.get("product")
    plain = matcher is None

    files = []
    for o in outputs:
        p = Path(o)
        files += sorted(x for x in p.iterdir()) if p.is_dir() else [p]

    report = {
        "pass_threshold": PASS,
        "plain_product": plain,
        "vlm": {"used": use_vlm, "model": vlm.MODEL if use_vlm else None, "all": bool(use_vlm and vlm_all)},
        "spec": {"keep": keep, "confirmed": spec_doc.get("confirmed")} if spec_doc else None,
        "images": [],
        "videos": [],
    }
    for f in files:
        ext = f.suffix.lower()
        if ext in IMG:
            img = cv2.imread(str(f))
            if img is None:
                continue
            if plain:
                r = {"good": 0, "inliers": 0, "pass": False}
            else:
                r = matcher.draw(img, out_dir / "vis" / f"{f.stem}.jpg") if draw else matcher.score(img)
            row = {"file": f.name, **r, "verdict": "match" if r["pass"] else ("no_print" if plain else "unverified")}
            row["final"] = row["verdict"] if r["pass"] else "unverified"
            if strong:
                full = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                st = strong_mod.inspect(full, img, matcher, expect, ocr=ocr, detect=detect)
                row["strong"] = {**st,
                                 "print_box": list(st["print_box"]) if st["print_box"] is not None else None,
                                 "product_box": list(st["product_box"]) if st["product_box"] is not None else None}
                row["final"] = st["verdict"]
            if use_vlm and (vlm_all or not r["pass"]):
                extra = {}
                if strong:
                    extra = {"zoom": strong_mod.zoom_crops(full, st["print_box"], st["product_box"]),
                             "checks": strong_mod.checks(expect)}
                row["vlm"] = second_pass(refs, product, _pil(img), found=r["pass"], plain=plain, keep=keep, **extra)
                if strong:
                    vv = row["vlm"]["verdict"]
                    if vv == "different" or (row["final"] == "unverified" and vv in ("match", "not_visible")):
                        row["final"] = vv
                elif row["vlm"]["verdict"] == "different" or not r["pass"]:
                    row["final"] = row["vlm"]["verdict"] or row["final"]
            report["images"].append(row)
        elif ext in VID:
            rows, small = [], []
            for t, fr in frames(f, fps):
                s = {"inliers": 0} if plain else matcher.score(fr)
                rows.append({"t": round(t, 2), "inliers": s["inliers"], **({"print": s["print"]} if "print" in s else {})})
                small.append(_pil(fr, 1024) if use_vlm else None)
            vals = [r["inliers"] for r in rows]
            hi = [v for v in vals if v >= PASS]
            dips = [r["t"] for r in rows if r["inliers"] < PASS] if hi else []
            v = {
                "file": f.name,
                "frames": rows,
                "min": min(vals) if vals else 0,
                "median": sorted(vals)[len(vals) // 2] if vals else 0,
                # A dip only means something when the print was matched in other frames.
                "dip_at": dips,
                "verdict": "no_print" if plain else "match" if vals and min(vals) >= PASS else ("dip" if dips else "unverified"),
            }
            v["final"] = "match" if v["verdict"] == "match" else "unverified"
            if use_vlm and rows and (vlm_all or v["verdict"] != "match"):
                # The lowest frame of every dip; three spread frames when nothing matched
                # (five for a plain product); with --vlm-all also three spread frames.
                picks = set()
                if hi:
                    picks |= {min(seg, key=lambda i: rows[i]["inliers"]) for seg in dip_segments(rows)}
                else:
                    picks |= set(spread(len(rows), 5 if plain else 3))
                if vlm_all:
                    picks |= set(spread(len(rows), 3))
                judged = []
                for i in sorted(picks):
                    found = rows[i]["inliers"] >= PASS
                    jv = second_pass(refs, product, small[i], found=found, plain=plain, keep=keep)
                    frame_file = f"{f.stem}_t{rows[i]['t']:.1f}.jpg"
                    small[i].save(out_dir / "vis" / frame_file, quality=85)
                    judged.append({"t": rows[i]["t"], "inliers": rows[i]["inliers"], "frame": f"vis/{frame_file}", **jv})
                v["vlm"] = judged
                verdicts = [j["verdict"] for j in judged]
                v["hidden_at"] = [j["t"] for j in judged if j["verdict"] == "not_visible"]
                v["different_at"] = [j["t"] for j in judged if j["verdict"] == "different"]
                if v["different_at"]:
                    v["final"] = "different"
                elif None in verdicts:
                    v["final"] = "unverified"
                elif hi or "match" in verdicts:
                    v["final"] = "match"
                else:
                    v["final"] = "not_visible"
            report["videos"].append(v)
    finals = [i["final"] for i in report["images"]] + [v["final"] for v in report["videos"]]
    report["summary"] = {k: finals.count(k) for k in ("match", "not_visible", "different", "unverified")}
    if strong:
        sv = [i["strong"]["verdict"] for i in report["images"]]
        report["summary"]["strong"] = {k: sv.count(k) for k in dict.fromkeys(sv)}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refset", required=True)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--no-vlm", action="store_true", help="first pass only, even when an API key is set")
    ap.add_argument("--vlm-all", action="store_true", help="also send SIFT-matched images to the VLM (small text, parts)")
    ap.add_argument("--spec", default=None, help="production spec (spec.json or its folder): its keep conditions become VLM checks")
    ap.add_argument("--strong", action="store_true", help="also check text and count (OCR + OWLv2)")
    a = ap.parse_args()
    r = run(a.refset, a.outputs, a.out, a.fps, not a.no_draw, use_vlm=not a.no_vlm, vlm_all=a.vlm_all, spec=a.spec, strong=a.strong)
    for i in r["images"]:
        why = f"  {i['vlm'].get('reason', '')}" if "vlm" in i else ""
        print(f"{i['file'][:40]:40s} inliers {i['inliers']:3d}  {i['verdict']:10s} -> {i['final']}{why}")
    for v in r["videos"]:
        print(f"{v['file'][:40]:40s} frames {[x['inliers'] for x in v['frames']]}  {v['verdict']} -> {v['final']}")
    print("summary", r["summary"], "| vlm", "on" if r["vlm"]["used"] else "off (no API key or --no-vlm)")


if __name__ == "__main__":
    main()

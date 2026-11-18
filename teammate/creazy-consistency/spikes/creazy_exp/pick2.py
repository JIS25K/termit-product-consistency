"""Second experiment: pick 2 of 4 B candidates per scene with the same ranking as pick.py, then run
the overlay step on the kept ones, and save outputs/B/<P>_<scene>_B_s<k>.png with notes.

Ranking (same as the first experiment): anchor words found (read at 0/90/270 degrees), fewer
garbled words, no count excess, more SIFT inliers against the print, lower candidate number.
View check (added after the first run, which picked front-facing candidates for the angle and
back scenes; that first pick is kept in B/picks_v1.json): when the scene sentence asks for a view
(proxy_bottle.VIEW_RULES), candidates are ordered first by whether they show it. The turn of the
generated bottle comes from the same cylinder fit as the overlay. Back (180): the front label must
not be found. Other views: the turn must be at least 15 degrees, closest to the asked angle first.
Overlay: overlay_bottle.overlay on every kept cut; it decides by itself and records why not.

  python spikes/creazy_exp/pick2.py
"""
import json, shutil, subprocess, os, sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pick  # noqa: E402
import overlay_bottle  # noqa: E402
import proxy_bottle  # noqa: E402

EXP = ROOT / "out" / "creazy_exp2"
if len(sys.argv) > 1:
    EXP = Path(sys.argv[1]).resolve()
WORK = EXP / "B"
CAND = WORK / "candidates"
KEEP = 2


def check():
    files = sorted(CAND.glob("*.png"))
    out = WORK / "check_B"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"); env.pop("OPENAI_API_KEY", None)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "check.run", "--refset", str(WORK / "refset" / "refset.json"),
                    "--outputs", *map(str, files), "--out", str(out), "--no-vlm", "--strong"],
                   cwd=ROOT, check=True, capture_output=True, env=env)
    return json.loads((out / "report.json").read_text())


def main():
    pick.CAND = CAND
    rep = check()
    anchors = json.loads((WORK / "refset" / "expect.json").read_text())["anchors"]
    scenes = {}
    for row in rep["images"]:
        scene = row["file"].rsplit("_c", 1)[0]
        key, info = pick.rank(row, anchors)
        # product count: the checker's own count read this photo as 2 (a baby bottle drawn on the
        # label), so the experiment counts label-matched bottles instead (count_product.py)
        import count_product
        n_prod = len(count_product.product_boxes(cv2.imread(str(CAND / row["file"])), str(overlay_bottle.SRC)))
        key = (key[0], key[1], int(n_prod > 1)) + key[3:]
        info = {**info, "count_over": n_prod > 1, "product_count": n_prod}
        scenes.setdefault(scene, []).append((key, row["file"], info))
    picks = {}
    dst = EXP / "outputs" / "B"; dst.mkdir(parents=True, exist_ok=True)
    raw = WORK / "selected_raw"; raw.mkdir(exist_ok=True)
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    asked = {f"{p['id']}_{t['id']}": proxy_bottle.view_for(t["scene"]) for p in cfg["products"] for t in p["targets"]}
    rb, ra, cx, R, yb0, yb1 = overlay_bottle.real()
    for scene, rows in sorted(scenes.items()):
        v = asked.get(scene)
        if v is not None:
            new_rows = []
            for key, f, info in rows:
                pp, qq = overlay_bottle.matches(rb, ra, cv2.imread(str(CAND / f)))
                n, par = overlay_bottle.fit_cylinder(pp, qq, cx, R) if len(pp) >= overlay_bottle.MIN_INLIERS else (0, None)
                found = par is not None and n >= overlay_bottle.MIN_INLIERS
                turn = par[0] if found else None
                if v[0] == 180:
                    ok, err = (not found), 0
                else:
                    ok = found and abs(turn) >= 15
                    err = abs(abs(turn) - v[0]) if found else 999
                info = {**info, "asked_view_deg": v[0], "front_label_found": found, "turn_deg": turn, "shows_asked_view": ok}
                new_rows.append(((not ok, err) + key, f, info))
            rows = new_rows
        rows.sort()
        picks[scene] = [{"file": f, "kept": i < KEEP, **info} for i, (_, f, info) in enumerate(rows)]
        for s, r in enumerate([r for r in picks[scene] if r["kept"]], 1):
            name = f"{scene}_B_s{s}.png"
            shutil.copy(CAND / r["file"], raw / name)
            raw_img = cv2.imread(str(CAND / r["file"]))
            out, oinfo = overlay_bottle.overlay(raw_img)
            if oinfo.get("applied"):
                # check after the fix: the pasted label must look more like the real label than
                # the generated one did (label SIFT inliers on the product crop); otherwise undo
                import count_product
                boxes = count_product.product_boxes(raw_img, str(overlay_bottle.SRC))
                if boxes:
                    (x0, y0, x1, y1), _ = max(boxes, key=lambda b: b[1])
                    pw, ph = int(0.1 * (x1 - x0)), int(0.05 * (y1 - y0))
                    sl = (slice(max(y0 - ph, 0), y1 + ph), slice(max(x0 - pw, 0), x1 + pw))
                    before = count_product.label_inliers(raw_img[sl], str(overlay_bottle.SRC))
                    chk, _ = overlay_bottle.overlay(raw_img, finish=False)  # same placement, no blur or grain
                    after = count_product.label_inliers(chk[sl], str(overlay_bottle.SRC))
                    oinfo["label_match_before_after"] = [before, after]
                    if after < 1.1 * before:
                        out = raw_img
                        oinfo.update(applied=False, why=f"after pasting the label matched the real one no better ({before} -> {after}), undone")
            cv2.imwrite(str(dst / name), out)
            r["overlay"] = oinfo
            print(f"{name} <- {r['file']}  view_ok={r.get('shows_asked_view')} turn={r.get('turn_deg')} inliers={r['inliers']} overlay={oinfo}")
    (WORK / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

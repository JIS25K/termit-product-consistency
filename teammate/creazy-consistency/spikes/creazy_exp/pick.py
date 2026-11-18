"""Condition B, step 3: check every candidate with our checker and keep the best 2 per scene.

Ranking, in this order (the reasons are in the experiment notes):
  1. text: how many anchor words (read in the seller's photos) are found, then fewer garbled
     words. The image is read upright and turned 90 degrees both ways and the best reading
     counts: a bottle held level for pouring carries its label sideways, and the checker's
     upright-only reading called those labels "missing" (seen on the A outputs).
  2. count: the product was counted more often than in the seller's photos (a weak signal:
     on a multi-part product such as the car holder OWLv2 also boxes the arm or the cradle
     alone, so it only breaks ties after the text).
  3. print match: SIFT inliers against the print region of the reference set.
  4. candidate number (earlier first), so the choice is reproducible.

  python spikes/creazy_exp/pick.py
"""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "out" / "creazy_exp_run" / "B"
CAND = WORK / "candidates"
KEEP = 2


def check(pid):
    files = sorted(CAND.glob(f"{pid}_*.png"))
    out = WORK / f"check_B_{pid}"
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "check.run", "--refset", str(WORK / f"refset_{pid}" / "refset.json"),
           "--outputs", *map(str, files), "--out", str(out), "--no-vlm", "--strong"]
    import os
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("OPENAI_API_KEY", None)
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, env=env)
    return json.loads((out / "report.json").read_text())


def text_any_rotation(path, anchors):
    from PIL import Image
    from check import text
    im = Image.open(path).convert("RGB")
    best = None
    for rot in (0, 90, 270):
        r = text.compare(anchors, text.read(im.rotate(rot, expand=True)))
        key = (len(r["found"]), -len(r["garbled"]))
        if best is None or key > best[0]:
            best = (key, rot, r)
    return best[1], best[2]


def rank(row, anchors):
    s = row.get("strong") or {}
    rot, t = (0, s.get("text") or {})
    if anchors:
        rot, t = text_any_rotation(CAND / row["file"], anchors)
    found, garbled = len(t.get("found") or []), len(t.get("garbled") or [])
    over = 1 if (s.get("count") or {}).get("over") else 0
    cand = int(row["file"].rsplit("_c", 1)[1].split(".")[0])
    return (-found, garbled, over, -row.get("inliers", 0), cand), dict(
        found=t.get("found"), missing=t.get("missing"), garbled=t.get("garbled"), text_rotation=rot,
        count_over=bool(over), inliers=row.get("inliers", 0), checker_final=row.get("final"))


def main():
    picks = {}
    for pid in ("P1", "P2"):
        rep = check(pid)
        anchors = json.loads((WORK / f"refset_{pid}" / "expect.json").read_text())["anchors"]
        scenes = {}
        for row in rep["images"]:
            scene = row["file"].rsplit("_c", 1)[0]
            key, info = rank(row, anchors)
            scenes.setdefault(scene, []).append((key, row["file"], info))
        for scene, rows in sorted(scenes.items()):
            rows.sort()
            picks[scene] = [{"file": f, "kept": i < KEEP, **info} for i, (_, f, info) in enumerate(rows)]
            for i, (_, f, info) in enumerate(rows):
                mark = "KEEP" if i < KEEP else "    "
                print(f"{mark} {f:22s} found={info['found']} garbled={info['garbled']} rot={info['text_rotation']} over={int(info['count_over'])} inliers={info['inliers']:3d} checker={info['checker_final']}")
    (WORK / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    sel = WORK / "selected"
    sel.mkdir(exist_ok=True)
    for scene, rows in picks.items():
        for s, r in enumerate([r for r in rows if r["kept"]], 1):
            shutil.copy(CAND / r["file"], sel / f"{scene}_B_s{s}.png")
    print("selected ->", sel)


if __name__ == "__main__":
    main()

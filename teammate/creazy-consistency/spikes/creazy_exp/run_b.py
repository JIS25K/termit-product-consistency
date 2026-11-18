"""Creazy product-change experiment, condition B (our system).

Controlled exactly like condition A (run_baseline.py in the experiment folder):
  same input photos in the same order, the scene sentence unchanged, gpt-image-2,
  1024x1024, quality medium, reference photos never sent to the model.
Our steps on top (agreed with the experiment owner):
  1. Before generation: reference set and anchor words from the input photos
     (refset.build, check.expect), and the rule-based keep lines of spec.make,
     appended after the scene sentence.
  2. Generation: 4 candidates per scene (A made 2).
  3. Check: our checker ranks the candidates (pick.py) and the best 2 are kept.
  4. Overlay with real pixels where the view and resolution allow (overlay step, separate).
No 3D guide image is added to the inputs.

  python spikes/creazy_exp/run_b.py --dry      # print the prompts, no API call
  python spikes/creazy_exp/run_b.py --pilot    # one call, to check the real cost
  python spikes/creazy_exp/run_b.py            # all candidates (files that exist are skipped)

The OpenAI key is read from OPENAI_API_KEY (environment or the repo's .env). It is never printed.
"""
import argparse, base64, csv, json, os, sys, time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "out" / "creazy_exp_run" / "creazy_exp"
WORK = ROOT / "out" / "creazy_exp_run" / "B"
CAND = WORK / "candidates"
LOG = WORK / "log_B.csv"
CANDIDATES = 4
CAP_USD = 2.0  # approved: about $1.8 for 28 calls
LOG_COLS = ["time", "product", "target", "candidate", "file", "model", "quality", "size",
            "text_in_tokens", "image_in_tokens", "out_tokens", "cost_usd", "total_spent_usd", "error"]


def load_env():
    """KEY=VALUE lines of the repo's .env; when a key appears twice the last line wins
    (the most recent entry). Values already in the environment are kept."""
    p = ROOT / ".env"
    if not p.exists():
        return
    found = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            found[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in found.items():
        os.environ.setdefault(k, v)


def keep_block(pid):
    """Our additions after the scene sentence: the rule-based keep line of spec.make and the
    words our OCR read in the seller's photos (check.expect anchors)."""
    kit = (WORK / f"spec_{pid}" / "kit" / "extra_request.txt").read_text(encoding="utf-8")
    lines = [l for l in kit.splitlines() if l.startswith("[반드시 유지]")]
    exp = json.loads((WORK / f"refset_{pid}" / "expect.json").read_text(encoding="utf-8"))
    words = [a["word"] for a in exp.get("anchors", [])]
    if words:
        lines.append("[실물 사진에서 읽힌 글자] " + ", ".join(words) + ": 철자, 순서, 위치를 실물 사진 그대로")
    return "\n".join(lines)


def prompt_for(cfg, pid, target):
    scene = cfg["prompt_template"].format(scene=target["scene"], angle=target["angle"])
    return scene + "\n\n" + keep_block(pid)


def spent_so_far():
    if not LOG.exists():
        return 0.0
    with open(LOG, encoding="utf-8-sig") as f:
        return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))


def write_log(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def cost_of(usage, price):
    usage = usage or {}
    det = usage.get("input_tokens_details") or {}
    ti, ii = det.get("text_tokens", 0) or 0, det.get("image_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    return ti, ii, out, (ti * price["text_in"] + ii * price["image_in"] + out * price["image_out"]) / 1e6


def call(cfg, paths, prompt):
    mime = lambda p: "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    files = [("image[]", (p.name, open(p, "rb"), mime(p))) for p in paths]  # upload order kept
    try:
        r = requests.post("https://api.openai.com/v1/images/edits",
                          headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                          data={"model": cfg["model"], "prompt": prompt, "size": cfg["size"],
                                "quality": cfg["quality"], "n": "1"},
                          files=files, timeout=300)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    jobs = [(p, t, c) for p in cfg["products"] for t in p["targets"] for c in range(1, CANDIDATES + 1)]
    if args.dry:
        for p in cfg["products"]:
            for t in p["targets"]:
                print(f"--- {p['id']}_{t['id']}  inputs={p['inputs']}\n{prompt_for(cfg, p['id'], t)}\n")
        print(f"{len(jobs)} calls planned, cap ${CAP_USD}")
        return
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없어요. 레포의 .env에 OPENAI_API_KEY=... 한 줄을 넣어 주세요.")
    CAND.mkdir(parents=True, exist_ok=True)
    spent = spent_so_far()
    print(f"지금까지 B에 쓴 금액: ${spent:.3f} / 상한 ${CAP_USD}")
    if args.pilot:
        jobs = jobs[:1]
    for p, t, c in jobs:
        fname = f"{p['id']}_{t['id']}_c{c}.png"
        fpath = CAND / fname
        if fpath.exists():
            continue
        if spent >= CAP_USD:
            print("상한에 도달해서 멈춰요.")
            break
        row = {"time": datetime.now().isoformat(timespec="seconds"), "product": p["id"], "target": t["id"],
               "candidate": c, "file": fname, "model": cfg["model"], "quality": cfg["quality"],
               "size": cfg["size"], "error": ""}
        try:
            res = call(cfg, [EXP / x for x in p["inputs"]], prompt_for(cfg, p["id"], t))
            fpath.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
            ti, ii, oo, usd = cost_of(res.get("usage"), cfg["price_per_million"])
            spent += usd
            row.update(text_in_tokens=ti, image_in_tokens=ii, out_tokens=oo,
                       cost_usd=round(usd, 4), total_spent_usd=round(spent, 4))
            print(f"[완료] {fname}  ${usd:.3f}  (누적 ${spent:.3f})", flush=True)
        except Exception as e:
            row.update(cost_usd=0, total_spent_usd=round(spent, 4), error=str(e)[:300])
            print(f"[실패] {fname}: {str(e)[:200]}", flush=True)
        write_log(row)
        time.sleep(1)
    if args.pilot:
        print(f"한 장 기준 전체 {CANDIDATES * sum(len(p['targets']) for p in cfg['products'])}회 예상: 약 ${spent_so_far() * 28:.2f}")


if __name__ == "__main__":
    main()

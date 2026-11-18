"""Second experiment (one shop photo of a pump bottle): generate condition A or B.

  python spikes/creazy_exp/gen2.py A          # baseline: input photo + scene sentence, 2 per scene
  python spikes/creazy_exp/gen2.py B --dry    # print B prompts and inputs
  python spikes/creazy_exp/gen2.py B          # ours: 4 candidates per scene

A is run exactly like the first experiment's run_baseline.py (images.edit, scene sentence only).
B adds, on top of the same photo and the same sentence:
  - the keep lines (spec.make rules) and the words our OCR read on the photo,
  - for a scene that asks for a view the photo does not show, a guide image rendered from our
    3D model of the bottle (out/creazy_exp2/B/guides/<scene>.png), sent after the seller's photo
    with one sentence saying what it is.
"""
import argparse, base64, csv, json, os, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_b import call, cost_of, load_env, LOG_COLS  # same request and cost code as the first experiment

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "out" / "creazy_exp2"
GUIDE_NOTE = ("[안내 그림] 마지막 사진은 위 제품을 요청한 각도에서 본 모양을 우리 3D 모형으로 그린 안내 그림이다. "
              "병과 펌프의 모양, 라벨의 위치와 방향은 이 그림을 따르고, 색과 질감과 라벨 그림은 첫 번째 사진의 실물을 따른다.")


PROMPT_MIN_CONF = 0.9  # a misread word in the prompt would make the generator write it


def reliable_words():
    """Words our OCR reads on the seller's photo with high confidence. Only these go into the
    prompt; the checker still uses the looser anchor list."""
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from check import text
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    out = []
    for rel in cfg["products"][0]["inputs"]:
        for r in text.read(Image.open(EXP / rel).convert("RGB")):
            t = r["text"].strip()
            if r["conf"] >= PROMPT_MIN_CONF and len(t) >= 2 and t not in out:
                out.append(t)  # as printed (case kept), not the checker's normalized form
    return out


def keep_block():
    kit = (EXP / "B" / "spec" / "kit" / "extra_request.txt").read_text(encoding="utf-8")
    lines = [l for l in kit.splitlines() if l.startswith("[반드시 유지]")]
    words = reliable_words()
    if words:
        lines.append("[실물 사진에서 읽힌 글자] " + ", ".join(words) + ": 철자, 순서, 위치를 실물 사진 그대로")
    return "\n".join(lines)


def job_inputs(cond, product, target):
    paths = [EXP / x for x in product["inputs"]]
    prompt = target["scene"]
    if cond == "B":
        prompt += "\n\n" + keep_block()
        g = EXP / "B" / "guides" / f"{target['id']}.png"
        if g.exists():
            paths.append(g)
            prompt += "\n" + GUIDE_NOTE
    return paths, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", choices=["A", "B"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="scene ids")
    ap.add_argument("--exp", default=None, help="experiment folder (default out/creazy_exp2)")
    args = ap.parse_args()
    global EXP
    if args.exp:
        EXP = Path(args.exp).resolve()
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    n = cfg["samples"] if args.cond == "A" else cfg["b_candidates"]
    outdir = EXP / "outputs" / "A" if args.cond == "A" else EXP / "B" / "candidates"
    log = EXP / f"log_{args.cond}.csv"
    jobs = [(p, t, s) for p in cfg["products"] for t in p["targets"] for s in range(1, n + 1)
            if not args.only or t["id"] in args.only]
    if args.dry:
        for p in cfg["products"]:
            for t in p["targets"]:
                paths, prompt = job_inputs(args.cond, p, t)
                print(f"--- {p['id']}_{t['id']} inputs={[x.name for x in paths]}\n{prompt}\n")
        print(len(jobs), "calls planned")
        return
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 없어요.")
    outdir.mkdir(parents=True, exist_ok=True)
    spent_all = 0.0
    for lf in EXP.glob("log_*.csv"):
        with open(lf, encoding="utf-8-sig") as f:
            spent_all += sum(float(r["cost_usd"] or 0) for r in csv.DictReader(f))
    for p, t, s in jobs:
        tag = f"_A_s{s}" if args.cond == "A" else f"_c{s}"
        fname = f"{p['id']}_{t['id']}{tag}.png"
        fpath = outdir / fname
        if fpath.exists():
            continue
        if spent_all >= cfg["budget_usd"]:
            print("예산에 도달해서 멈춰요."); break
        paths, prompt = job_inputs(args.cond, p, t)
        row = {"time": datetime.now().isoformat(timespec="seconds"), "product": p["id"], "target": t["id"],
               "candidate": s, "file": fname, "model": cfg["model"], "quality": cfg["quality"], "size": cfg["size"], "error": ""}
        try:
            res = call(cfg, paths, prompt)
            fpath.write_bytes(base64.b64decode(res["data"][0]["b64_json"]))
            ti, ii, oo, usd = cost_of(res.get("usage"), cfg["price_per_million"])
            spent_all += usd
            row.update(text_in_tokens=ti, image_in_tokens=ii, out_tokens=oo, cost_usd=round(usd, 4), total_spent_usd=round(spent_all, 4))
            print(f"[완료] {fname}  ${usd:.3f}  (전체 누적 ${spent_all:.3f})", flush=True)
        except Exception as e:
            row.update(cost_usd=0, total_spent_usd=round(spent_all, 4), error=str(e)[:300])
            print(f"[실패] {fname}: {str(e)[:200]}", flush=True)
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLS)
            if new:
                w.writeheader()
            w.writerow(row)
        time.sleep(1)


if __name__ == "__main__":
    main()

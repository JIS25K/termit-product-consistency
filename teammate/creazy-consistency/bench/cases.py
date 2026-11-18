"""Run the whole pipeline over a folder of cases (CS cases, or our own test products).

  python -m bench.cases --cases data/cs --out out/cs
  python -m bench.cases --cases data/cs --only 6 7 8 20 22 35 37      # the video CS cases

Each case is a folder:
  <case>/case.json     {"name": "seller's product name", "query": "a mug" (optional when a VLM
                        key is set), "print_query": optional, "cs": "what the CS was about"}
  <case>/uploads/      the seller's uploads, file names as Creazy stores them
                        (info-..., reference-..., the rest are product photos)
  <case>/outputs/      generated images and videos (.png .jpg .webp .mp4 .mov .webm)
  <case>/labels.json   optional hand labels (see data/mug/labels.json) -> scored

Customer data stays in data/cs/ (git-ignored). Reports carry case numbers only, never names.
"""
import argparse
import json
import traceback
from pathlib import Path

from check import run as check_run
from refset import build, vlm

from . import score as bench_score


def run_case(case_dir, out_dir, vlm_all=False, strong=False):
    meta = json.loads((case_dir / "case.json").read_text()) if (case_dir / "case.json").exists() else {}
    ref = build.build(
        case_dir / "uploads",
        meta.get("query"),
        out_dir / "refset",
        print_query=meta.get("print_query"),
        name=meta.get("name"),
    )
    row = {
        "case": case_dir.name,
        "cs": meta.get("cs", ""),
        "uploads": len(ref["items"]) + len(ref["excluded"]) + len(ref["dropped"]),
        "crops": len(ref["items"]),
        "excluded": [i["file"] for i in ref["excluded"]],
        "missing_faces": [f"{m['face_ko']} ({m['why']})" for m in ref["missing_faces"]],
        "prints": len(ref.get("prints") or []),
    }
    outputs = case_dir / "outputs"
    if outputs.exists() and (ref["print"]["path"] or vlm.available()):
        rep = check_run.run(out_dir / "refset" / "refset.json", [outputs], out_dir / "check", vlm_all=vlm_all, strong=strong)
        row["summary"] = rep["summary"]
        row["different"] = [
            {"file": i["file"], "why": i.get("vlm", {}).get("differences") or i.get("vlm", {}).get("reason"),
             **({"strong": i["strong"]["why"]} if i.get("strong") else {})}
            for i in rep["images"]
            if i["final"] == "different"
        ] + [{"file": v["file"], "at": v.get("different_at")} for v in rep["videos"] if v["final"] == "different"]
        row["videos"] = [{"file": v["file"], "min": v["min"], "dip_at": v["dip_at"], "final": v["final"]} for v in rep["videos"]]
        if (case_dir / "labels.json").exists():
            s = bench_score.score(json.loads((case_dir / "labels.json").read_text()), [rep])
            row["score"] = s["policies"]
    elif outputs.exists():
        row["note"] = "no print region (plain product): SIFT has nothing to match and no API key is set"
    return row


def scoreable(rows, cases_root):
    """The cases to score: those whose check ran in this run (their row has a summary). A case
    that errored, or whose check did not run, may still have an old report under --out from
    an earlier run; it is not scored. Returns (case dirs, labelled cases left out)."""
    root = Path(cases_root)
    ran = [root / r["case"] for r in rows if "summary" in r]
    left = [r["case"] for r in rows if "summary" not in r and (root / r["case"] / "labels.json").exists()]
    return ran, left


def markdown(rows):
    lines = [
        "| case | CS | uploads | crops | excluded | missing faces | match | not visible | different | unverified |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r.get("summary", {})
        lines.append(
            f"| {r['case']} | {r.get('cs', '')} | {r.get('uploads', '')} | {r.get('crops', '')} | {len(r.get('excluded', []))} | "
            f"{'; '.join(r.get('missing_faces', [])) or '-'} | {s.get('match', '-')} | {s.get('not_visible', '-')} | "
            f"{s.get('different', '-')} | {s.get('unverified', '-')} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--vlm-all", action="store_true")
    ap.add_argument("--strong", action="store_true", help="also check text and count")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in sorted(p for p in Path(a.cases).iterdir() if p.is_dir() and (p / "uploads").exists()):
        if a.only and case.name not in a.only:
            continue
        print("==", case.name)
        try:
            rows.append(run_case(case, out / case.name, a.vlm_all, a.strong))
        except (Exception, SystemExit) as e:
            traceback.print_exc()
            rows.append({"case": case.name, "error": f"{type(e).__name__}: {e}"})
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (out / "summary.md").write_text(markdown(rows))
    print(markdown(rows))
    ran, left = scoreable(rows, a.cases)
    s = bench_score.score_cases(ran, out)
    if s is not None:
        (out / "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
        print(bench_score.table(s))
    if left:
        print(f"labelled cases not scored (error, or no check in this run): {left}")


if __name__ == "__main__":
    main()

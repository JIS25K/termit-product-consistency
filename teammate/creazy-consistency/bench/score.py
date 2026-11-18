"""Score check reports against hand labels.

  python -m bench.score --labels data/mug/labels.json --report out/mug/check_all/report.json

Labels: match / not_visible / different per file (see data/mug/labels.json). A "different"
label is a real problem a person should see. Policies compared; "flagged" is the number of
outputs each one sends to a person (what someone has to look at):

  first_pass_flag_nothing      SIFT only, unverified is let through
  first_pass_flag_unverified   SIFT only, every unverified output goes to a person
  two_passes                   SIFT + VLM, only outputs judged "different" go to a person
                               (listed when the VLM judged at least one labelled output)
  strong_no_key                strong check (SIFT, text, count; no API key), only outputs it
                               calls "different" go to a person, unverified is let through
  strong_flag_not_match        strong check, every output it does not call "match" (different
                               and unverified) goes to a person

The two strong policies are listed when the report carries strong verdicts (check.run
--strong). The strong check covers still images only; outputs without a strong verdict
(videos) keep their first-pass verdict under strong_flag_not_match.

by_reason: for each label reason in "_why" (text, count, shape, ...), how many of the
"different" labels with that reason each policy flagged.
"""
import argparse
import json
from pathlib import Path


def flat_labels(labels):
    out = {}
    for k, v in labels.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out


def rows_of(reports):
    rows = {}
    for rep in reports:
        for i in rep["images"]:
            rows[i["file"]] = {"first": i["verdict"], "final": i.get("final", i["verdict"]), "inliers": i["inliers"],
                               "strong": (i.get("strong") or {}).get("verdict"), "vlm": "vlm" in i}
        for v in rep["videos"]:
            rows[v["file"]] = {"first": v["verdict"], "final": v.get("final", v["verdict"]), "inliers": v["min"],
                               "strong": (v.get("strong") or {}).get("verdict"), "vlm": "vlm" in v}
    return rows


def prf(flagged, truth):
    tp = len(flagged & truth)
    p = tp / len(flagged) if flagged else None
    r = tp / len(truth) if truth else None
    return {"flagged": len(flagged), "caught": tp, "problems": len(truth), "precision": p, "recall": r}


def _flagged(policy, row):
    if policy == "first_pass_flag_nothing":
        return False
    if policy == "first_pass_flag_unverified":
        return row["first"] != "match"
    if policy == "two_passes":
        return row["final"] == "different"
    if policy == "strong_flag_not_match":
        return (row["strong"] or row["first"]) != "match"
    return row["strong"] == "different"  # strong_no_key


def score(labels, reports):
    lab = flat_labels(labels)
    rows = rows_of(reports)
    keys = [k for k in lab if k in rows]
    truth = {k for k in keys if lab[k] == "different"}
    first_match_ok = sum(1 for k in keys if (rows[k]["first"] == "match") == (lab[k] == "match"))
    names = ["first_pass_flag_nothing", "first_pass_flag_unverified"]
    # A strong run sets final from the strong verdict, so final alone does not say a VLM ran.
    vlm_ran = any(rows[k]["vlm"] for k in keys)
    if vlm_ran:
        names.append("two_passes")
    if any(rows[k]["strong"] for k in keys):
        names += ["strong_no_key", "strong_flag_not_match"]
    flagged_by = {
        name: {k for k in keys if _flagged(name, rows[k])} for name in names
    }
    policies = {name: prf(flagged_by[name], truth) for name in names}
    why = labels.get("_why", {})
    by_reason = {}
    for k in sorted(truth):
        for reason in why.get(k, []):
            r = by_reason.setdefault(reason, {"problems": 0, **{name: 0 for name in policies}})
            r["problems"] += 1
            for name in policies:
                r[name] += k in flagged_by[name]
    confusion = {}
    for k in keys:
        key = (lab[k], rows[k]["final"])
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "n": len(keys),
        "missing_from_report": [k for k in lab if k not in rows],
        "labels": {c: sum(1 for k in keys if lab[k] == c) for c in ("match", "not_visible", "different")},
        "first_pass_match_agreement": f"{first_match_ok}/{len(keys)}",
        "policies": policies,
        "by_reason": by_reason,
        "confusion_label_vs_final": {f"{a} -> {b}": n for (a, b), n in sorted(confusion.items())},
        "items": {k: {"label": lab[k], **rows[k]} for k in keys},
    }


def fmt(x):
    return "-" if x is None else f"{x:.0%}"


def score_cases(case_dirs, out_root):
    """Merge the labels of every case that has labels.json and a check report.

    None when no case has labels. "skipped_no_report" lists the labelled cases that have no
    report under out_root; they are not scored. Labels and report rows are merged by file
    name, so a file name labelled in one case that also appears in another case's labels or
    report raises ValueError instead of mixing the two."""
    labels, whys, reports, skipped, names = {}, {}, [], [], {}
    for case in case_dirs:
        case = Path(case)
        lab_path = case / "labels.json"
        rep_path = Path(out_root) / case.name / "check" / "report.json"
        if not lab_path.exists():
            continue
        if not rep_path.exists():
            skipped.append(case.name)
            continue
        lab = json.loads(lab_path.read_text())
        rep = json.loads(rep_path.read_text())
        own = set(flat_labels(lab))
        names[case.name] = (own, own | set(rows_of([rep])))
        labels.update(flat_labels(lab))
        whys.update(lab.get("_why", {}))
        reports.append(rep)
    for a, (own, _) in names.items():
        for b, (_, seen) in names.items():
            if a != b and own & seen:
                raise ValueError(f"case {a} labels {sorted(own & seen)}, and case {b} has the same file names in its "
                                 "labels or report; scores merge by file name, so rename the files or score the cases apart")
    if not names and not skipped:
        return None
    return {**score({"_why": whys, "outputs": labels}, reports), "skipped_no_report": skipped}


def table(s):
    lines = [
        f"n={s['n']}  labels={s['labels']}  first-pass match agreement {s['first_pass_match_agreement']}",
        f"{'policy':30s} flagged  caught/problems  precision  recall   (flagged = outputs sent to a person)",
    ]
    for name, p in s["policies"].items():
        lines.append(f"{name:30s} {p['flagged']:7d}  {p['caught']:6d}/{p['problems']:<8d} {fmt(p['precision']):>9s}  {fmt(p['recall']):>6s}")
    if s.get("by_reason"):
        lines.append("by reason (problems  policy:caught ...)")
        for reason, r in s["by_reason"].items():
            caught = " ".join(f"{k}:{v}" for k, v in r.items() if k != "problems")
            lines.append(f"  {reason:14s} {r['problems']:3d}  {caught}")
    lines.append(f"label -> final: {s['confusion_label_vs_final']}")
    if s["missing_from_report"]:
        lines.append(f"not in report: {s['missing_from_report']}")
    if s.get("skipped_no_report"):
        lines.append(f"labelled cases with no report, not scored: {s['skipped_no_report']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--report", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="write the score as JSON here")
    a = ap.parse_args()
    s = score(json.loads(Path(a.labels).read_text()), [json.loads(Path(r).read_text()) for r in a.report])
    print(table(s))
    if a.out:
        Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

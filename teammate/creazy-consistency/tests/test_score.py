import json

import pytest

from bench import score

LAB = {"_why": {"b.png": ["text"], "c.png": ["count"]}, "outputs": {"a.png": "match", "b.png": "different", "c.png": "different"}}
REP = {"images": [
    {"file": "a.png", "verdict": "match", "final": "match", "inliers": 40, "strong": {"verdict": "match"}},
    # b.png is the one the VLM judged (a --vlm-all run): its row carries "vlm"
    {"file": "b.png", "verdict": "match", "final": "different", "inliers": 30, "strong": {"verdict": "different"},
     "vlm": {"verdict": "different"}},
    {"file": "c.png", "verdict": "unverified", "final": "unverified", "inliers": 3, "strong": {"verdict": "unverified"}},
], "videos": []}


def test_strong_policy_and_reasons():
    s = score.score(LAB, [REP])
    p = s["policies"]["strong_no_key"]
    assert (p["flagged"], p["caught"], p["problems"]) == (1, 1, 2)
    assert s["by_reason"]["text"]["strong_no_key"] == 1
    assert s["by_reason"]["count"] == {"problems": 1, "first_pass_flag_nothing": 0, "first_pass_flag_unverified": 1,
                                       "two_passes": 0, "strong_no_key": 0, "strong_flag_not_match": 1}


def test_keyless_strong_run_has_no_two_passes_row():
    # final carries the strong verdict, but no VLM ran: there is no second pass to score
    rep = {"images": [{k: v for k, v in i.items() if k != "vlm"} for i in REP["images"]], "videos": []}
    s = score.score(LAB, [rep])
    assert "two_passes" not in s["policies"]
    assert all("two_passes" not in r for r in s["by_reason"].values())
    assert "two_passes" not in score.table(s)


def test_strong_flag_not_match_counts_what_a_person_has_to_look_at():
    lab = {"_why": {"b.png": ["text"], "c.png": ["count"], "v.mp4": ["shape"]},
           "outputs": {"a.png": "match", "b.png": "different", "c.png": "different", "d.png": "not_visible",
                       "v.mp4": "different", "w.mp4": "match"}}
    vids = [{"file": "v.mp4", "verdict": "dip", "final": "unverified", "min": 3},
            {"file": "w.mp4", "verdict": "match", "final": "match", "min": 30}]
    rep = {"images": REP["images"] + [_img("d.png", "unverified", "unverified", "unverified")], "videos": vids}
    s = score.score(lab, [rep])
    p = s["policies"]
    # strong says match / different / unverified / unverified for a b c d; videos have no strong
    # verdict and keep their first pass (v dips, w matches)
    assert (p["strong_no_key"]["flagged"], p["strong_no_key"]["caught"]) == (1, 1)
    assert (p["strong_flag_not_match"]["flagged"], p["strong_flag_not_match"]["caught"]) == (4, 3)
    assert p["first_pass_flag_unverified"]["flagged"] == 3
    assert s["by_reason"]["count"]["strong_flag_not_match"] == 1 and s["by_reason"]["count"]["strong_no_key"] == 0
    assert s["by_reason"]["shape"]["strong_flag_not_match"] == 1
    lines = {l.split()[0]: l.split() for l in score.table(s).splitlines() if l.split()}
    # every policy row prints how many outputs go to a person, side by side
    assert lines["strong_flag_not_match"][1] == "4" and lines["strong_no_key"][1] == "1"
    assert lines["first_pass_flag_unverified"][1] == "3"


def test_null_strong_is_no_strong_verdict():
    rep = {"images": [{"file": "a.png", "verdict": "match", "inliers": 30, "strong": None}],
           "videos": [{"file": "v.mp4", "verdict": "match", "min": 30, "strong": None}]}
    rows = score.rows_of([rep])
    assert rows["a.png"]["strong"] is None and rows["v.mp4"]["strong"] is None
    assert "strong_no_key" not in score.score({"outputs": {"a.png": "match"}}, [rep])["policies"]


def test_old_reports_score_as_before():
    rep = {"images": [{k: v for k, v in i.items() if k != "strong"} for i in REP["images"]], "videos": []}
    assert "strong_no_key" not in score.score(LAB, [rep])["policies"]


def _img(f, first, final, strong):
    return {"file": f, "verdict": first, "final": final, "inliers": 10, "strong": {"verdict": strong}}


def test_score_cases_merges_labelled_cases(tmp_path):
    cases, out = tmp_path / "cs", tmp_path / "out"
    for name in ("1", "2", "3"):
        (cases / name / "uploads").mkdir(parents=True)
        (out / name / "check").mkdir(parents=True)
    (cases / "1" / "labels.json").write_text(json.dumps(
        {"_why": {"x1.png": ["text"]}, "outputs": {"x0.png": "match", "x1.png": "different"}}))
    (cases / "2" / "labels.json").write_text(json.dumps(
        {"_why": {"y1.png": ["count"]}, "outputs": {"y1.png": "different"}}))
    # case 3: no labels, report must be ignored
    reps = {
        "1": [_img("x0.png", "match", "match", "match"), _img("x1.png", "match", "different", "different")],
        "2": [_img("y1.png", "unverified", "unverified", "different")],
        "3": [_img("z.png", "match", "match", "match")],
    }
    for name, imgs in reps.items():
        (out / name / "check" / "report.json").write_text(json.dumps({"images": imgs, "videos": []}))
    dirs = [cases / n for n in ("1", "2", "3")]
    s = score.score_cases(dirs, out)
    assert s["n"] == 3
    assert s["by_reason"]["text"]["strong_no_key"] == 1
    assert s["by_reason"]["count"]["strong_no_key"] == 1
    assert "z.png" not in s["items"]
    assert s["skipped_no_report"] == []
    assert "text" in score.table(s)
    # a labelled case with no report is skipped and listed; no labels at all -> None
    (out / "2" / "check" / "report.json").unlink()
    s2 = score.score_cases(dirs, out)
    assert s2["n"] == 2 and s2["skipped_no_report"] == ["2"]
    assert "['2']" in score.table(s2).splitlines()[-1]
    only = score.score_cases([cases / "2"], out)
    assert only["n"] == 0 and only["skipped_no_report"] == ["2"]
    assert score.score_cases([cases / "3"], out) is None


def test_score_cases_refuses_a_file_name_in_two_cases(tmp_path):
    cases, out = tmp_path / "cs", tmp_path / "out"
    for name in ("1", "2"):
        (cases / name / "uploads").mkdir(parents=True)
        (out / name / "check").mkdir(parents=True)
        (out / name / "check" / "report.json").write_text(json.dumps(
            {"images": [_img("same.png", "match", "match", "match"), _img(f"own{name}.png", "match", "match", "match")],
             "videos": []}))
    (cases / "1" / "labels.json").write_text(json.dumps({"outputs": {"same.png": "match"}}))
    (cases / "2" / "labels.json").write_text(json.dumps({"outputs": {"same.png": "different"}}))
    dirs = [cases / "1", cases / "2"]
    with pytest.raises(ValueError, match="same.png"):
        score.score_cases(dirs, out)
    # labelled in one case only, but the other case's report has the same name: rows would mix too
    (cases / "2" / "labels.json").write_text(json.dumps({"outputs": {"own2.png": "match"}}))
    with pytest.raises(ValueError, match="same.png"):
        score.score_cases(dirs, out)
    (cases / "1" / "labels.json").write_text(json.dumps({"outputs": {"own1.png": "match"}}))
    assert score.score_cases(dirs, out)["n"] == 2


def test_cases_scores_only_cases_checked_in_this_run(tmp_path):
    from bench import cases
    for n in ("1", "2", "3", "4"):
        (tmp_path / n).mkdir()
    for n in ("1", "2", "3"):
        (tmp_path / n / "labels.json").write_text("{}")
    rows = [{"case": "1", "summary": {"match": 1}}, {"case": "2", "error": "RuntimeError: x"},
            {"case": "3", "note": "no print region"}, {"case": "4", "error": "RuntimeError: y"}]
    ran, left = cases.scoreable(rows, tmp_path)
    # an errored case (or one whose check did not run) may have an old report: not scored, listed
    assert [p.name for p in ran] == ["1"] and left == ["2", "3"]

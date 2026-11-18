import json
import pytest
from fastapi import HTTPException
from PIL import Image
from demo import app as demo

@pytest.fixture
def cs(tmp_path, monkeypatch):
    d = tmp_path / "cs" / "10"
    (d / "uploads").mkdir(parents=True); (d / "outputs").mkdir()
    Image.new("RGB", (64, 64)).save(d / "uploads" / "p1.webp")
    Image.new("RGB", (64, 64)).save(d / "uploads" / "info-x.webp")
    for n in ("a.png", "b.png"):
        Image.new("RGB", (64, 64)).save(d / "outputs" / n)
    (d / "outputs" / "v.mp4").write_bytes(b"")
    (d / "case.json").write_text(json.dumps({"query": "a bottle"}))
    rep = tmp_path / "out" / "10" / "check"; rep.mkdir(parents=True)
    rep.joinpath("report.json").write_text(json.dumps({"images": [{"file": "a.png", "verdict": "match", "final": "match", "inliers": 30}], "videos": []}))
    monkeypatch.setattr(demo, "CS", tmp_path / "cs")
    monkeypatch.setattr(demo, "LABEL_REPORTS", [tmp_path / "out"])
    return d

def test_case_lists_stills_and_product_photos_only(cs):
    c = demo.label_case("10")
    assert [o["file"] for o in c["outputs"]] == ["a.png", "b.png"]
    assert len(c["refs"]) == 1 and all(o["tool"] is None for o in c["outputs"])

def test_tool_verdict_only_after_labelling(cs):
    r = demo.save_label("10", {"file": "a.png", "label": "match", "why": []})
    assert r["tool"]["first"] == "match"
    outs = {o["file"]: o for o in demo.label_case("10")["outputs"]}
    assert outs["a.png"]["tool"] and not outs["b.png"].get("tool")

def test_relabel_overwrites_and_clears_why(cs):
    demo.save_label("10", {"file": "a.png", "label": "different", "why": ["text"]})
    demo.save_label("10", {"file": "a.png", "label": "match", "why": []})
    lab = json.loads((cs / "labels.json").read_text())
    assert lab["outputs"]["a.png"] == "match" and "a.png" not in lab["_why"]

def test_bad_input_is_refused(cs):
    for case, body in [("99", {"file": "a.png", "label": "match", "why": []}),
                       ("10", {"file": "zz.png", "label": "match", "why": []}),
                       ("10", {"file": "a.png", "label": "maybe", "why": []}),
                       ("10", {"file": "a.png", "label": "different", "why": []}),
                       ("10", {"file": "../case.json", "label": "match", "why": []})]:
        with pytest.raises(HTTPException):
            demo.save_label(case, body)

def test_labels_file_scores_with_the_existing_scorer(cs):
    from bench import score
    demo.save_label("10", {"file": "a.png", "label": "match", "why": []})
    lab = json.loads((cs / "labels.json").read_text())
    assert score.flat_labels(lab) == {"a.png": "match"}

def test_label_file_serves_only_known_kinds_and_names(cs, tmp_path):
    assert demo.label_file("10", "output", "a.png").path == cs / "outputs" / "a.png"
    assert demo.label_file("10", "upload", "p1.webp").path == cs / "uploads" / "p1.webp"
    # a traversal name is cut to its last part, which is not an output
    for case, kind, name in [("10", "output", "../case.json"), ("10", "output", "..%2Fcase.json"),
                             ("10", "output", "case.json"), ("10", "secret", "a.png"),
                             ("10", "print", "print.png"), ("99", "output", "a.png"),
                             ("2", "output", "a.png")]:
        with pytest.raises(HTTPException) as e:
            demo.label_file(case, kind, name)
        assert e.value.status_code == 404

def test_print_is_served_from_the_report_root_refset(cs, tmp_path):
    ref = tmp_path / "out" / "10" / "refset"; ref.mkdir()
    Image.new("RGB", (8, 8)).save(ref / "print.png")
    (ref / "refset.json").write_text(json.dumps({"prints": [{"path": "print.png"}]}))
    c = demo.label_case("10")
    assert "/api/label-file/10/print/print.png" in c["refs"]
    assert demo.label_file("10", "print", "../../print.png").path == ref / "print.png"

def test_tool_carries_strong_verdict_and_why(cs, tmp_path):
    rep = tmp_path / "out" / "10" / "check" / "report.json"
    rep.write_text(json.dumps({"images": [{"file": "a.png", "verdict": "match", "final": "different", "inliers": 3,
                                           "strong": {"verdict": "different", "why": ["text"]}}]}))
    r = demo.save_label("10", {"file": "a.png", "label": "different", "why": ["text"]})
    assert r["tool"] == {"first": "match", "final": "different", "strong": "different", "why": ["text"]}
    assert demo.save_label("10", {"file": "b.png", "label": "match", "why": []})["tool"] is None

def test_unreadable_labels_file_is_refused_not_overwritten(cs):
    demo.save_label("10", {"file": "a.png", "label": "match", "why": []})
    for bad in ('{"outputs": {"a.png": "match"', "[]", '{"outputs": ["a.png"]}'):
        (cs / "labels.json").write_text(bad)
        with pytest.raises(HTTPException) as e:
            demo.save_label("10", {"file": "b.png", "label": "match", "why": []})
        assert e.value.status_code == 409 and "labels.json" in e.value.detail
        assert (cs / "labels.json").read_text() == bad
        # the pages still open; they just show no labels
        assert demo.label_case("10")["outputs"][0]["label"] is None
        assert demo.label_cases()[0]["done"] == 0
    # a missing file is a fresh start, not an error
    (cs / "labels.json").unlink()
    assert demo.save_label("10", {"file": "b.png", "label": "match", "why": []})["label"] == "match"

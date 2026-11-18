"""End-to-end run of build + check with a fake Claude endpoint (no API key, no cost).

The fake endpoint answers from the schema the request asks for, so the test checks that the
requests are well formed and that the VLM answers land in refset.json and report.json.

  .venv/bin/python -m pytest tests -q
"""
import json
import os
from pathlib import Path

import anthropic
import httpx2
import pytest

from check import run as check_run
from refset import build, vlm

ROOT = Path(__file__).resolve().parent.parent
MUG = ROOT / "data" / "mug"


def fake_answer(body):
    schema = body["output_config"]["format"]["schema"]["title"]
    content = body["messages"][0]["content"]
    n_photos = sum(1 for c in content if c["type"] == "text" and c["text"].startswith("Photo "))
    n_images = sum(1 for c in content if c["type"] == "image")
    if schema == "Queries":
        return {"product": "머그컵", "query": "a mug", "print_query": "a cat line drawing"}
    if schema == "SetReview":
        assert n_images == n_photos
        views = [{"index": i, "face": "front" if i == 1 else "handle", "same_product": True, "note": "테스트"} for i in range(1, n_photos + 1)]
        return {"product": "머그컵", "views": views, "missing": [{"face": "back", "why": "회전 영상", "shot": "프린트 반대쪽을 정면에서 찍기"}]}
    if schema == "Spec":
        assert any(c["type"] == "text" and "Seller's request" in c["text"] for c in content)
        return {
            "product": "머그컵", "scene": "카페 테이블", "keep": ["흰 고양이 선화와 Faire une pause 문구, 프린트 정면일 때 손잡이 오른쪽"],
            "reference_use": ["배경"], "reference_ignore": [], "changeable": ["배경"], "needed_faces": ["front"],
            "conflicts": [], "notes": [], "decisions": [], "request_ko": "[반드시 유지] 테스트", "prompt_en": "test",
        }
    if schema == "Verdict":
        # reference crops + print region + the generated image
        assert n_images >= 3
        assert any(c.get("cache_control") for c in content if c["type"] == "image")
        return {"product_present": True, "verdict": "not_visible", "differences": [], "reason": "프린트 면이 보이지 않음"}
    raise AssertionError(schema)


@pytest.fixture
def fake_vlm(monkeypatch):
    class Calls(list):
        """Schema titles of the requests, in order; .bodies holds the request bodies."""

    calls = Calls()
    calls.bodies = bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        assert body["model"] == vlm.MODEL
        assert body["fallbacks"] == "default"
        calls.append(body["output_config"]["format"]["schema"]["title"])
        out = fake_answer(body)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    vlm.set_client(anthropic.Anthropic(api_key="test-key", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))))
    yield calls
    vlm.set_client(None)


def test_build_and_check_with_vlm(tmp_path, fake_vlm):
    ref = build.build(MUG / "uploads", None, tmp_path / "mug", name="고양이 라인드로잉 머그컵", use_vlm=True)
    assert ref["query"] == "a mug" and ref["print_query"] == "a cat line drawing"
    assert ref["name"] == "고양이 라인드로잉 머그컵"
    assert ref["representative"] == "crops/info-detail_cut.png"
    # fake review: photo 1 is the front, the rest are all "handle" -> one view per face
    assert ref["composite"]["views"] == ["crops/info-detail_cut.png", "crops/plain_back.png"]
    assert (tmp_path / "mug" / "composite.jpg").exists()
    assert all(i["face_ko"] for i in ref["items"])
    assert ref["missing_faces"][0]["face_ko"] == "뒷면"
    assert ref["vlm"]["steps"]["review"] == "ok"

    out = check_run.run(tmp_path / "mug" / "refset.json", [MUG / "outputs" / "run9"], tmp_path / "check", draw=False, use_vlm=True)
    imgs = {i["file"]: i for i in out["images"]}
    assert all(i["final"] == "match" for i in imgs.values() if i["verdict"] == "match")
    assert all(i["final"] == "not_visible" and "vlm" in i for i in imgs.values() if i["verdict"] == "unverified")
    assert out["summary"]["unverified"] == 0
    assert fake_vlm.count("Verdict") == sum(1 for i in imgs.values() if i["verdict"] == "unverified")


def test_without_key_runs_first_pass_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(vlm, "_load_env", lambda: None)
    ref = build.build(MUG / "uploads", "a mug", tmp_path / "mug")
    assert ref["vlm"]["used"] is False and "skipped" in ref["vlm"]
    comp = ref["composite"]
    assert comp["path"] == "composite.jpg" and comp["views"][0] == ref["representative"]
    assert len(comp["views"]) == len(set(comp["views"])) >= 2
    out = check_run.run(tmp_path / "mug" / "refset.json", [MUG / "outputs" / "run8"], tmp_path / "check", draw=False)
    assert out["vlm"]["used"] is False
    assert all(i["final"] in ("match", "unverified") for i in out["images"])


def test_dip_segments():
    rows = [{"inliers": v} for v in (40, 3, 0, 45, 50, 2, 41, 0)]
    assert check_run.dip_segments(rows) == [[1, 2], [5], [7]]


def test_plain_product_and_video_with_vlm(tmp_path, fake_vlm):
    ref = build.build(MUG / "uploads", "a mug", tmp_path / "mug", use_vlm=True)
    # pretend the product had no print: the check must fall back to the VLM alone
    rj = tmp_path / "mug" / "refset.json"
    data = json.loads(rj.read_text())
    data["print"]["path"], data["prints"] = None, []
    rj.write_text(json.dumps(data))
    out = check_run.run(rj, [MUG / "outputs" / "run6" / "a01_778eef14.png", MUG / "real_video"], tmp_path / "check", use_vlm=True)
    assert out["plain_product"] is True
    assert out["images"][0]["verdict"] == "no_print" and out["images"][0]["final"] == "not_visible"
    assert len(out["videos"][0]["vlm"]) == 5


def test_video_dips_go_to_vlm(tmp_path, fake_vlm):
    build.build(MUG / "uploads", "a mug", tmp_path / "mug", use_vlm=True)
    out = check_run.run(tmp_path / "mug" / "refset.json", [MUG / "real_video"], tmp_path / "check", use_vlm=True)
    v = out["videos"][0]
    assert v["verdict"] == "dip"
    assert v["hidden_at"] and v["final"] == "match"  # rotation out of view is not a failure


def test_composite_views(tmp_path):
    import numpy as np
    from PIL import Image

    def item(name, detail, face=None, fill=0.3, full=False, emb=(1, 0, 0), size=(300, 700)):
        e = np.array(emb, float)
        return {"crop": name, "detail": detail, "face": face, "full_frame": full, "frame_fill": fill, "crop_size": size,
                "_emb": e / np.linalg.norm(e), "_img": Image.new("RGB", size, "gray")}

    # with face names: the front goes first even when the back label carries more print,
    # one view per face, close-ups last
    back = item("back", 80, "back")
    items = [back, item("front", 50, "front"), item("front2", 40, "front"), item("head", 20, "top", fill=0.73),
             item("side", 30, "side")]
    assert [v["crop"] for v in build.pick_views(items, back)] == ["front", "back", "side", "head"]

    # without face names: least similar first, near-duplicates and close-ups skipped while whole views remain
    rep = item("rep", 80, emb=(1, 0, 0))
    items = [rep, item("dup", 70, emb=(1, 0.01, 0)), item("other", 30, emb=(0, 1, 0)),
             item("close", 10, emb=(0, 0, 1), full=True)]
    views = build.pick_views(items, rep)
    assert [v["crop"] for v in views] == ["rep", "other", "close"]
    meta = build.composite(views, tmp_path / "c.jpg")
    assert meta["layout"] == "row" and (tmp_path / "c.jpg").exists()
    assert build.composite(views[:1], tmp_path / "d.jpg") is None


def _uploads_with_reference_and_capture(tmp_path):
    import shutil

    up = tmp_path / "up"
    up.mkdir()
    for p in (MUG / "uploads").iterdir():
        shutil.copy(p, up / p.name)
    # a Run 9 output with an invented cat: a scene reference holding a look-alike mug
    shutil.copy(MUG / "outputs" / "run9" / "AI-Generated-68036f43-e5de-476e-bb1c-29661d5b6a63.png", up / "reference-desk.png")
    shutil.copy(MUG / "real_video" / "mug_rotate_real.mp4", up / "capture_turn.mp4")
    return up


def test_main_reference_and_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    up = _uploads_with_reference_and_capture(tmp_path)
    ref = build.build(up, "a mug", tmp_path / "ref", main="plain_back.jpg")
    files = [i["file"] for i in ref["items"]]
    assert "reference-desk.png" not in files  # scene references never become product evidence
    assert ref["references"][0]["file"] == "reference-desk.png" and ref["references"][0]["has_product"]
    assert ref["main"] == "crops/plain_back.png"
    assert ref["composite"]["views"][0] == "crops/plain_back.png"  # the main photo leads the composite
    caps = [i for i in ref["items"] if i["role"] == "capture"]
    assert 2 <= len(caps) <= 6 and ref["capture"][0]["kept"]


def test_spec_rules_kit_and_check(tmp_path, monkeypatch):
    from spec import make as spec_make

    monkeypatch.setattr(vlm, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    up = _uploads_with_reference_and_capture(tmp_path)
    ref = build.build(up, "a mug", tmp_path / "ref")
    rj = tmp_path / "ref" / "refset.json"

    doc = spec_make.make(rj, "뒷면 컷도 넣고 1+1 세트로, 좌우 반전도", tmp_path / "spec", use_vlm=False)
    s = doc["spec"]
    kinds = {c["kind"] for c in s["conflicts"]}
    assert {"multiple_units", "flip", "reference_has_product"} <= kinds
    assert "back" in s["needed_faces"] and s["notes"]  # faces unknown without the VLM
    assert any("프린트" in k for k in s["keep"])
    kit = tmp_path / "spec" / "kit"
    assert sorted(p.name for p in (kit / "1_product").iterdir()) == ["faces_composite.jpg", "print_face.png"]
    assert (kit / "2_info" / "info-detail_cut.jpg").exists() and (kit / "3_design_reference").iterdir()
    assert "제출 전에 정할 것" in (kit / "README.txt").read_text()

    # the seller drops the look-alike reference and keeps one unit
    decisions = []
    for c in s["conflicts"]:
        decisions.append("이 참고 사진 빼기" if c["kind"] == "reference_has_product" else c["options"][0])
    doc = spec_make.confirm(tmp_path / "spec", {"decisions": decisions}, rj)
    assert doc["confirmed"] and doc["references"] == []
    assert not list((kit / "3_design_reference").iterdir())
    assert "[정한 것]" in (kit / "extra_request.txt").read_text()

    out = check_run.run(rj, [MUG / "outputs" / "run9" / "AI-Generated-68036f43-e5de-476e-bb1c-29661d5b6a63.png"], tmp_path / "check", draw=False, spec=tmp_path / "spec")
    assert out["spec"]["confirmed"] and out["spec"]["keep"] == doc["spec"]["keep"]


def test_spec_with_vlm_and_keep_reaches_the_judge(tmp_path, fake_vlm):
    from spec import make as spec_make

    build.build(MUG / "uploads", "a mug", tmp_path / "ref", use_vlm=True)
    rj = tmp_path / "ref" / "refset.json"
    doc = spec_make.make(rj, "카페 테이블", tmp_path / "spec", use_vlm=True)
    assert doc["vlm"]["used"] and "Faire une pause" in doc["spec"]["keep"][0]
    check_run.run(rj, [MUG / "outputs" / "run8"], tmp_path / "check", draw=False, use_vlm=True, spec=tmp_path / "spec")
    judged = [b for b in fake_vlm.bodies if b["output_config"]["format"]["schema"]["title"] == "Verdict"]
    assert judged and all(any(c["type"] == "text" and "Faire une pause" in c["text"] for c in b["messages"][0]["content"]) for b in judged)


def test_reference_choices_and_hiding(tmp_path):
    from PIL import Image

    from spec import make as m

    # a reference mostly made of the product offers dropping first, otherwise hiding
    assert m.ref_options({"product_fill": 0.5})[0] == m.REF_DROP
    assert m.ref_options({"product_fill": 0.1})[0] == m.REF_HIDE
    # the seller's words, older wording and a VLM's rewording all map to an action
    assert [m.ref_action(c) for c in (m.REF_HIDE, m.REF_DROP, m.REF_KEEP, "구도, 배경, 조명만 참고하고 사진 속 제품은 무시하기", "참고 사진을 빼고 진행", "")] == [
        "hide", "drop", "keep", "hide", "drop", None]

    spec = m.Spec(product="", scene="", keep=[], reference_use=[], reference_ignore=[], changeable=[], needed_faces=[], conflicts=[], request_ko="[참고 사진] 구도만 참고\n[장면] 테스트", prompt_en="test")
    refs = [{"file": "a.png", "path": "references/a.jpg", "has_product": False},
            {"file": "b.png", "path": "references/b.jpg", "has_product": True, "product_boxes": [[0, 0, 10, 10]], "product_fill": 0.6}]
    m.treat_references(spec, refs)
    assert refs[0]["treatment"]["action"] == "keep" and refs[0]["treatment"]["by"] == "rule"
    assert refs[1]["treatment"]["action"] == "drop" and refs[1]["treatment"]["kit"] is None
    m.mark_hidden(spec, refs)
    assert m.HIDE_KO not in spec.request_ko  # nothing was hidden, so nothing to explain

    # hiding takes the color out of the box and leaves the rest of the photo alone
    img = Image.new("RGB", (200, 200), (220, 40, 40))
    out = m.hide_products(img, [[60, 60, 140, 140]])
    r, g, b = out.getpixel((100, 100))
    assert abs(r - g) < 4 and abs(g - b) < 4
    assert out.getpixel((5, 5)) == (220, 40, 40)


def test_reference_product_is_hidden_in_the_kit(tmp_path, monkeypatch):
    import shutil

    import numpy as np
    from PIL import Image

    from spec import make as m

    monkeypatch.setattr(vlm, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    up = tmp_path / "up"
    shutil.copytree(MUG / "uploads", up)
    # a Run 9 output: a look-alike mug with an invented cat on a desk
    shutil.copy(MUG / "outputs" / "run9" / "AI-Generated-68036f43-e5de-476e-bb1c-29661d5b6a63.png", up / "reference-desk.png")
    ref = build.build(up, "a mug", tmp_path / "ref")
    rj = tmp_path / "ref" / "refset.json"
    r0 = ref["references"][0]
    assert r0["product_boxes"] and 0 < r0["product_fill"] < m.DROP_FILL

    doc = m.make(rj, "카페 테이블", tmp_path / "spec", use_vlm=False)
    s, t = doc["spec"], doc["references"][0]["treatment"]
    c = next(c for c in s["conflicts"] if c["kind"] == "reference_has_product")
    assert c["options"][0] == m.REF_HIDE and c["ref_file"] == "reference-desk.png"
    assert t["action"] == "hide" and t["by"] == "default"
    kit = tmp_path / "spec" / "kit"
    orig = np.asarray(Image.open(tmp_path / "ref" / r0["path"]).convert("HSV")).astype(float)
    hidden = np.asarray(Image.open(kit / t["kit"]).convert("HSV")).astype(float)
    x0, y0, x1, y1 = r0["product_boxes"][0]
    assert hidden[y0:y1, x0:x1, 1].mean() < 8 < orig[y0:y1, x0:x1, 1].mean()  # no color left on the product
    assert np.abs(hidden[:40, :40, 2] - orig[:40, :40, 2]).mean() < 3  # far from the box, the photo is unchanged
    assert m.HIDE_KO in s["request_ko"] and m.HIDE_EN in s["prompt_en"]
    assert "제품 가림" in (kit / "README.txt").read_text()

    # the seller says it is their own product in the same color: uploaded as it is
    doc = m.confirm(tmp_path / "spec", {"decisions": [m.REF_KEEP]}, rj)
    t = doc["references"][0]["treatment"]
    assert t["action"] == "keep" and t["by"] == "seller"
    assert (kit / t["kit"]).read_bytes() == (tmp_path / "ref" / r0["path"]).read_bytes()
    assert m.HIDE_KO not in doc["spec"]["request_ko"] and m.HIDE_EN not in doc["spec"]["prompt_en"]

    # a spec written before this change said "무시하기": it now means hide
    doc = m.confirm(tmp_path / "spec", {"decisions": ["구도, 배경, 조명만 참고하고 사진 속 제품은 무시하기"]}, rj)
    assert doc["references"][0]["treatment"]["action"] == "hide" and (kit / doc["references"][0]["treatment"]["kit"]).exists()


def _fake_ocr(pil, box=None):
    x0, y0, x1, y1 = box or (0, 0, pil.width, pil.height)
    return [{"text": "faire une pause", "conf": 0.9, "box": (x0 + 1, y0 + 1, x1 - 1, y1 - 1)}]


def _fake_detect(pil, queries):
    return [{"score": 0.9, "box": (10.0, 10.0, pil.width - 10.0, pil.height - 10.0)}]


def test_strong_run_adds_a_verdict_and_zoomed_vlm_input(fake_vlm, tmp_path):
    build.build(MUG / "uploads", "a mug", tmp_path / "mug", use_vlm=True)
    rep = check_run.run(tmp_path / "mug" / "refset.json", [MUG / "outputs" / "run9"], tmp_path / "chk", draw=False,
                        use_vlm=True, strong=True, ocr=_fake_ocr, detect=_fake_detect)
    assert rep["images"] and all("strong" in i for i in rep["images"])
    assert "strong" in rep["summary"]
    json.dumps(rep)
    verdict_bodies = [b for b in fake_vlm.bodies if b["output_config"]["format"]["schema"]["title"] == "Verdict"]
    texts = [c["text"] for b in verdict_bodies for c in b["messages"][0]["content"] if c["type"] == "text"]
    assert any(t.startswith("Close-up of the generated image") for t in texts)
    assert any("Also check these" in t for t in texts)


def test_strong_off_keeps_the_old_report(fake_vlm, tmp_path):
    build.build(MUG / "uploads", "a mug", tmp_path / "mug", use_vlm=True)
    rep = check_run.run(tmp_path / "mug" / "refset.json", [MUG / "outputs" / "run9"], tmp_path / "chk", draw=False, use_vlm=True)
    assert all("strong" not in i for i in rep["images"]) and "strong" not in rep["summary"]

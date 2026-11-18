"""Fix proposals (check/suggest.py): the tool, cost and feasibility a seller sees before paying.

  .venv/bin/python -m pytest tests/test_suggest.py -q
"""
import json
from pathlib import Path

import cv2

from check import revision, suggest

ROOT = Path(__file__).resolve().parent.parent
MUG = ROOT / "data" / "mug" / "outputs"
EDITS = MUG / "run10b_edit"
PEN = (140, 240, 335, 315)
MUG_BOX = [478, 116, 891, 476]  # OWLv2 "a mug" in the pen cut


def ref(faces=(), missing=(), prints=True):
    """A minimal reference set: one crop per face, the print crop when the product has one."""
    items = [{"crop": f"crops/{f}.png", "face": f} for f in faces] or [{"crop": "crops/a.png", "face": None}]
    return {
        "items": items,
        "missing_faces": [{"face": f} for f in missing],
        "representative": items[0]["crop"],
        "prints": [{"path": "print.png"}] if prints else [],
        "print": {"path": "print.png" if prints else None},
    }


def test_text_fixes_never_touch_the_image():
    """CS 36: "text only" on a text element is the free text edit and the card says the image
    stays. The same words on an image element are pixels: an area edit, 5 credits."""
    c = suggest.propose("텍스트를 더 간략하게, 라운드 도형 대신 심플한 텍스트로", target="text_layer")
    assert c["kind"] == "text" and c["method"] == "text_edit" and c["cost"] == 0
    assert any("이미지" in k and "빼거나" in k for k in c["keep"]) and "이미지는 빼거나" in c["request_ko"]
    assert any(a["method"] == "self_edit" for a in c["alternatives"])
    c = suggest.propose("텍스트만 수정해 주세요", target="image")
    assert c["method"] == "area_text" and c["cost"] == 5 and c["possible"] == "ask"  # no painted place
    assert any("그림 속 픽셀" in w for w in c["warnings"])
    assert suggest.propose("영어로 번역해 주세요", target="text_layer")["method"] == "text_edit"
    assert suggest.propose("영어로 번역해 주세요", target="image", box=[0, 0, 100, 40])["method"] == "area_text"
    assert suggest.propose("섹션 순서를 바꿔 주세요", target="page")["method"] == "layout_edit"


def test_letters_on_the_product_are_its_print():
    """Run 10b's cat close-up lost "Faire une pause". Painted on the mug, the letters are the
    real print: the fix attaches the real print instead of redrawing them as page text."""
    c = suggest.propose("고양이 그림 위에 있던 문구 Faire une pause가 빠졌어요", box=[500, 150, 800, 260], product_box=MUG_BOX, refset=ref())
    assert c["kind"] == "print" and c["method"] == "area_print" and c["attach"][0]["path"] == "print.png"
    assert any("제품의 프린트" in w for w in c["warnings"])
    off = suggest.propose("상단 문구가 잘렸어요", box=[0, 0, 300, 60], product_box=MUG_BOX, refset=ref())
    assert off["kind"] == "text" and off["method"] == "area_text"


def test_no_kind_is_assumed_from_a_complaint():
    """CS 2: "editing stopped working" is not a request to edit. Nothing is run; the seller
    picks a kind or edits the text by hand."""
    c = suggest.propose("갑자기 또 수정이 안된다", target="text_layer")
    assert c["kind"] is None and c["method"] == "choose" and c["possible"] == "ask" and c["cost"] == 0
    assert c.get("request_ko") is None  # nothing to send to the AI
    assert "text" in c["choices"] and "extra" not in c["choices"]
    assert c["alternatives"][0]["method"] == "self_edit"
    assert any("이미지" in k for k in c["keep"])


def test_object_removal_goes_to_the_area_edit_and_is_restored():
    """Our chat test (docs/revision.md 4.6): removing the pen by chat went to whole-image
    generation. The card sends it to the area edit, keeps everything outside the box, and
    plans the restore."""
    c = suggest.propose("왼쪽 책 위에 놓인 펜만 지워 주세요", box=list(PEN), product_box=MUG_BOX)
    assert c["kind"] == "extra" and c["kind_from"] == "guessed" and "지워" in c["kind_words"]
    assert c["method"] == "area_erase" and c["cost"] == 5 and c["possible"] == "yes"
    assert c["keep"][0].startswith("칠한 곳 밖은 원본") and c["after"]["restore"]
    assert any("참조 기반 이미지 생성" in w for w in c["warnings"])
    assert not any("CS 38" in w for w in c["warnings"])  # the pen is not on the mug
    on = suggest.propose("주름 제거해 주세요", box=[500, 200, 700, 400], product_box=MUG_BOX)
    assert on["kind"] == "extra" and any("CS 38" in w for w in on["warnings"])
    assert "칠한 곳" in c["card_ko"] and "5크레딧" in c["card_ko"]


def test_the_spot_left_behind_is_not_a_position_fix():
    """Our CS area edits all read "..., 그 자리는 ...로 채우기". That "자리" is where the removed
    thing was, so the card has no second fix (seen in the demo on the CS 26 tote)."""
    for text in ("뒤에 앉아 있는 사람 제거, 그 자리는 커튼과 벽으로 채우기", "향수병 제거, 그 자리는 흰 바닥으로 채우기", "펜을 지우고 빈자리는 책 표지로 메워 주세요"):
        c = suggest.propose(text, box=[0, 0, 268, 464])
        assert c["kind"] == "extra" and c["method"] == "area_erase" and c["possible"] == "yes"
        assert not any("다른 고칠 것" in w for w in c["warnings"]), text
    assert suggest.guess_kinds("로고 자리가 달라요")[0] == ("position", ["자리"])
    assert [k for k, _ in suggest.guess_kinds("그 자리에 있던 로고를 오른쪽으로 옮겨 주세요")][0] == "position"


def test_position_needs_the_right_place():
    """CS 28, 31: a direction in words is ambiguous; the card asks for a click."""
    c = suggest.propose("로고 위치가 반대쪽이에요", box=[100, 100, 300, 300])
    assert c["kind"] == "position" and c["method"] == "move" and c["possible"] == "ask"
    assert not any("다른 고칠 것" in w for w in c["warnings"])  # "로고" is the object, not a second fix
    c = suggest.propose("로고 위치가 반대쪽이에요", box=[100, 100, 300, 300], point=[600, 200])
    assert c["possible"] == "yes" and "정답 자리 (600, 200)" in c["card_ko"]


def test_a_side_no_photo_shows_is_refused_before_paying():
    """CS 21: the seller paid 5 credits before hearing the back cannot be drawn. Here the card
    says so at 0 credits; with a photo of that side, it is attached."""
    c = suggest.propose("뒷면이 보이게 바꿔 주세요", refset=ref(["front", "handle"], missing=["back"]))
    assert c["kind"] == "face" and c["method"] == "shoot" and c["possible"] == "no" and c["cost"] == 0
    c = suggest.propose("뒷면이 보이게 바꿔 주세요", refset=ref(["front", "back"]))
    assert c["method"] == "regen_face" and c["possible"] == "yes" and c["cost"] == 5
    assert [a["path"] for a in c["attach"]] == ["crops/back.png"]
    c = suggest.propose("슬리퍼 앞뒤를 바꿔 주세요", refset=ref())  # face names unknown (VLM off)
    assert c["method"] == "regen_face" and c["possible"] == "ask"


def test_count_and_video_change_the_input():
    """CS 32: four runs of the same input, four times the same two knives. A video fix starts
    from a fixed start image; after two failures the real photo comes first."""
    c = suggest.propose("칼 2자루를 1자루로 바꿔 주세요", target="video", refset=ref(), tried=1)
    assert c["kind"] == "count" and c["method"] == "video_regen" and c["cost"] == 35
    assert c["first"]["method"] == "regen_input" and "정확히 1개만" in c["first"]["request_ko"]
    assert any("한 번 실패" in w for w in c["warnings"]) and any("CS 34, 38" in w for w in c["warnings"])
    c = suggest.propose("칼 2자루를 1자루로 바꿔 주세요", refset=ref(), tried=2)
    assert c["alternatives"][0]["method"] == "replace_real"
    assert suggest.propose("칼 2자루를 1자루로 바꿔 주세요")["possible"] == "ask"  # nothing new to put in


def test_flip_keeps_the_print_unmirrored():
    assert suggest.propose("좌우 반전해 주세요", refset=ref())["method"] == "regen_camera"
    plain = suggest.propose("좌우 반전해 주세요", refset=ref(prints=False))
    assert plain["method"] == "flip_tool" and plain["cost"] == 0


def test_cards_from_a_check_report():
    """different -> a card (print words -> the print fix); unverified -> look at it; match ->
    nothing. On the real Run 10b report (first pass only) every item is a look item."""
    report = {
        "images": [
            {"file": "a.png", "inliers": 60, "final": "match"},
            {"file": "b.png", "inliers": 4, "final": "unverified"},
            {"file": "c.png", "inliers": 3, "final": "different", "vlm": {"verdict": "different", "differences": ["프린트의 고양이가 강아지로 바뀜"]}},
            {"file": "d.png", "inliers": 0, "final": "different", "vlm": {"verdict": "different", "differences": ["손잡이 모양이 다름"]}},
        ],
        "videos": [
            {"file": "v.mp4", "final": "different", "different_at": [1.5], "vlm": [{"t": 1.5, "verdict": "different", "differences": ["없던 로고가 생김"]}]},
            {"file": "w.mp4", "final": "unverified", "dip_at": [2.0, 2.5]},
        ],
    }
    items = {i["file"]: i for i in suggest.from_check(report, ref())}
    assert set(items) == {"b.png", "c.png", "d.png", "v.mp4", "w.mp4"}
    assert items["c.png"]["card"]["method"] == "area_print" and items["c.png"]["card"]["attach"][0]["path"] == "print.png"
    assert items["d.png"]["card"]["method"] == "replace_real" and items["d.png"]["card"]["cost"] == 0
    assert items["v.mp4"]["card"]["method"] == "video_regen" and "1.5초" in items["v.mp4"]["card"]["card_ko"]
    assert "2초" in items["w.mp4"]["look_ko"] and "구분하지 못해요" in items["b.png"]["look_ko"]
    real = json.loads((ROOT / "out" / "mug" / "check_run10b" / "report.json").read_text()) if (ROOT / "out" / "mug" / "check_run10b" / "report.json").exists() else None
    if real:
        got = suggest.from_check(real)
        assert len(got) == real["summary"]["unverified"] and all("look_ko" in i for i in got)


def test_after_fix_on_real_creazy_edits():
    """The next step after a fix: the cookie area edit changed the mug outside the box ->
    restore; restored -> show; the Run 8 fix recomposed the picture -> nothing to paste back."""
    e = {x["name"]: x for x in json.loads((EDITS / "edits.json").read_text())["edits"]}["edit2_cookies"]
    before, after = cv2.imread(str(EDITS / e["before"])), cv2.imread(str(EDITS / e["after"]))
    rev = revision.compare(before, after, [e["box"]], product_boxes=[[483, 230, 907, 591]])
    rev.pop("_vis")
    nxt = suggest.after_fix({"before": "b.png", "after": "a.png", **rev})
    assert nxt["next"] == "restore" and "--box 22,389,375,568" in nxt["command"]
    from check import restore

    out, taken, _ = restore.restore(before, after, [e["box"]], product_boxes=[[483, 230, 907, 591]])
    again = revision.compare(before, out, mask=taken.astype("uint8") * 255, product_boxes=[[483, 230, 907, 591]])
    again.pop("_vis")
    assert suggest.after_fix(again)["next"] == "show"
    r8 = revision.compare(cv2.imread(str(MUG / "run8" / "c03_4e01d295.png")), cv2.imread(str(MUG / "run8" / "fix_3e87861f.png")), [[230, 250, 700, 830]])
    r8.pop("_vis", None)
    assert suggest.after_fix(r8)["next"] == "redrawn"
    assert suggest.after_fix(r8, method="regen_input")["next"] == "check_real"


def test_after_fix_reads_the_print_score():
    """The chat pen edit: nothing outside changed, the print score fell 26 percent (50 -> 37).
    At the old 35 percent note it passed unflagged; at 20 it is restored."""
    base = {"before": "b.png", "after": "a.png", "verdict": "pass", "outside": {"level": "same", "areas": []}, "inside": {"verdict": "changed"}, "region": {"boxes": [list(PEN)]}}
    nxt = suggest.after_fix({**base, "print": {"before": 50, "after": 37, "verdict": "kept"}})
    assert nxt["next"] == "restore" and "하락" in nxt["why"][0]
    assert suggest.after_fix({**base, "print": {"before": 50, "after": 47, "verdict": "kept"}})["next"] == "show"  # resampling
    assert suggest.after_fix({**base, "inside": {"verdict": "unchanged"}})["next"] == "retry"
    auto = {"before": "b.png", "after": "a.png", "verdict": "changes", "outside": {"level": "minor", "areas": [{}]}}
    assert "--auto" in suggest.after_fix(auto)["command"]

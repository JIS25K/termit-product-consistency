"""Demo steps 8 and 9 (demo/app.py): the confirmation card, then the fixed image checked and
the original pasted back. The route functions are called directly (no HTTP client needed).

  .venv/bin/python -m pytest tests/test_demo_fix.py -q
"""
import json
from pathlib import Path

import cv2
import pytest
from fastapi import HTTPException

from demo import app as demo

ROOT = Path(__file__).resolve().parent.parent
EDITS = ROOT / "data" / "mug" / "outputs" / "run10b_edit"
SID = "0123456789"


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A demo session with a minimal reference set: the mug's print cut from the cookie cut.
    No OWLv2 query, so no product box is searched for."""
    monkeypatch.setattr(demo, "OUT", tmp_path)
    ref = tmp_path / SID / "refset"
    ref.mkdir(parents=True)
    before = cv2.imread(str(EDITS / "../run10b/AI-Generated-3271d5a2-bf88-4931-9461-8e2473ee318b.png"))
    cv2.imwrite(str(ref / "print.png"), before[300:560, 560:800])
    (ref / "refset.json").write_text(json.dumps({
        "items": [{"crop": "print.png", "face": None}], "representative": "print.png",
        "prints": [{"path": "print.png"}], "print": {"path": "print.png"}, "missing_faces": [],
    }))
    (tmp_path / SID / "meta.json").write_text(json.dumps({"example": "mug"}))
    return tmp_path / SID


def card(**kw):
    args = dict(sid=SID, text="", kind="", target="image", box="", point="", tried=0, example_fix="", output="", file=None)
    return demo.suggest_fix(**{**args, **kw})


def test_card_then_the_creazy_area_edit_restored(session):
    j = card(text="접시 위 쿠키만 지우고 빈 접시로 두기", box="22,389,375,568", example_fix="edit2_cookies")
    c = j["card"]
    assert c["method"] == "area_erase" and c["possible"] == "yes" and c["cost"] == 5 and j["how"] == "area"
    assert (session / f"fix_{j['fix']}" / "before.png").exists()
    r = demo.check_fix(sid=SID, fix=j["fix"], how="area", example_after=True, file=None)
    assert r["restore"]["restored"]["outside_changed"] == 0
    assert r["as_is"]["verdict"] == "review" and r["next_as_is"]["next"] == "restore"
    assert r["checked"]["verdict"] == "pass" and r["next_checked"]["next"] == "show"
    assert r["checked"]["print"]["after"] >= r["as_is"]["print"]["after"]
    assert r["images"]["restored"].endswith("check/restored.png")


def test_card_for_a_chat_edit_found_without_a_painted_box(session):
    j = card(text="왼쪽 책 위에 놓인 펜만 지워 주세요", box="140,240,335,315", example_fix="chat2_pen")
    assert j["how"] == "chat" and j["card"]["method"] == "area_erase"
    r = demo.check_fix(sid=SID, fix=j["fix"], how="chat", example_after=True, file=None)
    assert r["restore"]["auto"]["boxes"] and r["restore"]["restored"]["outside_changed"] == 0
    assert r["next_checked"]["next"] == "show"


def test_card_needs_an_image_and_a_card_first(session):
    with pytest.raises(HTTPException):
        card(text="펜만 지워 주세요")  # no image picked
    with pytest.raises(HTTPException):
        card(text="펜만 지워 주세요", example_fix="edit2_cookies", box="10,10,5,5")
    with pytest.raises(HTTPException):
        demo.check_fix(sid=SID, fix="abcdef", how="area", example_after=True, file=None)

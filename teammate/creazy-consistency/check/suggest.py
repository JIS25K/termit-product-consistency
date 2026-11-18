"""Fix proposals: what to do about a wrong output, before any credits are spent
(docs/revision.md, section 4 steps 3 and 4).

Termit's chat picks the tool for a fix by itself and does not wait for the seller: a text fix
removed the image (CS 36), "I can't edit this" was taken as "edit it for me" (CS 2), removing
an object in the picture went to whole-image generation instead of the area edit (our test,
docs/revision.md 4.6), and a side no photo shows was charged before the seller heard it cannot
be drawn (CS 21). Here the tool comes from a fixed table, and every proposal is a confirmation
card the seller reads before paying:

  where     the place to fix (painted box, the right place for a position fix, video times)
  kind      what kind of problem, picked by the seller or guessed from the request text
  change    what is wrong and what it should be
  keep      what must not change: the production spec's keep conditions, plus the rest of
            the image (area edits are restored outside the box) or the image itself (text and
            layout fixes)
  attach    real photos to give the editor instead of the generated image
  method    the Termit tool, its menu path, the request text to paste, the cost in credits
  possible  yes / ask (something is missing: a painted box, the right place, a photo) / no
  after     how the fix is checked (check/revision.py, check/restore.py) and what happens when
            it fails

Kind -> method (target: text_layer = page text, page = page element, image, video)
  text, translate  text_layer: text edit, free. image: area edit of the letters only, 5,
                   with a note that letters in the picture are pixels (CS 36, 2)
  layout           page: order, margin and spacing values instead of dragging, free
  not_real         replace with the real photo, free (CS 19, 39)
  print            area edit with the real print crops attached, 5 (CS 10, 24)
  extra            area edit to erase, 5, then the original pasted back outside (CS 34)
  position         erase the wrong place and draw at the right one (the seller clicks it), 5
                   to 10, or a real-photo composite (CS 28, 31)
  face             the real photo of that side attached, 5; no photo of that side: not
                   possible, 0 credits, ask for the photo (CS 21)
  count            change the input (start image, references) and regenerate, 5; the same
                   input is not run again (CS 32)
  flip             regenerate from the other camera side, 5; the flip tool mirrors the print
  any, on video    fix the start image first, then regenerate the clip, 30 (CS 34, 38, 32)
Two failures with the same method move the proposal to the real photo (CS 19, 39, 29).

Entry points
  propose(...)        one request from the seller
  from_check(report)  every result check.run judged different (unverified ones become
                      "look at it" items: the first pass cannot tell a hidden print side
                      from a wrong print)
  after_fix(rev)      the next step once a fix was checked with check/revision.py

Rules only, no API key. Credits are those seen in Creazy (docs/revision.md section 3); the
video cost comes from the CS records (38: 30 per edit, 32: GIF 30).

Usage
  python -m check.suggest --text "왼쪽 책 위의 펜만 지워 주세요" --box 140,240,335,315 --refset out/mug/refset.json --out out/suggest/pen
  python -m check.suggest --text "텍스트만 간단하게 바꿔 주세요" --target text_layer --out out/suggest/text
  python -m check.suggest --report out/mug/check_run10b/report.json --refset out/mug/refset.json --out out/suggest/run10b
  python -m check.suggest --revision out/restore/edit2_cookies/as_is/revision.json --out out/suggest/after_cookies
"""
import argparse
import json
import re
from pathlib import Path

from .revision import ON_PRODUCT, PRINT_DROP, parse_box

AREA_TOOL = "이미지 선택 → AI 생성 → 이미지 생성 → 특정 부분만 생성하기"

TARGET_KO = {"text_layer": "페이지 글자", "page": "페이지 요소", "image": "그림", "video": "영상, GIF"}

KIND_KO = {
    "text": "문구, 서식",
    "translate": "번역",
    "layout": "위치, 순서(페이지 요소)",
    "not_real": "실물과 다름",
    "print": "글자, 프린트가 틀림(그림 속)",
    "extra": "없던 게 생김",
    "position": "위치가 틀림(그림 속)",
    "face": "보이는 면 바꾸기",
    "count": "개수",
    "flip": "좌우 반전",
}

# Words that point to a kind, checked in this order: the first kind found leads, the rest are
# listed as other things in the same request.
WORDS = [
    ("translate", ["번역", "영어로", "영문으로", "일본어", "중국어", "한국어로", "한글로"]),
    ("face", ["앞뒤", "뒷면", "옆면", "윗면", "밑면", "바닥면", "다른 면", "반대쪽 면", "뒤쪽이 보이", "옆이 보이"]),
    ("flip", ["좌우 반전", "반전", "거울", "뒤집"]),
    ("count", ["개수"]),
    ("extra", ["지워", "지우", "없애", "제거", "빼 주", "빼줘", "빼 줘", "생겼", "없던", "남아", "남았"]),
    ("position", ["위치", "방향", "자리", "옮겨", "옮기", "반대쪽", "반대로", "왼쪽으로", "오른쪽으로"]),
    ("print", ["프린트", "로고", "라벨", "일러스트", "무늬", "패턴", "그림이 달", "글씨가 깨", "글자가 깨", "글자가 이상", "성분"]),
    ("not_real", ["실물과 달", "실물이랑 달", "실제와 달", "실제랑 달", "실제 제품", "모양이 달", "색이 달", "색상이 달", "다른 제품", "실물 사진으로"]),
    ("layout", ["순서", "여백", "간격", "정렬", "배치", "섹션", "크롭"]),
    ("text", ["문구", "텍스트", "글자", "글씨", "오타", "맞춤법", "글꼴", "폰트", "제목", "굵게", "줄바꿈"]),
]
COUNT_RE = re.compile(r"(\d+|한|두|세|네|하나|둘|셋|넷)\s*(개|자루|켤레|병|벌|쌍)")
# "자리" in a removal is the spot left behind, not a position fix: "그 자리는 커튼과 벽으로
# 채우기", "빈자리를 바닥으로 메워". Taken out before the words are matched.
FILL_RE = re.compile(r"(?:(?<=[그빈] )|(?<=[그빈]))자리|자리(?=[^,.]{0,20}?(?:채우|채워|메우|메워))")
NUM_KO = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4}
PAGE_KINDS = ("text", "translate", "layout")

# method -> what the card shows; cost in credits
METHODS = {
    "text_edit": ("텍스트 편집(페이지 글자만 바꿈)", "글자 선택 → 왼쪽 패널", 0),
    "self_edit": ("직접 고치기", "글자 선택 → 왼쪽 패널", 0),
    "layout_edit": ("순서, 여백, 간격 값 바꾸기(드래그 대신)", "섹션 머리 버튼, 또는 요소 선택 → 위치와 간격 값", 0),
    "replace_real": ("실물 사진으로 교체", "이미지 선택 → 이미지 교체", 0),
    "composite_real": ("실물 합성(배경을 지운 실물 사진을 얹기)", "실물 사진 추가 → 이미지 편집 → 배경 제거", 0),
    "area_erase": ("AI 영역 편집으로 지우기", AREA_TOOL, 5),
    "area_print": ("AI 영역 편집 + 실물 프린트 사진 첨부", AREA_TOOL, 5),
    "area_text": ("AI 영역 편집으로 그림 속 글자만 다시 그리기", AREA_TOOL, 5),
    "move": ("틀린 자리 지우기 + 정답 자리에 그리기", AREA_TOOL, 5),
    "regen_face": ("그 면의 실물 사진을 붙여 다시 생성", "채팅 + 사진 첨부(+)", 5),
    "shoot": ("지금은 불가: 그 면의 사진을 찍어 올리기", "상품 이미지 추가", 0),
    "regen_input": ("입력(시작 이미지, 참고 사진)을 실물로 바꾼 뒤 다시 생성", "채팅 + 사진 첨부(+)", 5),
    "regen_camera": ("카메라 방향을 바꿔 다시 생성(반전 도구는 쓰지 않음)", "채팅", 5),
    "flip_tool": ("반전 도구", "이미지 편집 패널 → 반전", 0),
    "video_regen": ("시작 이미지를 먼저 고친 뒤 영상 다시 생성", "영상 편집", 30),
    "choose": ("종류 고르기", "", 0),
}
AREA_METHODS = ("area_erase", "area_print", "area_text", "move")
NO_CHARGE = "\"이대로 진행\"을 누를 때만 차감하고, 수정 검수에서 실패하면 차감하지 않아요(테르밋 정책 제안)."
SELF_EDIT = "AI가 바꾸지 않아요. 글자를 선택하면 왼쪽 패널에서 바로 고칠 수 있어요(CS 2)."
CHAT_ROUTE = "채팅으로 요청하면 그림 전체를 새로 만드는 \"참조 기반 이미지 생성\"으로 처리돼요(우리 시험). 영역 편집 메뉴로 하세요."


# ---------------------------------------------------------------- reading the request


def guess_kinds(text, target="image"):
    """Kinds the request text points to, leading one first, each with the words that matched."""
    text = text or ""
    scan = FILL_RE.sub("", text)
    found = []
    for kind, words in WORDS:
        hit = [w for w in words if w in scan]
        if kind == "count":
            hit += ["".join(m) for m in COUNT_RE.findall(text)]
        if hit:
            found.append((kind, hit))
    if target in ("text_layer", "page"):
        # A text or page element was picked: "지워" there deletes letters, not an object.
        # Nothing is assumed when no word matched: "수정이 안 된다" is not "fix it for me" (CS 2).
        found = [f for f in found if f[0] in PAGE_KINDS]
    else:
        found = [f for f in found if f[0] != "layout"]
    return found


def wanted_count(text):
    """The number the seller asks for: the last count in the text ("2자루를 1자루로" -> 1)."""
    m = COUNT_RE.findall(text or "")
    if not m:
        return None
    n = m[-1][0]
    return int(n) if n.isdigit() else NUM_KO.get(n)


def wanted_faces(text):
    from spec.make import faces_in  # local import: spec.make pulls in PIL and the VLM module

    faces = [f for f in faces_in(text or "") if f not in ("front", "detail")]
    if not faces and any(w in (text or "") for w in ("앞뒤", "다른 면", "반대쪽 면")):
        faces = ["back"]
    return faces


# ---------------------------------------------------------------- reference set


def load_refset(refset):
    """(refset dict, folder) from a path, or (None, None)."""
    if not refset:
        return None, None
    if isinstance(refset, dict):
        return refset, None
    p = Path(refset)
    return json.loads(p.read_text()), p.parent


def _path(base, rel):
    return str(base / rel) if base else rel


def real_photos(ref, base, kind, faces=()):
    """Real photos to attach for this kind, with why."""
    if not ref:
        return []
    face_ko = _faces_ko()
    main = ref.get("main") or ref.get("representative")
    prints = ref.get("prints") or ([{"path": ref["print"]["path"]}] if (ref.get("print") or {}).get("path") else [])
    out = []
    if kind == "print":
        out += [{"path": _path(base, p["path"]), "why": "실물 프린트(이 모양과 글자 그대로 그리게 함)"} for p in prints]
    if kind == "face":
        for it in ref.get("items", []):
            if it.get("face") in faces:
                out.append({"path": _path(base, it["crop"]), "why": f"실물 {face_ko.get(it['face'], it['face'])} 사진"})
        return out
    if main and kind in ("print", "not_real", "position", "count", "flip"):
        out.append({"path": _path(base, main), "why": "실물 대표 사진" + ("(생성본 대신 참고 이미지로)" if kind in ("print", "position") else "")})
    return out


def _faces_ko():
    from refset.vlm import FACE_KO

    return FACE_KO


def keep_lines(ref, spec_doc, kind):
    """The product's keep conditions: the confirmed spec, or the rule spec of spec/make.py."""
    if spec_doc:
        keep = list(spec_doc["spec"]["keep"])
    elif ref:
        from spec.make import rule_spec

        keep = rule_spec(ref, "", []).keep
    else:
        keep = ["제품의 형태, 비율, 색, 부품, 프린트는 실물 사진 그대로"]
    if kind == "count":
        keep = [k for k in keep if "1개만" not in k]
    return keep


# ---------------------------------------------------------------- one request


def _box_ko(b):
    return f"({b[0]}, {b[1]})-({b[2]}, {b[3]})"


def _overlap(box, other):
    x0, y0 = max(box[0], other[0]), max(box[1], other[1])
    x1, y1 = min(box[2], other[2]), min(box[3], other[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _alt(method, when):
    name, tool, cost = METHODS[method]
    return {"method": method, "method_ko": name, "tool": tool, "cost": cost, "when": when}


def propose(text="", kind=None, target="image", box=None, point=None, refset=None, spec=None, tried=0, product_box=None, times=None):
    """A confirmation card for one fix.

    text: the seller's request (Korean). kind: picked by the seller; guessed from text if None.
    target: text_layer, page, image or video (the element picked in the editor).
    box: the painted place, x0,y0,x1,y1 in image pixels. point: the right place (position fix).
    refset: refset.json path or dict. spec: confirmed spec.json path or dict.
    tried: how many times this fix already failed with the same method and input.
    product_box: the product in the image; a box on it gets the pattern warning (CS 38).
    times: video times in seconds where the problem shows (check.run's different_at)."""
    ref, base = load_refset(refset)
    spec_doc = json.loads(Path(spec).read_text()) if isinstance(spec, (str, Path)) and spec else spec
    guessed = guess_kinds(text, target)
    if kind:
        kind_from, others = "selected", [k for k, _ in guessed if k != kind]
        words = []
    elif guessed:
        kind, words = guessed[0]
        # print and text words are mostly the object of another fix ("로고 위치", "라벨 지워")
        kind_from, others = "guessed", [k for k, _ in guessed[1:] if k not in ("print", "text")]
    else:
        kind_from, others, words = None, [], []
    on_product = bool(box and product_box and _overlap(box, product_box) >= ON_PRODUCT)
    # Letters painted on the product are its print: fix them from the real print, not as text.
    as_print = kind_from == "guessed" and kind == "text" and target == "image" and on_product
    if as_print:
        kind = "print"
    card = {
        "text": text,
        "target": target,
        "target_ko": TARGET_KO[target],
        "kind": kind,
        "kind_ko": KIND_KO.get(kind),
        "kind_from": kind_from,
        "kind_words": words,
        "where": {"box": box, "point": point, "times": times},
        "warnings": [],
        "alternatives": [],
    }
    if as_print:
        card["warnings"].append("칠한 곳이 제품 위라 페이지 글자가 아니라 제품의 프린트로 보고, 실물 프린트 사진을 붙여요.")
    if others:
        card["warnings"].append("요청에 다른 고칠 것도 있어요(" + ", ".join(KIND_KO[k] for k in dict.fromkeys(others)) + "). 한 번에 하나씩 고쳐야 결과를 확인하기 쉬워요.")
    if kind is None:
        card.update(method="choose", possible="ask", possible_ko="무엇을 고칠지 알 수 없어요. 고칠 곳을 지정하고 종류를 골라 주세요.", choices=[k for k in KIND_KO if (k in PAGE_KINDS) == (target in ("text_layer", "page"))])
        if target == "text_layer":
            card["alternatives"].append(_alt("self_edit", SELF_EDIT))
        return _finish(card, ref, spec_doc)

    plain = bool(ref) and not (ref.get("prints") or (ref.get("print") or {}).get("path"))
    faces = wanted_faces(text) if kind == "face" else []
    card["attach"] = real_photos(ref, base, kind, faces)
    W = card["warnings"]
    A = card["alternatives"]
    possible, why = "yes", ""

    if target == "video":
        method = "video_regen"
        W.append("영상은 칠한 곳만 고칠 수 없어 전체가 다시 만들어져요. 문제가 없던 무늬, 로고, 길이도 바뀔 수 있어요(CS 34, 38).")
        first = propose(text, kind, "image", box, point, refset, spec, tried, product_box)
        card["first"] = {k: first[k] for k in ("kind", "method", "method_ko", "cost", "possible", "possible_ko", "request_ko")}
        if first["possible"] == "no":
            possible, why = "no", first["possible_ko"]
        if tried == 1:
            W.append("같은 입력으로 한 번 실패했어요. 시작 이미지를 고치기 전에는 다시 만들지 않아요(CS 32).")
    elif kind in ("text", "translate"):
        if target == "text_layer":
            method = "text_edit"
            A.append(_alt("self_edit", SELF_EDIT))
        else:
            method = "area_text"
            W.append("이 글자는 그림 속 픽셀이라 무료 텍스트 편집으로는 고칠 수 없어요. 그 부분을 다시 그려야 해요(CS 36, 2).")
            A.append(_alt("area_erase", "그림 속 글자를 지우고 그 자리에 페이지 글자를 새로 얹기. 얹은 글자는 이후 무료로 고칠 수 있어요."))
    elif kind == "layout":
        method = "layout_edit"
        W.append("기본 배치 모드에서 드래그하면 순서가 바뀌며 서식이 깨질 수 있어요. 값으로 바꾸세요.")
    elif kind == "not_real":
        method = "replace_real"
        W.append("그림의 장면은 사라지고 실물 사진이 들어가요.")
        A.append(_alt("composite_real", "장면을 살리고 싶으면: 배경을 지운 실물 사진을 장면 위에 얹기."))
        if not plain:
            A.append(_alt("area_print", "틀린 곳만 고치고 싶으면: 칠한 곳을 실물 프린트 사진을 붙여 다시 그리기."))
    elif kind == "print":
        method = "area_print"
        W.append("제품 위를 다시 그리므로 칠한 곳 안의 무늬는 원본으로 덮어 지킬 수 없어요. 수정 뒤 실물 프린트 점수로 확인해요.")
        A.append(_alt("replace_real", "두 번 고쳐도 틀리면 실물 사진으로 교체(CS 19, 39)."))
    elif kind == "extra":
        method = "area_erase"
        W.append(CHAT_ROUTE)
        if on_product:
            W.append("칠한 곳이 제품 위예요. 칠한 곳 안의 실물 무늬도 다시 그려져 사라질 수 있어요(CS 38).")
    elif kind == "position":
        method = "move"
        A.append(_alt("composite_real", "두 번에 안 되면: 배경을 지운 실물 사진을 정답 자리에 얹기."))
        if not point:
            possible, why = "ask", "정답 위치를 화면에서 눌러 주세요. 글로 쓴 방향은 보는 사람 기준인지 제품 기준인지 모호해요(CS 28, 31)."
    elif kind == "face":
        method, possible, why = _face(ref, faces)
    elif kind == "count":
        method = "regen_input"
        if tried == 1:
            W.append("같은 입력으로 한 번 실패했어요. 시작 이미지나 참고 사진을 바꾸기 전에는 다시 만들지 않아요(CS 32).")
        if not card["attach"]:
            possible, why = "ask", "바꿔 넣을 실물 사진이 없어요. 같은 입력으로 다시 만들면 같은 결과가 나와요(CS 32). 실물 사진을 올려 주세요."
    elif kind == "flip":
        if plain:
            method = "flip_tool"
            W.append("프린트가 없는 제품이라 반전 도구를 써도 돼요. 작은 로고나 글자가 있으면 거울처럼 뒤집혀요.")
        else:
            method = "regen_camera"
            W.append("반전 도구는 프린트와 글자까지 거울처럼 뒤집어요. 카메라 방향을 바꿔 다시 만들어요.")

    if method in AREA_METHODS and not box:
        possible, why = "ask", "고칠 곳을 칠해 주세요. 칠한 곳만 다시 그리고, 나머지는 원본으로 덮어요."
    if tried >= 2 and method not in ("replace_real", "composite_real", "shoot", "text_edit", "layout_edit"):
        W.append(f"같은 방법으로 {tried}번 실패했어요. 같은 입력으로는 다시 만들지 않고, 실물 사진 교체나 합성을 먼저 권해요(CS 32, 19, 39, 29).")
        A.insert(0, _alt("replace_real", "같은 방법으로 두 번 넘게 실패했을 때"))
    name, tool, cost = METHODS[method]
    card.update(method=method, method_ko=name, tool=tool, cost=cost + (card.get("first", {}).get("cost") or 0))
    if method == "move":
        card["cost_ko"] = "5~10크레딧(한 번에 안 되면 지우기와 그리기를 나눠 2번)"
    elif method == "video_regen":
        card["cost_ko"] = f"{card['cost']}크레딧(시작 이미지 {card['first']['cost']} + 영상 30, 영상 비용은 CS 38, 32 기록 기준)"
    card["possible"] = possible
    card["possible_ko"] = why or ("가능해요." if possible == "yes" else "")
    card["request_ko"] = request_text(card, faces)
    return _finish(card, ref, spec_doc)


def _face(ref, faces):
    face_ko = _faces_ko()
    if not faces:
        return "regen_face", "ask", "어느 면이 보이게 할지 골라 주세요."
    names = ", ".join(face_ko.get(f, f) for f in faces)
    known = {i.get("face") for i in (ref or {}).get("items", []) if i.get("face")}
    missing = {m["face"] for m in (ref or {}).get("missing_faces") or []}
    if any(f in missing for f in faces) or (known and not all(f in known for f in faces)):
        return "shoot", "no", f"지금 실물 사진에는 {names}이 없어 그릴 수 없어요. 그대로 만들면 AI가 지어내요. {names} 사진을 찍어 올려 주세요(CS 21: 차감 뒤에 불가를 알림)."
    if known:
        return "regen_face", "yes", ""
    return "regen_face", "ask", f"면 이름이 없어(VLM 꺼짐) 실물 사진에 {names}이 있는지 확인하지 못했어요. 있으면 그 사진을 붙여 주세요. 없으면 찍어 올려야 해요."


def request_text(card, faces=()):
    """Korean text to paste into the area edit prompt or the chat. None for manual tools."""
    t = (card["text"] or "").strip().rstrip(".")
    lead = f"{t}. " if t else ""
    m = card["method"]
    if m == "text_edit":
        return f"{lead}선택한 글자만 고쳐 주세요. 이미지는 빼거나 바꾸거나 새로 만들지 마세요. 다른 글자와 요소는 그대로 두세요."
    if m == "layout_edit":
        return f"{lead}드래그하지 말고 순서와 여백, 간격 값만 바꿔 주세요. 글자와 이미지는 그대로 두세요."
    if m == "area_text":
        return f"{lead}칠한 곳의 글자만 고쳐 주세요. 글자 밖의 그림과 제품은 바꾸지 마세요."
    if m == "area_erase":
        return f"{lead}칠한 곳 안만 지우고, 그 자리를 주변 배경으로 자연스럽게 채워 주세요. 칠한 곳 밖과 제품은 바꾸지 마세요."
    if m == "area_print":
        return f"{lead}칠한 곳의 프린트를 첨부한 실물 프린트 사진과 똑같이 그려 주세요. 그림과 글자의 모양, 크기, 위치를 바꾸거나 새로 지어내지 마세요. 칠한 곳 밖은 바꾸지 마세요."
    if m == "move":
        return f"{lead}1) 칠한 자리의 것을 지우고 주변 배경으로 채워 주세요. 2) 표시한 정답 자리에 첨부한 실물 사진과 같은 모양으로 그려 주세요. 다른 곳은 바꾸지 마세요."
    if m == "regen_face":
        names = ", ".join(_faces_ko().get(f, f) for f in faces) or "그 면"
        return f"{lead}첨부한 실물 {names} 사진과 같은 면이 보이게 다시 만들어 주세요. 장면과 조명은 지금 결과처럼 두고, 제품은 첨부 사진 그대로 그려 주세요."
    if m == "regen_input":
        n = wanted_count(card["text"])
        return f"{lead}시작 이미지를 첨부한 실물 사진으로 바꿔서 다시 만들어 주세요." + (f" 제품은 정확히 {n}개만 나오게 해 주세요." if n else "")
    if m == "regen_camera":
        return f"{lead}반전 도구를 쓰지 말고, 카메라가 제품의 반대쪽에서 보도록 다시 만들어 주세요. 프린트와 글자는 실물 사진 그대로 두고 뒤집지 마세요."
    if m == "video_regen":
        return f"{lead}먼저 시작 이미지를 고친 뒤, 고친 이미지로 영상을 다시 만들어 주세요. 제품은 실물 사진 그대로 유지해 주세요."
    return None


def _finish(card, ref, spec_doc):
    m = card["method"]
    keep = keep_lines(ref, spec_doc, card["kind"])
    if m in ("text_edit", "layout_edit") or (m == "choose" and card["target"] in ("text_layer", "page")):
        keep = ["이 섹션의 이미지(빼거나 바꾸거나 새로 만들지 않음)", "고르지 않은 다른 글자와 요소"]
    elif m in AREA_METHODS:
        keep = ["칠한 곳 밖은 원본 그대로(수정 뒤 원본 픽셀로 덮어요)"] + keep
    elif m in ("replace_real", "composite_real"):
        keep = ["페이지 글자와 배치"]
    card["keep"] = keep
    card.setdefault("cost", 0)
    card.setdefault("attach", [])
    card["charge_ko"] = NO_CHARGE if card["cost"] else "무료"
    card["after"] = after_plan(card)
    card["card_ko"] = render(card)
    return card


def after_plan(card):
    """How the fix is checked once it comes back."""
    m = card["method"]
    if m in AREA_METHODS:
        return {
            "restore": True,
            "check_ko": "칠한 곳 밖을 원본 픽셀로 덮은 뒤 수정 검수: 칠한 곳 밖 변화 0, 칠한 곳 안 바뀜, 실물 프린트 점수 유지.",
            "fail_ko": "실패하면 차감 없이 다시. 같은 입력을 반복하지 않고 실물 사진을 붙이거나 방법을 바꿔요. 두 번 실패하면 실물 사진 교체(무료).",
        }
    if m in ("regen_face", "regen_input", "regen_camera", "video_regen"):
        return {
            "restore": False,
            "check_ko": "그림 전체가 다시 만들어지므로 덮을 원본이 없어요. 결과를 실물과 다시 대조해요" + ("(영상은 프레임마다)." if m == "video_regen" else "."),
            "fail_ko": "실패하면 차감 없이 다시 하되 같은 입력은 반복하지 않아요. 두 번 실패하면 실물 사진 교체나 합성.",
        }
    if m in ("text_edit", "layout_edit"):
        return {"restore": False, "check_ko": "섹션의 이미지가 수정 전과 같은지 비교해요.", "fail_ko": "이미지가 바뀌었으면 되돌리고 글자만 다시 고쳐요."}
    return {"restore": False, "check_ko": None, "fail_ko": None}


def render(card):
    """The card as Korean lines, the way the seller would read it."""
    L = ["[수정 확인 카드]"]
    w = card["where"]
    if card["target"] == "video" and w.get("times"):
        place = "영상 " + ", ".join(f"{t:g}초" for t in w["times"])
    elif w.get("box"):
        place = "칠한 곳 " + _box_ko(w["box"])
    elif card["target"] in ("text_layer", "page"):
        place = f"선택한 {card['target_ko']}"
    else:
        place = "지정 안 됨"
    if w.get("point"):
        place += f" / 정답 자리 ({w['point'][0]}, {w['point'][1]})"
    L.append(f"고칠 곳: {place}")
    if card["kind"]:
        how = {"selected": "직접 고름", "guessed": "요청 글에서 추정" + (": " + ", ".join(f'"{x}"' for x in card["kind_words"]) if card["kind_words"] else "") + ". 다르면 바꿔 주세요"}[card["kind_from"]]
        L.append(f"종류: {card['kind_ko']} ({how})")
    else:
        L.append("종류: 고르지 않음")
    if card["text"]:
        L.append(f"바꿀 내용: {card['text']}")
    L.append("유지할 것:")
    L += [f"- {k}" for k in card["keep"]]
    if card["attach"]:
        L.append("붙일 사진:")
        L += [f"- {a['path']}: {a['why']}" for a in card["attach"]]
    else:
        L.append("붙일 사진: 없음")
    L.append(f"처리 방식: {METHODS[card['method']][0]}" + (f" ({card['tool']})" if card.get("tool") else ""))
    if card.get("first"):
        L.append(f"- 먼저 시작 이미지: {card['first']['method_ko']}, {card['first']['cost']}크레딧")
    if card.get("request_ko"):
        L.append(f"요청문: {card['request_ko']}")
    L.append("비용: " + (card.get("cost_ko") or (f"{card['cost']}크레딧" if card["cost"] else "무료")) + (". " + card["charge_ko"] if card["cost"] else ""))
    L.append("가능 여부: " + {"yes": "가능", "ask": "확인 필요", "no": "불가"}[card["possible"]] + (f". {card['possible_ko']}" if card["possible_ko"] and card["possible"] != "yes" else ""))
    if card.get("choices"):
        L.append("고를 종류: " + ", ".join(KIND_KO[k] for k in card["choices"]))
    if card["warnings"]:
        L.append("주의:")
        L += [f"- {x}" for x in card["warnings"]]
    if card["alternatives"]:
        L.append("다른 방법:")
        L += [f"- {a['method_ko']}({a['cost']}크레딧): {a['when']}" if a["cost"] else f"- {a['method_ko']}(무료): {a['when']}" for a in card["alternatives"]]
    if card["after"].get("check_ko"):
        L.append(f"고친 뒤: {card['after']['check_ko']}")
    return "\n".join(L)


# ---------------------------------------------------------------- from the check report


def _kind_from_differences(diffs):
    text = " ".join(diffs)
    return "print" if any(w in text for w in dict(WORDS)["print"] + ["글자", "문구"]) else "not_real"


def from_check(report, refset=None, spec=None):
    """Cards for every output check.run judged different, and look items for the ones the
    first pass could not settle."""
    items = []
    for row in report.get("images", []):
        if row["final"] == "different":
            diffs = (row.get("vlm") or {}).get("differences") or []
            card = propose("; ".join(diffs) or "실물과 달라요", _kind_from_differences(diffs), "image", refset=refset, spec=spec)
            items.append({"file": row["file"], "final": "different", "card": card})
        elif row["final"] == "unverified":
            items.append({"file": row["file"], "final": "unverified", "look_ko": (
                f"1차 검수(SIFT 점수 {row['inliers']})로는 프린트 면이 안 보이는 건지 틀리게 그려진 건지 구분하지 못해요. "
                "직접 보고, 틀렸으면 고칠 곳을 칠하고 종류를 골라 주세요. API 키가 있으면 2차 검수(VLM)가 판정해요.")})
    for v in report.get("videos", []):
        if v["final"] == "different":
            diffs = [d for j in v.get("vlm", []) if j.get("verdict") == "different" for d in j.get("differences") or []]
            card = propose("; ".join(dict.fromkeys(diffs)) or "실물과 달라요", _kind_from_differences(diffs), "video", refset=refset, spec=spec, times=v.get("different_at"))
            items.append({"file": v["file"], "final": "different", "card": card})
        elif v["final"] == "unverified" and v.get("dip_at"):
            items.append({"file": v["file"], "final": "unverified", "look_ko": (
                "실물 프린트가 맞다가 " + ", ".join(f"{t:g}초" for t in v["dip_at"][:6]) + ("…" if len(v["dip_at"]) > 6 else "")
                + "에서 떨어져요. 그 구간에서 프린트 면이 돌아간 건지 틀리게 그려진 건지 직접 봐 주세요.")})
    return items


# ---------------------------------------------------------------- after a fix


REGEN = ("regen_face", "regen_input", "regen_camera", "video_regen", "replace_real", "composite_real")


def after_fix(rev, method=None):
    """Next step once a fix was checked (check/revision.py's report). method: the card's
    method; a regeneration is expected to change the composition.

      check_real    a regeneration came back: compare it with the real product (check.run)
      redrawn       the composition changed: nothing to paste back; redo with the area edit
                    (no charge) or replace with the real photo
      retry         the request did not land: redo without charge, with a new input
      restore       the composition is kept and something outside the region (or the print)
                    changed: paste the original back (check.restore), then check again
      find_region   no region was given and the image changed: find what was drawn or removed
                    (check.restore --auto, with the seller's box as --where)
      show          nothing else changed: show the fix"""
    v = rev["verdict"]
    out = rev.get("outside") or {}
    p = rev.get("print") or {}
    region = rev.get("region")
    before, after = rev.get("before"), rev.get("after")
    dropped = p.get("verdict") == "lost" or (p.get("verdict") == "kept" and p.get("before") and p["after"] <= (1 - PRINT_DROP) * p["before"])
    if method in REGEN:
        nxt = {"next": "check_real", "next_ko": "다시 생성한 수정이라 구도가 바뀌는 게 정상이에요. 덮지 않고 실물과 대조해 맞으면 보여 줘요."}
    elif v == "redrawn":
        nxt = {"next": "redrawn", "next_ko": "구도가 바뀌어 원본으로 덮을 수 없어요. 차감 없이 영역 편집으로 다시 하거나, 실물 사진으로 교체해요(무료).", "charge_ko": "이번 수정은 차감하지 않기(제안)"}
        if p.get("verdict") in ("kept", "gained"):
            nxt["next_ko"] += " 다만 실물 프린트는 맞아요. 다시 생성을 의도한 수정이었다면 실물과 대조한 뒤 보여 줘도 돼요."
    elif v == "unchanged" or (rev.get("inside") or {}).get("verdict") == "unchanged":
        nxt = {"next": "retry", "next_ko": "요청이 반영되지 않았어요. 같은 입력으로 다시 하지 말고 실물 사진을 붙이거나 처리 방식을 바꿔 다시 해요(CS 21, 32).", "charge_ko": "이번 수정은 차감하지 않기(제안)"}
    elif region and (out.get("level") in ("minor", "major") or dropped):
        boxes = " ".join(f"--box {','.join(map(str, b))}" for b in region.get("boxes") or [])
        nxt = {"next": "restore", "next_ko": "구도는 그대로라 칠한 곳 밖을 원본 픽셀로 덮을 수 있어요. 덮은 뒤 다시 검수해요.",
               "command": f"python -m check.restore --before {before} --after {after} {boxes or '--mask <칠한 영역>'} --refset <refset.json> --out <폴더>"}
    elif not region and v == "changes":
        nxt = {"next": "find_region", "next_ko": "칠한 영역이 없는 수정이에요. 그려지거나 지워진 곳만 가져오고 나머지는 원본으로 덮어요. 셀러가 가리킨 곳이 있으면 --where로 넘겨요.",
               "command": f"python -m check.restore --before {before} --after {after} --auto --refset <refset.json> --out <폴더>"}
    else:
        nxt = {"next": "show", "next_ko": "수정 전과 달라진 곳이 고친 곳뿐이에요. 결과를 보여 줘요."}
    reasons = []
    if out.get("level") in ("minor", "major"):
        reasons.append(f"{'칠한 곳 밖' if region else '이미지에서'} {len(out.get('areas') or [])}곳 바뀜" + ("(배경 질감만)" if out["level"] == "minor" else ""))
    if p.get("verdict") == "lost":
        reasons.append(f"실물 프린트 사라짐(점수 {p['before']} → {p['after']})")
    elif dropped:
        reasons.append(f"실물 프린트 점수 하락({p['before']} → {p['after']})")
    elif p.get("verdict") == "gained":
        reasons.append(f"수정 뒤 실물 프린트가 맞음(점수 {p['before']} → {p['after']})")
    nxt["why"] = reasons
    return nxt


# ---------------------------------------------------------------- command line


def _write(out, doc, text):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "suggest.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    (out / "card.txt").write_text(text + "\n")


def _point(s):
    v = [int(float(x)) for x in s.split(",")]
    if len(v) != 2:
        raise argparse.ArgumentTypeError(f"point must be x,y: {s}")
    return v


def main():
    ap = argparse.ArgumentParser(description="Fix proposal (confirmation card) for a wrong output.")
    ap.add_argument("--text", default="", help="the seller's request, in Korean")
    ap.add_argument("--kind", choices=list(KIND_KO), default=None, help="picked by the seller; guessed from --text if not given")
    ap.add_argument("--target", choices=list(TARGET_KO), default="image", help="the element picked in the editor")
    ap.add_argument("--box", type=parse_box, default=None, help="painted place x0,y0,x1,y1")
    ap.add_argument("--point", type=_point, default=None, help="the right place x,y (position fix)")
    ap.add_argument("--product-box", type=parse_box, default=None)
    ap.add_argument("--tried", type=int, default=0, help="failures so far with the same method and input")
    ap.add_argument("--refset", default=None)
    ap.add_argument("--spec", default=None, help="confirmed spec.json")
    ap.add_argument("--report", default=None, help="check.run report.json: a card for every different result")
    ap.add_argument("--revision", default=None, help="check.revision revision.json: the next step after a fix")
    ap.add_argument("--method", choices=list(METHODS), default=None, help="with --revision: the method the fix used")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = str(Path(a.spec) / "spec.json") if a.spec and Path(a.spec).is_dir() else a.spec
    if a.revision:
        nxt = after_fix(json.loads(Path(a.revision).read_text()), a.method)
        _write(a.out, nxt, nxt["next_ko"] + ("\n" + "\n".join(nxt["why"]) if nxt["why"] else "") + ("\n" + nxt["command"] if nxt.get("command") else ""))
        print(json.dumps(nxt, ensure_ascii=False, indent=2))
        return
    if a.report:
        items = from_check(json.loads(Path(a.report).read_text()), a.refset, spec)
        text = "\n\n".join(f"{i['file']}\n" + (i["card"]["card_ko"] if "card" in i else i["look_ko"]) for i in items) or "고칠 결과가 없어요."
        _write(a.out, items, text)
        print(text)
        return
    card = propose(a.text, a.kind, a.target, a.box, a.point, a.refset, spec, a.tried, a.product_box)
    _write(a.out, card, card["card_ko"])
    print(card["card_ko"])


if __name__ == "__main__":
    main()

"""Production spec and the Creazy input kit.

The flow this step sits in:
  main photo -> guided capture -> product reference set (refset.build)
  -> scene request and scene references -> production spec proposed here
  -> the seller confirms or edits it -> generation settings (the kit) -> generation -> check

Three things are kept apart, so a scene reference never changes the product:
  product reference set   shape, color, parts, print and label: what must not change
  scene references        composition, background, lighting: what may be borrowed
  production spec         the scene, what may change, what must hold, and the conflicts the
                          seller should settle before credits are spent

  python -m spec.make --refset out/mug/refset.json --request "아침 햇살 카페 무드, 정면 위주" --out out/mug/spec
  python -m spec.make ... --references my_scene.jpg          # extra scene references

Rules write the spec without an API key. With a key the VLM refines it: it sees the product
photos, the references and the request, describes the print in words and finds conflicts the
rules cannot (a scene that needs a side no photo shows, a reference that fights the product).

The kit is what to put into Creazy's own fields (Creazy cannot be changed from here):
  product images      the representative print face and the face composite. Not the plain
                      faces: uploaded next to them, Creazy attaches the plain ones to some cuts
                      and those cuts invent the print (Run 10: 9 of 15). The print face is
                      added because a chat "use my real photo" request places the uploaded
                      photo on the page as it is, and the composite alone looks like a collage.
  info images         the seller's info images, unchanged
  design references   the scene references, with any product of the seller's kind hidden
  extra requests      the spec as short Korean lines (Creazy's 추가 요청사항 field)

A product of the seller's kind inside a scene reference leaks into the outputs: a sentence
saying "ignore it" did not stop it (CS 9: a lavender bra in the reference, a lavender bra in
one new cut). So the photo itself is changed before it goes into Creazy:
  hide   gray and blur every detected box, so color and pattern are gone and the pose and
         layout stay (the default)
  drop   leave the photo out (offered first when the boxes cover most of the photo)
  keep   upload it as it is (only when the seller says it is their product in the same color)
Hiding is the default for every product found, the seller's own included: DINOv2 could not
tell the seller's product from a look-alike of the same kind (CS 9 lavender bra 0.81 against
the set, the seller's own photos 0.77 to 0.95), and hiding the seller's own product costs
little because the product comes from the product images anyway.
CS 9 regenerated with the hidden reference: the visor-model composition came back wearing the
seller's black bra, and 0 of 14 cuts were lavender (1 of 14 with the sentence alone).
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Literal, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field

from refset import vlm

FACE_WORDS = {
    "front": ["정면", "앞면", "앞모습", "전면"],
    "back": ["뒷면", "뒤에서", "뒤쪽", "뒷모습", "후면", "등판"],
    "side": ["옆면", "옆모습", "측면", "옆에서"],
    "handle": ["손잡이"],
    "top": ["윗면", "위에서", "탑뷰", "뚜껑", "펌프 헤드", "입구"],
    "bottom": ["바닥", "밑면", "아래에서", "각인"],
    "detail": ["클로즈업", "접사", "확대", "디테일"],
}
CHANGE_WORDS = ["다른 색", "색을 바꿔", "색 바꿔", "색상 변경", "로고 빼", "로고 없이", "로고를 지워", "글자 빼", "글자 없이", "모양을 바꿔", "디자인 바꿔", "디자인을 바꿔"]
SET_WORDS = ["세트", "1+1", "2개", "두 개", "여러 개", "커플", "묶음"]
FLIP_WORDS = ["좌우 반전", "반전", "거울", "뒤집"]
REF_ASPECTS = {
    "구도": ["구도", "배치", "앵글", "각도"],
    "배경": ["배경", "장소", "공간", "카페", "테이블"],
    "조명": ["조명", "햇살", "빛", "무드"],
    "색감": ["색감", "톤"],
    "소품": ["소품"],
    "포즈": ["포즈", "모델", "손으로"],
}
SPEC_FILE = "spec.json"

# what to do with a scene reference that shows a product of the seller's kind
REF_HIDE = "사진 속 제품을 흐리게 가리고 구도, 배경, 조명만 참고하기"
REF_DROP = "이 참고 사진 빼기"
REF_KEEP = "가리지 않고 그대로 쓰기 (판매 상품과 같은 색의 같은 제품일 때만)"
DROP_FILL = 0.35  # boxes cover this much of the photo or more: hiding leaves little to borrow
HIDE_PAD = 0.12  # the detector's box stops short of straps and handles
HIDE_KO = "[가린 부분] 디자인 참고 사진에서 흐리게 가린 자리는 다른 제품을 지운 곳이에요. 그 자리에는 상품 이미지의 제품만 넣어 주세요."
HIDE_EN = "Gray blurred areas in the scene references mark a removed product: put only the product from the product reference photos there."
ACTION_KO = {"hide": "제품 가림", "drop": "뺌", "keep": "그대로"}


class Conflict(BaseModel):
    kind: Literal["missing_face", "changes_product", "multiple_units", "flip", "reference_has_product", "other"]
    detail: str = Field(description="Korean, what conflicts with what, one sentence")
    options: List[str] = Field(description="Korean, two or three ways the seller can settle it, the safest first")
    ref_file: Optional[str] = Field(default=None, description="for reference_has_product: the reference file name, unchanged")


class Spec(BaseModel):
    product: str = Field(description="Korean, the product being sold")
    scene: str = Field(description="Korean, the scene the seller wants, one or two sentences")
    keep: List[str] = Field(description="Korean, conditions that must hold in every output: shape, color, print and label (describe them), parts, number of units")
    reference_use: List[str] = Field(description="Korean, what to borrow from the scene references (구도, 배경, 조명, 색감, 소품, 포즈)")
    reference_ignore: List[str] = Field(description="Korean, what in the references must not be used (for example the product in them)")
    changeable: List[str] = Field(description="Korean, what may change freely")
    needed_faces: List[str] = Field(description="Faces of the product the scene needs: front, back, side, handle, top, bottom, detail")
    conflicts: List[Conflict]
    notes: List[str] = Field(default_factory=list, description="Korean, anything the seller should know that is not a conflict")
    decisions: List[str] = Field(default_factory=list, description="Korean, how the seller settled each conflict (filled at confirmation)")
    request_ko: str = Field(description="Korean text for Creazy's extra-request field: short lines, the keep conditions first")
    prompt_en: str = Field(description="English generation prompt that states the keep conditions and the scene")


def _has(text, words):
    return [w for w in words if w in text]


def faces_in(text):
    return [f for f, words in FACE_WORDS.items() if _has(text, words)]


def rule_spec(ref, request, references):
    """The spec from rules alone (no API key)."""
    product = ref.get("product") or ref.get("name") or "상품"
    prints = ref.get("prints") or []
    known = {i.get("face") for i in ref["items"] if i.get("face")}
    missing_listed = {m["face"] for m in ref.get("missing_faces") or []}
    needed = faces_in(request)

    keep = ["제품의 형태, 비율, 색, 부품은 상품 이미지(실물 사진) 그대로"]
    if prints:
        keep.append("프린트와 라벨은 실물 사진 그대로: 그림, 문구, 작은 글자, 위치를 바꾸거나 빼거나 새로 그리지 않기")
    keep.append("제품은 1개만")
    if known:
        keep.append("보이는 면은 실물 사진에 있는 면만: " + ", ".join(vlm.FACE_KO[f] for f in sorted(known) if f in vlm.FACE_KO))

    use = [a for a, words in REF_ASPECTS.items() if _has(request, words)]
    if references and not use:
        use = ["구도", "배경", "조명"]
    ignore = ["참고 사진 속 제품의 모양, 색, 프린트"] if references else []
    changeable = ["배경, 조명, 소품", "카메라 각도와 구도(실물 사진에 있는 면 안에서)"]

    conflicts, notes = [], []
    for f in needed:
        if known and f not in known or f in missing_listed:
            conflicts.append(
                Conflict(
                    kind="missing_face",
                    detail=f"장면에 {vlm.FACE_KO[f]} 사진이 필요한데 실물 사진에 없어요. 그대로 만들면 AI가 지어냅니다.",
                    options=[f"{vlm.FACE_KO[f]} 사진을 찍어 추가하기", f"{vlm.FACE_KO[f]} 없이 가능한 구도로 바꾸기", "그대로 진행하고 검수 대상으로 표시하기"],
                )
            )
    if needed and not known:
        notes.append("면 이름이 없어(VLM 꺼짐) 장면에 필요한 면이 사진에 있는지는 확인하지 못했어요: " + ", ".join(vlm.FACE_KO[f] for f in needed))
    if _has(request, CHANGE_WORDS):
        conflicts.append(
            Conflict(
                kind="changes_product",
                detail="요청이 제품 자체(색, 로고, 글자, 모양)를 바꾸라고 해요. 실물과 다른 상세페이지가 됩니다.",
                options=["제품은 그대로 두고 배경, 조명만 바꾸기", "실제로 다른 옵션 상품이면 그 상품 사진을 올리기"],
            )
        )
    if _has(request, SET_WORDS):
        conflicts.append(
            Conflict(
                kind="multiple_units",
                detail="요청에 세트나 여러 개가 있는데 실물 사진은 1개짜리예요. 셀러가 주지 않은 구성을 지어낼 수 있어요(CS 33).",
                options=["1개만 나오게 하기", "실제 세트 사진을 올리기"],
            )
        )
    if _has(request, FLIP_WORDS):
        conflicts.append(
            Conflict(
                kind="flip",
                detail="좌우 반전은 프린트와 글자까지 거울처럼 뒤집어요.",
                options=["반전하지 않고 카메라 방향만 바꾸기", "프린트가 안 보이는 면에서만 반전하기"],
            )
        )
    for r in references:
        if r.get("has_product"):
            conflicts.append(reference_conflict(r))

    scene = request.strip() or "상품을 돋보이게 하는 기본 연출"
    request_ko = kit_text(keep, use, ignore, scene)
    prompt_en = (
        "Show exactly the product in the product reference photos: same shape, proportions, color and parts"
        + (", the same printed design and lettering in the same place, small text included" if prints else "")
        + ". One unit only. "
        + ("From the scene reference photos take only " + ", ".join(use) + "; ignore any product in them. " if references else "")
        + f"Scene: {scene}"
    )
    return Spec(
        product=product,
        scene=scene,
        keep=keep,
        reference_use=use,
        reference_ignore=ignore,
        changeable=changeable,
        needed_faces=needed,
        conflicts=conflicts,
        notes=notes,
        request_ko=request_ko,
        prompt_en=prompt_en,
    )


def ref_options(r):
    """The seller's three choices for a reference with a product in it, the safest first."""
    if (r.get("product_fill") or 0) >= DROP_FILL:
        return [REF_DROP, REF_HIDE, REF_KEEP]
    return [REF_HIDE, REF_DROP, REF_KEEP]


def reference_conflict(r):
    n = len(r.get("product_boxes") or []) or 1
    return Conflict(
        kind="reference_has_product",
        detail=f"참고 사진 {r['file']}에 같은 종류 제품이 {n}곳 보여요. 판매 상품과 다른 제품이거나 다른 색이면 그 색과 모양이 생성에 섞여요(CS 9, CS 11).",
        options=ref_options(r),
        ref_file=r["file"],
    )


def ref_action(choice):
    """A choice in the seller's words to hide, drop or keep. Older specs said "무시하기" for
    hide, and a VLM may reword the options, so this reads keywords, not exact strings."""
    if not choice:
        return None
    if "빼" in choice:
        return "drop"
    if "그대로" in choice:
        return "keep"
    return "hide"


def treat_references(spec, refs):
    """Decide hide, drop or keep for each reference and say why. A decision the seller made
    wins; otherwise the first (safest) option of its conflict."""
    for r in refs:
        stem, name = Path(r["path"]).stem, Path(r["path"]).name
        if not r.get("has_product"):
            r["treatment"] = {"action": "keep", "by": "rule", "reason": "같은 종류 제품을 찾지 못해 그대로 씁니다.", "kit": f"3_design_reference/{name}"}
            continue
        at = next(
            (k for k, c in enumerate(spec.conflicts) if c.kind == "reference_has_product" and (c.ref_file == r["file"] or r["file"] in c.detail)),
            None,
        )
        choice = spec.decisions[at] if at is not None and at < len(spec.decisions) else ""
        action, by = ref_action(choice), "seller"
        if action is None:
            options = spec.conflicts[at].options if at is not None else ref_options(r)
            action, by = ref_action(options[0]), "default"
        n, fill = len(r.get("product_boxes") or []), r.get("product_fill") or 0
        if action == "hide":
            reason = f"같은 종류 제품 {n}곳을 흐리게 가렸어요. 사진 속 제품의 색과 무늬가 생성에 섞이지 않게 합니다."
        elif action == "drop":
            reason = "판매자가 이 사진을 빼기로 정했어요." if by == "seller" else f"제품이 사진의 {fill:.0%}를 차지해 가리면 참고할 것이 거의 남지 않아요. 빼는 쪽을 먼저 제안합니다."
        else:
            reason = "판매자가 그대로 쓰기로 정했어요. 사진 속 제품이 판매 상품과 다르면 섞일 수 있어요."
        kit = None if action == "drop" else f"3_design_reference/{stem}_hidden.jpg" if action == "hide" else f"3_design_reference/{name}"
        r["treatment"] = {"action": action, "by": by, "reason": reason, "kit": kit}
    return refs


def mark_hidden(spec, refs):
    """Tell the generator what the gray blur is, only when a photo was actually hidden."""
    hidden = any((r.get("treatment") or {}).get("action") == "hide" for r in refs)
    ko = [line for line in spec.request_ko.split("\n") if not line.startswith("[가린 부분]")]
    if hidden:
        at = next((k + 1 for k, line in enumerate(ko) if line.startswith("[참고 사진]")), len(ko))
        ko.insert(at, HIDE_KO)
    spec.request_ko = "\n".join(ko)
    en = spec.prompt_en.replace(HIDE_EN, "").strip()
    spec.prompt_en = f"{en} {HIDE_EN}" if hidden else en
    return spec


def hide_products(img, boxes, pad=HIDE_PAD):
    """Gray and blur each product box with a soft edge: color and pattern go, the pose and the
    layout stay. A flat gray fill was tried too; it hid the pose as well."""
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    short = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        b = (max(0, int(x0 - w * pad)), max(0, int(y0 - h * pad)), min(img.width, int(x1 + w * pad)), min(img.height, int(y1 + h * pad)))
        draw.rectangle(b, fill=255)
        short.append(min(b[2] - b[0], b[3] - b[1]))
    if not short:
        return img.copy()
    s = min(short)
    mask = mask.filter(ImageFilter.GaussianBlur(max(4, s * 0.04)))
    gray = ImageOps.grayscale(img).convert("RGB").filter(ImageFilter.GaussianBlur(max(6, s * 0.06)))
    return Image.composite(gray, img, mask)


def inspect_missing(refs, ref, base):
    """References from a refset.json written before every box was kept get their boxes now."""
    from refset import build

    for r in refs:
        if r.get("has_product") and "product_boxes" not in r:
            r.update(build.inspect_reference(build.open_rgb(base / r["path"]), ref["query"], 0.3))
    return refs


def kit_text(keep, use, ignore, scene, decisions=()):
    lines = ["[반드시 유지] " + " / ".join(keep)]
    if use:
        lines.append("[참고 사진] " + ", ".join(use) + "만 참고" + (". 쓰지 않을 것: " + ", ".join(ignore) if ignore else ""))
    lines.append("[장면] " + scene)
    decisions = [d for d in decisions if d]
    if decisions:
        lines.append("[정한 것] " + " / ".join(decisions))
    return "\n".join(lines)


SPEC_PROMPT = """You prepare a production spec for an AI product detail page, before any credits are spent.

The first images are the seller's real product (the reference set: representative print face, and a composite of the faces we have). Then come scene reference photos, if any. Then the seller's request.

Rules from the service:
- The product must look exactly like the reference set. Scene references may only lend composition, background, lighting, color mood, props or poses; any product inside them must be ignored.
- Every output shows one unit unless the seller uploaded a set.
- A side of the product that no reference photo shows will be invented by the generator.

Seller's request (Korean): {request}

A draft written by rules is below. Improve it:
- keep: describe the print, label and distinctive parts in words (what is drawn or written, where it sits, e.g. which way the handle points when the print faces the camera), plus shape, color, number of units.
- needed_faces and conflicts: which sides the requested scene needs, whether the reference set has them, and anything in the request or references that fights the product. Give options the seller can pick, safest first.
- request_ko: short Korean lines for the extra-request field, keep conditions first. prompt_en: an English prompt stating the keep conditions, then the scene.
Write all Korean fields in plain standard Korean.

Draft:
{draft}"""


def vlm_spec(ref, base, request, references, draft):
    content = []
    for rel, label in ((ref.get("representative"), "representative print face"), ((ref.get("composite") or {}).get("path"), "composite of the faces we have")):
        if rel:
            content.append(vlm.text_block(f"Real product, {label}:"))
            content.append(vlm.image_block(Image.open(base / rel).convert("RGB")))
    for r in references:
        content.append(vlm.text_block(f"Scene reference {r['file']}:"))
        content.append(vlm.image_block(Image.open(base / r["path"]).convert("RGB"), side=768))
    content.append(vlm.text_block(SPEC_PROMPT.format(request=request or "(none)", draft=draft.model_dump_json(indent=2))))
    return vlm.ask(content, Spec)


def add_references(paths, ref, base):
    """Extra scene references given here rather than as reference- uploads."""
    from refset import build

    out = []
    (base / "references").mkdir(exist_ok=True)
    for p in paths:
        p = Path(p)
        img = build.open_rgb(p)
        rel = f"references/{p.stem}.jpg"
        img.save(base / rel, quality=90)
        out.append({"file": p.name, "path": rel, **build.inspect_reference(img, ref["query"], 0.3)})
    return out


def make(refset_path, request, out, references=(), use_vlm=None):
    refset_path = Path(refset_path)
    base = refset_path.parent
    ref = json.loads(refset_path.read_text())
    refs = inspect_missing([dict(r) for r in ref.get("references") or []], ref, base) + add_references(references, ref, base)
    use_vlm = vlm.available() if use_vlm is None else (use_vlm and vlm.available())
    spec = rule_spec(ref, request, refs)
    log = {"used": use_vlm}
    if use_vlm:
        try:
            spec = vlm_spec(ref, base, request, refs, spec)
            log["model"] = vlm.MODEL
        except Exception as e:  # the rule spec still stands
            log["failed"] = f"{type(e).__name__}: {e}"
    # every reference with a product keeps a choice for the seller, whatever the VLM wrote
    for r in refs:
        if r.get("has_product") and not any(c.kind == "reference_has_product" and (c.ref_file == r["file"] or r["file"] in c.detail) for c in spec.conflicts):
            spec.conflicts.append(reference_conflict(r))
    mark_hidden(spec, treat_references(spec, refs))
    doc = {"request": request, "references": refs, "spec": spec.model_dump(), "confirmed": False, "vlm": log}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SPEC_FILE).write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    write_kit(doc, refset_path, out / "kit")
    return doc


def confirm(spec_dir, edited, refset_path):
    """The seller's edited spec replaces the proposal; the kit is rebuilt from it."""
    spec_dir = Path(spec_dir)
    doc = json.loads((spec_dir / SPEC_FILE).read_text())
    spec = Spec(**{**doc["spec"], **edited})
    # decisions come in conflict order; a dropped reference leaves the kit, a hidden one goes
    # in with its products grayed out
    inspect_missing(doc["references"], json.loads(Path(refset_path).read_text()), Path(refset_path).parent)
    treat_references(spec, doc["references"])
    doc["references"] = [r for r in doc["references"] if r["treatment"]["action"] != "drop"]
    if not doc["references"]:
        spec.reference_use, spec.reference_ignore = [], []
    spec.request_ko = kit_text(spec.keep, spec.reference_use, spec.reference_ignore, spec.scene, spec.decisions)
    mark_hidden(spec, doc["references"])
    doc.update(spec=spec.model_dump(), confirmed=True)
    (spec_dir / SPEC_FILE).write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    write_kit(doc, refset_path, spec_dir / "kit")
    return doc


def write_kit(doc, refset_path, kit):
    refset_path = Path(refset_path)
    base = refset_path.parent
    ref = json.loads(refset_path.read_text())
    kit = Path(kit)
    if kit.exists():
        shutil.rmtree(kit)
    for d in ("1_product", "2_info", "3_design_reference"):
        (kit / d).mkdir(parents=True)
    files = {"product": [], "info": [], "design_reference": []}
    # the seller's main photo leads; without one, the representative (densest print) face
    rep = ref.get("main") or ref["representative"]
    shutil.copy(base / rep, kit / "1_product" / ("print_face" + Path(rep).suffix))
    files["product"].append("1_product/print_face" + Path(rep).suffix)
    comp = (ref.get("composite") or {}).get("path")
    if comp:
        shutil.copy(base / comp, kit / "1_product" / "faces_composite.jpg")
        files["product"].append("1_product/faces_composite.jpg")
    uploads = Path(ref.get("uploads") or "")
    for it in ref["items"] + ref.get("excluded", []) + ref.get("dropped", []):
        src = uploads / it["file"]
        if it.get("role") == "info" and src.exists() and not (kit / "2_info" / src.name).exists():
            shutil.copy(src, kit / "2_info" / src.name)
            files["info"].append(f"2_info/{src.name}")
    for r in doc["references"]:
        t = r.get("treatment") or {"action": "keep", "kit": f"3_design_reference/{Path(r['path']).name}"}
        if t["action"] == "drop":
            continue
        if t["action"] == "hide":
            hide_products(Image.open(base / r["path"]).convert("RGB"), r["product_boxes"]).save(kit / t["kit"], quality=92)
        else:
            shutil.copy(base / r["path"], kit / t["kit"])
        files["design_reference"].append(t["kit"])
    spec = doc["spec"]
    (kit / "extra_request.txt").write_text(spec["request_ko"] + "\n")
    (kit / "prompt_en.txt").write_text(spec["prompt_en"] + "\n")
    open_conflicts = [c["detail"] for c in spec["conflicts"]] if not doc.get("confirmed") else []
    readme = [
        "크리에이지에 넣는 법",
        "1. 상품 이미지 칸: 1_product의 두 장만 올린다 (대표 면, 면 합본). 무지 면 사진은 올리지 않는다.",
        "2. 상품정보용 이미지 칸: 2_info",
        "3. 디자인 기획 단계의 디자인 참고 사진: 3_design_reference",
        "4. 추가 요청사항 칸: extra_request.txt 내용을 붙여 넣는다",
        "5. 생성 후 결과물을 check.run에 넣어 검수한다 (--spec으로 이 명세를 함께 준다)",
    ]
    treated = [r for r in doc["references"] if r.get("treatment")]
    if treated:
        readme += ["", "참고 사진 처리"] + [f"- {r['file']}: {ACTION_KO[r['treatment']['action']]}. {r['treatment']['reason']}" for r in treated]
    if open_conflicts:
        readme += ["", "제출 전에 정할 것"] + [f"- {c}" for c in open_conflicts]
    (kit / "README.txt").write_text("\n".join(readme) + "\n")
    (kit / "kit.json").write_text(json.dumps(files, ensure_ascii=False, indent=2))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refset", required=True)
    ap.add_argument("--request", default="", help="the scene the seller wants, in Korean")
    ap.add_argument("--references", nargs="*", default=[], help="extra scene reference photos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    doc = make(a.refset, a.request, a.out, a.references, use_vlm=not a.no_vlm)
    s = doc["spec"]
    print("유지:", *s["keep"], sep="\n  ")
    print("참고:", s["reference_use"], "/ 무시:", s["reference_ignore"])
    print("필요한 면:", s["needed_faces"])
    print("충돌:", *(f"[{c['kind']}] {c['detail']} -> {c['options']}" for c in s["conflicts"]) or ["없음"], sep="\n  ")
    for n in s["notes"]:
        print("참고:", n)
    print("키트:", Path(a.out) / "kit")


if __name__ == "__main__":
    main()

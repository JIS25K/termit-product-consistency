"""Demo server. The flow follows the team's production flow:

  1 product photos (main photo, guided capture video)  -> 2 product reference set
  3 scene request and scene references                 -> 4 production spec, confirmed by the seller
  5 generation settings (the Creazy input kit)         -> 6 check outputs -> 7 report
  8 fix request: the confirmation card (check/suggest.py) before any credit is spent
  9 the fixed image checked against the one before it, the original pasted back outside the
    fixed part (check/restore.py, check/revision.py)

  .venv/bin/python -m demo.app            # http://127.0.0.1:8024
  /label                                  # the human labeling page (data/cs/<case>/labels.json)

Everything a session makes is written under out/demo/<session id>/ and served from /files.
Built-in examples (our own mug and pump photos) make the demo work without uploads.
"""
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from check import restore as check_restore
from check import revision as check_revision
from check import run as check_run
from check import suggest as check_suggest
from refset import build, vlm
from spec import make as spec_make

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "demo"
STATIC = Path(__file__).resolve().parent / "static"
OUT.mkdir(parents=True, exist_ok=True)
EDITS = ROOT / "data/mug/outputs/run10b_edit"
# Where the seller pointed in the two chat edits, which came without a painted region.
CHAT_POINTED = {"chat1_succulent": [128, 110, 402, 506], "chat2_pen": [140, 240, 335, 315]}


def _mug_fixes():
    """Real Creazy fixes of Run 10b cuts (data/mug/outputs/run10b_edit, 25 credits spent):
    the request, the place, the image before and the image Creazy returned."""
    doc = json.loads((EDITS / "edits.json").read_text())
    label = {"edit1_clay": "점토 덩어리 지우기", "edit2_cookies": "접시 위 쿠키 지우기", "edit3_succulent": "다육식물 화분 지우기",
             "chat1_succulent": "화분 지우기 (채팅)", "chat2_pen": "책 위 펜 지우기 (채팅)"}
    out = []
    for e in doc["edits"]:
        out.append({"id": e["name"], "label": label.get(e["name"], e["name"]), "how": "area", "text": e["prompt"],
                    "box": e["box"], "before": EDITS / e["before"], "after": EDITS / e["after"]})
    for e in doc.get("chat", []):
        out.append({"id": e["name"], "label": label.get(e["name"], e["name"]), "how": "chat", "text": e["prompt"],
                    "box": CHAT_POINTED.get(e["name"]), "before": EDITS / e["before"], "after": EDITS / e["after"]})
    return out

EXAMPLES = {
    "mug": {
        "label": "고양이 라인드로잉 머그컵",
        "name": "고양이 라인드로잉 머그컵 Faire une pause",
        "query": "a mug",
        "uploads": ROOT / "data/mug/uploads",
        "main": "info-detail_cut.jpg",
        "capture": ROOT / "data/mug/real_video/mug_rotate_real.mp4",
        "scene": {
            "request": "아침 햇살이 드는 카페 테이블, 정면 컷 위주로 따뜻한 무드",
            # a Run 9 output with an invented cat: a scene reference that holds a look-alike mug
            "references": [ROOT / "data/mug/outputs/run9/AI-Generated-68036f43-e5de-476e-bb1c-29661d5b6a63.png"],
        },
        "fixes": _mug_fixes(),
        "outputs": {
            "run10b": ("Run 10b: 합본만 올린 생성", ROOT / "data/mug/outputs/run10b"),
            "run9": ("Run 9: 잘라낸 프린트 사진을 추가한 생성", ROOT / "data/mug/outputs/run9"),
            "run8": ("Run 8: 프린트 사진이 정보용으로 빠진 생성", ROOT / "data/mug/outputs/run8"),
            "run6": ("Run 6: 손잡이 사진만 붙은 컷이 섞인 생성", ROOT / "data/mug/outputs/run6"),
            "real": ("실물을 손으로 돌린 영상 (대조군)", ROOT / "data/mug/real_video"),
        },
    },
    "pump": {
        "label": "펌프형 바디로션",
        "name": "부케가르니 베이비파우더 바디로션 1000ml",
        "query": "a pump bottle",
        "uploads": ROOT / "data/pump/uploads",
        "main": "p03.jpg",
        "capture": ROOT / "data/pump/real_video/pump_rotate_real.mp4",
        "scene": {"request": "욕실 선반 위 아침 루틴, 손으로 펌프를 누르는 장면도 하나", "references": []},
        "outputs": {
            "real": ("실물을 손으로 돌린 영상 (대조군)", ROOT / "data/pump/real_video"),
        },
    },
}
ROLE_PREFIX = {"product": "", "info": "info-", "reference": "reference-"}

app = FastAPI(title="creazy-consistency demo")
app.mount("/files", StaticFiles(directory=OUT), name="files")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
lock = threading.Lock()  # one model run at a time; the demo is single-user


def _safe(name):
    name = Path(name).name
    return re.sub(r"[^\w.\-가-힣]", "_", name) or "file"


def _bare(name):
    for pre in ROLE_PREFIX.values():
        if pre and name.startswith(pre):
            return name[len(pre):]
    return name


def _session(sid=None):
    if sid is None:
        sid = uuid.uuid4().hex[:10]
    if not re.fullmatch(r"[0-9a-f]{10}", sid):
        raise HTTPException(400, "잘못된 세션입니다. 처음부터 다시 시작하세요.")
    d = OUT / sid
    d.mkdir(parents=True, exist_ok=True)
    return sid, d


def _url(sid, rel):
    return f"/files/{sid}/{rel}"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"vlm": vlm.available(), "model": vlm.MODEL if vlm.available() else None}


@app.get("/api/examples")
def examples():
    return [
        {
            "id": k,
            "label": e["label"],
            "name": e["name"],
            "main": e.get("main"),
            "capture": bool(e.get("capture")),
            "uploads": sorted(p.name for p in e["uploads"].iterdir() if p.suffix.lower() in build.IMG_EXT),
            "scene": {"request": e["scene"]["request"], "references": [p.name for p in e["scene"]["references"]]},
            "outputs": [{"id": o, "label": lab} for o, (lab, _) in e["outputs"].items()],
            "fixes": [{k: f[k] for k in ("id", "label", "how", "text", "box")} | {"src": f"/api/fix-file/{k}/{f['id']}/before"} for f in e.get("fixes", [])],
        }
        for k, e in EXAMPLES.items()
    ]


@app.get("/api/example-file/{ex}/{name}")
def example_file(ex: str, name: str):
    e = EXAMPLES.get(ex)
    if not e or name != _safe(name):
        raise HTTPException(404)
    for p in [e["uploads"] / name] + list(e["scene"]["references"]):
        if p.name == name and p.exists():
            return FileResponse(p)
    raise HTTPException(404)


def _refset_payload(sid, d, ref):
    base = "refset"
    return {
        "sid": sid,
        "refset": ref,
        "sheet": _url(sid, f"{base}/sheet.jpg"),
        "crops": {i["crop"]: _url(sid, f"{base}/{i['crop']}") for i in ref["items"] + ref["excluded"]},
        "prints": [_url(sid, f"{base}/{p['path']}") for p in ref.get("prints") or []],
        "composite": _url(sid, f"{base}/{ref['composite']['path']}") if (ref.get("composite") or {}).get("path") else None,
        "references": [{**r, "url": _url(sid, f"{base}/{r['path']}")} for r in ref.get("references") or []],
    }


@app.post("/api/refset")
def make_refset(
    files: List[UploadFile] = File(default=[]),
    roles: List[str] = Form(default=[]),
    name: str = Form(""),
    query: str = Form(""),
    example: str = Form(""),
    main: str = Form(""),
    with_capture: bool = Form(False),
):
    sid, d = _session()
    up = d / "uploads"
    up.mkdir(exist_ok=True)
    if example:
        e = EXAMPLES.get(example)
        if not e:
            raise HTTPException(404, "없는 예시입니다.")
        for p in e["uploads"].iterdir():
            if p.suffix.lower() in build.IMG_EXT:
                shutil.copy(p, up / p.name)
        if with_capture and e.get("capture"):
            shutil.copy(e["capture"], up / "capture_turn.mp4")
        name = name or e["name"]
        query = query or e["query"]
        main = main or e.get("main") or ""
    for i, f in enumerate(files):
        role = roles[i] if i < len(roles) else "product"
        fname = _bare(_safe(f.filename))
        (up / (ROLE_PREFIX.get(role, "") + fname)).write_bytes(f.file.read())
    if not any(up.iterdir()):
        raise HTTPException(400, "제품 사진을 한 장 이상 올리세요.")
    if not query and not (name and vlm.available()):
        raise HTTPException(400, "VLM이 꺼져 있어 탐지어를 만들 수 없습니다. 영어 탐지어(예: a mug)를 적어 주세요.")
    main_file = next((p.name for p in up.iterdir() if main and _bare(p.name) == _bare(_safe(main))), None)
    with lock:
        try:
            ref = build.build(up, query or None, d / "refset", name=name or None, main=main_file)
        except SystemExit as e:
            raise HTTPException(422, f"제품 기준 자료를 만들지 못했습니다: {e}")
    (d / "meta.json").write_text(json.dumps({"example": example, "name": name}, ensure_ascii=False))
    return _refset_payload(sid, d, ref)


def _spec_payload(sid, d, doc):
    kit = d / "spec" / "kit"
    files = json.loads((kit / "kit.json").read_text())
    return {
        "sid": sid,
        "doc": doc,
        "kit": {k: [_url(sid, f"spec/kit/{p}") for p in v] for k, v in files.items()},
        "kit_zip": f"/api/kit/{sid}.zip",
    }


@app.post("/api/spec")
def propose_spec(
    sid: str = Form(...),
    request: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    example_refs: bool = Form(False),
):
    sid, d = _session(sid)
    refset_path = d / "refset" / "refset.json"
    if not refset_path.exists():
        raise HTTPException(400, "제품 기준 자료를 먼저 만드세요.")
    sd = d / "scene_refs"
    if sd.exists():
        shutil.rmtree(sd)
    sd.mkdir()
    paths = []
    if example_refs:
        meta = json.loads((d / "meta.json").read_text())
        e = EXAMPLES.get(meta.get("example") or "")
        for p in (e or {}).get("scene", {}).get("references", []):
            shutil.copy(p, sd / p.name)
            paths.append(sd / p.name)
    for f in files:
        p = sd / _safe(f.filename)
        p.write_bytes(f.file.read())
        paths.append(p)
    with lock:
        doc = spec_make.make(refset_path, request, d / "spec", references=paths)
    return _spec_payload(sid, d, doc)


@app.post("/api/spec/confirm")
def confirm_spec(payload: dict = Body(...)):
    sid, d = _session(payload.get("sid"))
    if not (d / "spec" / "spec.json").exists():
        raise HTTPException(400, "제작 명세를 먼저 제안받으세요.")
    allowed = {"scene", "keep", "reference_use", "reference_ignore", "changeable", "decisions"}
    edited = {k: v for k, v in (payload.get("spec") or {}).items() if k in allowed}
    doc = spec_make.confirm(d / "spec", edited, d / "refset" / "refset.json")
    return _spec_payload(sid, d, doc)


@app.get("/api/kit/{sid}.zip")
def kit_zip(sid: str):
    sid, d = _session(sid)
    kit = d / "spec" / "kit"
    if not kit.exists():
        raise HTTPException(404, "키트가 없습니다.")
    z = shutil.make_archive(str(d / "creazy_kit"), "zip", kit)
    return FileResponse(z, filename="creazy_kit.zip")


@app.post("/api/check")
def check(
    sid: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    example_outputs: str = Form(""),
    vlm_all: bool = Form(False),
):
    sid, d = _session(sid)
    if not (d / "refset" / "refset.json").exists():
        raise HTTPException(400, "제품 기준 자료를 먼저 만드세요.")
    run_id = uuid.uuid4().hex[:6]
    od = d / f"outputs_{run_id}"
    od.mkdir()
    if example_outputs:
        meta = json.loads((d / "meta.json").read_text())
        e = EXAMPLES.get(meta.get("example") or "")
        if not e or example_outputs not in e["outputs"]:
            raise HTTPException(404, "이 기준 자료에 맞는 예시 결과물이 없습니다.")
        for p in e["outputs"][example_outputs][1].iterdir():
            if p.suffix.lower() in check_run.IMG | check_run.VID:
                shutil.copy(p, od / p.name)
    for f in files:
        (od / _safe(f.filename)).write_bytes(f.file.read())
    if not any(od.iterdir()):
        raise HTTPException(400, "검수할 결과물을 올리세요.")
    spec = d / "spec" if (d / "spec" / "spec.json").exists() else None
    with lock:
        try:
            rep = check_run.run(d / "refset" / "refset.json", [od], d / f"check_{run_id}", vlm_all=vlm_all, spec=spec)
        except SystemExit as e:
            raise HTTPException(422, str(e))
    vis = f"check_{run_id}/vis"
    for i in rep["images"]:
        i["src"] = _url(sid, f"outputs_{run_id}/{i['file']}")
        stem = Path(i["file"]).stem
        if (d / vis / f"{stem}.jpg").exists():
            i["vis"] = _url(sid, f"{vis}/{stem}.jpg")
    for v in rep["videos"]:
        v["src"] = _url(sid, f"outputs_{run_id}/{v['file']}")
        for j in v.get("vlm", []):
            j["frame_url"] = _url(sid, f"check_{run_id}/{j['frame']}")
    return {"sid": sid, "report": rep}


# ---------------------------------------------------------------- 8, 9: fix request and fix check


def _fix_example(ex, fid):
    e = EXAMPLES.get(ex) or {}
    f = next((x for x in e.get("fixes", []) if x["id"] == fid), None)
    if not f:
        raise HTTPException(404, "없는 수정 예시입니다.")
    return f


@app.get("/api/fix-file/{ex}/{fid}/{which}")
def fix_file(ex: str, fid: str, which: str):
    if which not in ("before", "after"):
        raise HTTPException(404)
    return FileResponse(_fix_example(ex, fid)[which])


@app.get("/api/fix-options")
def fix_options():
    return {"kinds": check_suggest.KIND_KO, "targets": check_suggest.TARGET_KO}


def _parse_ints(s, n):
    if not s:
        return None
    try:
        v = [int(float(x)) for x in s.split(",")]
    except ValueError:
        raise HTTPException(400, "좌표 형식이 잘못됐습니다.")
    if len(v) != n:
        raise HTTPException(400, "좌표 형식이 잘못됐습니다.")
    return v


def _file_url(p):
    try:
        return "/files/" + str(Path(p).resolve().relative_to(OUT.resolve()))
    except ValueError:
        return None


@app.post("/api/suggest")
def suggest_fix(
    sid: str = Form(...),
    text: str = Form(""),
    kind: str = Form(""),
    target: str = Form("image"),
    box: str = Form(""),
    point: str = Form(""),
    tried: int = Form(0),
    example_fix: str = Form(""),
    output: str = Form(""),
    file: UploadFile = File(default=None),
):
    """Step 8: the confirmation card for one fix. The image to fix is a built-in Creazy fix
    example, an output checked in step 7, or an uploaded image."""
    sid, d = _session(sid)
    refset_path = d / "refset" / "refset.json"
    if not refset_path.exists():
        raise HTTPException(400, "1단계에서 제품 기준 자료를 먼저 만드세요.")
    if kind and kind not in check_suggest.KIND_KO:
        raise HTTPException(400, "없는 종류입니다.")
    if target not in check_suggest.TARGET_KO:
        raise HTTPException(400, "없는 요소입니다.")
    fid = uuid.uuid4().hex[:6]
    fd = d / f"fix_{fid}"
    fd.mkdir()
    meta = {"example": None, "how": None}
    if example_fix:
        ex = json.loads((d / "meta.json").read_text()).get("example") or ""
        f = _fix_example(ex, example_fix)
        src = f["before"]
        meta.update(example={"ex": ex, "id": f["id"]}, how=f["how"])
    elif output:
        src = (d / output).resolve()
        if d.resolve() not in src.parents or not src.exists():
            raise HTTPException(400, "고칠 결과물을 찾지 못했습니다.")
    elif file is not None and file.filename:
        src = fd / ("upload_" + _safe(file.filename))
        src.write_bytes(file.file.read())
    else:
        raise HTTPException(400, "고칠 이미지를 고르세요.")
    before = fd / ("before" + Path(src).suffix.lower())
    shutil.copy(src, before)
    bx, pt = _parse_ints(box, 4), _parse_ints(point, 2)
    if bx and (bx[2] <= bx[0] or bx[3] <= bx[1]):
        raise HTTPException(400, "칠한 곳의 좌표가 잘못됐습니다.")
    spec = d / "spec" / "spec.json"
    product_box = None
    with lock:
        if target == "image" and Path(before).suffix in check_run.IMG:
            img = check_revision._read(before)
            found = check_revision.find_product(refset_path, img)
            product_box = found[0] if found else None
        card = check_suggest.propose(text, kind or None, target, bx, pt, refset_path, spec if spec.exists() else None, tried, product_box)
    for a in card["attach"]:
        a["url"] = _file_url(a["path"])
    meta.update(before=before.name, box=bx, point=pt, product_box=product_box, card=card)
    (fd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"sid": sid, "fix": fid, "before": _url(sid, f"fix_{fid}/{before.name}"), "product_box": product_box, "card": card, "how": meta["how"]}


def _rev(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


@app.post("/api/fix")
def check_fix(
    sid: str = Form(...),
    fix: str = Form(...),
    how: str = Form("area"),
    example_after: bool = Form(False),
    file: UploadFile = File(default=None),
):
    """Step 9: the image Creazy returned for the fix, checked against the image before it.
    how=area: the fix came from the area edit, the painted box is taken from it. how=chat:
    no painted region; what was drawn or removed is found, narrowed to the pointed box."""
    sid, d = _session(sid)
    if not re.fullmatch(r"[0-9a-f]{6}", fix) or not (d / f"fix_{fix}" / "meta.json").exists():
        raise HTTPException(400, "8단계에서 확인 카드를 먼저 만드세요.")
    fd = d / f"fix_{fix}"
    meta = json.loads((fd / "meta.json").read_text())
    if example_after:
        if not meta.get("example"):
            raise HTTPException(400, "이 이미지에는 예시 수정 결과가 없습니다. 크리에이지에서 고친 결과를 올리세요.")
        src = _fix_example(meta["example"]["ex"], meta["example"]["id"])["after"]
    elif file is not None and file.filename:
        src = fd / ("upload_after_" + _safe(file.filename))
        src.write_bytes(file.file.read())
    else:
        raise HTTPException(400, "크리에이지에서 고친 결과를 넣으세요.")
    after = fd / ("after" + Path(src).suffix.lower())
    shutil.copy(src, after)
    before = fd / meta["before"]
    box = meta.get("box")
    if how == "area" and not box:
        raise HTTPException(400, "영역 편집으로 고쳤다면 8단계에서 칠한 곳이 있어야 합니다.")
    refset_path = d / "refset" / "refset.json"
    product = [meta["product_box"]] if meta.get("product_box") else None
    method = meta["card"].get("method")
    out = fd / "check"
    if out.exists():
        shutil.rmtree(out)
    with lock:
        try:
            if how == "area":
                rep = check_restore.run(before, after, out, [box], None, refset_path, product)
            else:
                rep = check_restore.run(before, after, out, (), None, refset_path, product, auto=True, where=[box] if box else None)
        except ValueError:
            # the composition changed: nothing to paste back, only the check of the fix as it came
            check_revision.run(before, after, out / "as_is", [box] if box else (), None, refset_path, product_boxes=product)
            rep = {"restored": None, "notes_ko": ["수정 전후의 구도가 달라 원본으로 덮을 수 없어요. 영역 편집으로 다시 하거나 실물 사진으로 교체해야 해요."]}
    as_is, checked = _rev(out / "as_is" / "revision.json"), _rev(out / "check" / "revision.json")
    files = lambda rel: _url(sid, f"fix_{fix}/{rel}") if (fd / rel).exists() else None
    return {
        "sid": sid,
        "fix": fix,
        "method": method,
        "images": {"before": files(before.name), "after": files(after.name), "restored": files("check/restored.png"),
                   "as_is_vis": files("check/as_is/revision.jpg"), "check_vis": files("check/check/revision.jpg")},
        "restore": rep,
        "as_is": {k: as_is.get(k) for k in ("verdict", "outside", "inside", "print", "notes_ko", "align")} if as_is else None,
        "checked": {k: checked.get(k) for k in ("verdict", "outside", "inside", "print", "notes_ko")} if checked else None,
        "next_as_is": check_suggest.after_fix(as_is, method) if as_is else None,
        "next_checked": check_suggest.after_fix(checked, method) if checked else None,
    }


# ---- human labeling page: team members label the outputs of the customer-service cases ----
CS = ROOT / "data" / "cs"
LABEL_REPORTS = [ROOT / "out" / "cs_strong", ROOT / "out" / "cs_all"]
LABEL_CASES = ["1", "10", "14", "16", "19", "24", "32", "35", "37", "39"]
LABEL_REASONS = ["text", "count", "shape", "color", "other_product"]
LABEL_LABELS = ["match", "not_visible", "different"]
LABEL_ABOUT = ("일치: 실물과 같은 제품으로 보이고, 보이는 글자, 프린트, 개수, 모양이 실물과 맞는다. "
               "안 보임: 판단할 면이 보이지 않지만, 보이는 것 중 실물과 어긋나는 것도 없다. "
               "다름: 보이는 것 중 실물과 어긋나는 것이 있다(이유를 하나 이상 고른다).")
LABEL_STILL = {".png", ".jpg", ".jpeg", ".webp"}


def _label_case_dir(case):
    if case not in LABEL_CASES or not (CS / case).is_dir():
        raise HTTPException(404, "없는 케이스입니다.")
    return CS / case


def _label_root(case):
    for root in LABEL_REPORTS:
        if (root / case / "check" / "report.json").exists():
            return root
    return None


def _label_report(case):
    root = _label_root(case)
    if root is None:
        return {}
    try:
        doc = json.loads((root / case / "check" / "report.json").read_text())
    except (OSError, ValueError):
        return {}
    return {r["file"]: r for r in doc.get("images", []) if isinstance(r, dict) and "file" in r}


def _label_tool(rows, name):
    r = rows.get(name)
    if r is None:
        return None
    strong = r.get("strong") or {}
    return {"first": r.get("verdict"), "final": r.get("final"),
            "strong": strong.get("verdict"), "why": strong.get("why") or []}


LABEL_UNREADABLE = ("labels.json을 읽을 수 없어 저장하지 않았습니다. 지금 있는 라벨을 덮어쓰지 않으려고 멈췄습니다. "
                    "파일을 고치거나 다른 곳으로 옮긴 뒤 다시 저장하세요.")


def _label_read(d, strict=False):
    """(outputs, _why) of the case's labels.json. A missing file is an empty start. A file that
    exists but cannot be read shows as empty on the pages; with strict (the save path) it is
    refused with 409, so a save never overwrites labels it could not read."""
    try:
        doc = json.loads((d / "labels.json").read_text())
    except FileNotFoundError:
        doc = {}
    except (OSError, ValueError):
        if strict:
            raise HTTPException(409, LABEL_UNREADABLE)
        doc = {}
    if not isinstance(doc, dict) or any(k in doc and not isinstance(doc[k], dict) for k in ("outputs", "_why")):
        if strict:
            raise HTTPException(409, LABEL_UNREADABLE)
        doc = doc if isinstance(doc, dict) else {}
    outs = doc.get("outputs") if isinstance(doc.get("outputs"), dict) else {}
    why = doc.get("_why") if isinstance(doc.get("_why"), dict) else {}
    return outs, why


def _label_outputs(d):
    folder = d / "outputs"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in LABEL_STILL)


def _label_prints(case):
    root = _label_root(case)
    if root is None:
        return []
    try:
        doc = json.loads((root / case / "refset" / "refset.json").read_text())
    except (OSError, ValueError):
        return []
    items = doc.get("prints") or ([doc["print"]] if isinstance(doc.get("print"), dict) else [])
    names = [Path(i["path"]).name for i in items if isinstance(i, dict) and i.get("path")]
    return [n for n in names if (root / case / "refset" / n).is_file()]


@app.get("/label")
def label_page():
    return FileResponse(STATIC / "label.html")


@app.get("/api/label/cases")
def label_cases():
    out = []
    for case in LABEL_CASES:
        d = CS / case
        if not d.is_dir():
            continue
        try:
            query = json.loads((d / "case.json").read_text()).get("query")
        except (OSError, ValueError, AttributeError):
            query = None
        files = _label_outputs(d)
        labeled = _label_read(d)[0]
        out.append({"case": case, "query": query, "n": len(files), "done": sum(1 for f in files if f in labeled)})
    return out


@app.get("/api/label/{case}")
def label_case(case: str):
    d = _label_case_dir(case)
    labeled, why = _label_read(d)
    rows = _label_report(case)
    refs = []
    up = d / "uploads"
    if up.is_dir():
        refs += [f"/api/label-file/{case}/upload/{p.name}" for p in sorted(up.iterdir())
                 if p.is_file() and p.suffix.lower() in LABEL_STILL and build.role_of(p.name) == "product"]
    refs += [f"/api/label-file/{case}/print/{n}" for n in _label_prints(case)]
    outputs = []
    for name in _label_outputs(d):
        lab = labeled.get(name)
        outputs.append({"file": name, "url": f"/api/label-file/{case}/output/{name}", "label": lab,
                        "why": why.get(name, []), "tool": _label_tool(rows, name) if lab else None})
    return {"case": case, "about": LABEL_ABOUT, "reasons": LABEL_REASONS, "refs": refs, "outputs": outputs}


@app.get("/api/label-file/{case}/{kind}/{name}")
def label_file(case: str, kind: str, name: str):
    d = _label_case_dir(case)
    name = Path(name).name
    if kind == "upload":
        p = d / "uploads" / name
    elif kind == "output":
        p = d / "outputs" / name
    elif kind == "print":
        root = _label_root(case)
        if root is None:
            raise HTTPException(404)
        p = root / case / "refset" / name
    else:
        raise HTTPException(404)
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/label/{case}")
def save_label(case: str, payload: dict = Body(...)):
    d = _label_case_dir(case)
    name = Path(str(payload.get("file") or "")).name
    label = payload.get("label")
    reasons = payload.get("why") or []
    if not name or name not in _label_outputs(d):
        raise HTTPException(404, "없는 파일입니다.")
    if label not in LABEL_LABELS:
        raise HTTPException(400, "잘못된 라벨입니다.")
    if not isinstance(reasons, list) or any(r not in LABEL_REASONS for r in reasons):
        raise HTTPException(400, "잘못된 이유입니다.")
    if label == "different" and not reasons:
        raise HTTPException(400, "다름은 이유를 하나 이상 골라야 합니다.")
    reasons = [r for r in LABEL_REASONS if r in reasons] if label == "different" else []
    with lock:
        outs, why = _label_read(d, strict=True)
        outs[name] = label
        if reasons:
            why[name] = reasons
        else:
            why.pop(name, None)
        tmp = d / "labels.json.tmp"
        tmp.write_text(json.dumps({"_about": LABEL_ABOUT, "_why": why, "outputs": outs}, ensure_ascii=False, indent=2))
        os.replace(tmp, d / "labels.json")
    return {"file": name, "label": label, "why": reasons, "tool": _label_tool(_label_report(case), name)}


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8024)


if __name__ == "__main__":
    main()

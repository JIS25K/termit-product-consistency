"""Local pages for the CS validation (docs/validation_cs_plan.md): human ground-truth labels and
the blind evaluation. Serves on 127.0.0.1 only; answers are saved as JSON under out/validation/.

  python spikes/validation/server.py            # http://127.0.0.1:8792, this round's C cases
  python spikes/validation/server.py --round a  # the A cases (next round)
  /label?who=NAME   label every image of the round's cases: same / different / not visible and
                    the error type; for A cases with a shape error also the A conditions (angle
                    in the reference, hand, real photo, one product with nothing on it)
  /eval?who=NAME    blind evaluation of out/validation/eval/items.json (built by run.py evalset):
                    is the product the same as the reference, and does it look pasted (1-5)

Customer images never leave this machine. Nothing here calls an external service.
"""
import argparse
import hashlib
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "out" / "validation"
ROUNDS = {"c": [37, 32, 39, 19, 10, 28], "a": [13, 1]}  # docs/validation_cs_plan.md section 1
A_CASES = {13, 1}
CASES = ROUNDS["c"]
EXT = (".png", ".jpg", ".jpeg", ".webp")
LOCK = threading.Lock()


def case_files():
    return {c: sorted(f.name for f in (ROOT / "data" / "cs" / str(c) / "outputs").iterdir() if f.suffix.lower() in EXT) for c in CASES}


def uploads(case):
    d = ROOT / "data" / "cs" / str(case) / "uploads"
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in EXT) if d.exists() else []


ORIGINALS = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>원본 사진</title><style>__STYLE__ .gal { display:grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap:12px; }
.gal img { width:100%; height:auto; background:#fff; border-radius:8px; border:1px solid var(--line); }</style></head><body><div class="wrap">
<div class="top"><h1>셀러 원본 사진</h1><span class="muted">사례 __CASE__ · __N__장 · 원래 크기 그대로</span></div><div class="gal" style="margin-top:12px">__IMGS__</div></div></body></html>"""


def label_items():
    return [{"case": c, "file": f, "a": c in A_CASES} for c, fs in case_files().items() for f in fs]


def eval_items(who):
    p = VAL / "eval" / "items.json"
    if not p.exists():
        return []
    items = json.loads(p.read_text())
    seed = int(hashlib.sha256(who.encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(items)
    return [{"id": x["id"], "case": x["case"]} for x in items]


def safe_name(who):
    s = "".join(ch for ch in who if ch.isalnum() or ch in "-_")
    return s[:40] or "anon"


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_answer(path, key, value):
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = load(path)
        d[key] = value
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        tmp.replace(path)


STYLE = """
:root { --bg:#f5f4f1; --card:#fff; --fg:#1d1c1a; --muted:#6a655f; --line:#dedad4; --on:#24594a; --on-soft:#dcece6; --warn:#a2402f; }
@media (prefers-color-scheme: dark) { :root { --bg:#191817; --card:#222120; --fg:#ebe9e6; --muted:#a39e98; --line:#383532; --on:#7fc6ae; --on-soft:#1f3a31; --warn:#e08a78; color-scheme: dark; } }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 "Apple SD Gothic Neo", system-ui, sans-serif; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }
.top { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; } .top h1 { font-size:20px; margin:0; } .muted { color:var(--muted); }
.grid { display:grid; grid-template-columns: minmax(0,3fr) minmax(0,2fr); gap:16px; margin-top:12px; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; min-width:0; }
.panel img { width:100%; height:auto; max-height:72vh; object-fit:contain; display:block; background:#fff; border-radius:6px; }
.q { margin:10px 0; } .q b { display:block; margin-bottom:6px; }
.opts { display:flex; flex-wrap:wrap; gap:6px; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:6px 12px; cursor:pointer; }
button[aria-pressed="true"] { background:var(--on-soft); border-color:var(--on); color:var(--on); font-weight:600; }
button:focus-visible { outline:2px solid var(--on); outline-offset:2px; }
.nav { display:flex; gap:8px; margin-top:12px; } .save { color:var(--on); } .err { color:var(--warn); }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }
@media (max-width: 860px) { .grid { grid-template-columns: minmax(0,1fr); } }
#img { cursor: zoom-in; }
.thumbs { display:grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap:6px; }
.thumbs img { width:100%; aspect-ratio:1; object-fit:contain; background:#fff; border:1px solid var(--line); border-radius:6px; cursor:zoom-in; max-height:none; }
.thumbs img:hover { outline:2px solid var(--on); }
.ov { position:fixed; inset:0; background:rgba(10,10,10,.92); z-index:50; display:flex; flex-direction:column; padding:10px max(10px, env(safe-area-inset-right,0px)) 10px max(10px, env(safe-area-inset-left,0px)); }
.ov .bar { display:flex; gap:10px; align-items:center; color:#eee; margin-bottom:8px; flex-wrap:wrap; }
.ov .bar button { background:#2a2a2a; color:#eee; border-color:#555; }
.ov .panes { flex:1; display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:10px; min-height:0; }
.ov .pane { overflow:auto; background:#fff; border-radius:8px; display:flex; align-items:center; justify-content:center; min-height:0; }
.ov .pane img { max-width:100%; max-height:100%; object-fit:contain; cursor:zoom-in; }
.ov .pane.full { display:block; } .ov .pane.full img { max-width:none; max-height:none; cursor:zoom-out; }
.ov .cap { color:#bbb; font-size:13px; }
@media (max-width: 860px) { .ov .panes { grid-template-columns: minmax(0,1fr); } }
"""

LABEL_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>정답 라벨</title><style>__STYLE__</style></head><body><div class="wrap">
<div class="top"><h1>정답 라벨</h1><span class="muted" id="who"></span><span class="muted" id="pos"></span><span id="state"></span></div>
<div class="grid">
 <div class="panel"><img id="img" alt="생성 이미지"></div>
 <div class="panel">
  <b>셀러 원본 사진</b> <span class="muted" style="font-size:13px">누르면 생성 이미지와 나란히 크게 열려요 (←, → 넘기기, 사진 누르면 원래 크기, Esc 닫기)</span>
  <div class="thumbs" id="thumbs" style="margin-top:6px"></div>
  <div class="q"><b>제품이 기준 사진과 같은가 <span class="muted">(1, 2, 3)</span></b><div class="opts" data-k="label">
   <button data-v="match">같음</button><button data-v="different">다름</button><button data-v="not_visible">안 보임</button></div></div>
  <div class="q" id="errq"><b>다르다면 무엇이 <span class="muted">(i, s, b)</span></b><div class="opts" data-k="error">
   <button data-v="print">인쇄(글자, 로고, 그림)</button><button data-v="shape">모양(형태, 부품, 비율, 색)</button><button data-v="both">둘 다</button></div></div>
  <div id="aq">
  <div class="q"><b>기준 사진에 이 각도(30도 안)의 실물이 있나</b><div class="opts" data-k="angle">
   <button data-v="yes">있음</button><button data-v="no">없음</button></div></div>
  <div class="q"><b>손이 제품을 복잡하게 쥐고 있나</b><div class="opts" data-k="hand">
   <button data-v="yes">예</button><button data-v="no">아니오</button></div></div>
  <div class="q"><b>실사 장면인가(설명 그림, 도식이 아닌가)</b><div class="opts" data-k="scene">
   <button data-v="photo">실사</button><button data-v="drawing">설명 그림</button></div></div>
  <div class="q"><b>제품이 하나이고 그 위·안에 다른 물건이 없나</b><div class="opts" data-k="single">
   <button data-v="yes">예</button><button data-v="no">아니오</button></div></div>
  </div>
  <div class="nav"><button id="prev">이전</button><button id="next">다음 <kbd>Enter</kbd></button><button id="skip">안 한 것으로</button></div>
 </div>
</div></div>
<script>
const who = new URLSearchParams(location.search).get("who") || "";
let items = [], done = {}, i = 0, cur = {};
async function init() {
  if (!who) { document.body.innerHTML = '<div class="wrap"><p>주소 끝에 ?who=이름 을 붙여 여세요.</p></div>'; return; }
  document.getElementById("who").textContent = "라벨러: " + who;
  items = await (await fetch("/api/label_items")).json();
  done = await (await fetch("/api/labels?who=" + encodeURIComponent(who))).json();
  i = items.findIndex(x => !done[x.case + "/" + x.file]); if (i < 0) i = 0;
  show();
}
function key(x) { return x.case + "/" + x.file; }
function show() {
  const x = items[i]; cur = Object.assign({}, done[key(x)] || {});
  document.getElementById("img").src = "/cs/" + x.case + "/" + encodeURIComponent(x.file);
  loadThumbs(x.case);
  document.getElementById("pos").textContent = (i + 1) + " / " + items.length + "  (사례 " + x.case + ", 끝낸 것 " + Object.keys(done).length + ")";
  paint();
}
function paint() {
  document.querySelectorAll(".opts").forEach(g => g.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", cur[g.dataset.k] === b.dataset.v ? "true" : "false")));
  document.getElementById("errq").style.opacity = cur.label === "different" ? 1 : 0.4;
  document.getElementById("aq").hidden = !needA();
}
function needA() { return items[i].a && cur.label === "different" && (cur.error === "shape" || cur.error === "both"); }
function ready() { return cur.label && (cur.label !== "different" || cur.error) && (!needA() || (cur.angle && cur.hand && cur.scene && cur.single)); }
async function save() {
  if (!ready()) { document.getElementById("state").innerHTML = '<span class="err">빠진 답이 있어요</span>'; return false; }
  const x = items[i];
  const r = await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({who, case: x.case, file: x.file, ...cur})});
  if (!r.ok) { document.getElementById("state").innerHTML = '<span class="err">저장하지 못했어요</span>'; return false; }
  done[key(x)] = Object.assign({}, cur); document.getElementById("state").innerHTML = '<span class="save">저장됨</span>'; return true;
}
document.querySelectorAll(".opts").forEach(g => g.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; cur[g.dataset.k] = b.dataset.v; if (g.dataset.k === "label" && b.dataset.v !== "different") delete cur.error; paint(); }));
document.getElementById("next").onclick = async () => { if (await save() && i < items.length - 1) { i++; show(); } };
document.getElementById("prev").onclick = () => { if (i > 0) { i--; show(); } };
document.getElementById("skip").onclick = () => { if (i < items.length - 1) { i++; show(); } };
document.addEventListener("keydown", e => {
  const m = {"1": ["label","match"], "2": ["label","different"], "3": ["label","not_visible"], "i": ["error","print"], "s": ["error","shape"], "b": ["error","both"]};
  if (m[e.key]) { cur[m[e.key][0]] = m[e.key][1]; if (m[e.key][0] === "label" && m[e.key][1] !== "different") delete cur.error; paint(); }
  else if (e.key === "Enter") document.getElementById("next").click();
});

let ups = [], upCase = null, ovEl = null, oi = 0;
async function loadThumbs(c) {
  if (upCase === c) return; upCase = c;
  ups = await (await fetch("/api/uploads?case=" + c)).json();
  const t = document.getElementById("thumbs"); t.innerHTML = "";
  ups.forEach((n, k) => { const im = document.createElement("img"); im.src = "/upload/" + c + "/" + encodeURIComponent(n); im.alt = "셀러 원본 사진 " + (k + 1); im.loading = "lazy"; im.onclick = () => openOv(k); t.appendChild(im); });
}
function ovOpen() { return ovEl && !ovEl.hidden; }
function openOv(k) {
  if (!ups.length) return;
  oi = (k + ups.length) % ups.length;
  if (!ovEl) {
    ovEl = document.createElement("div"); ovEl.className = "ov";
    ovEl.innerHTML = '<div class="bar"><button id="ovprev">← 이전 원본</button><button id="ovnext">다음 원본 →</button><span class="cap" id="ovcap"></span><span style="flex:1"></span><button id="ovclose">닫기 (Esc)</button></div>' +
      '<div class="panes"><div class="pane" id="ovl"><img id="ovgen" alt="생성 이미지"></div><div class="pane" id="ovr"><img id="ovup" alt="셀러 원본 사진"></div></div>';
    document.body.appendChild(ovEl);
    ovEl.querySelector("#ovprev").onclick = () => openOv(oi - 1);
    ovEl.querySelector("#ovnext").onclick = () => openOv(oi + 1);
    ovEl.querySelector("#ovclose").onclick = closeOv;
    ["ovl", "ovr"].forEach(id => ovEl.querySelector("#" + id).addEventListener("click", e => { if (e.target.tagName === "IMG") e.currentTarget.classList.toggle("full"); }));
  }
  ovEl.hidden = false;
  ovEl.querySelector("#ovgen").src = document.getElementById("img").src;
  ovEl.querySelector("#ovup").src = "/upload/" + upCase + "/" + encodeURIComponent(ups[oi]);
  ovEl.querySelector("#ovcap").textContent = "왼쪽: 생성 이미지 · 오른쪽: 셀러 원본 " + (oi + 1) + " / " + ups.length;
}
function closeOv() { if (ovEl) { ovEl.hidden = true; ovEl.querySelectorAll(".pane").forEach(p => p.classList.remove("full")); } }
document.getElementById("img").addEventListener("click", () => openOv(0));
document.addEventListener("keydown", e => {
  if (!ovOpen()) return;
  if (e.key === "Escape") closeOv(); else if (e.key === "ArrowLeft") openOv(oi - 1); else if (e.key === "ArrowRight") openOv(oi + 1);
  e.stopImmediatePropagation(); e.preventDefault();
}, true);
init();
</script></body></html>"""

EVAL_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>블라인드 평가</title><style>__STYLE__</style></head><body><div class="wrap">
<div class="top"><h1>블라인드 평가</h1><span class="muted" id="who"></span><span class="muted" id="pos"></span><span id="state"></span></div>
<p class="muted">어떤 사진이 처리된 것인지 알려 주지 않아요. 보이는 그대로 답해 주세요.</p>
<div class="grid">
 <div class="panel"><img id="img" alt="평가할 사진"></div>
 <div class="panel">
  <b>셀러 원본 사진</b> <span class="muted" style="font-size:13px">누르면 생성 이미지와 나란히 크게 열려요 (←, → 넘기기, 사진 누르면 원래 크기, Esc 닫기)</span>
  <div class="thumbs" id="thumbs" style="margin-top:6px"></div>
  <div class="q"><b>사진 속 제품이 기준 사진의 제품과 같은가 <span class="muted">(1, 2, 3)</span></b><div class="opts" data-k="same">
   <button data-v="same">같음</button><button data-v="different">다름</button><button data-v="unsure">판단 어려움</button></div></div>
  <div class="q"><b>제품을 붙여 넣은 티가 나는가 <span class="muted">(q, w, e, r, t)</span></b><div class="opts" data-k="pasted">
   <button data-v="1">1 전혀</button><button data-v="2">2</button><button data-v="3">3</button><button data-v="4">4</button><button data-v="5">5 확실히</button></div></div>
  <div class="nav"><button id="prev">이전</button><button id="next">다음 <kbd>Enter</kbd></button></div>
 </div>
</div></div>
<script>
const who = new URLSearchParams(location.search).get("who") || "";
let items = [], done = {}, i = 0, cur = {};
async function init() {
  if (!who) { document.body.innerHTML = '<div class="wrap"><p>주소 끝에 ?who=이름 을 붙여 여세요.</p></div>'; return; }
  document.getElementById("who").textContent = "평가자: " + who;
  items = await (await fetch("/api/eval_items?who=" + encodeURIComponent(who))).json();
  if (!items.length) { document.body.innerHTML = '<div class="wrap"><p>평가할 사진이 아직 없어요(run.py evalset을 먼저 실행).</p></div>'; return; }
  done = await (await fetch("/api/evals?who=" + encodeURIComponent(who))).json();
  i = items.findIndex(x => !done[x.id]); if (i < 0) i = 0;
  show();
}
function show() {
  const x = items[i]; cur = Object.assign({}, done[x.id] || {});
  document.getElementById("img").src = "/evalimg/" + x.id;
  loadThumbs(x.case);
  document.getElementById("pos").textContent = (i + 1) + " / " + items.length + "  (끝낸 것 " + Object.keys(done).length + ")";
  paint();
}
function paint() { document.querySelectorAll(".opts").forEach(g => g.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", cur[g.dataset.k] === b.dataset.v ? "true" : "false"))); }
async function save() {
  if (!cur.same || !cur.pasted) { document.getElementById("state").innerHTML = '<span class="err">두 질문 모두 답해 주세요</span>'; return false; }
  const x = items[i];
  const r = await fetch("/api/eval", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({who, id: x.id, ...cur})});
  if (!r.ok) { document.getElementById("state").innerHTML = '<span class="err">저장하지 못했어요</span>'; return false; }
  done[x.id] = Object.assign({}, cur); document.getElementById("state").innerHTML = '<span class="save">저장됨</span>'; return true;
}
document.querySelectorAll(".opts").forEach(g => g.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; cur[g.dataset.k] = b.dataset.v; paint(); }));
document.getElementById("next").onclick = async () => { if (await save()) { if (i < items.length - 1) { i++; show(); } else document.getElementById("state").innerHTML = '<span class="save">모두 끝났어요. 고맙습니다</span>'; } };
document.getElementById("prev").onclick = () => { if (i > 0) { i--; show(); } };
document.addEventListener("keydown", e => {
  const m = {"1": ["same","same"], "2": ["same","different"], "3": ["same","unsure"], "q": ["pasted","1"], "w": ["pasted","2"], "e": ["pasted","3"], "r": ["pasted","4"], "t": ["pasted","5"]};
  if (m[e.key]) { cur[m[e.key][0]] = m[e.key][1]; paint(); } else if (e.key === "Enter") document.getElementById("next").click();
});

let ups = [], upCase = null, ovEl = null, oi = 0;
async function loadThumbs(c) {
  if (upCase === c) return; upCase = c;
  ups = await (await fetch("/api/uploads?case=" + c)).json();
  const t = document.getElementById("thumbs"); t.innerHTML = "";
  ups.forEach((n, k) => { const im = document.createElement("img"); im.src = "/upload/" + c + "/" + encodeURIComponent(n); im.alt = "셀러 원본 사진 " + (k + 1); im.loading = "lazy"; im.onclick = () => openOv(k); t.appendChild(im); });
}
function ovOpen() { return ovEl && !ovEl.hidden; }
function openOv(k) {
  if (!ups.length) return;
  oi = (k + ups.length) % ups.length;
  if (!ovEl) {
    ovEl = document.createElement("div"); ovEl.className = "ov";
    ovEl.innerHTML = '<div class="bar"><button id="ovprev">← 이전 원본</button><button id="ovnext">다음 원본 →</button><span class="cap" id="ovcap"></span><span style="flex:1"></span><button id="ovclose">닫기 (Esc)</button></div>' +
      '<div class="panes"><div class="pane" id="ovl"><img id="ovgen" alt="생성 이미지"></div><div class="pane" id="ovr"><img id="ovup" alt="셀러 원본 사진"></div></div>';
    document.body.appendChild(ovEl);
    ovEl.querySelector("#ovprev").onclick = () => openOv(oi - 1);
    ovEl.querySelector("#ovnext").onclick = () => openOv(oi + 1);
    ovEl.querySelector("#ovclose").onclick = closeOv;
    ["ovl", "ovr"].forEach(id => ovEl.querySelector("#" + id).addEventListener("click", e => { if (e.target.tagName === "IMG") e.currentTarget.classList.toggle("full"); }));
  }
  ovEl.hidden = false;
  ovEl.querySelector("#ovgen").src = document.getElementById("img").src;
  ovEl.querySelector("#ovup").src = "/upload/" + upCase + "/" + encodeURIComponent(ups[oi]);
  ovEl.querySelector("#ovcap").textContent = "왼쪽: 생성 이미지 · 오른쪽: 셀러 원본 " + (oi + 1) + " / " + ups.length;
}
function closeOv() { if (ovEl) { ovEl.hidden = true; ovEl.querySelectorAll(".pane").forEach(p => p.classList.remove("full")); } }
document.getElementById("img").addEventListener("click", () => openOv(0));
document.addEventListener("keydown", e => {
  if (!ovOpen()) return;
  if (e.key === "Escape") closeOv(); else if (e.key === "ArrowLeft") openOv(oi - 1); else if (e.key === "ArrowRight") openOv(oi + 1);
  e.stopImmediatePropagation(); e.preventDefault();
}, true);
init();
</script></body></html>"""

INDEX = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>CS 검증</title><style>__STYLE__</style></head><body><div class="wrap">
<h1>CS 검증</h1><p class="muted">기준: docs/validation_cs_plan.md</p>
<p>정답 라벨: <code>/label?who=이름</code> · 블라인드 평가: <code>/eval?who=이름</code></p></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def file(self, path):
        if not path.exists():
            return self.send(404, "{}")
        ctype = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
        self.send(200, path.read_bytes(), ctype)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [p for p in u.path.split("/") if p]
        html = lambda t: self.send(200, t.replace("__STYLE__", STYLE), "text/html; charset=utf-8")
        if not parts:
            return html(INDEX)
        if parts == ["label"]:
            return html(LABEL_PAGE)
        if parts == ["eval"]:
            return html(EVAL_PAGE)
        if parts == ["api", "uploads"] and q.get("case", "").isdigit() and int(q["case"]) in CASES:
            return self.send(200, json.dumps(uploads(int(q["case"]))))
        if parts == ["api", "label_items"]:
            return self.send(200, json.dumps(label_items()))
        if parts == ["api", "labels"]:
            return self.send(200, json.dumps(load(VAL / "labels" / f"{safe_name(q.get('who', ''))}.json"), ensure_ascii=False))
        if parts == ["api", "eval_items"]:
            return self.send(200, json.dumps(eval_items(safe_name(q.get("who", "")))))
        if parts == ["api", "evals"]:
            return self.send(200, json.dumps(load(VAL / "eval" / f"answers_{safe_name(q.get('who', ''))}.json"), ensure_ascii=False))
        if len(parts) == 3 and parts[0] == "cs" and parts[1].isdigit() and int(parts[1]) in CASES:
            name = Path(parts[2]).name
            if name in case_files()[int(parts[1])]:
                return self.file(ROOT / "data" / "cs" / parts[1] / "outputs" / name)
        if len(parts) == 2 and parts[0] == "originals" and parts[1].isdigit() and int(parts[1]) in CASES:
            c = int(parts[1]); names = uploads(c)
            imgs = "".join(f'<a href="/upload/{c}/{n}" target="_blank" rel="noopener"><img src="/upload/{c}/{n}" alt="셀러 원본 사진"></a>' for n in names)
            return html(ORIGINALS.replace("__CASE__", str(c)).replace("__N__", str(len(names))).replace("__IMGS__", imgs))
        if len(parts) == 3 and parts[0] == "upload" and parts[1].isdigit() and int(parts[1]) in CASES:
            name = Path(parts[2]).name
            if name in uploads(int(parts[1])):
                return self.file(ROOT / "data" / "cs" / parts[1] / "uploads" / name)
        if len(parts) == 2 and parts[0] == "ref" and parts[1].isdigit():
            return self.file(VAL / "refs" / f"case{int(parts[1])}.jpg")
        if len(parts) == 2 and parts[0] == "evalimg" and parts[1].isalnum():
            return self.file(VAL / "eval" / "images" / f"{parts[1]}.jpg")
        return self.send(404, "{}")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self.send(400, "{}")
        who = safe_name(str(body.get("who", "")))
        if u.path == "/api/label":
            c, f = int(body.get("case", 0)), Path(str(body.get("file", ""))).name
            if c not in CASES or f not in case_files()[c] or body.get("label") not in ("match", "different", "not_visible"):
                return self.send(400, "{}")
            v = {k: body.get(k) for k in ("label", "error", "angle", "hand", "scene", "single") if body.get(k) is not None}
            save_answer(VAL / "labels" / f"{who}.json", f"{c}/{f}", v)
            return self.send(200, "{}")
        if u.path == "/api/eval":
            i = str(body.get("id", ""))
            if not i.isalnum() or body.get("same") not in ("same", "different", "unsure") or str(body.get("pasted")) not in "12345":
                return self.send(400, "{}")
            save_answer(VAL / "eval" / f"answers_{who}.json", i, {"same": body["same"], "pasted": int(body["pasted"])})
            return self.send(200, "{}")
        return self.send(404, "{}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=sorted(ROUNDS), default="c")
    CASES = ROUNDS[ap.parse_args().round]
    srv = ThreadingHTTPServer(("127.0.0.1", 8792), Handler)
    print("http://127.0.0.1:8792  (label: /label?who=NAME, eval: /eval?who=NAME)", flush=True)
    srv.serve_forever()

"""Comparison page for the second experiment (one shop photo of a pump bottle): input | A | B.

  python spikes/creazy_exp/make_compare4.py
writes out/compare/creazy_exp4.html (images from out/compare/img_exp4/) and
out/compare/creazy_exp4_team.html (one file, images inlined, for sending to the team).
"""
import base64, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "compare"
IMG = OUT / "img_exp4"
PICKS = json.loads((ROOT / "out" / "creazy_exp4" / "B" / "picks.json").read_text())

PRODUCTS = [
    {"id": "P3", "title": "P3 하늘색 바디로션 펌프 병 1L", "sub": "입력: 쇼핑몰 정면 사진 1장(500x500, 배경 투명). 실물을 구할 수 없는 판매자",
     "inputs": ["in_P3_1"], "ref": "guide_angle",
     "check": ["하늘색 불투명 원통 병, 둥근 어깨, 원본과 같은 비율", "하늘색 펌프 머리, 노즐이 한쪽으로 뻗은 모양", "위가 둥근 아치형 라벨 창",
               "라벨 가운데 곰 인형(파란 리본)과 둘레 소품 배치가 원본과 같음", "아치 위쪽의 휘어진 작은 글자가 뭉개지지 않음",
               "로고 글자와 아래 제품명 줄이 정확함", "손이나 소품 대비 병 크기가 자연스러움 (1L 병)"],
     "scenes": [
         ("fullbody", "전신 화보", "이 바디로션을 들고 있는 여자 전신 화보 만들어줘, 밝은 침실 배경", "", ""),
         ("couple", "커플 장면", "커플이 세면대 앞에서 이 바디로션을 같이 쓰는 장면 만들어줘", "", ""),
         ("bedside", "침실 협탁", "여자가 침대에 앉아 다리에 바디로션을 바르고, 이 바디로션은 옆 협탁 위에 놓인 장면 만들어줘", "", ""),
         ("bathroom", "욕실 전체", "이 바디로션이 놓인 밝은 욕실 전체 인테리어 컷 만들어줘", "", ""),
     ]},
]


import re


def overlay_ko(o):
    if not o:
        return "실물 덮기: 기록 없음"
    if o.get("applied"):
        ba = o.get("label_match_before_after")
        extra = f", 실물 라벨 대조점 {ba[0]} -> {ba[1]}" if ba else ""
        return f"실물 덮기 적용: 병이 {abs(o['turn_deg'])}도 돌고 {abs(o.get('tilt_deg', 0))}도 기울었다고 계산, 실물 사진을 {o['scale']}배로 맞춰 감았다{extra}"
    why = o.get("why", "")
    m = re.search(r"enlarged ([0-9.]+)x", why)
    if m:
        return f"실물 덮기 안 함: 실물 사진을 {m.group(1)}배로 키워야 해서(한도 2배) 흐려진다"
    m = re.search(r"covers ([0-9]+)%", why)
    if m:
        return f"실물 덮기 안 함: 손이 병 몸통의 {m.group(1)}%를 가린다"
    m = re.search(r"\(([0-9]+) -> ([0-9]+)\), undone", why)
    if m:
        return f"실물 덮기 되돌림: 덮은 뒤 실물 라벨과 닮은 정도가 나아지지 않았다(대조점 {m.group(1)} -> {m.group(2)})"
    m = re.search(r"spills out of the product \(([0-9]+)% inside\)", why)
    if m:
        return f"실물 덮기 안 함: 덮을 자리가 병 영역을 벗어난다(병 안 {m.group(1)}%)"
    if "few matches" in why or "not found" in why:
        return "실물 덮기 안 함: 생성된 그림에서 실물 라벨을 확실히 찾지 못했다(라벨이 안 보이거나 다르게 그려짐)"
    return "실물 덮기 안 함: " + why


def view_ko(r):
    if r.get("asked_view_deg") is None:
        return ""
    if r["asked_view_deg"] == 180:
        return "앞 라벨이 보이지 않아 뒷면으로 판단. " if r.get("shows_asked_view") else "앞 라벨이 보여 뒷면이 아님. "
    if r.get("turn_deg") is None:
        return "병의 회전 각도를 재지 못함. "
    return f"생성된 병이 {abs(r['turn_deg'])}도 돌아감(요청 {r['asked_view_deg']}도). "


def line_for(r):
    o = r.get("overlay") or {}
    if r.get("asked_view_deg") == 180 and not o.get("applied"):
        ov = "실물 덮기 안 함: 뒷면은 실물 사진에 없는 면이라 덮을 실물이 없다"
    else:
        ov = overlay_ko(o)
    return checks_ko(r) + view_ko(r) + ov


def checks_ko(r):
    f = r.get("found") or []
    return f"검수: 로고 단어 {len(f)}개 읽힘, 제품 {r.get('product_count', '?')}개. "


def pick_line(pid, sid):
    rows = [r for r in PICKS[f"{pid}_{sid}"] if r["kept"]]
    return " / ".join(f"s{i}: 후보 {r['file'].rsplit('_c', 1)[1].split('.')[0]}번. {line_for(r)}" for i, r in enumerate(rows, 1))


CSS = """
:root { --bg: #f4f3f1; --card: #ffffff; --fg: #1f1d1b; --muted: #6b6661; --line: #e2dfdb;
  --real: #4a5a78; --real-soft: #e4e8f0; --a: #b0413e; --a-soft: #f6e4e3; --b: #2f6f5e; --b-soft: #e1efe9; --chip: #efedea; }
@media (prefers-color-scheme: dark) { :root { --bg: #181716; --card: #222120; --fg: #ecebe9; --muted: #a29d97; --line: #353331;
  --real: #9fb0d4; --real-soft: #262c38; --a: #e07a76; --a-soft: #3a2524; --b: #7cc4ad; --b-soft: #1f322c; --chip: #2c2a28; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.6 "Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif; word-break: keep-all; }
.wrap { max-width: 1480px; margin: 0 auto; padding: 32px max(16px, env(safe-area-inset-right, 0px)) 64px max(16px, env(safe-area-inset-left, 0px)); }
header h1 { font-size: 26px; margin: 0 0 6px; }
header p { margin: 0; color: var(--muted); max-width: 80ch; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 12px; font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.notice { margin-top: 10px; font-size: 13px; color: var(--muted); }
nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 20px 0 26px; }
nav a { text-decoration: none; color: var(--fg); background: var(--card); border: 1px solid var(--line); padding: 6px 12px; border-radius: 999px; font-size: 13px; }
.product { margin-bottom: 36px; }
.phead { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; align-items: start; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }
.phead h2 { margin: 0; font-size: 21px; }
.phead .sub { color: var(--muted); font-size: 13px; margin: 2px 0 10px; }
.phead ul { margin: 0; padding-left: 1.2em; font-size: 13.5px; columns: 2; column-gap: 28px; }
.phead .refbox { text-align: center; font-size: 12px; color: var(--muted); max-width: 200px; }
.phead .refbox img { width: 120px; display: inline-block !important; margin: 0 2px 4px; max-width: 100%; border-radius: 8px; border: 1px solid var(--line); cursor: zoom-in; display: block; margin-bottom: 4px; }
.scene { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin-bottom: 18px; scroll-margin-top: 16px; }
.scene h3 { margin: 0 0 4px; font-size: 17px; }
.req { margin: 0 0 14px; font-size: 14px; color: var(--muted); }
.req q { color: var(--fg); }
.grid { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 2fr) minmax(0, 2fr); gap: 14px; }
.group { min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.tag { align-self: flex-start; padding: 2px 9px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.real .tag { background: var(--real-soft); color: var(--real); }
.a .tag { background: var(--a-soft); color: var(--a); }
.b .tag { background: var(--b-soft); color: var(--b); }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stack { display: grid; gap: 8px; }
figure { margin: 0; }
.frame { border-radius: 10px; overflow: hidden; border: 2px solid var(--line); background: #fff; aspect-ratio: 1 / 1; max-width: 100%; }
.real .frame { border-color: var(--real-soft); } .a .frame { border-color: var(--a-soft); } .b .frame { border-color: var(--b-soft); }
.frame img { display: block; width: 100%; height: 100%; object-fit: contain; cursor: zoom-in; }
figcaption { font-size: 12px; color: var(--muted); margin-top: 3px; }
.note { font-size: 13.5px; margin: 0; }
.note b { font-weight: 600; }
.btop { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px; }
.rawbtn { border: 1px solid var(--line); background: var(--card); color: var(--fg); font: inherit; font-size: 12px; padding: 2px 10px; border-radius: 6px; cursor: pointer; }
.rawbtn[aria-pressed="true"] { background: var(--b-soft); color: var(--b); border-color: var(--b); }
.zoom { position: fixed; inset: 0; background: rgba(10, 10, 10, .9); display: flex; align-items: center; justify-content: center; padding: 20px; z-index: 10; cursor: zoom-out; }
.zoom img { max-width: 100%; max-height: 100%; border-radius: 6px; }
@media (max-width: 640px) { .grid { grid-template-columns: minmax(0, 1fr); } .phead { grid-template-columns: minmax(0, 1fr); } .phead ul { columns: 1; }
  .real .stack { grid-template-columns: repeat(2, minmax(0, 140px)); } }
"""

JS = """
document.addEventListener('click', function (e) {
  var b = e.target.closest('.rawbtn'); if (!b) return;
  var on = b.getAttribute('aria-pressed') !== 'true';
  b.setAttribute('aria-pressed', on ? 'true' : 'false');
  b.textContent = on ? '덮은 결과 보기' : '덮기 전 보기';
  b.closest('.group').querySelectorAll('img[data-final]').forEach(function (im) {
    var k = on ? im.dataset.raw : im.dataset.final;
    im.src = (typeof IMG !== 'undefined') ? IMG[k] : 'img_exp4/' + k + '.jpg';
  });
});
document.addEventListener('click', function (e) {
  var img = e.target.closest('.frame img, .refbox img');
  if (!img) return;
  var z = document.createElement('div'); z.className = 'zoom'; z.innerHTML = '<img alt="">';
  z.firstChild.src = img.src; z.addEventListener('click', function () { z.remove(); }); document.body.appendChild(z);
});
document.addEventListener('keydown', function (e) { if (e.key === 'Escape') document.querySelectorAll('.zoom').forEach(function (z) { z.remove(); }); });
"""


def img(key, alt, src, alt_key=None):
    extra = f' data-final="{key}" data-raw="{alt_key}"' if alt_key else ""
    return f'<img {src(key)}{extra} alt="{html.escape(alt)}">'


def page(src, notice, extra_js=""):
    e = html.escape
    nav = "".join(f'<a href="#{p["id"]}-{s[0]}">{p["id"]} {s[1]}</a>' for p in PRODUCTS for s in p["scenes"])
    body = []
    for p in PRODUCTS:
        body.append(f'<section class="product" id="{p["id"]}"><div class="phead"><div><h2>{e(p["title"])}</h2>'
                    f'<p class="sub">{e(p["sub"])}</p><ul>' + "".join(f"<li>{e(c)}</li>" for c in p["check"]) +
                    '</ul><p class="sub" style="margin:8px 0 0">위 목록은 채점 체크리스트를 요약한 것이다.</p></div>'
                    f'<div class="refbox">{img("in_P3_1", "판매자 사진", src)}판매자 사진 1장<br>500x500, 배경 투명</div></div>')
        for sid, sname, req, a_note, b_note in p["scenes"]:
            ins = "".join(f'<figure><div class="frame">{img(k, "입력 사진", src)}</div><figcaption>입력 {i}</figcaption></figure>'
                          for i, k in enumerate(p["inputs"], 1))
            a = "".join(f'<figure><div class="frame">{img(f"{p["id"]}_{sid}_A_s{s}", f"A {s}번", src)}</div><figcaption>A s{s}</figcaption></figure>' for s in (1, 2))
            b = "".join(f'<figure><div class="frame">{img(f"{p["id"]}_{sid}_B_s{s}", f"B {s}번", src, alt_key=f"{p["id"]}_{sid}_B_s{s}_raw")}</div><figcaption>B s{s}</figcaption></figure>' for s in (1, 2))
            a_txt = f'<p class="note"><b>예건 메모</b> {e(a_note)}</p>' if a_note else ""
            body.append(
                f'<div class="scene" id="{p["id"]}-{sid}"><h3>{p["id"]} {e(sname)}</h3><p class="req">요청 문장 <q>{e(req)}</q></p>'
                f'<div class="grid"><div class="group real"><span class="tag">원본 제품 사진</span><div class="stack">{ins}</div></div>'
                f'<div class="group a"><span class="tag">A 범용 GPT (그대로 넣음)</span><div class="pair">{a}</div>{a_txt}</div>'
                f'<div class="group b"><div class="btop"><span class="tag">B 우리 시스템</span><button type="button" class="rawbtn" aria-pressed="false">덮기 전 보기</button></div><div class="pair">{b}</div>'
                f'<p class="note"><b>고른 후보</b> {e(pick_line(p["id"], sid))}</p></div></div></div>')
        body.append("</section>")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>제품 변형 실험 3 A와 B 비교</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>제품 변형 실험 3: 사람이 들어간 장면, A와 B 비교</h1>
<p>같은 입력 사진과 같은 요청 문장으로 만든 결과다. A는 사진과 문장을 GPT 이미지 모델(gpt-image-2, 1024px, medium)에 그대로 넣었고, 사람이 주인공이고 제품이 작게 나오는 장면이다. 정찰에서 범용 AI가 이런 장면에서 로고를 깨뜨리거나 제품을 두 개로 그렸다. B는 장면 문장 뒤에 고정 지시를 붙이고, 후보 4장을 검수(로고 글자, 제품 개수, 라벨 대조)로 골라 2장을 남긴 뒤, 실물 라벨을 생성된 병의 각도와 기울기에 맞춰 덮었다. 덮은 뒤 실물과 덜 닮으면 되돌렸다. 사진을 누르면 크게 볼 수 있다.</p>
<div class="legend"><span><i style="background: var(--real)"></i>원본 제품 사진: 판매자가 올린 입력</span><span><i style="background: var(--a)"></i>A: 그대로 넣은 결과</span><span><i style="background: var(--b)"></i>B: 우리 시스템 (유석)</span></div>
<p class="notice">{e(notice)}</p></header>
<nav>{nav}</nav>{''.join(body)}</div><script>{extra_js}{JS}</script></body></html>"""


def main():
    keys = sorted(p.stem for p in IMG.glob("*.jpg"))
    local = page(lambda k: f'src="img_exp4/{k}.jpg"',
                 "A와 B 이름이 보이는 확인용 페이지다. A와 B 모두 이 세션에서 같은 기준 스크립트 방식으로 생성했다. 2026년 11월 17일 결과.")
    (OUT / "creazy_exp4.html").write_text(local, encoding="utf-8")
    table = "var IMG = {" + ",".join(f'"{k}":"data:image/jpeg;base64,{base64.b64encode((IMG / f"{k}.jpg").read_bytes()).decode()}"' for k in keys) + "};"
    loader = "document.querySelectorAll('img[data-img]').forEach(function (im) { im.src = IMG[im.dataset.img]; });"
    team = page(lambda k: f'data-img="{k}"',
                "팀 내부용. A와 B 이름이 보이는 확인용이며, 블라인드 채점은 이름을 가린 파일로 따로 한다. 외부 서비스에 올리지 않는다. 2026년 11월 17일 결과.",
                extra_js=table + loader)
    (OUT / "creazy_exp4_team.html").write_text(team, encoding="utf-8")
    print("written", len(keys), "images;", round((OUT / "creazy_exp4_team.html").stat().st_size / 1e6, 2), "MB team file")


if __name__ == "__main__":
    main()

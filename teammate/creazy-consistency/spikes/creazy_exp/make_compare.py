"""Comparison page for the experiment: seller's input photos | A (plain GPT) | B (our system).

  python spikes/creazy_exp/make_compare.py
writes out/compare/creazy_exp.html (images from out/compare/img_exp/) and
out/compare/creazy_exp_team.html (one file, images inlined, for sending to the team).
"""
import base64, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "compare"
IMG = OUT / "img_exp"
PICKS = json.loads((ROOT / "out" / "creazy_exp_run" / "B" / "picks.json").read_text())

PRODUCTS = [
    {"id": "P1", "title": "P1 참기름 350ml", "sub": "입력: 정면 사진 1장 (입력이 부족한 판매자)",
     "inputs": ["in_P1_1"], "ref": "ref_P1",
     "check": ["빨간 뚜껑 모양과 색", "가늘고 긴 병 목, 둥근 어깨 등 병 모양", "갈색 네모 로고 안의 세로 글자와 물방울 그림",
               "큰 글자 '참기름'", "작은 문구와 '350ml'이 뭉개지지 않음", "손과 그릇 대비 병 크기", "기름이 따라지는 모습의 물리"],
     "scenes": [
         ("use", "사용 장면", "이 참기름을 비빔밥 위에 손으로 따르는 장면 만들어줘. 상세페이지에 쓸 거야",
          "로고 글자 순서가 섞이고 작은 글자가 뭉개짐.",
          "병을 눕혀 들어 라벨도 옆으로 누움. 돌려 읽으니 고른 2장 모두 기준 글자 4개가 읽힘."),
         ("angle", "새 각도", "이 참기름병을 뒤쪽에서 비스듬히 내려다본 각도로 부엌 조리대 위에 놓인 연출컷 만들어줘",
          "2장 모두 정면으로 그림. 사진이 정면뿐이라 각도를 못 바꿈.",
          "후보 4장이 모두 정면. 3D 안내 그림 없이 문장만으로는 B도 각도를 못 바꿈."),
         ("closeup", "클로즈업", "이 참기름 라벨 부분 클로즈업 컷 만들어줘",
          "거의 정확.",
          "후보 4장 모두 기준 글자 4개가 읽힘. 프린트 대조 점수가 높은 2장을 고름."),
     ]},
    {"id": "P2", "title": "P2 차량용 휴대폰 거치대", "sub": "입력: 전체 사진 + 팔 부분 사진 2장 (사진이 있어도 각도가 바뀌면 변형되는지)",
     "inputs": ["in_P2_1", "in_P2_2"], "ref": "ref_P2",
     "check": ["판 가운데 은색 원형 버튼 1개", "로고 글자가 정확함", "좌우 집게 2개와 아래 받침 2개(개수 그대로)",
               "팔 위쪽 조임 노브 1개", "늘어나는 팔, 꺾이는 관절 1곳, 끝 볼 조인트(없는 관절 추가 없음)",
               "원형 흡착판 받침 모양", "휴대폰, 차 내부 대비 크기"],
     "scenes": [
         ("use", "사용 장면", "운전 중에 이 거치대에 휴대폰 꽂아서 내비 보는 장면 만들어줘",
          "",
          "s1은 휴대폰이 거치대 판 뒤에 끼워진 컷. 검수가 버튼과 로고가 잘 보이는 컷을 높게 쳐서 골랐음(알려진 문제, 규칙은 바꾸지 않음)."),
         ("angle", "새 각도", "이 거치대를 옆에서 본 각도로, 팔을 길게 뺀 상태로 차 대시보드에 붙어 있는 연출컷 만들어줘",
          "",
          "후보 4장 모두 옆 시점에 팔을 뺀 모습. 프린트 대조 점수로 2장을 고름."),
         ("closeup", "클로즈업", "휴대폰 잡아주는 집게 부분 클로즈업 컷 만들어줘",
          "거의 정확.",
          "후보 4장 모두 판 앞면이 보임. 프린트 대조 점수가 높은 2장을 고름."),
         ("back", "뒷면", "이 거치대를 뒤쪽에서 본 모습으로, 차 앞유리에 붙어 있는 연출컷 만들어줘",
          "원본에 없는 꺾이는 부분이 생기고 노브 위치가 바뀜.",
          "뒷면은 입력 어디에도 없어 B도 지어냄. 뒷면에는 대조할 프린트가 없어 검수 점수가 모두 낮고, 고르는 근거가 약함."),
     ]},
]


def pick_line(pid, sid):
    rows = [r for r in PICKS[f"{pid}_{sid}"] if r["kept"]]
    return " / ".join(f"s{i}: 후보 {r['file'].rsplit('_c', 1)[1].split('.')[0]}번" for i, r in enumerate(rows, 1))


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
.phead .refbox img { width: 180px; max-width: 100%; border-radius: 8px; border: 1px solid var(--line); cursor: zoom-in; display: block; margin-bottom: 4px; }
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
.zoom { position: fixed; inset: 0; background: rgba(10, 10, 10, .9); display: flex; align-items: center; justify-content: center; padding: 20px; z-index: 10; cursor: zoom-out; }
.zoom img { max-width: 100%; max-height: 100%; border-radius: 6px; }
@media (max-width: 640px) { .grid { grid-template-columns: minmax(0, 1fr); } .phead { grid-template-columns: minmax(0, 1fr); } .phead ul { columns: 1; }
  .real .stack { grid-template-columns: repeat(2, minmax(0, 140px)); } }
"""

JS = """
document.addEventListener('click', function (e) {
  var img = e.target.closest('.frame img, .refbox img');
  if (!img) return;
  var z = document.createElement('div'); z.className = 'zoom'; z.innerHTML = '<img alt="">';
  z.firstChild.src = img.src; z.addEventListener('click', function () { z.remove(); }); document.body.appendChild(z);
});
document.addEventListener('keydown', function (e) { if (e.key === 'Escape') document.querySelectorAll('.zoom').forEach(function (z) { z.remove(); }); });
"""


def img(key, alt, src):
    return f'<img {src(key)} alt="{html.escape(alt)}">'


def page(src, notice, extra_js=""):
    e = html.escape
    nav = "".join(f'<a href="#{p["id"]}-{s[0]}">{p["id"]} {s[1]}</a>' for p in PRODUCTS for s in p["scenes"])
    body = []
    for p in PRODUCTS:
        body.append(f'<section class="product" id="{p["id"]}"><div class="phead"><div><h2>{e(p["title"])}</h2>'
                    f'<p class="sub">{e(p["sub"])}</p><ul>' + "".join(f"<li>{e(c)}</li>" for c in p["check"]) +
                    '</ul><p class="sub" style="margin:8px 0 0">위 목록은 채점 체크리스트를 요약한 것이다.</p></div>'
                    f'<div class="refbox">{img(p["ref"], "브랜드 실제 연출 사진", src)}브랜드 실제 사진<br>채점 기준용, 모델에 넣지 않음</div></div>')
        for sid, sname, req, a_note, b_note in p["scenes"]:
            ins = "".join(f'<figure><div class="frame">{img(k, "입력 사진", src)}</div><figcaption>입력 {i}</figcaption></figure>'
                          for i, k in enumerate(p["inputs"], 1))
            a = "".join(f'<figure><div class="frame">{img(f"{p["id"]}_{sid}_A_s{s}", f"A {s}번", src)}</div><figcaption>A s{s}</figcaption></figure>' for s in (1, 2))
            b = "".join(f'<figure><div class="frame">{img(f"{p["id"]}_{sid}_B_s{s}", f"B {s}번", src)}</div><figcaption>B s{s}</figcaption></figure>' for s in (1, 2))
            a_txt = f'<p class="note"><b>예건 메모</b> {e(a_note)}</p>' if a_note else ""
            body.append(
                f'<div class="scene" id="{p["id"]}-{sid}"><h3>{p["id"]} {e(sname)}</h3><p class="req">요청 문장 <q>{e(req)}</q></p>'
                f'<div class="grid"><div class="group real"><span class="tag">원본 제품 사진</span><div class="stack">{ins}</div></div>'
                f'<div class="group a"><span class="tag">A 범용 GPT (그대로 넣음)</span><div class="pair">{a}</div>{a_txt}</div>'
                f'<div class="group b"><span class="tag">B 우리 시스템</span><div class="pair">{b}</div>'
                f'<p class="note"><b>고른 후보</b> {e(pick_line(p["id"], sid))}. {e(b_note)}</p></div></div></div>')
        body.append("</section>")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>제품 변형 실험 A와 B 비교</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>제품 변형 실험: A와 B 비교</h1>
<p>같은 입력 사진과 같은 요청 문장으로 만든 결과다. A는 사진과 문장을 GPT 이미지 모델(gpt-image-2, 1024px, medium)에 그대로 넣었고, B는 같은 조건에서 우리 단계를 더했다. 장면 문장 뒤에 고정 지시를 붙였고, 후보 4장을 뽑아 검수로 2장을 골랐다. 실물 덮기와 3D 안내 그림은 이번 결과에 쓰이지 않았다. 사진을 누르면 크게 볼 수 있다.</p>
<div class="legend"><span><i style="background: var(--real)"></i>원본 제품 사진: 판매자가 올린 입력</span><span><i style="background: var(--a)"></i>A: 그대로 넣은 결과 (예건)</span><span><i style="background: var(--b)"></i>B: 우리 시스템 (유석)</span></div>
<p class="notice">{e(notice)}</p></header>
<nav>{nav}</nav>{''.join(body)}</div><script>{extra_js}{JS}</script></body></html>"""


def main():
    keys = sorted(p.stem for p in IMG.glob("*.jpg"))
    local = page(lambda k: f'src="img_exp/{k}.jpg"',
                 "A와 B 이름이 보이는 확인용 페이지다. 블라인드 채점은 이름을 가린 파일로 따로 한다. 2026년 11월 17일 결과.")
    (OUT / "creazy_exp.html").write_text(local, encoding="utf-8")
    table = "var IMG = {" + ",".join(f'"{k}":"data:image/jpeg;base64,{base64.b64encode((IMG / f"{k}.jpg").read_bytes()).decode()}"' for k in keys) + "};"
    loader = "document.querySelectorAll('img[data-img]').forEach(function (im) { im.src = IMG[im.dataset.img]; });"
    team = page(lambda k: f'data-img="{k}"',
                "팀 내부용. A와 B 이름이 보이는 확인용이며, 블라인드 채점은 이름을 가린 파일로 따로 한다. 외부 서비스에 올리지 않는다. 2026년 11월 17일 결과.",
                extra_js=table + loader)
    (OUT / "creazy_exp_team.html").write_text(team, encoding="utf-8")
    print("written", len(keys), "images;", round((OUT / "creazy_exp_team.html").stat().st_size / 1e6, 2), "MB team file")


if __name__ == "__main__":
    main()

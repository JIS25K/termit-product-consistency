"""ChatGPT kit for experiment 5: what to attach and paste for A and B, scene by scene, and where to
save the results.

  python spikes/creazy_exp/make_kit.py
writes out/compare/chatgpt_kit/ (kit.html, the product photo, guide images, prompt files and empty
result folders). After the user saves the images, collect_kit() moves them into
out/creazy_exp5/outputs/A and out/creazy_exp5/B/candidates with the experiment's names.
"""
import html, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen2  # noqa: E402

EXP = ROOT / "out" / "creazy_exp5"
KIT = ROOT / "out" / "compare" / "chatgpt_kit"
NAMES = {"grippump": "펌프누르기", "holdface": "얼굴옆광고컷", "showcam": "카메라로내밀기", "angle30": "30도비스듬히", "angle45": "45도비스듬히"}
N_A, N_B = 2, 4


def build():
    gen2.EXP = EXP
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    KIT.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXP / "inputs" / "P3" / "1.png", KIT / "제품사진.png")
    cards = []
    for i, t in enumerate(cfg["products"][0]["targets"], 1):
        d = KIT / f"{i:02d}_{NAMES[t['id']]}"
        (d / "A_결과").mkdir(parents=True, exist_ok=True)
        (d / "B_결과").mkdir(parents=True, exist_ok=True)
        _, a_prompt = gen2.job_inputs("A", cfg["products"][0], t)
        b_paths, b_prompt = gen2.job_inputs("B", cfg["products"][0], t)
        (d / "A_문장.txt").write_text(a_prompt, encoding="utf-8")
        (d / "B_문장.txt").write_text(b_prompt, encoding="utf-8")
        guide = None
        if len(b_paths) > 1:
            guide = f"{d.name}/B_안내그림.png"
            shutil.copy(b_paths[1], KIT / guide)
        cards.append((i, t, d.name, a_prompt, b_prompt, guide))
    e = html.escape
    sections = []
    for i, t, dn, ap, bp, guide in cards:
        b_attach = '<img src="제품사진.png" alt="제품 사진">' + (f'<img src="{guide}" alt="3D 안내 그림">' if guide else "")
        b_files = "제품사진.png" + (f", 그다음 {dn}/B_안내그림.png" if guide else "")
        sections.append(f"""
<section class="scene" id="s{i}"><h2>{i}. {e(NAMES[t['id']])}</h2>
<div class="cols">
 <div class="col a"><h3>A 범용 (그대로) <span>{N_A}장</span></h3>
  <ol><li><b>A 전용 채팅</b>에서 이어서 한다(1번 장면 때 새 임시 채팅을 하나 열고, 5번까지 그 채팅을 쓴다).</li><li><b>제품사진.png</b> 한 장만 첨부한다. 장면마다 다시 첨부한다.</li>
  <li>아래 문장을 그대로 붙여 넣고 보낸다.</li><li>나온 그림을 저장하고, <b>다시 생성</b>을 눌러 한 장 더 저장한다.</li>
  <li>저장 위치: <code>{dn}/A_결과/</code> (파일 이름은 아무거나)</li></ol>
  <div class="attach"><img src="제품사진.png" alt="제품 사진"></div>
  <div class="prompt"><pre id="a{i}">{e(ap)}</pre><button type="button" data-copy="a{i}">문장 복사</button></div>
 </div>
 <div class="col b"><h3>B 우리 엔진 <span>{N_B}장</span></h3>
  <ol><li><b>B 전용 채팅</b>에서 이어서 한다(1번 장면 때 A와 다른 새 임시 채팅을 하나 열고, 5번까지 그 채팅을 쓴다).</li><li>{e(b_files)} 순서로 첨부한다. 장면마다 다시 첨부한다.</li>
  <li>아래 문장을 그대로 붙여 넣고 보낸다.</li><li>나온 그림을 저장하고, <b>다시 생성</b>을 3번 더 눌러 모두 {N_B}장 저장한다.</li>
  <li>저장 위치: <code>{dn}/B_결과/</code> (파일 이름은 아무거나)</li></ol>
  <div class="attach">{b_attach}</div>
  <div class="prompt"><pre id="b{i}">{e(bp)}</pre><button type="button" data-copy="b{i}">문장 복사</button></div>
 </div>
</div></section>""")
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChatGPT 생성 준비물</title><style>
:root {{ --bg:#f4f3f1; --card:#fff; --fg:#1f1d1b; --muted:#6b6661; --line:#e2dfdb; --a:#b0413e; --a-soft:#f6e4e3; --b:#2f6f5e; --b-soft:#e1efe9; --code:#efedea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#181716; --card:#222120; --fg:#ecebe9; --muted:#a29d97; --line:#353331; --a:#e07a76; --a-soft:#3a2524; --b:#7cc4ad; --b-soft:#1f322c; --code:#2c2a28; color-scheme: dark; }} }}
* {{ box-sizing: border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 "Pretendard","Apple SD Gothic Neo",system-ui,sans-serif; word-break: keep-all; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:28px max(16px, env(safe-area-inset-right, 0px)) 60px max(16px, env(safe-area-inset-left, 0px)); }}
h1 {{ margin:0 0 6px; font-size:24px; }} .lead {{ color:var(--muted); margin:0 0 6px; max-width:80ch; }}
.rules {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin:16px 0 24px; }}
.rules li {{ margin:2px 0; }}
.scene {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; margin-bottom:18px; }}
.scene h2 {{ margin:0 0 10px; font-size:18px; }}
.cols {{ display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:16px; }}
.col h3 {{ margin:0 0 6px; font-size:15px; }} .col h3 span {{ font-size:12px; padding:1px 8px; border-radius:6px; margin-left:6px; }}
.a h3 span {{ background:var(--a-soft); color:var(--a); }} .b h3 span {{ background:var(--b-soft); color:var(--b); }}
.col ol {{ margin:0 0 10px; padding-left:1.2em; font-size:14px; }}
.attach {{ display:flex; gap:8px; margin-bottom:8px; }} .attach img {{ height:110px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
.prompt {{ position:relative; }} pre {{ background:var(--code); border-radius:8px; padding:10px 12px; white-space:pre-wrap; font-size:13px; margin:0; max-height:240px; overflow:auto; }}
button {{ margin-top:6px; border:1px solid var(--line); background:var(--card); color:var(--fg); font:inherit; font-size:13px; padding:4px 12px; border-radius:6px; cursor:pointer; }}
button:focus-visible {{ outline:2px solid var(--b); outline-offset:2px; }}
code {{ background:var(--code); padding:1px 5px; border-radius:4px; font-size:12.5px; }}
@media (max-width:760px) {{ .cols {{ grid-template-columns: minmax(0,1fr); }} }}
</style></head><body><div class="wrap">
<h1>ChatGPT 생성 준비물</h1>
<p class="lead">장면마다 A(범용, 그대로) {N_A}장과 B(우리 엔진) {N_B}장을 만든다. 장면 {len(cards)}개, 모두 {len(cards) * (N_A + N_B)}장. 저장한 뒤 알려 주면 검수, 고르기, 실물 덮기, 비교 페이지까지 이어서 처리한다.</p>
<div class="rules"><b>공통 규칙</b><ul>
<li>채팅은 두 개만 쓴다. <b>A 전용 임시 채팅 하나</b>에서 1~5번 장면의 A를, <b>B 전용 임시 채팅 하나</b>에서 1~5번 장면의 B를 순서대로 만든다. A와 B를 같은 채팅에 섞지 않는다.</li>
<li>장면마다 <b>제품 사진을 다시 첨부</b>한다. 앞에서 그린 그림이 아니라 원본 사진을 보게 하려는 것이다.</li>
<li>"다시 생성"은 방금 장면의 그림만 다시 뽑는다. 장수를 채운 뒤 다음 장면으로 넘어간다.</li>
<li>문장은 <b>복사한 그대로</b> 붙여 넣는다. 고치거나 덧붙이지 않는다.</li>
<li>첨부 순서를 지킨다: 제품 사진이 먼저, 안내 그림이 있으면 그다음.</li>
<li>ChatGPT가 질문을 되묻거나 그림 대신 글로 답하면, "그대로 만들어줘"라고만 보낸다.</li>
<li>저장은 그림을 크게 연 상태에서 원본으로 받는다. 파일 이름은 신경 쓰지 않아도 된다.</li>
</ul></div>
{''.join(sections)}
</div><script>
document.addEventListener('click', function (e) {{
  var b = e.target.closest('button[data-copy]'); if (!b) return;
  var t = document.getElementById(b.dataset.copy).textContent;
  var done = function () {{ b.textContent = '복사됨'; setTimeout(function () {{ b.textContent = '문장 복사'; }}, 1500); }};
  if (navigator.clipboard) navigator.clipboard.writeText(t).then(done, function () {{ sel(b); }}); else sel(b);
  function sel(btn) {{ var r = document.createRange(); r.selectNodeContents(document.getElementById(btn.dataset.copy)); var s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = '선택됨, Cmd+C'; }}
}});
</script></body></html>"""
    (KIT / "kit.html").write_text(page, encoding="utf-8")
    return cards


def collect_kit():
    """Move saved images into the experiment folders with the experiment's names (by save order)."""
    cfg = json.loads((EXP / "experiment.json").read_text(encoding="utf-8"))
    report = {}
    for i, t in enumerate(cfg["products"][0]["targets"], 1):
        d = KIT / f"{i:02d}_{NAMES[t['id']]}"
        for cond, sub, n in (("A", "A_결과", N_A), ("B", "B_결과", N_B)):
            files = sorted([p for p in (d / sub).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")], key=lambda p: p.stat().st_mtime)
            report[f"{t['id']}_{cond}"] = len(files)
            for k, f in enumerate(files[:n], 1):
                from PIL import Image
                im = Image.open(f).convert("RGB")
                dst = (EXP / "outputs" / "A" / f"P3_{t['id']}_A_s{k}.png") if cond == "A" else (EXP / "B" / "candidates" / f"P3_{t['id']}_c{k}.png")
                im.save(dst)
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        print(collect_kit())
    else:
        for i, t, dn, ap, bp, guide in build():
            print(i, dn, "guide" if guide else "")

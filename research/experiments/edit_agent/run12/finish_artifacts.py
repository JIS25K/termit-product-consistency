"""Validate run12 execution and build a static view of original images."""
import json, hashlib, html
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads((ROOT / path).read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = {}
for manifest in ('frozen_sha256.json', 'C_execution_sha256.json', 'C_eval_sha256.json'):
    for path, expected in read(manifest).items():
        checks[path] = digest(ROOT / path) == expected
assert all(checks.values()), checks
calls = read('outputs/image_calls.json')
assert len(calls) == 14
assert len({c['dest'] for c in calls}) == 14
images = []
for call in calls:
    path = Path(call['dest'])
    with Image.open(path) as im:
        im.load()
        size = list(im.size)
    assert path.is_file() and all(Path(p).is_file() for p in call['refs'])
    if 'plan_file' in call:
        assert digest(Path(call['plan_file'])) == call['plan_sha256']
    images.append({'path':str(path.relative_to(ROOT)), 'sha256':digest(path),'size':size})
assert len({i['sha256'] for i in images}) == 14
for step in (1, 2, 3):
    jobs = read(f'plans/step{step}_jobs.json')
    assert len(jobs) == 4
    for j in jobs:
        expected_base = ROOT / 'images' / j['product'] / ('shared' if step == 1 else j['arm']) / f'v{step-1}.png'
        assert Path(j['refs'][0]) == expected_base
        matching = [c for c in calls if c['dest'] == j['dest']]
        assert len(matching) == 1
        assert matching[0]['prompt'] == j['prompt'] and matching[0]['refs'] == j['refs']
        assert len(j['refs']) <= 4

parts = ['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>run12 · 편집 에이전트 비교</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f3ee;color:#172927;margin:0;padding:28px;max-width:1600px;margin:auto}h1{font-size:28px}h2{margin-top:40px}p{max-width:1000px}table{width:100%;border-collapse:collapse;table-layout:fixed;background:white}td,th{padding:10px;border:1px solid #ddd;vertical-align:top;text-align:left}th:first-child{width:85px}img{width:100%;display:block}a{color:#15615b}small{color:#52615c}.refs{display:flex;gap:16px;max-width:550px;margin:16px 0}.refs figure{margin:0;width:50%}details{margin:20px 0}.notice{padding:14px;background:#e6eee7;border-radius:8px}@media(max-width:700px){body{padding:12px}table{min-width:780px}.scroll{overflow:auto}}</style>
<h1>수정을 이어갈 때, 상태 장부가 도움이 됐나?</h1>
<p>일반 방식과 명시적 상태 관리 방식 모두 전체 요청 이력과 과거 이미지에 접근했습니다. 각 단계는 한 번만 생성했습니다. 아래 이미지를 누르면 원본을 엽니다.</p>
<p class="notice">평가 결과는 <a href="결과_보고서.md">통합 보고서</a>에 기록했습니다. 작은 글자·픽셀 차이는 제외하고 요청 이행과 주요 제품 보존을 따로 평가합니다.</p>''']
protocol=read('frozen/protocol.json')
for product,label in [('character','캐릭터'),('bottle','식혜병')]:
    parts.append(f'<h2>{label}</h2><div class="refs">')
    for key,desc in [('front','실제 정면'),('side','실제 목표 측면')]:
        # masked_review holds byte-identical copies, without external file dependencies.
        path=f'masked_review/{product}/actual_{key}.jpg'
        assert (ROOT/path).is_file()
        parts.append(f'<figure><a href="{path}"><img src="{path}" alt="{label} {desc}"></a><figcaption>{desc}</figcaption></figure>')
    parts.append('</div><div class="scroll"><table><thead><tr><th>방식</th><th>v0 · 공통 출발<br><small>정면 / 밝은 카페 / 따뜻한 빛</small></th><th>v1 · 측면 회전<br><small>배경·조명 유지</small></th><th>v2 · 저녁으로 변경<br><small>제품도 차가운 푸른빛</small></th><th>v3 · 조명만 복원<br><small>측면·어두운 카페 유지</small></th></tr></thead><tbody>')
    for arm,arm_label in [('baseline','일반 방식'),('agent','상태 관리')]:
        parts.append(f'<tr><th>{arm_label}</th>')
        for step in range(4):
            path=f'images/{product}/{"shared" if step==0 else arm}/v{step}.png'
            assert (ROOT/path).is_file()
            parts.append(f'<td><a href="{path}"><img src="{path}" alt="{label} {arm_label} v{step}"></a></td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
parts.append('<details><summary>조건과 원본 기록</summary><p><a href="실험설계.md">설계</a> · <a href="frozen/protocol.json">고정 조건</a> · <a href="outputs/image_calls.json">실제 프롬프트·입력·호출 기록</a> · <a href="outputs/AB_결과.md">A/B 평가</a> · <a href="outputs/evaluation_c.json">C 평가</a></p></details><p><small>두 제품에서 조건별 한 경로씩 실행한 탐색 실험입니다. 실제 크리에이지 서비스에서 실행하지 않았습니다. 이미지는 가공 없이 원본 파일을 표시합니다.</small></p></html>')
(ROOT/'비교.html').write_text('\n'.join(parts))
result={'status':'execution_validated','image_calls':14,'unique_images':14,'edits':12,'shared_initial_images':2,'quality_rerolls':0,'frozen_hash_checks':checks,'image_files':images,'prompt_and_reference_lineage_verified':True,'comparison_html':'비교.html','comparison_validation':'Referenced image paths checked; browser rendering not verified.'}
(ROOT/'outputs/execution_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'images':len(images),'frozen_checks':len(checks),'unique_sizes':sorted({tuple(i['size']) for i in images})},ensure_ascii=False))

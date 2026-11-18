from pathlib import Path
import json,shutil,hashlib
r=Path(__file__).resolve().parent;p=r/'review';p.mkdir(exist_ok=True)
jobs={x['id']:x for x in json.loads((r/'execution_jobs.json').read_text())};order=['agent_D','baseline_C','baseline_D','agent_C'];cases=[];mapping=[];refs={}
def cp(src):
 if src not in refs:
  f=Path(src);dst=p/f'reference_{len(refs)+1:02d}{f.suffix}';shutil.copy2(src,dst);refs[src]=str(dst)
 return refs[src]
for i,jid in enumerate(order,1):
 j=jobs[jid];out=p/f'candidate_{i:02d}.png';shutil.copy2(j['dest'],out)
 cases.append({'id':f'result_{i:02d}','candidate':str(out),'before':cp(j['refs'][0]),'product_references':[cp(s) for s in j['refs'][1:]],'request':'마킹펜의 주요 제품 구조를 원본에 맞게 복구하되 기존 장면·시점·배경·손·동작(있는 경우)을 유지. 작은 인쇄 글자 및 작은 제품명 누락은 허용. 큰 로고·큰 형태·주요부품은 보존. 원본에서 보이지 않는 면이나 물리적 mm 길이는 추측하여 감점하지 않음.'});mapping.append({'review_id':f'result_{i:02d}','job_id':jid})
(p/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));(r/'private/review_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2));(r/'review_sha256.json').write_text(json.dumps({str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()},indent=2))
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>새 수정 결과 4건</title><style>body{background:#f4f5f7;color:#182231;font:17px/1.6 system-ui}main{max-width:1120px;margin:auto;padding:24px}section{background:white;border:1px solid #ccd4df;border-radius:12px;margin:24px 0;padding:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;height:430px;object-fit:contain;background:#f7f7f7}a{color:#155bc2}@media(max-width:650px){.pair{grid-template-columns:1fr}img{height:auto}}</style><main><h1>수정 결과 4건</h1><p>왼쪽 기준 제품과 오른쪽 수정 결과의 큰 형태·주요 부품이 맞는지 확인해 주세요. 작은 글자는 제외합니다. 사진을 누르면 확대됩니다. 방식 이름과 AI 판정은 숨겼습니다.</p>']
for i,c in enumerate(cases,1):
 ref=Path(c['product_references'][0]).as_uri();img=Path(c['candidate']).as_uri();before=Path(c['before']).as_uri();parts +=[f'<section><h2>{i}번 결과</h2><div class="pair"><figure><figcaption>기준 제품</figcaption><a href="{ref}" target="_blank"><img src="{ref}" alt="기준 제품"></a></figure><figure><figcaption>수정 결과</figcaption><a href="{img}" target="_blank"><img src="{img}" alt="{i}번 결과"></a></figure></div><p><a href="{before}" target="_blank">수정 전 보기</a> · '+ ' · '.join(f'<a href="{Path(f).as_uri()}" target="_blank">참조 {j+1}</a>' for j,f in enumerate(c['product_references']))+'</p><p>판정: 통과 / 실패 / 자료 부족</p></section>']
parts.append('<p>대화에 예를 들어 “1 통과, 2 실패, 3 통과, 4 통과”처럼 보내 주세요.</p></main></html>');(r/'새결과_확인.html').write_text('\n'.join(parts))
print('Prepared 4 masked outputs and reference/before images.')

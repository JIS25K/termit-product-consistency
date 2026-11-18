from pathlib import Path
import json,hashlib,collections
r=Path(__file__).resolve().parent
load=lambda p:json.loads((r/p).read_text())
cases=load('public/cases.json');m={x['id']:x for x in load('private/mapping.json')}
a={x['id']:x for x in load('outputs/detection.json')['rows']};b={x['id']:x for x in load('outputs/independent_review.json')['rows']}
ids={x['id'] for x in cases};assert ids==set(a)==set(b)
assert len(a)==len(b)==21
for data in [a,b]:
 for row in data.values():assert row['decision'] in ['pass','fail','uncertain'] and row['evidence']
frozen=load('frozen_sha256.json');assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
rows=[{'id':i,'product':m[i]['product'],'phase':m[i]['phase'],'detector':a[i]['decision'],'reviewer':b[i]['decision'],'agree':a[i]['decision']==b[i]['decision'],'review_required':a[i]['decision']!=b[i]['decision'] or 'uncertain' in [a[i]['decision'],b[i]['decision']]} for i in sorted(ids)]
source_review_ids=['case_10','case_12','case_18']
spotcheck_ids=['case_04','case_15','case_05','case_06','case_09']
for row in rows:
 row['source_incomplete']=row['id'] in source_review_ids
 row['review_required']=row['review_required'] or row['id'] in source_review_ids+spotcheck_ids
result={'source_incomplete_ids':source_review_ids,'spotcheck_ids':spotcheck_ids,'cases':21,'products':3,'detector_counts':dict(collections.Counter(x['decision'] for x in a.values())),'reviewer_counts':dict(collections.Counter(x['decision'] for x in b.values())),'agreement_count':sum(x['agree'] for x in rows),'priority_human_review_count':sum(x['review_required'] for x in rows),'rows':rows,'human_gold_accuracy':None,'reason':'Independent AI agreement only; no image-level human normal/failure gold labels.','frozen_inputs_unchanged':True,'new_images_generated':0,'service_mutations':0}
(r/'outputs/comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# 사람 확인용 — 우선 8건', '', '개별 AI 판정은 표시하지 않았습니다. 원본과 후보를 비교하여 통과·실패·자료 부족을 적어 주세요. 냄비 후속3건은 당시 추가 참조 확인이 먼저 필요합니다.', '']
cs={c['id']:c for c in cases}
for i in source_review_ids+spotcheck_ids:
 c=cs[i];lines += [f'## {i}', '',c['request'],'',' · '.join(f'[원본 {j+1}]({s})' for j,s in enumerate(c['references'])),f'![후보 {i}]({c["candidate"]})','','사람 판정: [ ] 통과 [ ] 실패 [ ] 자료 부족 / 근거 및 올바른 기준 사진:','']
(r/'사람_확인용.md').write_text('\n'.join(lines))

from pathlib import Path
import json,hashlib,shutil
r=Path(__file__).resolve().parent
load=lambda p:json.loads((r/p).read_text())
mp={x['id']:x for x in load('private/mapping.json')};cases={x['id']:x for x in load('frozen/cases.json')};jobs=[];keep=[];scores={}
common='''Use case: precise-object-edit. 이미지 1은 편집할 현재 광고/제품 이미지입니다. 이미지 2부터는 기준 제품 사진입니다. 참조 사진의 광고 문구나 배경을 복제하지 말고 제품 구조만 참고하세요. 현재 이미지의 장면, 시점, 배치, 조명, 배경과 화면 비율을 유지하고 제품의 잘못된 주요 구조만 바로잡으세요. 작은 인쇄 글자는 정확한 복원 대상이 아닙니다. 새 소품, 새 문구, 워터마크를 추가하지 마세요. 결과 한 장만 생성하세요.\n\n수정 지시:\n'''
(r/'images').mkdir(exist_ok=True)
for arm,path in [('baseline','outputs/baseline_plan.json'),('agent','outputs/agent_plan.json')]:
 rows=load(path)['rows'];assert len(rows)==8 and {x['id'] for x in rows}==set(cases)
 scored=[]
 for row in rows:
  m=mp[row['id']];target={'pass':'keep','fail':'repair_once','uncertain':'request_reference'}[m['human_decision']];scored.append({'id':row['id'],'display_id':m['display_id'],'action':row['action'],'expected':target,'matches':row['action']==target})
  if m['display_id'] not in ['A','B','C','D']:continue
  c=cases[row['id']];dest=r/'images'/f'{arm}_{m["display_id"]}{Path(c["candidate"]).suffix if row["action"]=="keep" else ".png"}'
  if row['action']=='keep':
   shutil.copy2(c['candidate'],dest);keep.append({'arm':arm,'display_id':m['display_id'],'source':c['candidate'],'dest':str(dest),'action':'keep'})
  elif row['action']=='repair_once':
   assert row['repair_instruction'];jobs.append({'id':f'{arm}_{m["display_id"]}','arm':arm,'display_id':m['display_id'],'prompt':common+row['repair_instruction'],'refs':[c['candidate']]+c['references'],'dest':str(dest)})
  else:keep.append({'arm':arm,'display_id':m['display_id'],'action':'request_reference','dest':None})
 scores[arm]={'matches':sum(x['matches'] for x in scored),'total':8,'rows':scored,'unnecessary_repairs':sum(x['expected']=='keep' and x['action']=='repair_once' for x in scored),'missed_failures':sum(x['expected']=='repair_once' and x['action']=='keep' for x in scored)}
assert len(jobs)<=8
# interleave by case and alternate arm starting order; fixed before generation
jobs.sort(key=lambda j:(j['display_id'],(0 if j['arm']==('baseline' if j['display_id']=='C' else 'agent') else 1)))
for name,obj in [('execution_jobs.json',jobs),('outputs/kept_outputs.json',keep),('outputs/action_score.json',scores)]: (r/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
(r/'execution_sha256.json').write_text(json.dumps({p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in ['execution_jobs.json','outputs/baseline_plan.json','outputs/agent_plan.json']},indent=2))
print(json.dumps({'action_scores':{k:v['matches'] for k,v in scores.items()},'kept':len(keep),'image_jobs':len(jobs),'job_ids':[x['id'] for x in jobs]},ensure_ascii=False))

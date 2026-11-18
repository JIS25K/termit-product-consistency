from pathlib import Path
import json,sys,hashlib
p=Path(__file__).resolve().parent;step=int(sys.argv[1]);proto=json.loads((p/'frozen/protocol.json').read_text());jobs=[]
for arm in ['baseline','agent']:
 source=p/'plans'/f'{arm}_step{step}_output.json';out=json.loads(source.read_text());inputs=json.loads((p/'plans'/f'{arm}_step{step}_input.json').read_text())
 assert len(out['plans'])==2
 for product in ['character','bottle']:
  m=[x for x in out['plans'] if x['product']==product];assert len(m)==1
  plan=m[0];inp=next(x for x in inputs if x['product']==product)
  assert plan['step']==step
  assert plan['action']=='edit',('not silently overriding model action',plan)
  assert plan['edit_base'] in inp['versions']
  assert 1<=len(plan['references'])<=4
  assert len(plan['prompt'])<=2500
  assert plan['references'][0]['asset_id']==plan['edit_base']
  catalog={k:v['path'] for k,v in inp['versions'].items()}|inp['actual_photos']
  refs=[]
  for ref in plan['references']:
   assert ref['asset_id'] in catalog
   assert str(Path(ref['path']).resolve())==str(Path(catalog[ref['asset_id']]).resolve())
   assert Path(ref['path']).exists();refs.append(ref['path'])
  history='초기 요청: '+inp['initial_request']+'\n'+'\n'.join(f'{i+1}번째 수정 요청: {request}' for i,request in enumerate(inp['history']))
  prompt='전체 사용자 요청 이력:\n'+history+'\n\n현재 단계 실행 지시:\n'+plan['prompt']+'\n\n공통 제약:\n'+proto['track_C']['common_suffix']+' '+inp['product_invariants']
  jobs.append({'product':product,'arm':arm,'step':step,'prompt':prompt,'refs':refs,'dest':str((p/'images'/product/arm/f'v{step}.png').resolve()),'plan_file':str(source.resolve()),'plan_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'action':plan['action']})
# Counterbalance execution order across steps; both arms retain their own result chain.
order=['baseline','agent'] if step%2 else ['agent','baseline']
jobs.sort(key=lambda j:(['character','bottle'].index(j['product']),order.index(j['arm'])))
(p/'plans'/f'step{step}_jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2));print('validated jobs',len(jobs),'step',step)

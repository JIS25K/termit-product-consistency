import json,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
checks={}
for manifest in ['frozen_sha256.json','repair_execution_sha256.json','repair_evaluation_sha256.json']:
    for p,h in read(manifest).items():checks[p]=sha(ROOT/p)==h
assert all(checks.values())
calls=read('outputs/image_calls.json'); jobs=read('frozen/repair_jobs.json')
assert len(calls)==4 and len({c['dest'] for c in calls})==4
assert read('outputs/detection_score.json')['gate_passed']
common=(ROOT/'frozen/repair_common_prompt.txt').read_text()
feedback=next(r['repair_feedback'].strip() for r in read('outputs/detection.json')['rows'] if r['decision']=='fail')
refs=read('frozen/repair_references.json');images=[]
for call in calls:
    expected=next(j for j in jobs if j['arm']==call['arm'] and j['repeat']==call['repeat'])
    assert all(call[k]==expected[k] for k in expected)
    assert call['refs']==refs
    assert call['prompt']==common+('\n\n검수에서 확인된 수정 피드백:\n'+feedback if call['arm']=='feedback_retry' else '')
    assert all(Path(p).is_file() for p in refs)
    path=Path(call['dest'])
    with Image.open(path) as image:image.load();size=list(image.size)
    images.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'size':size})
assert len({i['sha256'] for i in images})==4
mapping=read('private/repair_mask_mapping.json')
review=read('outputs/repair_masked_review.json')
root={(r['arm'],r['repeat']):r for r in read('outputs/repair_root_review.json')['rows']}
rows=[]
assert len(review['rows'])==4 and {r['id'] for r in review['rows']}==set(mapping)
for original in review['rows']:
    r={**original,**mapping[original['id']]}
    assert set(r['checks'])=={'target_side','warm_soft_product_light','dark_cafe','major_product_identity','center_full_product'}
    vals=list(r['checks'].values());assert all(v in ['pass','fail','uncertain'] for v in vals)
    overall='fail' if 'fail' in vals else 'uncertain' if 'uncertain' in vals else 'pass'
    assert overall==r['overall']
    r['root_agrees']=root[(r['arm'],r['repeat'])]['overall']==r['overall']
    rows.append(r)
summary={arm:{label:sum(r['overall']==label for r in rows if r['arm']==arm) for label in ['pass','fail','uncertain']} for arm in ['ordinary_retry','feedback_retry']}
save('outputs/repair_evaluation.json',{'rows':rows,'summary':summary,'method':'Fresh masked model reviewer; arm mapping applied after review saved. Root recorded visual labels independently before masked review. Same model family, no human blind evaluation.','root_agreement':sum(r['root_agrees'] for r in rows)})
save('outputs/execution_validation.json',{'frozen_hash_checks':checks,'same_refs_and_common_prompt':True,'only_prompt_difference':'verbatim detector feedback append','image_calls':4,'technical_failures':0,'quality_rerolls':0,'images':images,'source_reference_hashes':{p:sha(Path(p)) for p in refs},'asset_copies_identical':all(sha(ROOT/p)==v['sha256'] for p,v in read('private/assets.json').items())})
save('completion.json',{'status':'complete','date':'2026-11-17','detection':read('outputs/detection_score.json')['counts'],'repair':summary,'fresh_images':4,'root_masked_agreement':sum(r['root_agrees'] for r in rows),'validation':'Frozen hashes match; all four image files unique and decode; input references and exact prompt-only feedback difference verified.','execution':'Codex model sessions and built-in image_gen, no GPU or external API payment.','limits':['One known failure; 11 correlated normal cases','Only two repair draws per arm; random seeds unavailable','Reference labels and reviewers are models, not human gold standard','No Creage production integration or deployed automatic loop','No measured API token cost or end-to-end production latency'],'artifacts':['결과_보고서.md','실험설계.md','outputs/detection.json','outputs/detection_score.json','outputs/repair_evaluation.json','outputs/image_calls.json','outputs/execution_validation.json']})
print(json.dumps({'detection':read('outputs/detection_score.json')['counts'],'repair':summary,'root_agreement':sum(r['root_agrees'] for r in rows),'image_sizes':[i['size'] for i in images]},ensure_ascii=False))

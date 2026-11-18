import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads((ROOT/p).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in read('frozen_sha256.json').items():
    assert sha(ROOT/p)==h,p
oracle={r['id']:r for r in read('private/oracle.json')}
raw=read('outputs/detection.json')
rows=raw['rows']
assert len(rows)==len(oracle) and {r['id'] for r in rows}==set(oracle)
assert len({r['id'] for r in rows})==len(rows)
counts=dict(true_positive=0,false_negative=0,false_positive=0,true_negative=0,uncertain=0)
details=[]
for r in rows:
    assert r['decision'] in ('pass','fail','uncertain')
    assert set(r['checks'])=={'requested_change','preservation','product_identity'}
    assert all(v in ('pass','fail','uncertain') for v in r['checks'].values())
    assert r['evidence'].strip()
    expected=oracle[r['id']]['expected']
    if r['decision']=='uncertain': outcome='uncertain'
    elif r['decision']=='fail': outcome='true_positive' if expected=='fail' else 'false_positive'
    else: outcome='false_negative' if expected=='fail' else 'true_negative'
    counts[outcome]+=1
    details.append({'id':r['id'],'expected':expected,'actual':r['decision'],'outcome':outcome})
gate=counts==dict(true_positive=1,false_negative=0,false_positive=0,true_negative=11,uncertain=0)
score={'counts':counts,'gate_passed':gate,'rows':details,'detector_sha256':sha(ROOT/'outputs/detection.json'),'scope':'One known positive and 11 related negatives; not held-out general accuracy; oracle from run12 model visual review.'}
(ROOT/'outputs/detection_score.json').write_text(json.dumps(score,ensure_ascii=False,indent=2))
if gate:
    candidate=next(r for r in rows if r['decision']=='fail')
    feedback=candidate['repair_feedback'].strip()
    assert feedback and candidate['violations']
    common=(ROOT/'frozen/repair_common_prompt.txt').read_text()
    refs=read('frozen/repair_references.json')
    jobs=[]
    for arm,rep in [('ordinary_retry',1),('feedback_retry',1),('feedback_retry',2),('ordinary_retry',2)]:
        prompt=common+('\n\n검수에서 확인된 수정 피드백:\n'+feedback if arm=='feedback_retry' else '')
        jobs.append({'arm':arm,'repeat':rep,'prompt':prompt,'refs':refs,'dest':str(ROOT/'images'/f'{arm}_{rep}.png')})
    (ROOT/'frozen/repair_jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2))
    (ROOT/'repair_execution_sha256.json').write_text(json.dumps({'frozen/repair_jobs.json':sha(ROOT/'frozen/repair_jobs.json'),'outputs/detection.json':sha(ROOT/'outputs/detection.json')},indent=2))
print(json.dumps({'counts':counts,'gate_passed':gate},ensure_ascii=False))

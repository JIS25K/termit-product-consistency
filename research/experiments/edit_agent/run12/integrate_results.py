"""Unmask the completed independent review and combine existing evidence."""
import json, hashlib, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads((ROOT/p).read_text())
masked=read('outputs/C_masked_review_final.json')
mapping=read('private/masked_condition_map.json')
root=read('outputs/C_root_review.json')
root_rows={(r['product'],r['arm'],r['step']):r for r in root['rows']}
rows=[]
for original in masked['rows']:
    row=dict(original)
    row['arm']=mapping[row['product']][row['masked_condition']]
    key=(row['product'],row['arm'],row['step'])
    row['root_review_agrees']=root_rows[key]['overall']==row['conservative_overall']
    row['image']=f"images/{row['product']}/{row['arm']}/v{row['step']}.png"
    rows.append(row)
assert len(rows)==12 and len({(r['product'],r['arm'],r['step']) for r in rows})==12
summary={}
for arm in ('baseline','agent'):
    subset=[r for r in rows if r['arm']==arm]
    summary[arm]={metric:{label:sum((r[metric] if metric=='conservative_overall' else r[metric]['label'])==label for r in subset) for label in ('pass','fail','uncertain')} for metric in ('requested_change','preservation','product_identity','conservative_overall')}
evaluation={'method':'Root visual review and independently masked model review; same model family, not human blind evaluation. Masked reviewer had no access to plans, arm mapping, A/B scores or root findings. Mapping applied after review saved.','rows':rows,'summary':summary,'overall_agreement_rows':sum(r['root_review_agrees'] for r in rows),'scope':'Two products, one three-step trajectory per arm. Correlated steps, no statistical causal claim.'}
(ROOT/'outputs/evaluation_c.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2))
required=['실험설계.md','결과_보고서.md','비교.html','outputs/scores_ab.json','outputs/AB_결과.md','outputs/evaluation_c.json','outputs/image_calls.json','outputs/execution_validation.json']
assert all((ROOT/p).is_file() for p in required)
broken=[]
for doc in ('결과_보고서.md','비교.html'):
    text=(ROOT/doc).read_text()
    targets=re.findall(r'(?:href|src)="([^"]+)"',text) if doc.endswith('.html') else re.findall(r'\]\(([^)]+)\)',text)
    for target in targets:
        if target=='completion.json': continue
        if not target.startswith(('http:','https:','#')) and not (ROOT/target).exists():broken.append([doc,target])
assert not broken,broken
completion={'status':'complete','date':'2026-11-17','A_B':'8 shared cases per arm scored on two axes; both 8/8 on both','C':summary,'image_calls':14,'technical_failures':0,'quality_rerolls':0,'root_masked_agreement':evaluation['overall_agreement_rows'],'artifacts':required,'validation':'14 unique PNGs decoded; all 10 frozen hashes match; planner hashes, exact prompts, references and lineage checked; artifact links checked. HTML browser rendering not verified.','not_executed':['production deployment','Creage integration','repeated stochastic trials','automatic detection and repair loop'],'conclusion':'A/B tie; one bottle step succeeded only in state-management arm. Limited observed difference, no causal proof of ledger benefit.'}
(ROOT/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2))
print(json.dumps({'C':summary,'agreement':evaluation['overall_agreement_rows'],'links':'ok'},ensure_ascii=False))

from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    (ROOT/p).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def copy(source,dest):
    source=Path(source);dest=ROOT/dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
    assert sha(source)==sha(dest)
    return str(dest)
refs=read('frozen/repair_references.json')
common={name:copy(p,f'repair_review/ref_{i+1}{Path(p).suffix}') for i,(name,p) in enumerate(zip(['initial_candidate_to_edit','side_and_dark_cafe_reference','warm_light_only_reference','real_product_side'],refs))}
order=[('feedback_retry',2),('ordinary_retry',1),('ordinary_retry',2),('feedback_retry',1)]
cases=[];mapping={}
for i,(arm,rep) in enumerate(order,1):
    ident=f'candidate_{i:02d}'
    cases.append({'id':ident,'image':copy(ROOT/'images'/f'{arm}_{rep}.png',f'repair_review/{ident}.png')})
    mapping[ident]={'arm':arm,'repeat':rep}
save('repair_review/cases.json',{'goal':'실제 목표 측면, 따뜻하고 부드러운 제품 조명, 어두운 카페, 주요 제품 형태·디자인, 중앙 전체 구도','references':common,'candidates':cases})
save('private/repair_mask_mapping.json',mapping)
print('Prepared four masked candidates and four common references')

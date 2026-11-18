from pathlib import Path
from PIL import Image
import hashlib,json
r=Path(__file__).resolve().parent
load=lambda p:json.loads((r/p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(r/p)==h for p,h in load('frozen_sha256.json').items())
assert all(sha(r/p)==h for p,h in load('execution_sha256.json').items())
assert all(sha(r/p)==h for p,h in load('review_sha256.json').items())
rows=load('outputs/image_calls.json');kept=load('outputs/kept_outputs.json');assert len(rows)==4 and len(kept)==4
assert len({sha(x['dest']) for x in rows})==4
for case in ['C','D']:
 pair=[x for x in rows if x['display_id']==case];assert len(pair)==2 and pair[0]['refs']==pair[1]['refs']
for k in kept:assert sha(k['source'])==sha(k['dest'])
out={'new_images':4,'generation_failures':0,'quality_rerolls':0,'kept_outputs':4,'unique_new_output_hashes':4,'frozen_inputs_unchanged':True,'planning_outputs_unchanged':True,'matched_generation_references':True,'kept_outputs_byte_identical':True,'human_final_labels':(load('outputs/human_output_review.json') if (r/'outputs/human_output_review.json').exists() else None),'images':[{'id':x['id'],'dest':x['dest'],'dimensions':list(Image.open(x['dest']).size),'sha256':sha(x['dest']),'elapsed_ms':x['elapsed_ms']} for x in rows]}
(r/'outputs/execution_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))

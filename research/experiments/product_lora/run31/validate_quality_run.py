from pathlib import Path
import json,hashlib,math
from PIL import Image
import numpy as np
r=Path(__file__).resolve().parent;checks={}
for name in ['training_frozen.json','comparison_frozen.json']:
 f=json.loads((r/'outputs'/name).read_text());checks[name]=all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in f['files'].items())
summary=json.loads((r/'outputs/train/summary.json').read_text());checks['200_steps_completed']=summary['steps']==200;checks['finite_adapter']=summary['nonfinite_adapter_tensors']==0;checks['adapter_updated']=summary['nonzero_lora_B_elements']>0
m=json.loads((r/'data/manifest.json').read_text());train=[x for x in m['files'] if x['split']=='train'];held=[x for x in m['files'] if x['split']=='heldout'];checks['fixed_photo_counts']=len(train)==12 and len(held)==2;checks['no_identical_holdout']=not bool({x['sha256'] for x in train}&{x['sha256'] for x in held})
for arm in ['base','lora','ordinary']:
 paths=[r/f'outputs/{arm}/{v}_{i}.png' for v in ['A','B'] for i in [1,2]];checks[arm+'_four_outputs']=all(p.is_file() for p in paths)
 hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];checks[arm+'_four_distinct_outputs']=len(set(hashes))==4
 checks[arm+'_nonblank_and_equal_canvas']=all(Image.open(p).size==(768,768) and float(np.array(Image.open(p)).std())>5 for p in paths)
base=json.loads((r/'outputs/base/manifest.json').read_text());lora=json.loads((r/'outputs/lora/manifest.json').read_text());checks['same_model_comparison_conditions']=all((a['view'],a['sample'],a['seed'],a['prompt'])==(b['view'],b['sample'],b['seed'],b['prompt']) for a,b in zip(base,lora))
result={'checks':checks,'all_passed':all(checks.values()),'scope':'execution integrity only; not product identity, geometric accuracy or ad quality judgment','invalid_technical_outputs_excluded':4,'valid_comparison_outputs':12};(r/'outputs/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['all_passed']

"""Apply the post-experiment exclusion rule to existing Run 23 records only.

This is a policy replay, not another rendering experiment or independent test.
It preserves the frozen renderer and returns unedited input for excluded cases.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'run23'
OUT=ROOT/'outputs/scoped'
OUT.mkdir(exist_ok=True)
policy=json.loads((ROOT/'scope_policy.json').read_text())
prior=json.loads((PREV/'outputs/results.json').read_text())
records=[]
for case in prior['cases']:
    cid=case['id']
    before=cv2.imread(case['input'])
    support=cv2.imread(str(PREV/'outputs'/(cid+'_support.png')),0)>0
    y,x=np.where(support)
    height=int(y.max()-y.min()+1) if len(y) else 0
    decision=case['decision']
    if decision=='restored' and height<policy['minimum_native_ink_support_bbox_height_px']:
        decision='abstain_small_text'
        output=before.copy()
    elif decision=='restored':
        decision='candidate_for_visual_review'
        output=cv2.imread(str(PREV/'outputs'/(cid+'.png')))
    else:
        output=before.copy()
    path=OUT/(cid+'.png')
    cv2.imwrite(str(path),output)
    saved=cv2.imread(str(path))
    assert np.array_equal(saved,output)
    changed=int(np.any(saved!=before,axis=2).sum())
    if decision!='candidate_for_visual_review': assert changed==0
    records.append({'id':cid,'decision':decision,'native_ink_support_bbox_height_px':height,
                    'changed_pixels_vs_input':changed,'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'type':'replay of existing Run 23 cases, not new evaluation',
        'policy_sha256':hashlib.sha256((ROOT/'scope_policy.json').read_bytes()).hexdigest(),
        'cases':records}
(OUT/'decisions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))

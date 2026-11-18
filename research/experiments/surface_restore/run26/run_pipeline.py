"""Run the unchanged Run 23 engine with a newly registered product and Run 24 scope.

Source regions are registered once. No new target coordinates or parameter tuning.
"""
import datetime
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'run23'
OUT=ROOT/'outputs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(stage):
    path=ROOT/'timeline.json'
    data=json.loads(path.read_text())
    data['events'].append({'stage':stage,'utc':utc()})
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    registration=json.loads((ROOT/'registration/product.json').read_text())
    for f in protocol['frozen_files']:assert sha(f['path'])==f['sha256']
    event('normalize_and_pipeline_start')
    start=time.perf_counter()
    image=Image.open(ROOT/'inputs/generated_native.png').convert('RGB')
    native_size=list(image.size)
    image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    image.save(ROOT/'inputs/ad_before.png')

    # Import the frozen engine. Override data roots, never the algorithm or thresholds.
    spec=importlib.util.spec_from_file_location('run23_frozen_engine',PREV/'restore.py')
    engine=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    cfg=json.loads((PREV/'protocol.json').read_text())
    for key in ['source_label_quad','source_text_roi','presence_guard_roi']:
        cfg[key]=registration[key]
    cfg['reference']=registration['reference']
    cfg['cases']=[{'id':'new_ad','input':'inputs/ad_before.png',
                  'role':'second product ad with deliberately reserved black handwritten tagline'}]
    engine.ROOT=ROOT
    engine.OUT=OUT/'engine'
    engine.CFG=cfg
    engine.OUT.mkdir(exist_ok=True)
    (ROOT/'effective_engine_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    engine.main()

    raw=json.loads((engine.OUT/'results.json').read_text())
    record=raw['cases'][0]
    before=cv2.imread(str(ROOT/'inputs/ad_before.png'))
    repaired=cv2.imread(str(engine.OUT/'new_ad.png'))
    mask=cv2.imread(str(engine.OUT/'new_ad_support.png'),0)>0
    y,x=np.where(mask)
    ink_height=int(y.max()-y.min()+1) if len(y) else 0
    decision=record['decision']
    if decision=='restored' and ink_height<protocol['minimum_ink_height_px']:
        decision='abstain_small_text'
        repaired=before.copy()
        mask[:]=False
    changed=np.any(before!=repaired,axis=2)
    assert not np.any(changed&~mask)
    if decision!='restored':assert not changed.any()
    cv2.imwrite(str(OUT/'final_ad.png'),repaired)
    cv2.imwrite(str(OUT/'edit_support.png'),mask.astype(np.uint8)*255)
    assert np.array_equal(cv2.imread(str(OUT/'final_ad.png')),repaired)
    for f in protocol['frozen_files']:assert sha(f['path'])==f['sha256']
    event('automatic_pipeline_complete')
    data={
        'created_utc':utc(), 'decision':decision,
        'native_generated_size':native_size,'delivery_size':list(image.size),
        'normalization':'whole-image aspect-preserving Lanczos thumbnail before restoration; no crop or upscaling',
        'source_registration_count':1, 'source_region_annotations_reused':False,
        'new_manual_source_regions':2,'source_annotation_passes':1,'source_coordinate_revisions':0,'target_manual_alignment_edits':0,
        'manual_pixel_retouches':0,'generation_prompt_revisions':0,'image_generation_calls':1,
        'generation_retries':0,'renderer_parameter_changes':0,
        'native_ink_support_height_px':ink_height,
        'changed_pixels':int(changed.sum()),'changed_outside_support':0,
        'ink_bbox_xyxy':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(y) else None,
        'normalization_and_pipeline_seconds':time.perf_counter()-start,
        'engine_seconds':raw['elapsed_seconds'],
        'reference_sha256':sha(registration['reference']),
        'native_generated_sha256':sha(ROOT/'inputs/generated_native.png'),
        'before_sha256':sha(ROOT/'inputs/ad_before.png'),
        'final_sha256':sha(OUT/'final_ad.png'),
        'protocol_sha256':sha(ROOT/'protocol.json'),
        'effective_engine_config_sha256':sha(ROOT/'effective_engine_config.json'),
        'engine_record':record,
        'limits':['Planned empty black tagline, not accidental failure repair',
                  'Newly registered second product; agent source-annotation time measured, not human-user timing',
                  'Visual structure inspection only; no target-view real photograph']}
    (OUT/'workflow_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!='engine_record'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

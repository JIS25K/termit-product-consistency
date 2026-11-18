from pathlib import Path
import json,hashlib,datetime
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;O=R/'outputs'
frozen=json.loads((O/'frozen.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
production=json.loads((O/'production.json').read_text());review=json.loads((O/'visual_review.json').read_text())
layer=np.load(O/'fixed_layer.npz');opaque=layer['alpha']==1
candidate=cv2.imread(str(O/'candidate.png'));source=np.rint(layer['premult']).clip(0,255).astype(np.uint8)
production['strictly_opaque_changed_pixels']=int(np.any(candidate[opaque]!=source[opaque],axis=1).sum())
production['near_opaque_note']='The earlier alpha>0.999 check includes 6 semi-transparent boundary pixels affected by compositing rounding; strict alpha==1 is reported separately.'
assert production['manual_result_reposition']==0
expected=np.array(json.loads((R.parent/'run28/outputs/production.json').read_text())['affine']);actual=np.array(json.loads((O/'registration_check.json').read_text())['affine'])
assert np.array_equal(expected,actual)
report={'production':production,'registration_exact_match_run28':True,'visual_review':review,'frozen_files_verified':True,'calls':{'ordinary':1,'surround_scene':1,'retries':0},'limits':['No model version/seed control; single pipeline comparison.','Candidate receives an additional fixed layout guide derived from same references; human setup from Run28 is inherited.','Interior pixels locked by construction, not evidence of accurate new-view 3D.','Ball comes from scene reference; tee comes from real photo.','Output full-canvas resize is automatic; no positional fit or manual relocation.'],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
fontpath='/System/Library/Fonts/AppleSDGothicNeo.ttc';font=lambda n:ImageFont.truetype(fontpath,n)
sheet=Image.new('RGB',(1440,1135),'#f7f4ee');d=ImageDraw.Draw(sheet)
d.text((32,23),'Run 29 · 제품을 고정하고 주변 장면을 만들면?',font=font(32),fill='#1d3329')
for i,(name,label) in enumerate([('baseline_normalized.png','일반 호출 1회'),('candidate.png','제품 고정 → 주변 생성 → 원본 픽셀 복원')]):
 x=32+i*704;im=Image.open(O/name).convert('RGB')
 d.text((x,82),label,font=font(25),fill='#1d3329');sheet.paste(im.resize((672,504),Image.Resampling.LANCZOS),(x,125))
 d.text((x,650),'받침·잔디 접합 확대 · 같은 위치 / 같은 배율',font=font(22),fill='#42554b')
 sheet.paste(im.crop((345,590,681,730)).resize((672,280),Image.Resampling.LANCZOS),(x,690))
d.text((32,993),review['board_verdict'],font=font(26),fill='#7c382b')
d.text((32,1036),'기존 위치·크기 그대로 / 결과 위치 재조정 0회 / 각 경로 1회 생성 / 결과를 본 뒤 마스크·조명 튜닝 0회',font=font(22),fill='#42554b')
d.text((32,1077),'실제 원본 부품의 수동 등록을 재사용한 2D 실험입니다. 다른 각도의 구조 생성이나 완전 자동 등록은 검증하지 않았습니다.',font=font(21),fill='#42554b')
sheet.save(O/'comparison.png');print(json.dumps(report,ensure_ascii=False,indent=2))
# Additional debugging evidence: raw generated scene before source-pixel restoration.
debug=Image.new('RGB',(1536,530),'#f7f4ee');dd=ImageDraw.Draw(debug)
for i,(file,label) in enumerate([('scene_generated_normalized.png','주변 생성 원출력'),('candidate.png','고정 레이어 복원 후'),('baseline_normalized.png','일반 호출')]):
 dd.text((i*512+12,10),label,font=font(24),fill='#1d3329');im=Image.open(O/file).crop((340,275,680,735)).resize((340,460),Image.Resampling.LANCZOS);debug.paste(im,(i*512+85,50))
debug.save(O/'stage_comparison.png')

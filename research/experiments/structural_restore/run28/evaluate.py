from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;O=R/'outputs'
frozen=json.loads((O/'frozen.json').read_text())
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
# Visual endpoint annotations for evaluation only. Native baseline retained separately.
annotations={
 'reference':{'file':'inputs/reference.webp','shaft':[977,1029],'base':[890,1124],'height':[398,756]},
 'input':{'file':'inputs/target.png','shaft':[482,531],'base':[381,628],'height':[318,696]},
 'baseline':{'file':'outputs/baseline_normalized.png','shaft':[489,532],'base':[416,605],'height':[318,684]},
 'candidate':{'file':'outputs/candidate.png','shaft':[478,533],'base':[385,632],'height':[318,696]}}
result={}
before=cv2.imread(str(R/'inputs/target.png'));support=cv2.imread(str(O/'support.png'),0)>0
for name,a in annotations.items():
 shaft=a['shaft'][1]-a['shaft'][0];base=a['base'][1]-a['base'][0];height=a['height'][1]-a['height'][0]
 result[name]={'shaft_px':shaft,'base_px':base,'height_px':height,'shaft_to_base':shaft/base,'base_to_height':base/height}
 if name in ['baseline','candidate']:
  im=cv2.imread(str(R/a['file']));d=np.max(np.abs(im.astype(int)-before.astype(int)),axis=2)
  result[name].update({'outside_support_changed_gt8':int(((d>8)&~support).sum()),'ball_top_region_changed_gt8':int((d[:318]>8).sum())})
ref=result['reference']
for name in ['input','baseline','candidate']:
 for metric in ['shaft_to_base','base_to_height']:
  result[name][metric+'_error_pct']=100*(result[name][metric]/ref[metric]-1)
report={'measurements':result,'evaluation_only_annotations':annotations,'limits':['Approximate assistant-selected visible endpoints, roughly +/-3px. Not calibrated ground truth.','Baseline native 1448x1086 resized uniformly to 1024x768 for coordinate comparison; pixel deltas include resampling.','Candidate source shape locked by construction. This verifies implementation, not unseen-angle reconstruction.','Both paths repair green material and remove invented holes. Candidate has obvious inpainting smear; full visual gate fails.'],'gate':{'green_base_no_holes':'both pass','shaft_base_ratio_within_10pct':'both pass on approximate measurements','ball_scene_exact':'candidate only by construction; baseline resampled','no_visible_smear':'candidate fails; baseline visually cleaner'},'decision':'No overall win. Source geometry retained, but ordinary edit is more naturally integrated. Do not deploy structural repair as automatic product.'}
(O/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
fontpath='/System/Library/Fonts/AppleSDGothicNeo.ttc'
font=lambda size:ImageFont.truetype(fontpath,size)
sheet=Image.new('RGB',(1520,1070),'#f7f4ee');d=ImageDraw.Draw(sheet)
d.text((36,22),'Run 28 · 골프티 — 형태 보존과 광고 완성도는 달랐다',font=font(33),fill='#182e29')
columns=[('reference','원본 실사',[850,355,1170,795]),('input','공통 수정 입력',[350,280,670,720]),('baseline','일반 호출 1회',[350,280,670,720]),('candidate','원본 부품 배치',[350,280,670,720])]
for i,(key,title,box) in enumerate(columns):
 x=36+i*372;d.text((x,82),title,font=font(26),fill='#182e29')
 im=Image.open(R/annotations[key]['file']).convert('RGB').crop(box).resize((344,473),Image.Resampling.LANCZOS);sheet.paste(im,(x,130))
 notes={
 'reference':['넓게 퍼진 초록 받침','몸통 / 받침 폭 ≈ 0.22','받침 폭 / 부품 높이 ≈ 0.65'],
 'input':['은색 원판·구멍이 생김','실제 제품과 구조가 다름','기존 광고를 그대로 입력'],
 'baseline':['초록 받침 복원 성공','몸통 / 받침 폭 ≈ 0.23','받침 폭 / 부품 높이 ≈ 0.52'],
 'candidate':['원본 받침의 폭·곡선 유지','몸통 / 받침 폭 ≈ 0.22','받침 폭 / 부품 높이 ≈ 0.65']}[key]
 for j,line in enumerate(notes):d.text((x,625+j*39),line,font=font(22),fill='#314d42')
d.rounded_rectangle((36,775,1484,987),radius=16,fill='#eae2d4')
lines=['판정: 전체 품질 우위는 미입증. 일반 호출도 핵심 오류를 고쳤고, 합성은 잔디 번짐이 남았다.',
       '차이: 원본 배치는 실제 받침의 폭·형태와 공·로고를 보존했다. 일반 호출은 받침이 더 좁아졌다.',
       '한계: 비슷한 정면 사진의 수동 등록 실험. 새 시점 생성·자동 등록·전체 제품 해결이 아니다.',
       '다음 게이트: 등록한 형상을 유지하며 빈 배경에서 장면을 만들고, 위치 재조정 없이 접합되는지 1회 검증.']
for j,line in enumerate(lines):d.text((56,800+j*43),line,font=font(23),fill='#263c33')
d.text((36,1017),'원본은 부품 높이에 맞춰 표시. 나머지 세 장은 동일 좌표 크롭. 일반 출력은 동일 해상도로 정규화.',font=font(21),fill='#52675b')
sheet.save(O/'comparison.png')
print(json.dumps(report,ensure_ascii=False,indent=2))

"""Verify the abstained second-product output without changing the frozen engine."""
import datetime,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((ROOT/'protocol.json').read_text());w=json.loads((OUT/'workflow_results.json').read_text())
for f in cfg['frozen_files']:assert sha(f['path'])==f['sha256']
assert sha(ROOT/'protocol.json')==w['protocol_sha256']
assert sha(ROOT/'effective_engine_config.json')==w['effective_engine_config_sha256']
assert sha(ROOT/'generation_prompt.txt')==cfg['generation_prompt_sha256']
before=cv2.imread(str(ROOT/'inputs/ad_before.png'));after=cv2.imread(str(OUT/'final_ad.png'))
assert sha(OUT/'final_ad.png')==w['final_sha256']
assert sha(ROOT/'inputs/ad_before.png')==w['before_sha256']
assert sha(ROOT/'inputs/generated_native.png')==w['native_generated_sha256']
assert w['decision']=='abstain_alignment'
assert np.array_equal(before,after)
assert w['engine_record']['inliers_above_below_name']==[194,0]

# Check that second-product changes are data fields only, never tuning.
effective=json.loads((ROOT/'effective_engine_config.json').read_text())
previous=json.loads((ROOT.parent/'run23/protocol.json').read_text())
for key in ['alignment','presence']:assert effective[key]==previous[key]
reg=json.loads((ROOT/'registration/product.json').read_text())
for key in ['source_label_quad','source_text_roi','presence_guard_roi']:assert effective[key]==reg[key]
font=lambda n:ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',n)
panel=Image.new('RGB',(1120,510),'#f4f5f7');d=ImageDraw.Draw(panel)
d.text((20,15),'Run 26 | 두 번째 상품에서는 자동 위치 확신 부족 → 복원 보류',font=font(25),fill='#17212b')
d.text((20,65),'원본: 복원 대상 검은 손글씨',font=font(21),fill='#17212b')
d.text((580,65),'새 광고 결과: 문구 누락 그대로',font=font(21),fill='#9a3412')
ref=Image.open(reg['reference']).convert('RGB')
x0,y0,x1,y1=reg['source_text_roi']
crop=ref.crop((x0-12,y0-12,x1+12,y1+12));crop=crop.resize((crop.width*2,crop.height*2),Image.Resampling.NEAREST)
panel.paste(crop,(20,112))
q=np.array(w['engine_record']['projected_name_quad']);lo=np.floor(q.min(axis=0)).astype(int)-10;hi=np.ceil(q.max(axis=0)).astype(int)+10
target=Image.fromarray(cv2.cvtColor(after,cv2.COLOR_BGR2RGB)).crop(tuple(map(int,[*lo,*hi])))
target=target.resize((target.width*2,target.height*2),Image.Resampling.NEAREST)
panel.paste(target,(580,112))
d.text((20,398),'원본과 결과의 대응점: 문구 위 194개 / 아래 0개 · 아래쪽 위치를 검증할 근거 부족',font=font(20),fill='#9a3412')
d.text((20,435),'결과 좌표 수작업 0회 · 추가 튜닝 0회 · 입력과 출력 0픽셀 차이 (복원 성공을 뜻하지 않음)',font=font(18),fill='#4b5563')
panel.save(OUT/'comparison.png')
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':w['decision'],
 'frozen_files_verified':len(cfg['frozen_files']),'alignment_and_presence_thresholds_unchanged':True,
 'new_source_registration_applied':True,'input_output_hashes_verified':True,
 'pixel_identical_to_input':True,'changed_pixels':0,'inliers_above_below_print':[194,0],
 'blankness_check_reached':False,'ink_transfer_reached':False,
 'readability_result':'Target tagline absent, not restored. No successful-restoration readability claim.',
 'visual_qa':'Final full ad and original print crop inspected; requested black tagline missing.',
 'limits':'Abstention shows this pipeline did not establish registration confidence; does not prove forced placement would necessarily be wrong.'}
(OUT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))

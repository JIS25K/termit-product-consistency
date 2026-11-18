"""Saved-file verification and same-region before/after for Run 25."""
import datetime
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=json.loads((OUT/'workflow_results.json').read_text())
cfg=json.loads((ROOT/'protocol.json').read_text())
before=cv2.imread(str(ROOT/'inputs/ad_before.png'))
final=cv2.imread(str(OUT/'final_ad.png'))
support=cv2.imread(str(OUT/'edit_support.png'),0)>0
changed=np.any(before!=final,axis=2)
assert sha(OUT/'final_ad.png')==data['final_sha256']
assert sha(ROOT/'inputs/ad_before.png')==data['before_sha256']
assert sha(ROOT/'inputs/generated_native.png')==data['native_generated_sha256']
assert sha(ROOT/'effective_engine_config.json')==data['effective_engine_config_sha256']
assert sha(ROOT/'protocol.json')==data['protocol_sha256']
assert not np.any(changed&~support)
assert int(changed.sum())==data['changed_pixels']
for f in cfg['frozen_files']:assert sha(f['path'])==f['sha256']
font=lambda n:ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',n)
q=np.array(data['engine_record']['projected_name_quad'])
# Evaluation crop comes from the automatic projection, never feeds back into alignment.
lo=np.floor(q.min(axis=0)).astype(int)-np.array([20,45])
hi=np.ceil(q.max(axis=0)).astype(int)+np.array([20,35])
box=tuple(map(int,[*lo,*hi]))
panel=Image.new('RGB',(1100,485),'#f4f5f7');d=ImageDraw.Draw(panel)
d.text((20,15),'Run 25 | 새 광고의 예약된 제품명 영역에 원본 인쇄 자동 배치',font=font(25),fill='#17212b')
for i,(arr,title) in enumerate([(before,'새 광고 시안: 제품명 한 줄을 의도적으로 비움'),(final,'자동 처리 후: 아몬드 스프레드')]):
    x=20+550*i
    d.text((x,65),title,font=font(18),fill='#17212b')
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB))
    crop=im.crop(box)
    d.text((x,100),'동일 영역 2배 확대',font=font(16),fill='#4b5563')
    crop=crop.resize((crop.width*2,crop.height*2),Image.Resampling.NEAREST)
    panel.paste(crop,(x,130))
d.text((20,425),f"최종 1024×1024 · 인쇄 높이 {data['native_ink_support_height_px']}px · 수동 결과 위치 수정 0회 · 영역 밖 변경 0px",font=font(20),fill='#166534')
panel.save(OUT/'before_after.png')
verification={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'saved_png_shape':list(final.shape),'saved_png_changed_pixels':int(changed.sum()),
 'saved_png_changes_outside_ink_support':0,'frozen_file_hashes_verified':len(cfg['frozen_files']),
 'input_and_output_hashes_verified':True,'effective_config_hash_verified':True,
 'final_text_height_px':data['native_ink_support_height_px'],
 'minimum_scope_guard_px':cfg['minimum_ink_height_px'],
 'guard_pass':data['native_ink_support_height_px']>=cfg['minimum_ink_height_px']}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
print(json.dumps(verification,ensure_ascii=False,indent=2))

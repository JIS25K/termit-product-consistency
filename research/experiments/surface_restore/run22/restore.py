"""Small ink-transfer prototype. Source text pixels, not newly rendered glyphs."""
import datetime,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';CFG=json.loads((ROOT/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(CFG['reference'])==CFG['input_sha256']['reference']
assert sha(CFG['candidate'])==CFG['input_sha256']['candidate']
start=time.perf_counter()
reference=cv2.imread(CFG['reference'],cv2.IMREAD_COLOR)
candidate=cv2.imread(CFG['candidate'],cv2.IMREAD_COLOR)
p=CFG['manual_parameters'];h,w=candidate.shape[:2]
H=cv2.getPerspectiveTransform(np.array(p['source_label_quad'],np.float32),np.array(p['target_label_quad'],np.float32))
x0,y0,x1,y1=p['source_text_roi'];gray=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
alpha=np.zeros(gray.shape,np.float32)
alpha[y0:y1,x0:x1]=np.clip((170-gray[y0:y1,x0:x1].astype(float))/100,0,1)
warped_alpha=cv2.warpPerspective(alpha,H,(w,h),flags=cv2.INTER_LINEAR)
warped_color=cv2.warpPerspective(reference,H,(w,h),flags=cv2.INTER_LINEAR)
restored=np.rint(candidate*(1-warped_alpha[:,:,None])+warped_color*warped_alpha[:,:,None]).clip(0,255).astype(np.uint8)
support=warped_alpha>0;changed=np.any(candidate!=restored,axis=2)
assert not np.any(changed&~support)
assert np.count_nonzero(changed)>0
cv2.imwrite(str(OUT/'restored.png'),restored)
cv2.imwrite(str(OUT/'edit_support.png'),support.astype(np.uint8)*255)
cv2.imwrite(str(OUT/'source_ink_mask.png'),np.rint(alpha*255).astype(np.uint8))
ys,xs=np.where(support);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
font=lambda n:ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',n)

# Same crop and same 3x enlargement before/after; no independent recentering.
crop_box=(600,650,845,825)
pair=Image.new('RGB',(1500,610),'white');d=ImageDraw.Draw(pair)
for i,(im,label) in enumerate([(candidate,'복원 전: 제품명 누락'),(restored,'복원 후: 원본 사진의 인쇄만 재투영')]):
    rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    crop=rgb.crop(crop_box).resize((735,525),Image.Resampling.NEAREST)
    pair.paste(crop,(i*750,50));d.text((15+i*750,12),label,font=font(24),fill='black')
d.text((15,582),'동일 영역 3배 확대 · 나머지 장면과 병 구조는 유지 · 수작업 위치 지정',font=font(20),fill='#444444')
pair.save(OUT/'label_before_after.jpg',quality=95)
full=Image.new('RGB',(1024,1064),'white')
full.paste(Image.fromarray(cv2.cvtColor(restored,cv2.COLOR_BGR2RGB)),(0,40))
ImageDraw.Draw(full).text((15,8),'기존 생성 광고에 제품명만 복원한 결과',font=font(24),fill='black')
full.save(OUT/'restored_overview.jpg',quality=95)
src=Image.fromarray(cv2.cvtColor(reference[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)).resize(((x1-x0)*3,(y1-y0)*3),Image.Resampling.NEAREST)
src.save(OUT/'reference_text_crop.png')

record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,'input_hashes_unchanged':sha(CFG['reference'])==CFG['input_sha256']['reference'] and sha(CFG['candidate'])==CFG['input_sha256']['candidate'],'source_text_roi':p['source_text_roi'],'H_source_to_candidate':H.tolist(),'target_edit_bbox':bbox,'edit_support_pixels':int(support.sum()),'changed_pixels':int(changed.sum()),'total_pixels':h*w,'changed_fraction':float(changed.mean()),'changed_pixels_outside_support':int(np.count_nonzero(changed&~support)),'generative_calls':0,'retyped_text_pixels':0,'manual_inputs':'two label quadrilaterals and one source text ROI','outputs':{name:sha(OUT/name) for name in ['restored.png','edit_support.png','source_ink_mask.png']}}
(OUT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))

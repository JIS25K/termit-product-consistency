"""Post-run verification and comparison artifacts. Never imported by restore.py."""
import hashlib, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
results=json.loads((OUT/'results.json').read_text())
font=lambda n:ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc',n)
labels={
 'actual_missing':('실제 누락 사례 · Run 22와 동일한 입력',(600,650,845,825)),
 'controlled_missing':('다른 구도 · 제품명만 지운 통제 샘플',(635,247,1010,515)),
 'normal_control':('정상 대조군 · 이미 제품명이 있는 광고',(585,630,845,835))}
sheet=Image.new('RGB',(1460,1370),'#f4f5f7');d=ImageDraw.Draw(sheet)
d.text((24,16),'Run 23 | 결과 위치를 자동으로 찾아 제품명만 복원',font=font(28),fill='#17212b')
d.text((24,56),'원본 라벨·제품명 영역은 한 번 지정 · 결과별 위치 지정 0회 · 아래는 같은 영역의 전후 확대',font=font(20),fill='#4b5563')
checks=[]
for i,record in enumerate(results['cases']):
    cid=record['id']; label,box=labels[cid]
    before=Image.open(record['input']).convert('RGB')
    after=Image.open(OUT/(cid+'.png')).convert('RGB')
    b=np.asarray(before);a=np.asarray(after)
    support=np.asarray(Image.open(OUT/(cid+'_support.png')))>0
    changed=np.any(a!=b,axis=2)
    assert int(changed.sum())==record['changed_pixels']
    assert not np.any(changed&~support)
    assert hashlib.sha256(Path(record['input']).read_bytes()).hexdigest()==record['input_sha256']
    assert hashlib.sha256((OUT/(cid+'.png')).read_bytes()).hexdigest()==record['output_sha256']
    if cid=='normal_control':assert np.array_equal(a,b)
    ys,xs=np.where(changed)
    bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
    checks.append({'id':cid,'changed_pixels':int(changed.sum()),'changed_bbox':bbox,'changed_outside_support':0,'pixel_identical':bool(np.array_equal(a,b))})
    y=100+i*410
    d.text((24,y),label,font=font(24),fill='#17212b')
    for j,(im,heading) in enumerate([(before,'입력'),(after,'자동 처리 후')]):
        d.text((24+j*720,y+34),heading,font=font(18),fill='#4b5563')
        crop=im.crop(box)
        crop.thumbnail((690,310),Image.Resampling.LANCZOS) if crop.width>690 or crop.height>310 else None
        # Match the same enlarged crop and scaling within each row.
        cw,ch=box[2]-box[0],box[3]-box[1]
        scale=min(690/cw,310/ch)
        crop=im.crop(box).resize((round(cw*scale),round(ch*scale)),Image.Resampling.NEAREST)
        sheet.paste(crop,(24+j*720,y+60))
    verdict={'actual_missing':'자동 배치 · 글자 높이 약 8px: 획 손실 / 가독성 부족',
             'controlled_missing':'자동 배치 · 글자 높이 약 15px: 문구 읽기 가능',
             'normal_control':'기존 인쇄 감지 → 유지'}[cid]
    msg=f"{verdict} · 변경 {record['changed_pixels']} px · 영역 밖 변경 0 px"
    d.text((24,y+375),msg,font=font(19),fill='#9a3412' if cid=='actual_missing' else '#166534')
d.text((24,1337),'PNG 원본으로 픽셀 검사 · 통제 샘플의 삭제 위치/원본은 복원 코드에서 읽지 않음 · 실제 3D 구조 정확도는 별도',font=font(18),fill='#4b5563')
sheet.save(OUT/'comparison.jpg',quality=95)

# A separate comparison against the generated ad before controlled deletion.
prep=json.loads((ROOT/'evaluation/preparation.json').read_text())
original=Image.open(prep['original']).convert('RGB')
original_cv=cv2.imread(prep['original'])
fixed_cv=cv2.imread(str(OUT/'controlled_missing.png'))
mask=cv2.imread(str(ROOT/'evaluation/deletion_mask.png'),0)>0
original_ink=(cv2.cvtColor(original_cv,cv2.COLOR_BGR2GRAY)<100)&mask
fixed_ink=(cv2.cvtColor(fixed_cv,cv2.COLOR_BGR2GRAY)<100)&mask
def ink_stats(ink):
    y,x=np.where(ink)
    return {'pixels':len(x),'center_xy':[float(x.mean()),float(y.mean())], 'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]}
os=ink_stats(original_ink);fs=ink_stats(fixed_ink)
verification={'input_and_output_hashes_verified':True,'cases':checks,
 'controlled_generated_original_comparison':{
    'note':'Existing generated ad is a 2D restoration reference only, not target-view real-photo ground truth.',
    'original_dark_ink':os,'restored_dark_ink':fs,
    'ink_centroid_distance_px':float(np.linalg.norm(np.array(os['center_xy'])-fs['center_xy'])),
    'ink_mask_iou':float((original_ink&fixed_ink).sum()/(original_ink|fixed_ink).sum())}}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
pair=Image.new('RGB',(1440,470),'white');pd=ImageDraw.Draw(pair)
for i,(im,text) in enumerate([(original,'삭제 전 생성 광고 (실사 정답 아님)'),(Image.open(OUT/'controlled_missing.png'),'자동 복원 결과')]):
    pd.text((20+720*i,15),text,font=font(24),fill='black')
    pair.paste(im.crop((720,305,895,405)).resize((700,400),Image.Resampling.NEAREST),(10+720*i,55))
pair.save(OUT/'controlled_original_comparison.jpg',quality=95)
print(json.dumps(verification,ensure_ascii=False,indent=2))

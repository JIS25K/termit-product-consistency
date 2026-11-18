from pathlib import Path
import cv2,json,numpy as np,hashlib
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;O=R/'outputs'
frozen=json.loads((O/'frozen.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
# Evaluation-only approximate visible boundaries; these do not feed the renderer.
rows={'reference':{'path':'inputs/reference.webp','metal':[294,491],'body_x':620,'y':[245,320]},'input':{'path':'inputs/target.png','metal':[83,999],'body_x':1260,'y':[440,610]},'baseline':{'path':'outputs/baseline.png','metal':[805,999],'body_x':1260,'y':[440,610]},'candidate':{'path':'outputs/candidate.png','metal':[578,976],'body_x':1260,'y':[440,610]}}
results={}
for name,d in rows.items():
 im=cv2.imread(str(R/d['path']));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);y0,y1=d['y'];a,b=d['metal'];xbody=d['body_x']
 ys=np.flatnonzero(g[y0:y1,xbody]<180)+y0;diam=int(ys.max()-ys.min()+1)
 points=[];widths=[]
 for x in range(a+8,b-8):
  yy=np.flatnonzero(g[y0:y1,x]<180)+y0
  if len(yy)>=2: points.append([x,(yy[0]+yy[-1])/2]);widths.append(int(yy[-1]-yy[0]+1))
 pts=np.array(points);fit=np.polyfit(pts[:,0],pts[:,1],1);err=np.abs(pts[:,1]-np.polyval(fit,pts[:,0]));
 results[name]={'metal_length_px':b-a,'body_diameter_px':diam,'length_to_diameter':(b-a)/diam,'centerline_p95_px':float(np.quantile(err,.95)),'centerline_max_px':float(err.max()),'centerline_angle_degrees':float(np.degrees(np.arctan(fit[0]))),'metal_median_width_px':float(np.median(widths))}
ref=results['reference']['length_to_diameter'];before=cv2.imread(str(R/'inputs/target.png'));support=cv2.imread(str(O/'support.png'),0)>0
for k in ['input','baseline','candidate']:
 results[k]['ratio_error_pct']=100*(results[k]['length_to_diameter']/ref-1)
 if k!='input':
  im=cv2.imread(str(R/rows[k]['path']));assert im.shape==before.shape
  delta=np.max(np.abs(im.astype(int)-before.astype(int)),axis=2)
  results[k]['changed_pixels']=int((delta>0).sum());results[k]['outside_support_changes']=int(((delta>0)&~support).sum());results[k]['outside_support_changes_gt8']=int(((delta>8)&~support).sum());results[k]['main_body_changes']=int((delta[:,1165:]>0).sum());results[k]['main_body_changes_gt8']=int((delta[:,1165:]>8).sum())
report={'measures':results,'evaluation_annotations':rows,'measurement_limits':['Visual boundaries selected by assistant; approximate +/-2px. Source is product reference catalog image, not calibrated target ground truth.','Centerline from dark silhouette in selected visible tube span: not tip/connector joint verification or perceptual quality.','Input already nearly straight by this simple metric; do not claim a straightness win.','Body position and preserved region predefined; source rendering manual registration; not automated.'],'verdict':'Candidate preserves reference nib/body ratio substantially better on this selected side-view case. Straightness advantage not demonstrated. Candidate connector is softer from lower resolution reference; whole-product/user acceptance unverified.'}
(O/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
fontpath='/System/Library/Fonts/AppleSDGothicNeo.ttc';font=ImageFont.truetype(fontpath,30);small=ImageFont.truetype(fontpath,22)
W=1768;sheet=Image.new('RGB',(W,1065),'#f7f4ee');dr=ImageDraw.Draw(sheet)
dr.text((48,25),'Run 27 · 같은 입력, 원본 부품 비율을 고정했을 때',font=font,fill='#202b29')
labels=[('input','수정 전 · 과장된 촉'),('baseline','일반 호출 1회 · 촉을 너무 짧게 줄임'),('candidate','원본 부품 재투영 · 원본의 비율에 가까움')]
for i,(k,label) in enumerate(labels):
 y=90+i*285;dr.text((48,y),label,font=font,fill='#202b29');im=Image.open(R/rows[k]['path']).convert('RGB').crop((0,420,1672,620));sheet.paste(im,(48,y+45));m=results[k];dr.text((48,y+247),f"금속 촉 / 몸통 지름 ≈ {m['length_to_diameter']:.2f}   |   원본 기준 ≈ {ref:.2f}",font=small,fill='#42564d')
dr.text((48,970),'동일 크기·동일 좌표로 자른 비교. 원본 부품 등록은 수동이며 새 각도 3D 복원 실험이 아닙니다.',font=small,fill='#42564d')
dr.text((48,1005),'직선성은 두 방식 모두 유사합니다. 이번 차이는 촉 비율과 원래 몸통 보존입니다.',font=small,fill='#42564d')
sheet.save(O/'comparison.png')
print(json.dumps(report,ensure_ascii=False,indent=2))

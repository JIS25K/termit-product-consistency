"""Heldout scoring. Never aligns output to target or writes production files."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from common import ROOT,CFG,FACES,sha,dump,visibility,camera

OUT=ROOT/'outputs';EV=ROOT/'evaluation'
frozen=json.loads((OUT/'production_frozen.json').read_text())
for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
marks=json.loads((EV/'corners.json').read_text());logo_rois=json.loads((ROOT/'source/logo_regions.json').read_text())
model=json.loads((OUT/'model.json').read_text());vertices=np.array(model['vertices'])
info=json.loads((EV/'image_info.json').read_text())
sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.025)
matcher=cv2.BFMatcher(cv2.NORM_L2)
def summarize(values,diag):
 values=np.array(values,float)
 if not len(values):return {'n':0,'median_px':None,'p95_px':None,'median_pct_diag':None,'p95_pct_diag':None}
 return {'n':len(values),'median_px':float(np.median(values)),'p95_px':float(np.percentile(values,95)),'median_pct_diag':float(np.median(values)/diag*100),'p95_pct_diag':float(np.percentile(values,95)/diag*100)}
def boundary(mask):return cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
def point_line_dist(points,a,b):
 v=b-a;t=np.clip(((points-a)*v).sum(1)/(v@v),0,1)
 return np.linalg.norm(points-(a+t[:,None]*v),axis=1)
def fundamental(source,target):
 Ks,Es=camera(source);Kt,Et=camera(target);T=Et@np.linalg.inv(Es);t=T[:3,3]
 cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
 return np.linalg.inv(Kt).T@cross@T[:3,:3]@np.linalg.inv(Ks)
def epipolar_error(F,s,t):
 a=F@np.r_[s,1];b=F.T@np.r_[t,1];e=abs(np.r_[t,1]@a)
 return float(.5*e*(1/np.linalg.norm(a[:2])+1/np.linalg.norm(b[:2])))
rows=[];panels=[];all_logo=[];all_corner=[];all_seam=[]
for name in CFG['target_images']:
 stem=Path(name).stem;real=cv2.imread(str(EV/f'{stem}.png'));render=cv2.imread(str(OUT/f'{stem}_render.png'));predmask=cv2.imread(str(OUT/f'{stem}_mask.png'),0)
 meta=json.loads((OUT/f'{stem}_render.json').read_text());pred=np.array(meta['vertices_2d']);ann=marks[name]
 gtpts=np.array(list(ann.values()),np.float32);hull=cv2.convexHull(gtpts).astype(np.int32)
 gtmask=np.zeros(real.shape[:2],np.uint8);cv2.fillConvexPoly(gtmask,hull,255)
 box=cv2.boundingRect(hull);diag=float(np.hypot(box[2],box[3]))
 corner_errors=[np.linalg.norm(pred[int(k)]-p) for k,p in ann.items()]
 corner=summarize(corner_errors,diag);all_corner.extend(corner_errors)
 eb=boundary(gtmask);pb=boundary(predmask)
 dist_gt=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 dist_pred=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 outline=summarize(np.r_[dist_gt[pb],dist_pred[eb]],diag)
 overlap=(gtmask>0)&(predmask>0)
 iou=float(overlap.sum()/((gtmask>0)|(predmask>0)).sum())
 # Printed texture correspondences, with a separate subset for wordmark ROIs.
 feature_records=[];logo_records=[]
 rejected_epipolar=0;mutual_candidates=0
 overlay=real.copy()
 for face in meta['faces']:
  physical_ids=FACES[face['face']]
  if any(str(i) not in ann for i in physical_ids):continue
  target_face=np.zeros(real.shape[:2],np.uint8);cv2.fillConvexPoly(target_face,np.array([ann[str(i)] for i in physical_ids],np.int32),255)
  kp_t,desc_t=sift.detectAndCompute(cv2.cvtColor(real,cv2.COLOR_BGR2GRAY),cv2.erode(target_face,np.ones((5,5),np.uint8)))
  sname=face['source'];src=cv2.imread(str(ROOT/'source'/sname));sm=np.zeros(src.shape[:2],np.uint8)
  cv2.fillConvexPoly(sm,np.rint(face['source_quad']).astype(np.int32),255);sm=cv2.erode(sm,np.ones((9,9),np.uint8))
  ks,ds=sift.detectAndCompute(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),sm)
  if ds is None or desc_t is None:continue
  forward=matcher.knnMatch(ds,desc_t,k=2);reverse=matcher.knnMatch(desc_t,ds,k=2)
  rev={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.7*n.distance}
  H=np.array(face['H_source_to_target'])
  F=fundamental(sname,name)
  for m,n in forward:
   if not m.distance<.7*n.distance or rev.get(m.trainIdx)!=m.queryIdx:continue
   sp=np.array(ks[m.queryIdx].pt);tp=np.array(kp_t[m.trainIdx].pt);hp=H@np.r_[sp,1];pp=hp[:2]/hp[2]
   mutual_candidates+=1;epi=epipolar_error(F,sp,tp)
   if epi>2.0:rejected_epipolar+=1;continue
   error=float(np.linalg.norm(pp-tp))
   record={'face':face['face'],'source':sname,'source_xy':sp.tolist(),'predicted_xy':pp.tolist(),'observed_xy':tp.tolist(),'error_px':error,'epipolar_error_px':epi}
   feature_records.append(record)
   roi=logo_rois.get(face['face'])
   if roi and roi['source']==sname and cv2.pointPolygonTest(np.array(roi['polygon'],np.float32),tuple(sp),False)>=0:
    logo_records.append(record)
    cv2.line(overlay,tuple(np.rint(pp).astype(int)),tuple(np.rint(tp).astype(int)),(0,220,255),1)
    cv2.circle(overlay,tuple(np.rint(tp).astype(int)),2,(0,0,255),-1)
 vis=visibility(vertices,name);visible=[FACES[f] for f in FACES if vis[f]>.015]
 seams=[];seam_errors=[]
 for i,face_a in enumerate(visible):
  for face_b in visible[i+1:]:
   ids=sorted(set(face_a)&set(face_b))
   if len(ids)!=2 or any(str(j) not in ann for j in ids):continue
   a,b=np.array(ann[str(ids[0])],float),np.array(ann[str(ids[1])],float)
   pa,pb_=pred[ids]
   points=pa+np.linspace(0,1,100)[:,None]*(pb_-pa)
   errs=point_line_dist(points,a,b)
   seams.append({'vertices':ids,'gt_segment':[a.tolist(),b.tolist()],'pred_segment':[pa.tolist(),pb_.tolist()],**summarize(errs,diag)})
   seam_errors.extend(errs)
   cv2.line(overlay,tuple(a.astype(int)),tuple(b.astype(int)),(0,220,0),2)
   cv2.line(overlay,tuple(np.rint(pa).astype(int)),tuple(np.rint(pb_).astype(int)),(255,0,255),2)
 for k,p in ann.items():
  cv2.circle(overlay,tuple(p),3,(0,220,0),-1);cv2.circle(overlay,tuple(np.rint(pred[int(k)]).astype(int)),3,(255,0,255),-1)
 logo=summarize([r['error_px'] for r in logo_records],diag);feature=summarize([r['error_px'] for r in feature_records],diag);seam=summarize(seam_errors,diag)
 all_logo.extend([r['error_px'] for r in logo_records]);all_seam.extend(seam_errors)
 rmse=float(np.sqrt(np.mean((real[overlap].astype(float)-render[overlap].astype(float))**2)))
 statuses={'logo':'insufficient_matches' if logo['n']<6 else 'within_tolerance' if logo['median_pct_diag']<=.5 and logo['p95_pct_diag']<=1 else 'outside_tolerance','outline':'within_tolerance' if outline['p95_pct_diag']<=1 else 'outside_tolerance','seam':'unscored' if not seam['n'] else 'within_tolerance' if seam['p95_pct_diag']<=1 else 'outside_tolerance'}
 row={'target':name,'bbox_diagonal_px':diag,'corner':corner,'outline':outline,'seam':seam,'logo':logo,'all_printed_features':feature,'mutual_candidates':mutual_candidates,'rejected_epipolar':rejected_epipolar,'silhouette_iou':iou,'overlap_rgb_rmse_8bit':rmse,'status':statuses,'seams':seams,'corner_details':[{'corner':int(k),'observed':p,'predicted':pred[int(k)].tolist(),'error_px':float(np.linalg.norm(pred[int(k)]-p))} for k,p in ann.items()]}
 rows.append(row);dump(EV/f'{stem}_feature_matches.json',{'logo':logo_records,'all_printed':feature_records})
 cv2.imwrite(str(EV/f'{stem}_overlay.png'),overlay)
 x0,y0,x1,y1=info[name]['crop'];crops=[real[y0:y1,x0:x1],render[y0:y1,x0:x1],overlay[y0:y1,x0:x1]]
 panel=Image.new('RGB',(1200,600),'white');d=ImageDraw.Draw(panel)
 for j,crop in enumerate(crops):
  tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));tile.thumbnail((380,530));panel.paste(tile,(j*400+(400-tile.width)//2,35));d.text((j*400+12,12),['HELD-OUT PHOTO','SOURCE REPROJECTION','GT green / predicted magenta'][j],fill='black')
 d.text((12,575),f'{stem} | logo median {logo["median_px"]} px | corner p95 {corner["p95_px"]:.2f} px | seam p95 {seam["p95_px"]:.2f} px',fill='black')
 panel.save(EV/f'{stem}_comparison.jpg');panels.append(panel)
summary={'targets':rows,'aggregate_px':{'logo':summarize(all_logo,1),'corner':summarize(all_corner,1),'seam':summarize(all_seam,1)},'source_fit':{k:v for k,v in model.items() if k.startswith('source_') and k!='source_residuals'},'matching_method':'SIFT mutual ratio 0.70 within independently annotated physical target face; reject epipolar error >2px using published camera poses, NOT fitted to predicted output; no target-fitted homography or output alignment','evaluation_revision':'Unrestricted initial matching confused repeated wordmarks on different faces; raw results retained. Target corner annotations checked against target-only enlarged crops. Production outputs unchanged.','target_annotation':'manual physical corners on heldout-only grids; approximate uncertainty 2–5px, folded-rim points larger; no independent second annotator','production_hashes_still_match':all(sha(ROOT/k)==v for k,v in frozen['files'].items())}
# Aggregate percentages are undefined across varying box sizes; only per-case normalized values are valid.
for stats in summary['aggregate_px'].values():stats.pop('median_pct_diag');stats.pop('p95_pct_diag')
dump(EV/'metrics.json',summary)
print(json.dumps([{'target':r['target'],'logo':r['logo'],'corners':r['corner'],'seam':r['seam'],'status':r['status']} for r in rows],indent=2))

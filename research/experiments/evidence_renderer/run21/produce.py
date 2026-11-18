"""Production phase. Reads ONLY six source RGB photographs plus camera metadata."""
from pathlib import Path
import json,time
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from common import ROOT,CFG,SIGNS,FACES,sha,dump,projection,project,verts,visibility,camera

OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
READ_LOG=[]
def read_source(name):
 assert name in CFG['source_images'] and name not in CFG['target_images']
 p=ROOT/'source'/name;READ_LOG.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 return cv2.imread(str(p))
def reconstruct():
 annotations=json.loads((ROOT/'source/corners.json').read_text())
 observations=[];initial=[]
 for idx in range(8):
  equations=[]
  for name,marks in annotations.items():
   assert name in CFG['source_images']
   P=projection(name)
   if str(idx) not in marks:continue
   x,y=marks[str(idx)];equations.extend([x*P[2]-P[0],y*P[2]-P[1]])
   observations.append((name,idx,np.array([x,y],float)))
  assert len(equations)>=4, f'Need two source observations for corner {idx}'
  _,_,vt=np.linalg.svd(equations);v=vt[-1];initial.append(v[:3]/v[3])
 initial=np.array(initial)
 coeff=np.linalg.lstsq(np.c_[SIGNS,np.ones(8)],initial,rcond=None)[0]
 axes=coeff[:3].T;size=2*np.linalg.norm(axes,axis=0);R0=axes/(size/2)
 u,_,vt=np.linalg.svd(R0);R=u@vt
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
 p0=np.r_[coeff[3],Rotation.from_matrix(R).as_rotvec(),np.log(size)]
 projections={name:projection(name) for name in annotations}
 def residual(p):
  xyz=verts(p)
  return np.concatenate([project(xyz[[idx]],projections[name])[0]-xy for name,idx,xy in observations])
 fit=least_squares(residual,p0,loss='soft_l1',f_scale=2,max_nfev=4000,xtol=1e-12,ftol=1e-12,gtol=1e-12)
 xyz=verts(fit.x);errors=residual(fit.x).reshape(-1,2)
 result={'vertices':xyz.tolist(),'params':fit.x.tolist(),'dimensions':np.exp(fit.x[6:]).tolist(),
  'source_corner_reprojection_median_px':float(np.median(np.linalg.norm(errors,axis=1))),
  'source_corner_reprojection_p95_px':float(np.percentile(np.linalg.norm(errors,axis=1),95)),
  'source_observation_count':len(observations),'fit_success':bool(fit.success),
  'source_residuals':[{'image':n,'corner':i,'observed':xy.tolist(),'projected':project(xyz[[i]],projections[n])[0].tolist(),'error_px':float(np.linalg.norm(e))} for (n,i,xy),e in zip(observations,errors)]}
 dump(OUT/'model.json',result)
 for name in CFG['source_images']:
  im=read_source(name);uv=project(xyz,projection(name))
  vis=visibility(xyz,name)
  for face,ids in FACES.items():
   if vis[face]<=0:continue
   cv2.polylines(im,[np.rint(uv[ids]).astype(np.int32)],True,(0,255,0),2)
  for i,p in enumerate(uv):
   cv2.circle(im,tuple(np.rint(p).astype(int)),4,(255,0,0),-1)
   cv2.putText(im,str(i),tuple(np.rint(p+[5,-5]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
  cv2.imwrite(str(OUT/name.replace('.jpg','_source_fit.jpg')),im)
 return xyz

def render_target(xyz,name,sources):
 assert name in CFG['target_images']
 K,E=camera(name);P=K@E[:3];uv=project(xyz,P)
 h,w=next(iter(sources.values())).shape[:2]
 result=np.full((h,w,3),245,np.uint8);mask=np.zeros((h,w),np.uint8)
 idsmap=np.full((h,w),-1,np.int8);coords=np.full((h,w,2),np.nan,np.float32)
 face_map=np.full((h,w),-1,np.int8);records=[]
 target_vis=visibility(xyz,name)
 source_vis={s:visibility(xyz,s) for s in sources}
 for fid,(face,indices) in enumerate(FACES.items()):
  if target_vis[face]<=0.015:continue
  candidates=sorted(sources,key=lambda s:source_vis[s][face],reverse=True)
  source=candidates[0]
  if source_vis[source][face]<=0.10:continue
  sp=project(xyz[indices],projection(source));tp=uv[indices]
  H=cv2.getPerspectiveTransform(sp.astype(np.float32),tp.astype(np.float32))
  inverse=np.linalg.inv(H)
  poly=np.rint(tp).astype(np.int32);fm=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(fm,poly,255)
  yy,xx=np.where(fm>0);hom=np.c_[xx,yy,np.ones(len(xx))]@inverse.T;sxy=(hom[:,:2]/hom[:,2:]).astype(np.float32)
  vals=np.concatenate([cv2.remap(sources[source],chunk[:,0].reshape(-1,1),chunk[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)).reshape(-1,3) for chunk in np.array_split(sxy,max(1,int(np.ceil(len(sxy)/20000))))])
  result[yy,xx]=vals;mask[yy,xx]=255;idsmap[yy,xx]=list(sources).index(source);coords[yy,xx]=sxy;face_map[yy,xx]=fid
  records.append({'face':face,'source':source,'source_visibility':source_vis[source][face],'source_quad':sp.tolist(),'target_quad':tp.tolist(),'H_source_to_target':H.tolist()})
 stem=Path(name).stem
 cv2.imwrite(str(OUT/f'{stem}_render.png'),result);cv2.imwrite(str(OUT/f'{stem}_mask.png'),mask)
 np.savez_compressed(OUT/f'{stem}_provenance.npz',source_id=idsmap,source_xy=coords,face_id=face_map)
 dump(OUT/f'{stem}_render.json',{'target':name,'faces':records,'vertices_2d':uv.tolist(),'camera_metadata_only':True,'source_order':list(sources)})

def main():
 started=time.perf_counter()
 if (OUT/'eval_first_access.json').exists():raise RuntimeError('Heldout has been opened. Do not overwrite frozen production.')
 xyz=reconstruct()
 sources={name:read_source(name) for name in CFG['source_images']}
 assert all(x is not None for x in sources.values())
 for name in CFG['target_images']:render_target(xyz,name,sources)
 dump(OUT/'production_rgb_read_log.json',READ_LOG)
 files=[ROOT/'protocol.json',ROOT/'common.py',Path(__file__),ROOT/'source/corners.json',ROOT/'source/logo_regions.json',ROOT/'source/image_info.json',OUT/'model.json',ROOT/'data/ycb/calibration.h5',OUT/'production_rgb_read_log.json']
 files += [ROOT/'data/ycb/poses'/f'NP5_{Path(n).stem.split("_")[1]}_pose.h5' for n in CFG['source_images']+CFG['target_images']]
 files += [ROOT/'source'/n for n in CFG['source_images']]
 files += [ROOT/'data/ycb'/n for n in CFG['source_images']]
 files += sorted(OUT.glob('N2_*_render.*'))+sorted(OUT.glob('N2_*_mask.png'))+sorted(OUT.glob('N2_*_provenance.npz'))
 frozen={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
         'source_rgb_used':CFG['source_images'],'target_rgb_used':[],
         'target_camera_metadata_used':CFG['target_images'],'seconds':time.perf_counter()-started,
         'gpu_used':False,'generative_calls':0,
         'files':{str(f.relative_to(ROOT)):sha(f) for f in files}}
 dump(OUT/'production_frozen.json',frozen)
 print(json.dumps({'rendered':len(CFG['target_images']),'seconds':frozen['seconds'],'model':json.loads((OUT/'model.json').read_text())},indent=2))
if __name__=='__main__':main()

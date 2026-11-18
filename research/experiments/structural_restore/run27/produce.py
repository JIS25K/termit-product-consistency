"""Assisted reference-part registration. No generated product geometry.
Frozen before ordinary-edit output inspection; manual annotations in setup.json.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np, cv2
R=Path(__file__).resolve().parent;O=R/'outputs';O.mkdir(exist_ok=True)
t=time.perf_counter()
source=cv2.imread(str(R/'inputs/reference.webp'));before=cv2.imread(str(R/'inputs/target.png'))
setup={'source_part_roi':[270,245,588,322],'source_attachment':[584,282],'source_body_diameter_px':65,'target_attachment':[1165,523.5],'target_body_diameter_px':134,'target_old_front_roi':[10,450,1165,597],'background_rows':[400,635],'manual_regions':2,'manual_attachment_points':2,'manual_diameter_estimates':2,'target_registration_revisions':0,'source_mask_refinement_passes':1}
(R/'setup.json').write_text(json.dumps(setup,indent=2))
x0,y0,x1,y1=setup['source_part_roi'];patch=source[y0:y1,x0:x1]
gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY);binary=(gray<200).astype(np.uint8)*255
contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
mask=np.zeros(gray.shape,np.uint8);cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,255,-1)
# Cut at the chosen cylindrical body attachment, not at a generated part.
# Circular/stepped axial part: use the upper visible silhouette mirrored about
# the registered axis to exclude the photographed cast shadow below the part.
axis=282-y0
for col in range(mask.shape[1]):
 ys=np.flatnonzero(mask[:axis+1,col])
 if len(ys):
  lower=min(mask.shape[0]-1,2*axis-int(ys.min()))
  mask[lower+1:,col]=0
mask[:,584-x0+1:]=0
alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(3,3),.5)
scale=setup['target_body_diameter_px']/setup['source_body_diameter_px']
sx,sy=setup['source_attachment'];tx,ty=setup['target_attachment']
M=np.array([[scale,0,tx-scale*(sx-x0)],[0,scale,ty-scale*(sy-y0)]],np.float32)
h,w=before.shape[:2]
a=cv2.warpAffine(alpha,M,(w,h),flags=cv2.INTER_LINEAR)
prem=cv2.warpAffine(patch.astype(np.float32)*alpha[:,:,None],M,(w,h),flags=cv2.INTER_LINEAR)
a[:,1165:]=0;prem[:,1165:]=0
# Remove the previous generated front, restoring only its foreground and narrow halo.
old=np.zeros((h,w),np.uint8);x0,y0,x1,y1=setup['target_old_front_roi'];old[y0:y1,x0:x1]=(cv2.cvtColor(before[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)<210).astype(np.uint8)*255
old=cv2.dilate(old,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)));old[:,1165:]=0
bg=before.astype(np.float32).copy();top=np.median(before[390:410],axis=0);bottom=np.median(before[625:645],axis=0)
for y in range(h):
 f=np.clip((y-400)/(635-400),0,1);bg[y]=(1-f)*top+f*bottom
base=before.astype(np.float32).copy();base[old>0]=bg[old>0]
out=np.rint(base*(1-a[:,:,None])+prem).clip(0,255).astype(np.uint8)
support=(old>0)|(a>0)
# Preserve all unrelated pixels exactly; preserve main body pixels x>=1165 exactly.
out[~support]=before[~support]
cv2.imwrite(str(O/'candidate.png'),out);cv2.imwrite(str(O/'support.png'),support.astype(np.uint8)*255);cv2.imwrite(str(O/'source_mask.png'),mask)
cv2.imwrite(str(O/'source_part.png'),patch)
changed=np.any(out!=before,axis=2)
report={'method':'source front-assembly rigid similarity transform','scale':scale,'affine_matrix':M.tolist(),'changed_pixels':int(changed.sum()),'changed_outside_support':int((changed&~support).sum()),'changed_body_pixels_x_ge_1165':int(changed[:,1165:].sum()),'seconds':time.perf_counter()-t,'generated_product_pixels':0,'output_size':[w,h],'registration':setup}
(O/'production.json').write_text(json.dumps(report,indent=2))
freeze={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'protocol.json',R/'setup.json',R/'produce.py',O/'candidate.png',O/'support.png']}
(O/'frozen.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(report))

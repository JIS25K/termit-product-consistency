from pathlib import Path
import cv2,numpy as np,json,time,hashlib
R=Path(__file__).resolve().parent;O=R/'outputs';t=time.perf_counter()
src=cv2.imread(str(R/'inputs/reference.webp'));im=cv2.imread(str(R/'inputs/target.png'));h,w=im.shape[:2]
setup={'source_roi':[875,398,1135,760],'source_axis_x':1005,'source_top_y':398,'source_base_y':756,'target_axis_x':507,'target_top_y':318,'target_base_y':696,'old_outline':[[448,316],[564,316],[558,345],[535,369],[534,603],[548,624],[600,637],[623,650],[630,665],[621,681],[591,692],[539,700],[468,700],[413,692],[385,680],[379,663],[386,646],[414,637],[466,621],[477,601],[478,369],[459,345]],'manual_rois':1,'manual_target_contours':1,'manual_axis_registration_sets':1,'retuning_passes':0,'old_mask_cleanup_passes_before_baseline':1}
x0,y0,x1,y1=setup['source_roi'];patch=src[y0:y1,x0:x1];hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
binary=((hsv[:,:,0]>=25)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>65)&(hsv[:,:,2]>28)).astype(np.uint8)*255
contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);mask=np.zeros(binary.shape,np.uint8);cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,255,-1)
alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(3,3),.5)
s=(696-318)/(756-398);M=np.array([[s,0,507-s*(1005-x0)],[0,s,318]],np.float32)
a=cv2.warpAffine(alpha,M,(w,h));prem=cv2.warpAffine(patch.astype(np.float32)*alpha[:,:,None],M,(w,h));a[:318]=0;prem[:318]=0
old=np.zeros((h,w),np.uint8);cv2.fillPoly(old,[np.array(setup['old_outline'],np.int32)],255);old=cv2.dilate(old,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)));old[:318]=0
# Inpaint only pixels formerly occupied by the incorrect product. The locked ball remains intact.
base=cv2.inpaint(im,old,7,cv2.INPAINT_TELEA).astype(np.float32)
out=np.rint(base*(1-a[:,:,None])+prem).clip(0,255).astype(np.uint8);support=(old>0)|(a>0);out[~support]=im[~support]
cv2.imwrite(str(O/'candidate.png'),out);cv2.imwrite(str(O/'support.png'),support.astype(np.uint8)*255);cv2.imwrite(str(O/'source_mask.png'),mask);cv2.imwrite(str(O/'source_part.png'),patch)
changed=np.any(im!=out,axis=2)
report={'seconds':time.perf_counter()-t,'changed_pixels':int(changed.sum()),'changed_outside_support':int((changed&~support).sum()),'ball_region_changes_y_lt_318':int(changed[:318].sum()),'scale':s,'affine':M.tolist(),'source_size':list(src.shape[:2][::-1]),'target_size':[w,h],'manual':setup}
(R/'setup.json').write_text(json.dumps(setup,indent=2));(O/'production.json').write_text(json.dumps(report,indent=2));(O/'frozen.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'produce.py',R/'setup.json',R/'protocol.json',O/'candidate.png',O/'support.png']},indent=2));print(json.dumps(report))

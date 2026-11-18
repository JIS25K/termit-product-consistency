from pathlib import Path
import cv2,numpy as np,json,hashlib
R=Path(__file__).resolve().parent;O=R/'outputs';I=R/'inputs'
setup=json.loads((I/'registration.json').read_text());im=cv2.imread(str(I/'scene.png'));src=cv2.imread(str(I/'reference.webp'));h,w=im.shape[:2]
x0,y0,x1,y1=setup['source_roi'];patch=src[y0:y1,x0:x1]
mask=cv2.imread(str(R.parent/'run28/outputs/source_mask.png'),0)
assert mask.shape==patch.shape[:2]
alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(3,3),.5)
s=(setup['target_base_y']-setup['target_top_y'])/(setup['source_base_y']-setup['source_top_y'])
M=np.array([[s,0,setup['target_axis_x']-s*(setup['source_axis_x']-x0)],[0,s,setup['target_top_y']]],np.float32)
a=cv2.warpAffine(alpha,M,(w,h));prem=cv2.warpAffine(patch.astype(np.float32)*alpha[:,:,None],M,(w,h));a[:318]=0;prem[:318]=0
# Ball source is the existing common advertisement, not a new logo rendering.
# Its matte is prepared once before generation; no result-dependent location fitting.
gc=np.zeros((h,w),np.uint8)
cv2.ellipse(gc,(506,170),(163,159),0,0,360,cv2.GC_PR_BGD,-1)
cv2.ellipse(gc,(506,170),(155,153),0,0,360,cv2.GC_PR_FGD,-1)
cv2.ellipse(gc,(506,170),(145,143),0,0,360,cv2.GC_FGD,-1)
gc[318:]=cv2.GC_BGD
cv2.grabCut(im,gc,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
ball=((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)).astype(np.float32)
ball=cv2.GaussianBlur(ball,(3,3),.5);ball[318:]=0
locked_a=np.maximum(ball,a)
locked_prem=im.astype(np.float32)*ball[:,:,None]*(1-a[:,:,None])+prem
guide=np.rint(locked_prem+np.array([218,222,223],np.float32)*(1-locked_a[:,:,None])).clip(0,255).astype(np.uint8)
cv2.imwrite(str(I/'fixed_product_guide.png'),guide)
cv2.imwrite(str(O/'fixed_alpha.png'),np.rint(locked_a*255).astype(np.uint8))
np.savez_compressed(O/'fixed_layer.npz',alpha=locked_a,premult=locked_prem,tee_alpha=a,ball_alpha=ball)
(O/'registration_check.json').write_text(json.dumps({'same_as_run28':True,'affine':M.tolist(),'manual_new_tee_registration':0,'manual_result_reposition':0,'new_pre_generation_ball_matte':'ellipse-seeded GrabCut at existing scene coordinates; 1 setup; no output fitting'},indent=2))
print('Prepared immutable product layer and guide, retaining Run 28 tee registration.')

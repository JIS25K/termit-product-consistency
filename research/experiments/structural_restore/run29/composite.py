from pathlib import Path
import cv2,numpy as np,json,time,hashlib
R=Path(__file__).resolve().parent;O=R/'outputs'
f=json.loads((O/'frozen.json').read_text())
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in f.items())
t=time.perf_counter();layer=np.load(O/'fixed_layer.npz');a=layer['alpha'];prem=layer['premult']
bg=cv2.imread(str(O/'scene_generated.png'));native=list(bg.shape[:2][::-1]);bg=cv2.resize(bg,(1024,768),interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(str(O/'scene_generated_normalized.png'),bg)
out=np.rint(bg.astype(np.float32)*(1-a[:,:,None])+prem).clip(0,255).astype(np.uint8)
cv2.imwrite(str(O/'candidate.png'),out)
base=cv2.imread(str(O/'baseline.png'));baseline_native=list(base.shape[:2][::-1]);base=cv2.resize(base,(1024,768),interpolation=cv2.INTER_LANCZOS4);cv2.imwrite(str(O/'baseline_normalized.png'),base)
interior=a>.999;expected=np.rint(prem).clip(0,255).astype(np.uint8);changed=np.any(out!=bg,axis=2)
metrics={'cpu_seconds':time.perf_counter()-t,'scene_native_size':native,'baseline_native_size':baseline_native,'canvas':[1024,768],'manual_result_reposition':0,'post_generation_tuning':0,'locked_interior_changed_pixels':int(np.any(out[interior]!=expected[interior],axis=1).sum()),'outside_layer_changed_pixels':int((changed&(a==0)).sum()),'registration_reused':'Run 28 exact affine','no_inpainting':True}
(O/'production.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))

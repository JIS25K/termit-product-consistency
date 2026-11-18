"""Read-only pixel analysis. Does not create or alter images."""
from pathlib import Path
from PIL import Image
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent
src=p.parent/'run02/S1.png';dst=p/'background_skyblue.png'
a=np.asarray(Image.open(src).convert('RGB')).astype(np.int16)
b=np.asarray(Image.open(dst).convert('RGB')).astype(np.int16)
assert a.shape==b.shape
regions={'near_ear':[580,350,620,410],'head':[670,420,710,480],'yellow_bag':[430,650,580,730],'foot':[670,840,730,870],'pedestal':[450,960,750,1000],'background':[50,50,200,200]}
results={}
for name,(x1,y1,x2,y2) in regions.items():
 d=np.abs(a[y1:y2,x1:x2]-b[y1:y2,x1:x2])
 results[name]={'xyxy':[x1,y1,x2,y2],'mean_abs_channel_difference_0_255':round(float(d.mean()),4),'exactly_equal_pixels_percent':round(float((d==0).all(axis=2).mean()*100),3)}
r={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'size':[a.shape[1],a.shape[0]],'elapsed_seconds':15.491,'tool':'image_gen built-in','quality_rerolls':0,'method':'fixed interior rectangles; no alignment, no mask, no image alteration','limitations':'sampled interior regions only; not full object preservation score; differences do not alone imply shape changes','regions':results}
(p/'records.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))

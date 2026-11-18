"""First decode of heldout RGB happens after production hashes are verified."""
import datetime,json
import cv2,h5py,numpy as np
from PIL import Image,ImageDraw
from common import ROOT,CFG,sha,dump,camera,silhouette
OUT=ROOT/'outputs';EV=ROOT/'evaluation';EV.mkdir(exist_ok=True)
frozen=json.loads((OUT/'production_frozen.json').read_text())
for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest, name
assert frozen['target_rgb_used']==[]
if not (OUT/'eval_first_access.json').exists():
 dump(OUT/'eval_first_access.json',{'first_decode_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'production_frozen_sha256':sha(OUT/'production_frozen.json'),'all_frozen_hashes_verified_before_target_decode':True,'target_files':CFG['target_images']})
tiles=[];info={}
for name in CFG['target_images']:
 p=ROOT/'data/ycb'/name;im=cv2.imread(str(p));im=cv2.resize(im,(1600,1067),interpolation=cv2.INTER_AREA)
 K,_=camera(name)
 with h5py.File(ROOT/'data/ycb/calibration.h5') as f:d=f['N2_rgb_d'][:]
 im=cv2.undistort(im,K,d,None,K)
 cv2.imwrite(str(EV/name.replace('.jpg','.png')),im)
 mask,hull=silhouette(im);x,y,w,h=cv2.boundingRect(hull)
 # Color mask gives an inspection crop only. It is NOT final geometric truth.
 box=[max(0,x-45),max(0,y-45),min(1600,x+w+45),min(1067,y+h+45)]
 crop=im[box[1]:box[3],box[0]:box[2]].copy()
 for xx in range((box[0]//50+1)*50,box[2],50):
  cv2.line(crop,(xx-box[0],0),(xx-box[0],crop.shape[0]-1),(170,170,170),1)
  cv2.putText(crop,str(xx),(xx-box[0]+1,15),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1)
 for yy in range((box[1]//50+1)*50,box[3],50):
  cv2.line(crop,(0,yy-box[1]),(crop.shape[1]-1,yy-box[1]),(170,170,170),1)
  cv2.putText(crop,str(yy),(1,yy-box[1]-2),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1)
 cv2.imwrite(str(EV/name.replace('.jpg','_grid.png')),crop)
 tile=Image.fromarray(cv2.cvtColor(im[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2RGB));tile.thumbnail((380,480));panel=Image.new('RGB',(400,520),'white');panel.paste(tile,((400-tile.width)//2,10));ImageDraw.Draw(panel).text((10,495),name+' HELD OUT',fill='black');tiles.append(panel)
 info[name]={'crop':box,'original_sha256':sha(p),'working_sha256':sha(EV/name.replace('.jpg','.png'))}
out=Image.new('RGB',(1200,1040),'white')
for i,t in enumerate(tiles):out.paste(t,((i%3)*400,(i//3)*520))
out.save(EV/'heldout_contact.jpg');dump(EV/'image_info.json',info)
print(json.dumps({'decoded_heldout':len(tiles),'production_files_verified':len(frozen['files']),'no_registration_to_outputs':True}))

from pathlib import Path
import json
import cv2,h5py
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text())
(ROOT/'source').mkdir(exist_ok=True)
tiles=[]; info={}
for name in CFG['source_images']:
 p=ROOT/'data/ycb'/name
 if not p.exists():continue
 image=cv2.imread(str(p));scale=1600/image.shape[1]
 image=cv2.resize(image,(1600,round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
 with h5py.File(ROOT/'data/ycb/calibration.h5') as f:
  K=f['N2_rgb_K'][:].astype(float);K[:2]*=scale;dist=f['N2_rgb_d'][:]
 image=cv2.undistort(image,K,dist,None,K)
 hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
 mask=(((hsv[:,:,0]<28)|(hsv[:,:,0]>165))&(hsv[:,:,1]>65)&(hsv[:,:,2]>30)).astype(np.uint8)*255
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
 i=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);mask=(labels==i).astype(np.uint8)*255
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 contour=max(contours,key=cv2.contourArea)
 hull=cv2.convexHull(contour)
 mask[:]=0;cv2.fillConvexPoly(mask,hull,255)
 x,y,w,h=cv2.boundingRect(hull)
 cv2.imwrite(str(ROOT/'source'/name),image)
 cv2.imwrite(str(ROOT/'source'/name.replace('.jpg','_mask.png')),mask)
 # Only source photos; coordinates are in the 1600px working images.
 margin=30;x0=max(0,x-margin);y0=max(0,y-margin);x1=min(1600,x+w+margin);y1=min(image.shape[0],y+h+margin)
 crop=image[y0:y1,x0:x1].copy()
 for xx in range((x0//50+1)*50,x1,50):
  cv2.line(crop,(xx-x0,0),(xx-x0,crop.shape[0]-1),(190,190,190),1)
  cv2.putText(crop,str(xx),(xx-x0+1,15),cv2.FONT_HERSHEY_SIMPLEX,.34,(30,30,30),1)
 for yy in range((y0//50+1)*50,y1,50):
  cv2.line(crop,(0,yy-y0),(crop.shape[1]-1,yy-y0),(190,190,190),1)
  cv2.putText(crop,str(yy),(1,yy-y0-2),cv2.FONT_HERSHEY_SIMPLEX,.34,(30,30,30),1)
 cv2.imwrite(str(ROOT/'source'/name.replace('.jpg','_grid.png')),crop)
 info[name]={'original_size':list(cv2.imread(str(p)).shape[1::-1]),'working_size':list(image.shape[1::-1]),'scale':scale,'bbox':[x,y,w,h],'crop':[x0,y0,x1,y1],'hull':hull[:,0,:].tolist()}
 tile=Image.fromarray(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));tile.thumbnail((380,480))
 frame=Image.new('RGB',(400,520),'white');frame.paste(tile,((400-tile.width)//2,20));ImageDraw.Draw(frame).text((12,495),name,fill='black');tiles.append(frame)
if tiles:
 out=Image.new('RGB',(1200,520*((len(tiles)+2)//3)), 'white')
 for i,t in enumerate(tiles):out.paste(t,((i%3)*400,(i//3)*520))
 out.save(ROOT/'source'/'contact.jpg')
 (ROOT/'source'/'image_info.json').write_text(json.dumps(info,indent=2))
 print(json.dumps({k:{'bbox':v['bbox'],'crop':v['crop']} for k,v in info.items()},indent=2))

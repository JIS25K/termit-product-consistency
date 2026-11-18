from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
import cv2,numpy as np,json
r=Path(__file__).resolve().parent;m=json.loads((r/'data/manifest.json').read_text());records=[]
for item in m['files']:
 p=r/'data/photos'/item['name'];im=Image.open(p).convert('RGB');a=np.array(im);hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
 mask=(((hsv[:,:,0]<22)|(hsv[:,:,0]>170))&(hsv[:,:,1]>80)&(hsv[:,:,2]>35)).astype('uint8')*255
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((35,35),np.uint8));cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 largest=max(cv2.contourArea(c) for c in cs)
 kept=[]
 for c in cs:
  bx,by,bw,bh=cv2.boundingRect(c);cx=bx+bw/2;cy=by+bh/2
  if cv2.contourArea(c)>=.01*largest and .35*im.width<cx<.65*im.width and .10*im.height<cy<.72*im.height:kept.append(c)
 assert kept, p
 points=np.concatenate(kept);x,y,w,h=cv2.boundingRect(points)
 # Fixed generous padding covers black chuck and grip around red housing; no output-dependent fitting.
 pad_x=int(max(.50*w,.35*h));pad_y=int(.35*h);box=(max(0,x-pad_x),max(0,y-pad_y),min(im.width,x+w+pad_x),min(im.height,y+h+pad_y))
 crop=ImageOps.pad(im.crop(box),(768,768),method=Image.Resampling.LANCZOS,color=(238,238,238));d=r/'data'/item['split'];d.mkdir(exist_ok=True);dest=d/(p.stem+'.png');crop.save(dest);records.append({'source':item['name'],'split':item['split'],'box':box,'path':str(dest)})
(r/'data/crops.json').write_text(json.dumps(records,indent=2))
train=[v for v in records if v['split']=='train'];board=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(board)
for i,v in enumerate(sorted(train,key=lambda z:z['source'])):
 im=Image.open(v['path']);im.thumbnail((290,270));x=(i%4)*300;y=(i//4)*300;board.paste(im,(x,y+25));draw.text((x+8,y+5),v['source'],fill='black')
board.save(r/'outputs/training_contact.jpg',quality=90)
print('Prepared',len(records),'photos',flush=True)

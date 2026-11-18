from pathlib import Path
import hashlib,json
import cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text())
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]],float)
FACES={'front':[0,1,2,3],'right':[1,5,6,2],'back':[5,4,7,6],'left':[4,0,3,7],'top':[3,2,6,7],'bottom':[4,5,1,0]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def camera(name):
 cam,angle=Path(name).stem.split('_')
 with h5py.File(ROOT/'data/ycb/calibration.h5') as f:
  key=next(k for k in [cam+'_rgb_K',cam+'_K'] if k in f)
  K=f[key][:].astype(float)
  H=f[f'H_{cam}_from_NP5'][:].astype(float)
 with h5py.File(ROOT/'data/ycb/poses'/f'NP5_{angle}_pose.h5') as f:T=f['H_table_from_reference_camera'][:].astype(float)
 # The published Canon images are 4272x2848. The shared calibration is scaled
 # using the dimensions read from source photos, never target pixels.
 info=json.loads((ROOT/'source/image_info.json').read_text())
 ref=next(iter(info.values()));sx=1600/ref['original_size'][0]
 K[:2]*=sx
 E=H@np.linalg.inv(T)
 return K,E
def projection(name):
 K,E=camera(name);return K@E[:3]
def project(points,P):
 q=np.c_[points,np.ones(len(points))]@P.T
 return q[:,:2]/q[:,2:3]
def verts(params):
 c=params[:3];R=Rotation.from_rotvec(params[3:6]).as_matrix();size=np.exp(params[6:9])
 return (SIGNS*size/2)@R.T+c
def visibility(vertices,name):
 _,E=camera(name);center=np.linalg.inv(E)[:3,3]
 values={}
 for face,indices in FACES.items():
  q=vertices[indices];n=np.cross(q[1]-q[0],q[2]-q[1]);n/=np.linalg.norm(n)
  direction=center-q.mean(0);direction/=np.linalg.norm(direction)
  values[face]=float(n@direction)
 return values
def silhouette(image):
 hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
 mask=(((hsv[:,:,0]<28)|(hsv[:,:,0]>165))&(hsv[:,:,1]>65)&(hsv[:,:,2]>30)).astype(np.uint8)*255
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
 index=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);mask=(labels==index).astype(np.uint8)*255
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 hull=cv2.convexHull(max(contours,key=cv2.contourArea));mask[:]=0;cv2.fillConvexPoly(mask,hull,255)
 return mask,hull

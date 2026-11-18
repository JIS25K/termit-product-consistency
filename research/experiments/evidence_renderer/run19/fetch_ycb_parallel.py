"""Parallel range download, then safe selected extraction. No image inspection."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,io,json,pathlib,re,tarfile,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'data'/'ycb'; OUT.mkdir(parents=True,exist_ok=True)
TMP=pathlib.Path('/private/tmp/run19-ycb-parts');TMP.mkdir(exist_ok=True)
URL='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/003_cracker_box/003_cracker_box_berkeley_rgb_highres.tgz'
SIZE=1506806493;N=12;BLOCK=(SIZE+N-1)//N
def fetch(i):
 start=i*BLOCK;end=min(SIZE,(i+1)*BLOCK)-1;p=TMP/f'{i:02d}.part'
 if p.exists() and p.stat().st_size==end-start+1:return p
 for attempt in range(3):
  try:
   req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=60) as r,p.open('wb') as f:
    if r.status!=206:raise RuntimeError('Range ignored')
    while chunk:=r.read(1024*1024):f.write(chunk)
   assert p.stat().st_size==end-start+1
   return p
  except Exception:
   if attempt==2:raise
class Parts:
 def __init__(self,paths):self.paths=iter(paths);self.f=open(next(self.paths),'rb')
 def read(self,n):
  buf=b''
  while len(buf)<n:
   b=self.f.read(n-len(buf))
   if b:buf+=b
   else:
    self.f.close()
    try:self.f=open(next(self.paths),'rb')
    except StopIteration:break
  return buf
start=time.monotonic()
with ThreadPoolExecutor(max_workers=12) as pool:
 for f in as_completed([pool.submit(fetch,i) for i in range(N)]):print('Downloaded',f.result().name,flush=True)
records=[];count=0
with tarfile.open(fileobj=Parts([TMP/f'{i:02d}.part' for i in range(N)]),mode='r|gz') as archive:
 for m in archive:
  count+=1;n=m.name;p=pathlib.PurePosixPath(n)
  if not m.isfile() or '..' in p.parts or p.is_absolute():continue
  match=re.search(r'/(N[123])_(\d+)\.jpg$',n)
  source=bool(match and int(match[2])%60==0)
  target=bool(match and match[1]=='N2' and int(match[2]) in [30,90,150,210,270,330])
  meta=n.endswith(('.json','.h5','.txt')) and ('pose' in n.lower() or 'calibration' in n.lower())
  if source or target or meta:
   b=archive.extractfile(m).read();dest=OUT.joinpath(*p.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
   records.append({'name':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'role':'heldout' if target else 'source_pool' if source else 'camera_metadata'})
manifest={'url':URL,'license':'CC BY 4.0','publisher':'YCB Object and Model Set, Calli et al.','members':count,'elapsed_seconds':time.monotonic()-start,'files':records}
(ROOT/'data/download_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'completed':True,'members':count,'kept':len(records),'seconds':manifest['elapsed_seconds']}),flush=True)

from pathlib import Path
import urllib.request,concurrent.futures,time,json,os
r=Path(__file__).resolve().parent;out=r/'data/archive_parts';out.mkdir(exist_ok=True)
url='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/035_power_drill/035_power_drill_berkeley_rgb_highres.tgz';size=1588776814;chunk=16*1024*1024;t=time.monotonic()
def fetch(i):
 start=i*chunk;end=min(size,start+chunk)-1;p=out/f'{i:04d}.part'
 if p.exists() and p.stat().st_size==end-start+1:return p.stat().st_size
 for attempt in range(2):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=40) as resp:
    assert resp.status==206,(resp.status,i)
    assert resp.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),resp.headers['Content-Range']
    with p.with_suffix('.partial').open('wb') as f:
     while b:=resp.read(1024*1024):f.write(b)
   assert p.with_suffix('.partial').stat().st_size==end-start+1
   p.with_suffix('.partial').replace(p);return p.stat().st_size
  except Exception as e:
   if attempt:raise
   print(json.dumps({'retry_part':i,'error':str(e)}),flush=True)
   time.sleep(1)
count=(size+chunk-1)//chunk;done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for fut in concurrent.futures.as_completed([ex.submit(fetch,i) for i in range(count)]):
  done+=fut.result();print(json.dumps({'downloaded_mb':round(done/1e6,1),'total_mb':round(size/1e6,1),'seconds':round(time.monotonic()-t,1)}),flush=True)
with (r/'data/photos_archive_complete.tgz').open('wb') as f:
 for i in range(count):
  with (out/f'{i:04d}.part').open('rb') as src:
   while b:=src.read(1024*1024):f.write(b)
(r/'outputs/photo_download_complete.json').write_text(json.dumps({'url':url,'bytes':size,'seconds':time.monotonic()-t,'method':'4 concurrent HTTP byte ranges; Content-Range and length validated'},indent=2));print('ARCHIVE_READY',flush=True)

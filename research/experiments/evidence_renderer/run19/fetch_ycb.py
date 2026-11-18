"""Stream official real-photo archive; keep selected photos and calibration only."""
import hashlib,json,pathlib,re,tarfile,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'data'/'ycb'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/003_cracker_box/003_cracker_box_berkeley_rgb_highres.tgz'
start=time.monotonic(); records=[];count=0
with urllib.request.urlopen(URL,timeout=60) as response:
 with tarfile.open(fileobj=response,mode='r|gz') as archive:
  for member in archive:
   count+=1
   name=member.name; relative=pathlib.PurePosixPath(name)
   if not member.isfile() or '..' in relative.parts or relative.is_absolute(): continue
   match=re.search(r'/(N[123])_(\d+)\.jpg$',name)
   # N2 source every 60 degrees, N2 targets midway. N1/N3 source only for geometry.
   photo=bool(match and int(match[2])%60==0)
   target=bool(match and match[1]=='N2' and int(match[2]) in [30,90,150,210,270,330])
   metadata=name.endswith(('.json','.h5','.txt')) and ('pose' in name.lower() or 'calibration' in name.lower())
   if photo or target or metadata:
    data=archive.extractfile(member).read()
    dest=OUT.joinpath(*relative.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    records.append({'name':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'role':'heldout' if target else 'source' if photo else 'camera_metadata'})
   if count%100==0: print(json.dumps({'members':count,'kept':len(records),'seconds':round(time.monotonic()-start)}),flush=True)
(ROOT/'data'/'download_manifest.json').write_text(json.dumps({'url':URL,'license':'CC BY 4.0','publisher':'YCB Object and Model Set, Calli et al.','members':count,'elapsed_seconds':time.monotonic()-start,'files':records},indent=2))
print(json.dumps({'completed':True,'members':count,'kept':len(records),'seconds':time.monotonic()-start}),flush=True)

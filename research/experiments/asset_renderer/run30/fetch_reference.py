from pathlib import Path,PurePosixPath
import urllib.request,tarfile,re,json,time,hashlib,datetime
R=Path(__file__).resolve().parent;out=R/'data/photos';out.mkdir(exist_ok=True)
url='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/035_power_drill/035_power_drill_berkeley_rgb_highres.tgz'
t=time.monotonic();records=[];count=0;camera=None
with urllib.request.urlopen(url,timeout=60) as r:
 with tarfile.open(fileobj=r,mode='r|gz') as ar:
  for m in ar:
   if not m.isfile():continue
   match=re.search(r'/(N[1-5])_(\d+)\.jpg$',m.name)
   if count<5:print(m.name,flush=True)
   count+=1
   if not match:continue
   if camera is None:camera=match[1]
   angle=int(match[2])
   prior=[int(x['name'].split('_')[1].split('.')[0]) for x in records]
   if match[1]==camera and all(min(abs(angle-q),360-abs(angle-q))>=20 for q in prior):
    b=ar.extractfile(m).read();dest=out/Path(m.name).name;dest.write_bytes(b)
    records.append({'name':dest.name,'archive_member':m.name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    print(json.dumps({'saved':dest.name,'elapsed_s':time.monotonic()-t}),flush=True)
   if len(records)==3:break
(R/'data/reference_manifest.json').write_text(json.dumps({'url':url,'publisher':'YCB / Calli et al.','license':'CC BY 4.0','selected':'First encountered camera; first three archive photos at least 20 degrees apart. Random archive order discovered; fixed-angle streaming interrupted after N1_60 was saved; retained as additional reference. Inspection/shared references, not independent holdout.','files':records,'download_seconds':time.monotonic()-t,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('Reference extraction complete',flush=True)

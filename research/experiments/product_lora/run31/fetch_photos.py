from pathlib import Path
import urllib.request,tarfile,json,time,hashlib,datetime
r=Path(__file__).resolve().parent;out=r/'data/photos';out.mkdir(exist_ok=True)
train=[f'N1_{a}.jpg' for a in [0,30,90,120,150,180,210,270,300,330]]+['N2_0.jpg','N2_180.jpg'];held=['N1_60.jpg','N1_240.jpg'];wanted=set(train+held)
url='https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/035_power_drill/035_power_drill_berkeley_rgb_highres.tgz'
records=[];t=time.monotonic()
with urllib.request.urlopen(url,timeout=60) as resp:
 with tarfile.open(fileobj=resp,mode='r|gz') as ar:
  for m in ar:
   n=Path(m.name).name
   if m.isfile() and n in wanted:
    b=ar.extractfile(m).read();(out/n).write_bytes(b);wanted.remove(n)
    records.append({'name':n,'split':'train' if n in train else 'heldout','sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    print(json.dumps({'saved':n,'remaining':len(wanted),'seconds':round(time.monotonic()-t,1)}),flush=True)
   if not wanted:break
manifest={'source_url':url,'attribution':'YCB / Calli et al.; CC BY 4.0','train_names':train,'heldout_names':held,'min_train_heldout_azimuth_separation_degrees':30,'selection':'fixed angles before training; no quality-based substitutions','files':records,'missing':sorted(wanted),'seconds':time.monotonic()-t,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'data/manifest.json').write_text(json.dumps(manifest,indent=2));assert not wanted,wanted

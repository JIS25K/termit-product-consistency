from pathlib import Path
import time,json,subprocess,sys,os
r=Path(__file__).resolve().parent;start=time.monotonic();summary=r/'outputs/train/summary.json'
while not summary.exists():
 if time.monotonic()-start>2400:raise TimeoutError('Training completion not observed within40 minutes')
 time.sleep(3)
s=json.loads(summary.read_text());assert s['steps']==200 and s['nonfinite_adapter_tensors']==0,s
# Allow the completed trainer to release MPS buffers before starting inference.
time.sleep(5)
env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for arm in ['base','lora']:
 print('Starting valid comparison arm:',arm,flush=True)
 with (r/f'outputs/{arm}_inference.log').open('w') as log:
  subprocess.run([str(r/'.venv/bin/python'),str(r/'infer.py'),arm],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Completed:',arm,flush=True)
print('ALL_LOCAL_COMPARISONS_READY',flush=True)

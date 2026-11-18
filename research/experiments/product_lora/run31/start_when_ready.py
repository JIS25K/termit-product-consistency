from pathlib import Path
import time,subprocess,sys
r=Path(__file__).resolve().parent;start=time.monotonic()
while not (r/'outputs/model_download.json').exists():
 if time.monotonic()-start>240:raise TimeoutError('Model not ready within additional 4 minutes; no pilot started')
 time.sleep(2)
print('Model ready. Starting pretrained SDXL/real photo pilot.',flush=True)
code=subprocess.call(['/bin/zsh',str(r/'run_pilot.sh')],cwd=r)
sys.exit(code)

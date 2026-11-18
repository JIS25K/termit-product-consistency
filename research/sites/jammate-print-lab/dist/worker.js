let py;
self.onmessage=async({data})=>{try{if(!py){postMessage({stage:'처리 엔진 다운로드 중 · 첫 실행은 시간이 걸립니다'});importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');py=await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'});await py.loadPackage(['numpy','opencv-python']);py.FS.mkdirTree('/work/out');const code=await(await fetch('assets/engine.py')).text();py.FS.writeFile('/work/engine.py',code)}postMessage({stage:'원본 대응점과 빈 영역을 검사하고 있습니다'});const cfg=data.cfg;cfg.reference='/work/ref.png';cfg.cases=[{id:'result',input:'/work/input.png',role:'browser prototype input'}];py.FS.writeFile('/work/protocol.json',JSON.stringify(cfg));py.FS.writeFile('/work/ref.png',new Uint8Array(data.ref));py.FS.writeFile('/work/input.png',new Uint8Array(data.target));await py.runPythonAsync(`
import importlib.util,json,numpy as np,cv2
from pathlib import Path
spec=importlib.util.spec_from_file_location('engine','/work/engine.py')
e=importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
e.OUT=Path('/work/out')
e.main()
r=json.loads(Path('/work/out/results.json').read_text())['cases'][0]
before=cv2.imread('/work/input.png')
out=cv2.imread('/work/out/result.png')
mask=cv2.imread('/work/out/result_support.png',0)>0
y,x=np.where(mask)
r['ink_height']=int(y.max()-y.min()+1) if len(y) else 0
if r['decision']=='restored' and r['ink_height']<15:
 r['decision']='abstain_small_text'
 r['reasons']=['인쇄 높이가 잠정 최소 크기 15px 미만입니다']
 out=before.copy();mask[:]=False
r['changed_pixels']=int(np.any(out!=before,axis=2).sum())
r['changed_outside_support']=int((np.any(out!=before,axis=2)&~mask).sum())
r['opencv_version']=cv2.__version__
r['scope']='JamMate large dark print only; placement requires visual review'
cv2.imwrite('/work/out/final.png',out)
vis=before.copy();vis[mask]=[40,90,240]
cv2.imwrite('/work/out/mask.png',vis)
Path('/work/out/report.json').write_text(json.dumps(r,ensure_ascii=False))
`);const out=py.FS.readFile('/work/out/final.png');const mask=py.FS.readFile('/work/out/mask.png');const report=JSON.parse(py.FS.readFile('/work/out/report.json',{encoding:'utf8'}));postMessage({done:true,out,mask,report})}catch(e){postMessage({error:String(e)})}};

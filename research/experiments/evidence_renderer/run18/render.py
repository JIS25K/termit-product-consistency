"""CPU-only cylindrical label reprojection. NumPy + Pillow, no learned model.

Coordinates refer to 487 x 650 working photographs. Surface geometry and camera
poses are explicitly approximate; provenance is not a correctness certificate.
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import platform
import time
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / 'protocol.json').read_text())
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)
W, H = 320, 440

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    arrays, manifest = [], []
    for s in CFG['sources']:
        p = ROOT / s['path']
        im = ImageOps.exif_transpose(Image.open(p)).convert('RGB').resize((487, 650), Image.Resampling.LANCZOS)
        dest = HERE / 'source' / (s['id'] + '.jpg')
        im.save(dest, quality=92, subsampling=0)
        arrays.append(np.asarray(Image.open(dest)).copy())
        manifest.append({**s, 'original_sha256': digest(p), 'working_sha256': digest(dest),
                         'working_size': [487, 650], 'original_size': list(Image.open(p).size)})
    return arrays, manifest

def surface(angle):
    yy, xx = np.mgrid[:H, :W]
    nx = (xx - 159.5) / 134
    valid_x = np.abs(nx) < 1
    a = np.arcsin(np.clip(nx, -1, 1))
    v = (yy - 22 - 19 * np.cos(a)) / 370
    mask = valid_x & (v >= 0) & (v <= 1)
    theta = a + np.deg2rad(angle)
    return theta, v, mask

def correspondence(theta, v, s, pose_offset=0):
    pose = s['pose_deg'] + (pose_offset if s['id'] == 'side' else 0)
    d = (theta - np.deg2rad(pose) + np.pi) % (2*np.pi) - np.pi
    c = np.interp(v, [0, 1/3, 2/3, 1], s['cx'])
    r = np.interp(v, [0, 1/3, 2/3, 1], s['r'])
    x = c + r * np.sin(d)
    top = s['top_base'] + s['top_curve'] * np.cos(d)
    bottom = s['bottom_base'] + s['bottom_curve'] * np.cos(d)
    y = top + v * (bottom - top)
    support = (np.abs(d) <= np.deg2rad(CFG['source_support_half_angle_deg']))
    return x, y, np.where(support, np.cos(d), -np.inf)

def bilinear(im, x, y):
    x = np.clip(x, 0, im.shape[1]-1.001)
    y = np.clip(y, 0, im.shape[0]-1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x-x0)[..., None], (y-y0)[..., None]
    val = im[y0,x0]*(1-fx)*(1-fy) + im[y0,x0+1]*fx*(1-fy) + im[y0+1,x0]*(1-fx)*fy + im[y0+1,x0+1]*fx*fy
    return np.rint(val).astype(np.uint8)

def render(arrays, angle, count=2, pose_offset=0, save=None):
    theta, v, obj = surface(angle)
    maps = [correspondence(theta,v,s,pose_offset) for s in CFG['sources'][:count]]
    scores = np.stack([m[2] for m in maps])
    choice = np.argmax(scores,axis=0)
    known = obj & np.isfinite(np.max(scores,axis=0))
    ids = np.full((H,W), -2, np.int8)
    ids[obj] = -1
    ids[known] = choice[known]
    yy, xx = np.mgrid[:H,:W]
    checker = np.where(((xx//10+yy//10)%2)[...,None], [235,204,213], [249,231,235])
    rgb = np.full((H,W,3), 247, np.uint8)
    rgb[obj & ~known] = checker[obj & ~known]
    uv = np.full((H,W,2),np.nan,np.float32)
    for i,(x,y,_) in enumerate(maps):
        take = known & (choice == i)
        color = bilinear(arrays[i],x,y)
        rgb[take] = color[take]
        uv[take] = np.stack([x,y],axis=-1)[take]
    prov = np.full_like(rgb,247)
    prov[ids == -1] = checker[ids == -1]
    prov[ids == 0] = [74,130,173]
    prov[ids == 1] = [206,144,60]
    stats = {'angle_deg': angle,'sources':count,'pose_offset_deg':pose_offset,
             'surface_pixels':int(obj.sum()),'known_pixels':int(known.sum()),
             'unknown_fraction':float((obj & ~known).sum()/obj.sum()),
             'source_fractions':[float((ids==i).sum()/obj.sum()) for i in range(count)]}
    if save:
        Image.fromarray(rgb).save(OUT / f'{save}.png')
        Image.fromarray(prov).save(OUT / f'{save}_provenance.png')
        np.savez_compressed(OUT/f'{save}_map.npz',source_id=ids,source_xy=uv)
    return rgb, ids, uv, stats

def fraction(angle,count,offset):
    theta,v,obj=surface(angle)
    known=np.zeros_like(obj)
    for s in CFG['sources'][:count]:
        known |= np.isfinite(correspondence(theta,v,s,offset)[2])
    return float((obj & ~known).sum()/obj.sum())

def plan(request,tolerance,count):
    allowed = range(max(-60,request-tolerance),min(150,request+tolerance)+1)
    scored=[]
    for a in allowed:
        worst=max(fraction(a,count,o) for o in CFG['manual_pose_sensitivity_deg'])
        scored.append((a,worst))
    feasible=[(a,e) for a,e in scored if e<=CFG['acceptance_rule']['max_unknown_fraction']]
    if feasible:
        best=min(feasible,key=lambda x:(abs(x[0]-request),x[1]))
        return {'request':request,'tolerance':tolerance,'sources':count,'status':'preview_allowed',
                'chosen_angle':best[0],'worst_unknown_fraction':best[1]}
    best=min(scored,key=lambda x:(x[1],abs(x[0]-request)))
    return {'request':request,'tolerance':tolerance,'sources':count,'status':'need_more_evidence',
            'chosen_angle':None,'best_within_range':best[0],'worst_unknown_fraction':best[1]}

def verify(arrays, cases, manifest):
    checks=[]
    # Independent scalar reconstruction using persisted float32 coordinates.
    max_error=0
    sampled=0
    for name in cases:
        data=np.load(OUT/f'{name}_map.npz')
        ids,uv=data['source_id'],data['source_xy']
        rgb=np.asarray(Image.open(OUT/f'{name}.png'))
        ys,xs=np.where(ids>=0)
        for k in np.linspace(0,len(ys)-1,min(250,len(ys)),dtype=int):
            y,x=int(ys[k]),int(xs[k]); sid=int(ids[y,x]); sx,sy=map(float,uv[y,x])
            ix,iy=int(sx),int(sy); dx,dy=sx-ix,sy-iy
            im=arrays[sid]
            rec=np.rint(im[iy,ix].astype(float)*(1-dx)*(1-dy)+im[iy,ix+1]*dx*(1-dy)+im[iy+1,ix]*(1-dx)*dy+im[iy+1,ix+1]*dx*dy).astype(int)
            max_error=max(max_error,int(np.max(np.abs(rec-rgb[y,x].astype(int)))))
            sampled+=1
    checks.append({'name':'independent_provenance_reconstruction','sampled_pixels':sampled,'max_channel_error_8bit':max_error,'passed':max_error<=1})
    # Independent analytical expectation: one source, 0 degrees, support +-70.
    empirical=fraction(0,1,0)
    expected=1-np.sin(np.deg2rad(70))
    checks.append({'name':'analytic_single_view_coverage','measured_unknown':empirical,'analytic_unknown':float(expected),'passed':bool(abs(empirical-expected)<0.01)})
    checks.append({'name':'evidence_addition_cannot_reduce_coverage','passed':all(fraction(a,2,0)<=fraction(a,1,0) for a in range(-60,151,5))})
    plans=[plan(30,0,2),plan(90,0,2),plan(90,45,2),plan(30,0,1)]
    checks.append({'name':'strict_pose_request_with_insufficient_evidence_is_held','passed':plans[1]['chosen_angle'] is None})
    checks.append({'name':'tolerant_pose_stays_within_authorized_range','passed':plans[2]['chosen_angle'] is not None and abs(plans[2]['chosen_angle']-90)<=45})
    checks.append({'name':'input_originals_unchanged','passed':all(digest(ROOT/m['path'])==m['original_sha256'] for m in manifest)})
    return {'checks':checks,'all_passed':all(c['passed'] for c in checks),'plans':plans,
            'scope':'계산과 출처 기록 검증. 실제 제품 정확도 검증이 아님.'}

def main():
    start=time.perf_counter()
    arrays,manifest=prepare()
    metrics=[]; names=[]
    for count in [1,2]:
        for a in CFG['experiment_angles_deg']:
            name=f's{count}_a{a:+04d}'
            _,_,_,stats=render(arrays,a,count,save=name)
            stats['worst_unknown_fraction']=max(fraction(a,count,o) for o in CFG['manual_pose_sensitivity_deg'])
            metrics.append(stats); names.append(name)
    validation=verify(arrays,names,manifest)
    # Used by the CPU browser renderer; every source is a local working photo.
    ui={'config':CFG,'sources':[], 'metrics':metrics,'plans':validation['plans']}
    for m in manifest:
        p=HERE/'source'/(m['id']+'.jpg')
        ui['sources'].append({**m,'data_uri':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()})
    (OUT/'ui_data.json').write_text(json.dumps(ui,ensure_ascii=False))
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    runtime={'python':platform.python_version(),'numpy':np.__version__,'pillow':Image.__version__,
             'processor':platform.machine(),'seconds':time.perf_counter()-start,'gpu_used':False,
             'generative_model_calls':0,'code_sha256':digest(Path(__file__)), 'protocol_sha256':digest(HERE/'protocol.json')}
    (OUT/'runtime.json').write_text(json.dumps(runtime,indent=2))
    template=(HERE/'viewer.template.html').read_text()
    target=(Path(__file__).resolve().parent / 'preview.html')
    target.write_text(template.replace('__PAYLOAD__',json.dumps(ui,ensure_ascii=False)))
    print(json.dumps({'validation':validation,'runtime':runtime,'visualization_bytes':target.stat().st_size},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

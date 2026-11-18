"""Render and precommit a source-only view choice before any target RGB exists."""
import datetime,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
import common,produce
from common import ROOT,CFG,FACES,sha,dump,project,projection,visibility

OUT=ROOT/'outputs'
assert not (OUT/'eval_first_access.json').exists()
assert not (OUT/'production_frozen.json').exists()
assert not any((ROOT/'data/ycb'/n).exists() for n in CFG['target_images'])
prior=ROOT.parent/'run19'
for name in ['common.py','produce.py']:
    assert sha(ROOT/name)==sha(prior/name),'Existing renderer changed'
assert sha(OUT/'model.json')==sha(prior/'outputs/model.json')
xyz=np.array(json.loads((OUT/'model.json').read_text())['vertices'])
sources={name:produce.read_source(name) for name in CFG['source_images']}

def jacobian(H,p):
    x,y=p;den=H[2,0]*x+H[2,1]*y+H[2,2]
    num=H[:2,:2]@p+H[:2,2]
    return (H[:2,:2]*den-num[:,None]*H[2,:2])/den**2

start=time.perf_counter();rows=[];tiles=[]
for name in CFG['target_images']:
    produce.render_target(xyz,name,sources)
    stem=Path(name).stem;meta=json.loads((OUT/f'{stem}_render.json').read_text())
    points=project(xyz,projection(name));v=visibility(xyz,name)
    visible_areas={f:abs(cv2.contourArea(points[ids].astype(np.float32))) for f,ids in FACES.items() if v[f]>.015}
    total=sum(visible_areas.values());covered=0;weighted=0;details=[]
    for face in meta['faces']:
        H=np.array(face['H_source_to_target']);quad=np.array(face['source_quad'])
        risks=[]
        for u in np.linspace(.1,.9,5):
            for w in np.linspace(.1,.9,5):
                p=(1-u)*(1-w)*quad[0]+u*(1-w)*quad[1]+u*w*quad[2]+(1-u)*w*quad[3]
                singular=np.linalg.svd(jacobian(H,p),compute_uv=False)
                risks.append(float(np.abs(np.log2(singular)).mean()))
        area=visible_areas[face['face']];cost=float(np.mean(risks))
        covered+=area;weighted+=area*cost
        details.append({'face':face['face'],'area_fraction':area/total,'warp_cost':cost,'source':face['source']})
    with np.load(OUT/f'{stem}_provenance.npz') as z:
        valid=z['source_id']>=0;xy=z['source_xy'][valid]
        out=float(np.mean((xy[:,0]<0)|(xy[:,0]>=1600)|(xy[:,1]<0)|(xy[:,1]>=1067)))
    uncovered=max(0.,1-covered/total);warp=weighted/max(covered,1)
    angle=int(stem.split('_')[1]);distance=min(min(abs(angle-int(Path(s).stem.split('_')[1])),360-abs(angle-int(Path(s).stem.split('_')[1]))) for s in sources)
    front=visible_areas.get('front',0)/total;right=visible_areas.get('right',0)/total
    eligible=front>=.4 and right>=.05 and distance>=12
    score=10*(uncovered+out)+warp
    rows.append({'target':name,'angle':angle,'eligible':bool(eligible),'risk_score':score,'uncovered_fraction':uncovered,'out_of_bounds_fraction':out,'warp_cost':warp,'front_fraction':front,'right_fraction':right,'minimum_source_distance_degrees':distance,'faces':details})
    im=Image.open(OUT/f'{stem}_render.png');lo=points.min(0)-30;hi=points.max(0)+30
    crop=im.crop(tuple(np.r_[lo,hi].astype(int)));crop.thumbnail((290,420))
    tile=Image.new('RGB',(310,470),'white');tile.paste(crop,((310-crop.width)//2,25))
    d=ImageDraw.Draw(tile);d.text((10,6),f'{angle} deg | risk {score:.4f}',fill='black');d.text((10,448),'RENDER ONLY / target RGB unopened',fill='black');tiles.append(tile)

ranked=sorted([r for r in rows if r['eligible']],key=lambda r:(r['risk_score'],abs(r['angle']-33),r['angle']))
assert ranked,'No eligible view: record infeasible brief, do not select an invalid candidate'
choice={'chosen':ranked[0]['target'],'baseline':CFG['baseline'],'ranking':[r['target'] for r in ranked],'rows':rows,'selected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_rgb_accessed':False,'selection_seconds':time.perf_counter()-start}
dump(OUT/'selection.json',choice);dump(OUT/'production_rgb_read_log.json',produce.READ_LOG)
sheet=Image.new('RGB',(930,940),'white')
for i,t in enumerate(tiles):sheet.paste(t,((i%3)*310,(i//3)*470))
sheet.save(OUT/'candidates_before_ground_truth.jpg')
files=[ROOT/n for n in ['protocol.json','common.py','produce.py','run_selection.py','source/corners.json','source/logo_regions.json','source/image_info.json','data/ycb/calibration.h5']]
files += [ROOT/'source'/n for n in sources]
files += [ROOT/'data/ycb/poses'/f'NP5_{Path(n).stem.split("_")[1]}_pose.h5' for n in CFG['source_images']+CFG['target_images']]
files += sorted(OUT.iterdir())
frozen={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_rgb_used':[],'source_rgb_used':CFG['source_images'],'chosen':choice['chosen'],'baseline':choice['baseline'],'renderer_identical_to_run19':True,'model_identical_to_run19':True,'seconds':time.perf_counter()-start,'files':{str(f.relative_to(ROOT)):sha(f) for f in files}}
dump(OUT/'production_frozen.json',frozen)
print(json.dumps({'chosen':choice['chosen'],'baseline':choice['baseline'],'rows':rows,'files_frozen':len(files),'seconds':frozen['seconds']},ensure_ascii=False,indent=2))

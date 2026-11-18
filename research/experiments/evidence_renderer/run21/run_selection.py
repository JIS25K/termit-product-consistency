"""One predeclared structural-score addition, no target RGB and no renderer edits."""
import datetime,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import least_squares
import produce
from common import ROOT,CFG,FACES,sha,dump,project,projection,visibility

OUT=ROOT/'outputs';prior=ROOT.parent/'run19'
assert not (OUT/'production_frozen.json').exists()
assert not (OUT/'eval_first_access.json').exists()
assert not any((ROOT/'data/ycb'/n).exists() for n in CFG['target_images'])
for n in ['common.py','produce.py']:assert sha(ROOT/n)==sha(prior/n)
assert sha(OUT/'model.json')==sha(prior/'outputs/model.json')
xyz=np.array(json.loads((OUT/'model.json').read_text())['vertices'])
sources={n:produce.read_source(n) for n in CFG['source_images']}
annotations=json.loads((ROOT/'source/corners.json').read_text())
start=time.perf_counter();reference=[];fit_notes=[]
for corner in range(8):
    obs=[(n,np.array(a[str(corner)],float)) for n,a in annotations.items() if str(corner) in a]
    assert all(n in CFG['source_images'] for n,_ in obs)
    equations=[]
    for n,(x,y) in obs:
        P=projection(n);equations.extend([x*P[2]-P[0],y*P[2]-P[1]])
    _,_,vt=np.linalg.svd(equations);p0=vt[-1,:3]/vt[-1,3]
    def residual(p):return np.concatenate([project(p[None,:],projection(n))[0]-uv for n,uv in obs])
    fitted=least_squares(residual,p0,loss='soft_l1',f_scale=2,max_nfev=300)
    reference.append(fitted.x)
    fit_notes.append({'corner':corner,'observations':len(obs),'source_median_residual_px':float(np.median(np.linalg.norm(residual(fitted.x).reshape(-1,2),axis=1)))})
reference=np.array(reference)
dump(OUT/'source_reference_for_scoring.json',{'vertices':reference.tolist(),'source_fit':fit_notes,'used_for_rendering':False,'target_pixels_used':False})

def jacobian(H,p):
    den=H[2,:2]@p+H[2,2];num=H[:2,:2]@p+H[:2,2]
    return (H[:2,:2]*den-num[:,None]*H[2,:2])/den**2

def segment_distance(points,a,b):
    v=b-a;t=np.clip(((points-a)*v).sum(1)/(v@v),0,1)
    return np.linalg.norm(points-a-t[:,None]*v,axis=1)

rows=[];tiles=[]
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
for name in CFG['target_images']:
    produce.render_target(xyz,name,sources)
    stem=Path(name).stem;meta=json.loads((OUT/f'{stem}_render.json').read_text())
    points=project(xyz,projection(name));refpoints=project(reference,projection(name));v=visibility(xyz,name)
    visible={f:FACES[f] for f in FACES if v[f]>.015}
    areas={f:abs(cv2.contourArea(points[ids].astype(np.float32))) for f,ids in visible.items()}
    total=sum(areas.values());covered=0.;weighted=0.;details=[]
    for face in meta['faces']:
        H=np.array(face['H_source_to_target']);quad=np.array(face['source_quad']);risks=[]
        for u in np.linspace(.1,.9,5):
            for w in np.linspace(.1,.9,5):
                p=(1-u)*(1-w)*quad[0]+u*(1-w)*quad[1]+u*w*quad[2]+(1-u)*w*quad[3]
                risks.append(float(np.abs(np.log2(np.linalg.svd(jacobian(H,p),compute_uv=False))).mean()))
        area=areas[face['face']];cost=float(np.mean(risks));covered+=area;weighted+=area*cost
        details.append({'face':face['face'],'source':face['source'],'area_fraction':area/total,'warp_cost':cost})
    with np.load(OUT/f'{stem}_provenance.npz') as z:
        xy=z['source_xy'][z['source_id']>=0]
        out=float(np.mean((xy[:,0]<0)|(xy[:,0]>=1600)|(xy[:,1]<0)|(xy[:,1]>=1067)))
    uncovered=max(0.,1-covered/total);warp=weighted/max(covered,1);old_score=10*(uncovered+out)+warp
    ids=sorted({i for group in visible.values() for i in group})
    corner_errors=np.linalg.norm(points[ids]-refpoints[ids],axis=1)
    seams=[];seam_errors=[];groups=list(visible.values())
    for i,group in enumerate(groups):
        for other in groups[i+1:]:
            edge=sorted(set(group)&set(other))
            if len(edge)!=2:continue
            a,b=points[edge];ra,rb=refpoints[edge]
            sampled=a+np.linspace(0,1,100)[:,None]*(b-a)
            errors=segment_distance(sampled,ra,rb);seam_errors.extend(errors)
            seams.append({'vertices':edge,'median_displacement_px':float(np.median(errors))})
    corner=float(np.median(corner_errors));seam=float(np.median(seam_errors));structure=.5*(corner+seam)
    score=old_score+structure/5
    angle=int(stem.split('_')[1]);distance=min(min(abs(angle-int(Path(n).stem.split('_')[1])),360-abs(angle-int(Path(n).stem.split('_')[1]))) for n in sources)
    front=areas.get('front',0)/total;right=areas.get('right',0)/total;eligible=front>=.4 and right>=.05 and distance>=12
    rows.append({'target':name,'angle':angle,'eligible':bool(eligible),'risk_score':score,'old_score':old_score,'structure_px':structure,'reference_corner_displacement_px':corner,'reference_seam_displacement_px':seam,'uncovered_fraction':uncovered,'out_of_bounds_fraction':out,'warp_cost':warp,'front_fraction':front,'right_fraction':right,'minimum_source_distance_degrees':distance,'faces':details,'reference_seams':seams})
    im=Image.open(OUT/f'{stem}_render.png');lo=points.min(0)-30;hi=points.max(0)+30;crop=im.crop(tuple(np.r_[lo,hi].astype(int)));crop.thumbnail((360,460))
    tile=Image.new('RGB',(400,540),'white');tile.paste(crop,((400-crop.width)//2,60));d=ImageDraw.Draw(tile)
    d.text((12,8),f'{angle} deg | combined {score:.3f}',font=font,fill='black');d.text((12,28),f'Old {old_score:.3f} | source structure {structure:.2f}px',font=font,fill='black');tiles.append(tile)

rank=lambda key:sorted([r for r in rows if r['eligible']],key=lambda r:(r[key],abs(r['angle']-33),r['angle']))
newrank=rank('risk_score');oldrank=rank('old_score');assert newrank
choice={'chosen':newrank[0]['target'],'baseline':oldrank[0]['target'],'baseline_kind':'unchanged Run20 score on the same unseen pool','fixed_angle_control':CFG['fixed_angle_control'],'ranking':[r['target'] for r in newrank],'old_ranking':[r['target'] for r in oldrank],'rows':rows,'selected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_rgb_accessed':False,'selection_seconds':time.perf_counter()-start}
dump(OUT/'selection.json',choice);dump(OUT/'production_rgb_read_log.json',produce.READ_LOG)
sheet=Image.new('RGB',(800,1080),'white')
for i,t in enumerate(tiles):sheet.paste(t,((i%2)*400,(i//2)*540))
sheet.save(OUT/'candidates_before_ground_truth.jpg')
files=[ROOT/n for n in ['protocol.json','common.py','produce.py','run_selection.py','source/corners.json','source/logo_regions.json','source/image_info.json','data/ycb/calibration.h5']]
files += [ROOT/'source'/n for n in sources]
files += [ROOT/'data/ycb/poses'/f'NP5_{Path(n).stem.split("_")[1]}_pose.h5' for n in CFG['source_images']+CFG['target_images']]
files += sorted(OUT.iterdir())
frozen={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_rgb_used':[],'source_rgb_used':CFG['source_images'],'chosen':choice['chosen'],'baseline':choice['baseline'],'renderer_identical_to_run19':True,'model_identical_to_run19':True,'seconds':time.perf_counter()-start,'files':{str(f.relative_to(ROOT)):sha(f) for f in files}}
dump(OUT/'production_frozen.json',frozen)
print(json.dumps({'chosen':choice['chosen'],'old_score_winner':choice['baseline'],'fixed_angle_control':choice['fixed_angle_control'],'scores':[{'angle':r['angle'],'old':r['old_score'],'structure_px':r['structure_px'],'new':r['risk_score']} for r in rows],'frozen_files':len(files),'seconds':frozen['seconds']},indent=2))

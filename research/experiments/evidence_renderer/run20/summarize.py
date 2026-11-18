"""Small, shareable evidence figures from frozen renders and heldout evaluation."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,sha,dump

EV=ROOT/'evaluation';OUT=ROOT/'outputs'
selection=json.loads((OUT/'selection.json').read_text())
metrics=json.loads((EV/'metrics.json').read_text())
info=json.loads((EV/'image_info.json').read_text())
rows={r['target']:r for r in metrics['targets']}
frozen=json.loads((OUT/'production_frozen.json').read_text())
assert all(sha(ROOT/n)==digest for n,digest in frozen['files'].items())
fontpath='/System/Library/Fonts/AppleSDGothicNeo.ttc'
font=lambda size:ImageFont.truetype(fontpath,size)

def pair_panel(name,label):
    stem=Path(name).stem;row=rows[name];crop=info[name]['crop']
    panel=Image.new('RGB',(1000,690),'white');d=ImageDraw.Draw(panel)
    d.text((20,8),label,font=font(24),fill='#161616')
    for j,(p,title) in enumerate([(EV/f'{stem}.png','제작에 쓰지 않은 실사'),(OUT/f'{stem}_render.png','기존 렌더러 결과')]):
        im=Image.open(p).convert('RGB').crop(crop);im.thumbnail((460,550))
        panel.paste(im,(j*500+(500-im.width)//2,90))
        d.text((j*500+20,52),title,font=font(20),fill='#454545')
    line=f"중앙 오차: 로고 {row['logo']['median_px']:.1f}px · 모서리 {row['corner']['median_px']:.1f}px · 접합선 {row['seam']['median_px']:.1f}px"
    d.text((20,650),line,font=font(21),fill='#161616')
    return panel

chosen=selection['chosen'];baseline=selection['baseline']
panel=Image.new('RGB',(1000,1380),'white')
panel.paste(pair_panel(chosen,f"사전 선택: {Path(chosen).stem.split('_')[1]}°"),(0,0))
panel.paste(pair_panel(baseline,f"고정 비교 기준: {Path(baseline).stem.split('_')[1]}°"),(0,690))
panel.save(EV/'selection_vs_baseline.jpg',quality=92)

# Both crops use the same coordinates and magnification for truth and prediction.
top=Image.new('RGB',(1200,650),'white');d=ImageDraw.Draw(top)
for i,(name,label) in enumerate([(chosen,'사전 선택'),(baseline,'고정 기준')]):
    x0,y0,x1,y1=info[name]['crop'];box=[x0,y0,x1,y0+round((y1-y0)*.32)]
    stem=Path(name).stem
    d.text((i*600+20,12),f'{label} {stem.split("_")[1]}°',font=font(24),fill='black')
    for j,(p,label2) in enumerate([(EV/f'{stem}.png','실사'),(OUT/f'{stem}_render.png','재투영')]):
        im=Image.open(p).convert('RGB').crop(box)
        im=im.resize((560,round(im.height*560/im.width)),Image.Resampling.LANCZOS)
        top.paste(im,(i*600+20,85+j*280));d.text((i*600+20,52+j*280),label2,font=font(20),fill='black')
top.save(EV/'top_detail.jpg',quality=92)

summary={'selected_before_target_decode':chosen,'baseline':baseline,'frozen_files_unchanged':True,'views':[],'best_measured_by_metric':{}}
for name in selection['ranking']:
    r=rows[name];pred=next(x for x in selection['rows'] if x['target']==name)
    summary['views'].append({'target':name,'predicted_risk':pred['risk_score'],'logo_matches':r['logo']['n'],'logo_median_px':r['logo']['median_px'],'corner_median_px':r['corner']['median_px'],'seam_median_px':r['seam']['median_px'],'corner_p95_px':r['corner']['p95_px'],'seam_p95_px':r['seam']['p95_px']})
for key in ['logo','corner','seam']:
    valid=[r for r in metrics['targets'] if r[key]['median_px'] is not None and (key!='logo' or r[key]['n']>=6)]
    summary['best_measured_by_metric'][key]=min(valid,key=lambda r:r[key]['median_px'])['target']
summary['selected_minus_baseline_px']={key:rows[chosen][key]['median_px']-rows[baseline][key]['median_px'] for key in ['logo','corner','seam']}
dump(EV/'selection_result.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))

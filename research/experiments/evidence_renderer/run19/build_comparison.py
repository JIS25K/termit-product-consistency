"""Package existing, frozen images for inspection; never alter production."""
import base64
import io
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEST = (Path(__file__).resolve().parent / 'comparison.html')
metrics = json.loads((ROOT/'evaluation/metrics.json').read_text())
info = json.loads((ROOT/'evaluation/image_info.json').read_text())

def encoded(image):
    image.thumbnail((440,560),Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer,format='JPEG',quality=78,optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

payload = {}
for row in metrics['targets']:
    name = row['target']
    stem = Path(name).stem
    angle = stem.split('_')[1]
    crop = info[name]['crop']
    x,y,x1,y1 = crop
    real = Image.open(ROOT/'evaluation'/f'{stem}.png').convert('RGB').crop(crop)
    prediction = Image.open(ROOT/'outputs'/f'{stem}_render.png').convert('RGB').crop(crop)
    top = (0,0,real.width,round(real.height*.39))
    def local(point):
        return [round(point[0]-x,2),round(point[1]-y,2)]
    payload[angle] = {
        'width':real.width,'height':real.height,
        'realTop':encoded(real.crop(top)),
        'renderTop':encoded(prediction.crop(top)),
        'real':encoded(real.copy()),'render':encoded(prediction.copy()),
        'logoN':row['logo']['n'],'logoMedian':row['logo']['median_px'],
        'cornerP95':row['corner']['p95_px'],'seamP95':row['seam']['p95_px'],
        'corners':[{'observed':local(p['observed']),'predicted':local(p['predicted'])} for p in row['corner_details']],
        'seams':[{'observed':[local(p) for p in s['gt_segment']],'predicted':[local(p) for p in s['pred_segment']]} for s in row['seams']],
    }
template = (ROOT/'comparison.template.html').read_text()
assert template.count('__BOX_PAYLOAD__') == 1
fragment = template.replace('__BOX_PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
assert len(fragment.encode()) < 1_000_000
DEST.write_text(fragment)
print(json.dumps({'path':str(DEST),'bytes':len(fragment.encode()),'angles':list(payload),'bbox_diagonals':[r['bbox_diagonal_px'] for r in metrics['targets']]}))

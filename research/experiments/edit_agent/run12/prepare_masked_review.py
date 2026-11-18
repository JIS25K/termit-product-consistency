from pathlib import Path
import json,shutil,sys,hashlib
p=Path(__file__).resolve().parent;step=int(sys.argv[1]);proto=json.loads((p/'frozen/protocol.json').read_text());dest=p/'masked_review';dest.mkdir(exist_ok=True)
map_={'character':{'P':'agent','Q':'baseline'},'bottle':{'P':'baseline','Q':'agent'}}
(p/'private/masked_condition_map.json').write_text(json.dumps(map_,indent=2))
products={}
def copy(src,out):
 src=Path(src);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():shutil.copy2(src,out)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==hashlib.sha256(out.read_bytes()).hexdigest()
 return str(out)
for product,meta in proto['products'].items():
 assets={'actual_front':copy(meta['front'],dest/product/'actual_front.jpg'),'actual_side':copy(meta['side'],dest/product/'actual_side.jpg'),'v0':copy(p/'images'/product/'shared/v0.png',dest/product/'v0.png')}
 chains={}
 for label,arm in map_[product].items():
  chains[label]={f'v{i}':copy(p/'images'/product/arm/f'v{i}.png',dest/product/label/f'v{i}.png') for i in range(1,step+1)}
 products[product]={'actual_assets':assets,'target_side_description':meta['target'],'invariants':meta['invariants'],'chains':chains}
manifest={'completed_steps':step,'requests':proto['track_C']['requests'][:step],'products':products,'note':'P/Q are arbitrary masked condition labels; do not infer a shared identity across products. Evaluate each output, not which condition you prefer. Shared initial v0 differences from actual product are baseline defects, not arm-specific drift.'}
(dest/f'manifest_step{step}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Masked review prepared',step)

from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import json,html
r=Path(__file__).resolve().parent;o=r/'outputs'
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',25);small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
refs={'A':r/'data/heldout/N1_60.png','B':r/'data/heldout/N1_240.png'}
# All outputs retain original framing; only uniform display resizing is applied.
def board(arms,rows,path,title):
 cell=480;top=90;rowheight=520;im=Image.new('RGB',(cell*len(arms),top+rowheight*len(rows)),'#f3f3f1');d=ImageDraw.Draw(im);d.text((18,14),title,font=font,fill='#141414')
 for col,(key,label) in enumerate(arms):d.text((col*cell+16,55),label,font=small,fill='#303030')
 for row,(view,sample) in enumerate(rows):
  for col,(arm,label) in enumerate(arms):
   p=refs[view] if arm=='real' else o/arm/f'{view}_{sample}.png'
   a=Image.open(p).convert('RGB');a=ImageOps.contain(a,(cell-12,cell-12));x=col*cell+(cell-a.width)//2;y=top+row*rowheight+24;im.paste(a,(x,y));d.text((col*cell+12,top+row*rowheight),f'{view} / sample {sample}' if arm!='real' else f'{view} / held-out real photo',font=small,fill='#303030')
 im.save(path,quality=94)
board([('real','REAL REFERENCE'),('ordinary','ORDINARY CALL'),('lora','PRODUCT LoRA')],[('A',1),('B',1)],o/'comparison_primary.jpg','RUN 31 | First sample of each view (fixed in advance)')
board([('real','REAL REFERENCE'),('ordinary','ORDINARY CALL'),('lora','PRODUCT LoRA'),('base','SDXL WITHOUT LoRA')],[(v,s) for v in ['A','B'] for s in [1,2]],o/'comparison_all.jpg','RUN 31 | All first outputs: 2 views x 2 samples x 3 methods')
print('Saved primary and all-output comparison boards.')

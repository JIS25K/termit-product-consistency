from pathlib import Path
import torch,json,time,sys,hashlib,datetime
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig,set_peft_model_state_dict
from safetensors.torch import load_file
r=Path(__file__).resolve().parent;arm=sys.argv[1];assert arm in ['base','lora'];q=json.loads((r/'quality_protocol.json').read_text());o=r/'outputs'/arm;o.mkdir(exist_ok=True);torch.set_num_threads(4)
t=time.monotonic();pipe=StableDiffusionXLPipeline.from_pretrained(r/'model',variant='fp16',torch_dtype=torch.float16,use_safetensors=True,local_files_only=True,add_watermarker=False)
if arm=='lora':
 pipe.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
 state=load_file(r/'outputs/train/product_adapter.safetensors');result=set_peft_model_state_dict(pipe.unet,state,adapter_name='default');assert not result.unexpected_keys,result.unexpected_keys
 assert len([n for n,p in pipe.unet.named_parameters() if 'lora_' in n])==len(state)
pipe.to('mps',dtype=torch.float16);records=[]
for view in ['A','B']:
 prompt=q['generation_prompts'][view]
 assert all(len(tok(prompt).input_ids)<=tok.model_max_length for tok in [pipe.tokenizer,pipe.tokenizer_2]), 'Prompt truncation forbidden'
 for index,seed in enumerate(q['comparison']['sdxl_seeds'],1):
  start=time.monotonic();gen=torch.Generator(device='cpu').manual_seed(seed)
  im=pipe(prompt=prompt,negative_prompt='illustration, drawing, cartoon, cropped product, multiple drills, human hands, distorted geometry, extra handles, extra battery, extra parts, watermark, advertising headline',height=768,width=768,num_inference_steps=30,guidance_scale=7,generator=gen).images[0]
  dest=o/f'{view}_{index}.png';im.save(dest);row={'arm':arm,'view':view,'sample':index,'seed':seed,'seconds':time.monotonic()-start,'prompt':prompt,'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()};records.append(row);(o/'manifest.json').write_text(json.dumps(records,indent=2));print('OUTPUT '+json.dumps({k:v for k,v in row.items() if k!='prompt'}),flush=True)
print(json.dumps({'arm':arm,'total_seconds':time.monotonic()-t,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)

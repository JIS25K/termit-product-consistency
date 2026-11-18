from pathlib import Path
import torch,time,json,statistics,datetime,gc
from diffusers import UNet2DConditionModel
from peft import LoraConfig
r=Path(__file__).resolve().parent;o=r/'outputs/architecture_probe';o.mkdir(exist_ok=True)
torch.manual_seed(31);torch.set_num_threads(4);t=time.monotonic()
print('Constructing SDXL architecture with RANDOM weights: runtime only, no product learning claim',flush=True)
config=json.loads((r/'model/unet/config.json').read_text());torch.set_default_dtype(torch.float16)
unet=UNet2DConditionModel.from_config(config);torch.set_default_dtype(torch.float32)
unet.requires_grad_(False);unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
unet.to('mps',dtype=torch.float16)
for p in unet.parameters():
 if p.requires_grad:p.data=p.data.float()
unet.enable_gradient_checkpointing();unet.train();params=[p for p in unet.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4)
print(json.dumps({'stage':'constructed','seconds':time.monotonic()-t,'parameters':sum(p.numel() for p in unet.parameters()),'trainable':sum(p.numel() for p in params)}),flush=True)
rows=[]
for i in range(4):
 torch.mps.synchronize();start=time.monotonic();x=torch.randn(1,4,96,96,device='mps',dtype=torch.float16);target=torch.randn_like(x);context=torch.randn(1,77,2048,device='mps',dtype=torch.float16)
 pred=unet(x,torch.tensor([500],device='mps'),context,added_cond_kwargs={'text_embeds':torch.randn(1,1280,device='mps',dtype=torch.float16),'time_ids':torch.tensor([[768,768,0,0,768,768]],device='mps',dtype=torch.float16)}).sample
 loss=(pred.float()-target.float()).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();opt.zero_grad();torch.mps.synchronize()
 row={'step':i+1,'seconds':time.monotonic()-start,'loss':loss.item(),'mps_allocated_gb':torch.mps.current_allocated_memory()/1e9,'mps_driver_gb':torch.mps.driver_allocated_memory()/1e9};rows.append(row);print(json.dumps(row),flush=True);(o/'steps.json').write_text(json.dumps(rows,indent=2))
 if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
 if row['mps_driver_gb']>18:raise RuntimeError('18 GB gate exceeded')
 if i>=1 and statistics.median(a['seconds'] for a in rows[1:])>12:
  print('Runtime gate exceeded: stop early without further steps',flush=True);break
summary={'type':'random-initialized SDXL architecture + real LoRA backward/update, synthetic latents/text','product_training':False,'rows':rows,'median_seconds_after_warmup':statistics.median(a['seconds'] for a in rows[1:]),'nonzero_lora_B_elements':sum(torch.count_nonzero(p).item() for n,p in unet.named_parameters() if 'lora_B' in n),'projection_200_steps_seconds':statistics.median(a['seconds'] for a in rows[1:])*200,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(o/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE '+json.dumps(summary),flush=True)

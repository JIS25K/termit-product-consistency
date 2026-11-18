from pathlib import Path
from huggingface_hub import snapshot_download,HfApi
import json,time,datetime
r=Path(__file__).resolve().parent;t=time.monotonic();repo='stabilityai/stable-diffusion-xl-base-1.0'
sha=HfApi().model_info(repo).sha
print('Downloading pinned model',repo,sha,flush=True)
p=snapshot_download(repo,revision=sha,local_dir=r/'model',allow_patterns=['model_index.json','scheduler/*','tokenizer/*','tokenizer_2/*','text_encoder/config.json','text_encoder/model.fp16.safetensors','text_encoder_2/config.json','text_encoder_2/model.fp16.safetensors','unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors','vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors','LICENSE.md','README.md'],max_workers=4)
(r/'outputs/model_download.json').write_text(json.dumps({'repo':repo,'revision':sha,'seconds':time.monotonic()-t,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(p)},indent=2));print('MODEL READY',flush=True)

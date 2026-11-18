"""One-image diagnostic with separate encoder and denoiser processes.

Experimental T4 fallback: no simultaneous residency of the two large models.
Run encode then denoise with the same config/output; no model or prompt changes.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from run_experiment import ROOT, LORA_FILE, digest, prepare, execute, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['encode', 'denoise'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plan = prepare(args.config, args.out)
    c = plan['config']
    if not c.get('diagnostic_only') or len(plan['jobs']) != 1 or c['precision'] != 'nf4_fp32':
        raise ValueError('Only a single nf4_fp32 diagnostic is supported.')
    out = Path(args.out).resolve()
    job = plan['jobs'][0]
    base = ROOT / 'models/base'
    for spec in plan['models']['base']['files']:
        if spec['name'].endswith('.safetensors'):
            path = base / spec['name']
            if not path.exists() or path.stat().st_size != spec['size']:
                raise ValueError('Complete pinned base download required.')
    for record in plan['inputs']:
        if digest(record['path']) != record['sha256']:
            raise ValueError('Input changed after planning.')
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file, load_file
    from safetensors import safe_open
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as module
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    module.VAE_IMAGE_SIZE = c.get('reference_size', 384) ** 2
    images = [ImageOps.exif_transpose(Image.open(p)).convert('RGB') for p in job['inputs']]
    cache = out / 'prompt_embeddings.safetensors'
    started = time.monotonic()
    if args.phase == 'encode':
        from transformers import BitsAndBytesConfig
        print('Loading text/image encoder only', flush=True)
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base/'text_encoder'),
            torch_dtype=torch.float32, device_map={'':'cuda:0'}, low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32))
        pipe = QwenImageEditPlusPipeline(scheduler=None, vae=None, transformer=None,
            text_encoder=encoder, tokenizer=AutoTokenizer.from_pretrained(str(base/'tokenizer')),
            processor=AutoProcessor.from_pretrained(str(base/'processor')))
        references = []
        for im in images:
            w,h = module.calculate_dimensions(module.CONDITION_IMAGE_SIZE, im.width/im.height)
            references.append(pipe.image_processor.resize(im,h,w))
        values = {}
        with torch.inference_mode():
            for prefix, prompt in [('positive',job['prompt']),('negative',' ')]:
                embeds, mask = pipe._get_qwen_prompt_embeds(prompt, references, device=torch.device('cuda'), dtype=torch.float32)
                if not torch.isfinite(embeds).all().item():
                    raise RuntimeError('Non-finite encoder output.')
                values[prefix] = embeds.cpu().contiguous()
                values[prefix+'_mask'] = mask.cpu().contiguous()
        save_file(values, str(cache), metadata={'fingerprint':plan['fingerprint'], 'runner_sha256':digest(__file__)})
        write_json(out/'encode_runtime.json', {'seconds':time.monotonic()-started,
                   'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30})
        print('ENCODE COMPLETE; this process now exits to release encoder memory',flush=True)
        return
    with safe_open(str(cache), framework='pt') as f:
        if f.metadata() != {'fingerprint':plan['fingerprint'], 'runner_sha256':digest(__file__)}:
            raise ValueError('Stale prompt cache; rerun encode.')
    values = load_file(str(cache), device='cuda')
    from diffusers import BitsAndBytesConfig
    print('Loading image transformer only',flush=True)
    import diffusers.models.modeling_utils as loading
    loading._caching_allocator_warmup = lambda *args, **kwargs: None
    transformer = QwenImageTransformer2DModel.from_pretrained(str(base/'transformer'),
        torch_dtype=torch.float32, device_map={'':'cuda:0'}, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32,
            llm_int8_skip_modules=['transformer_blocks.0.img_mod']))
    # The first image modulation layer is sensitive to NF4; preserve original FP32 weights.
    vae = AutoencoderKLQwenImage.from_pretrained(str(base/'vae'), torch_dtype=torch.float32).to('cuda')
    vae.enable_tiling()
    vae.enable_slicing()
    pipe = QwenImageEditPlusPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(base/'scheduler')),
        vae=vae, transformer=transformer, text_encoder=None, tokenizer=None, processor=None)
    lora = ROOT/'models/lora'/LORA_FILE
    expected = next(s['sha256'] for s in plan['models']['lora']['files'] if s['name']==LORA_FILE)
    if digest(lora) != expected:
        raise ValueError('LoRA checksum mismatch.')
    pipe.load_lora_weights(str(lora.parent), weight_name=LORA_FILE, adapter_name='angles')
    pipe.set_adapters('angles', adapter_weights=c['lora_scale'])
    pipe.disable_lora()
    def check_latents(p,step,t,kw):
        if not torch.isfinite(kw['latents']).all().item():
            raise RuntimeError(f'Non-finite latents at step {step}')
        return kw
    class Backend:
        runtime = {'backend':'T4 staged NF4 FP32',
                   'quantization_skip_modules':['transformer_blocks.0.img_mod'], 'runner_sha256':digest(__file__),
                   'gpu':torch.cuda.get_device_name(0), 'cuda':torch.version.cuda,
                   'embedding_sha256':digest(cache), 'load_seconds':round(time.monotonic()-started,3),
                   'packages':{p:importlib.metadata.version(p) for p in ['torch','diffusers','transformers','bitsandbytes','peft','accelerate']}}
        def generate(self, job, paths, config):
            if job['lora']: pipe.enable_lora()
            try:
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    result=pipe(image=images, prompt_embeds=values['positive'], prompt_embeds_mask=values['positive_mask'],
                        negative_prompt_embeds=values['negative'], negative_prompt_embeds_mask=values['negative_mask'],
                        width=c['width'], height=c['height'], num_inference_steps=c['steps'],
                        true_cfg_scale=c['true_cfg_scale'], guidance_scale=1.0,
                        generator=torch.Generator(device='cpu').manual_seed(job['seed']),
                        callback_on_step_end=check_latents).images[0]
                return result, {'peak_cuda_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3)}
            finally: pipe.disable_lora()
    execute(plan,out,Backend())


if __name__ == '__main__':
    main()

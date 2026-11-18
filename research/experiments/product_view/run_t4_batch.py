"""T4 paired pilot: encode/denoise initial jobs, then encode/denoise dependent finals.

Each invocation loads only one large model. Results remain compatible with the
protocol runner, and every cached embedding is bound to its actual input files.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from run_experiment import ROOT, LORA_FILE, digest, execute, prepare, write_json


def selected_jobs(plan, out, wave):
    jobs = []
    for job in plan['jobs']:
        if bool(job['dependency']) != (wave == 'final'):
            continue
        record_path = out / 'records' / (job['id'] + '.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['status'] == 'complete':
                image = out / 'images' / (job['id'] + '.png')
                if not image.exists() or digest(image) != record['output_sha256']:
                    raise ValueError('Completed output changed: ' + job['id'])
                continue
        jobs.append(job)
    return jobs


def cache_spec(plan, job, out):
    paths = [Path(p) if Path(p).is_absolute() else out / p for p in job['inputs']]
    inputs = [{'path': str(p), 'sha256': digest(p)} for p in paths]
    if job['dependency']:
        record = json.loads((out / 'records' / (job['dependency'] + '.json')).read_text())
        if record['status'] != 'complete' or inputs[0]['sha256'] != record['output_sha256']:
            raise ValueError('Invalid dependency: ' + job['id'])
    elif [x['sha256'] for x in inputs] != [x['sha256'] for x in plan['inputs']]:
        raise ValueError('Input photo changed')
    spec = {'plan': plan['fingerprint'], 'runner': digest(__file__),
            'prompt': job['prompt'], 'negative_prompt': ' ', 'inputs': inputs}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return paths, spec, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['encode', 'denoise'])
    parser.add_argument('--wave', choices=['initial', 'final'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    plan = prepare(args.config, out)
    c = plan['config']
    if c['precision'] != 'nf4_fp32':
        raise ValueError('This runner is specifically for the T4 FP32/NF4 profile')
    jobs = selected_jobs(plan, out, args.wave)
    if not jobs:
        print('No pending jobs', flush=True)
        return
    (out / 'embeddings').mkdir(exist_ok=True)
    base = ROOT / 'models/base'
    for entry in plan['models']['base']['files']:
        if entry['name'].endswith('.safetensors'):
            p = base / entry['name']
            if not p.exists() or p.stat().st_size != entry['size']:
                raise ValueError('Complete pinned checkpoint required')

    import torch
    from PIL import Image, ImageOps
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from diffusers import (QwenImageEditPlusPipeline, QwenImageTransformer2DModel,
                           AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler)
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as module
    module.VAE_IMAGE_SIZE = c.get('reference_size', 384) ** 2

    def images_for(paths):
        return [ImageOps.exif_transpose(Image.open(p)).convert('RGB') for p in paths]

    if args.action == 'encode':
        pending = []
        for job in jobs:
            paths, spec, key = cache_spec(plan, job, out)
            cache = out / 'embeddings' / (key + '.safetensors')
            if cache.exists():
                with safe_open(str(cache), framework='pt') as f:
                    if f.metadata() != {'key': key}:
                        raise ValueError('Invalid embedding metadata')
                continue
            pending.append((job, paths, spec, key, cache))
        if not pending:
            print('All prompt caches verified', flush=True)
            return
        from transformers import (BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
                                  AutoProcessor, AutoTokenizer)
        print('Loading encoder once for', len(pending), 'jobs', flush=True)
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(base/'text_encoder'), torch_dtype=torch.float32,
            device_map={'': 'cuda:0'}, low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32))
        pipe = QwenImageEditPlusPipeline(scheduler=None, vae=None, transformer=None,
            text_encoder=encoder, tokenizer=AutoTokenizer.from_pretrained(str(base/'tokenizer')),
            processor=AutoProcessor.from_pretrained(str(base/'processor')))
        for job, paths, spec, key, cache in pending:
            if cache.exists():
                continue  # Same prompts and same images can share a cache.
            references = []
            for im in images_for(paths):
                w, h = module.calculate_dimensions(module.CONDITION_IMAGE_SIZE, im.width/im.height)
                references.append(pipe.image_processor.resize(im, h, w))
            values = {}
            with torch.inference_mode():
                for prefix, prompt in [('positive', job['prompt']), ('negative', ' ')]:
                    embeds, mask = pipe._get_qwen_prompt_embeds(prompt, references,
                        device=torch.device('cuda'), dtype=torch.float32)
                    if not torch.isfinite(embeds).all().item():
                        raise RuntimeError('Non-finite encoder output')
                    values[prefix] = embeds.cpu().contiguous()
                    values[prefix+'_mask'] = mask.cpu().contiguous()
            save_file(values, str(cache), metadata={'key': key})
            write_json(cache.with_suffix('.json'), spec)
            print('ENCODED', job['id'], key, flush=True)
        return

    from diffusers import BitsAndBytesConfig
    import diffusers.models.modeling_utils as loading
    loading._caching_allocator_warmup = lambda *a, **kw: None
    started = time.monotonic()
    print('Loading fixed denoiser once for', len(jobs), 'jobs', flush=True)
    transformer = QwenImageTransformer2DModel.from_pretrained(str(base/'transformer'),
        torch_dtype=torch.float32, device_map={'': 'cuda:0'}, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float32,
            llm_int8_skip_modules=['transformer_blocks.0.img_mod']))
    import bitsandbytes as bnb
    assert not isinstance(transformer.transformer_blocks[0].img_mod[1], bnb.nn.Linear4bit)
    assert transformer.transformer_blocks[0].img_mod[1].weight.dtype == torch.float32
    vae = AutoencoderKLQwenImage.from_pretrained(str(base/'vae'), torch_dtype=torch.float32).to('cuda')
    vae.enable_tiling()
    vae.enable_slicing()
    pipe = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(base/'scheduler')),
        vae=vae, transformer=transformer, text_encoder=None, tokenizer=None, processor=None)
    lora = ROOT / 'models/lora' / LORA_FILE
    expected = next(x['sha256'] for x in plan['models']['lora']['files'] if x['name'] == LORA_FILE)
    if digest(lora) != expected:
        raise ValueError('LoRA checksum mismatch')
    pipe.load_lora_weights(str(lora.parent), weight_name=LORA_FILE, adapter_name='angles')
    pipe.set_adapters('angles', adapter_weights=c['lora_scale'])
    pipe.disable_lora()

    class Backend:
        runtime = {'backend': 'T4 staged batch NF4 FP32', 'runner_sha256': digest(__file__),
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'quantization_skip_modules': ['transformer_blocks.0.img_mod'],
            'load_seconds': round(time.monotonic()-started, 3),
            'packages': {p: importlib.metadata.version(p) for p in
                ['torch', 'diffusers', 'transformers', 'bitsandbytes', 'peft', 'accelerate']}}

        def generate(self, job, paths, config):
            _, _, key = cache_spec(plan, job, out)
            cache = out / 'embeddings' / (key + '.safetensors')
            with safe_open(str(cache), framework='pt') as f:
                if f.metadata() != {'key': key}:
                    raise ValueError('Invalid prompt cache')
            values = load_file(str(cache), device='cuda')
            images = images_for(paths)
            pipe.enable_lora() if job['lora'] else pipe.disable_lora()
            t0 = time.monotonic()
            def progress(p, step, t, kw):
                if not torch.isfinite(kw['latents']).all().item():
                    raise RuntimeError('Non-finite latents')
                if (step+1) % 10 == 0:
                    print(job['id'], step+1, '/', c['steps'], round(time.monotonic()-t0, 1), 'sec', flush=True)
                return kw
            try:
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    image = pipe(image=images, prompt_embeds=values['positive'],
                        prompt_embeds_mask=values['positive_mask'],
                        negative_prompt_embeds=values['negative'],
                        negative_prompt_embeds_mask=values['negative_mask'],
                        width=c['width'], height=c['height'], num_inference_steps=c['steps'],
                        true_cfg_scale=c['true_cfg_scale'], guidance_scale=1.0,
                        generator=torch.Generator(device='cpu').manual_seed(job['seed']),
                        callback_on_step_end=progress).images[0]
                return image, {'peak_cuda_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 3),
                               'embedding_sha256': digest(cache)}
            finally:
                pipe.disable_lora()

    execute(dict(plan, jobs=jobs), out, Backend())


if __name__ == '__main__':
    main()

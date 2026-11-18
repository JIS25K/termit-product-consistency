"""Paired Qwen product-view experiments. Planning needs only Pillow; generation needs CUDA."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

ROOT = Path(__file__).resolve().parent
LORA_FILE = 'qwen-image-edit-2511-multiple-angles-lora.safetensors'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_config(path, smoke=False):
    from PIL import Image, ImageOps
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if 'REPLACE' in json.dumps(config):
        raise ValueError('Fill product_description and must_preserve in experiment.json first.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['product_id']):
        raise ValueError('product_id must use letters, digits, underscore or hyphen.')
    if not 1 <= len(config['inputs']) <= 3:
        raise ValueError('This first protocol uses 1–3 input images, identical in every arm.')
    if len(config['inputs']) != len(config['input_descriptions']):
        raise ValueError('One input description per input is required.')
    if not config['must_preserve'] or not config['targets'] or not config['seeds']:
        raise ValueError('Features, targets and seeds cannot be empty.')
    if len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Seeds must be unique.')
    if len({t['id'] for t in config['targets']}) != len(config['targets']):
        raise ValueError('Target IDs must be unique.')
    for t in config['targets']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', t['id']):
            raise ValueError('Target IDs must use letters, digits, underscore or hyphen.')
    for dimension in ['width', 'height']:
        if not isinstance(config[dimension], int) or not 256 <= config[dimension] <= 2048 or config[dimension] % 32:
            raise ValueError('Image dimensions must be multiples of 32 between 256 and 2048.')
    if config['precision'] not in ['nf4', 'bf16', 'nf4_fp32']:
        raise ValueError('precision must be nf4, bf16 or nf4_fp32.')
    if not 1 <= config['steps'] <= 100 or not 0 < config['lora_scale'] <= 2:
        raise ValueError('Invalid inference steps or LoRA scale.')
    reference_size = config.get('reference_size', 1024)
    if reference_size not in [256, 384, 512, 768, 1024]:
        raise ValueError('Unsupported reference_size.')
    if config.get('diagnostic_only') and (len(config['targets']) != 1 or len(config['seeds']) != 1):
        raise ValueError('Diagnostic requires exactly one target and seed.')
    records = []
    for name in config['inputs']:
        image_path = (path.parent / name).resolve()
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
            width, height = im.size
        if min(width, height) < 512:
            raise ValueError(f'Input too small (<512 px short edge): {name}')
        records.append({'path': str(image_path), 'sha256': digest(image_path), 'width': width, 'height': height})
    if len({r['sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate input images detected.')
    holdouts = []
    for target in config['targets']:
        name = target.get('holdout')
        if not name:
            continue
        p = (path.parent / name).resolve()
        with Image.open(p) as im:
            im.verify()
        h = digest(p)
        if h in {r['sha256'] for r in records}:
            raise ValueError('Held-out evaluation image duplicates an input.')
        holdouts.append({'target': target['id'], 'path': str(p), 'sha256': h})
    if smoke:
        config['seeds'] = config['seeds'][:1]
        config['targets'] = config['targets'][:1]
        holdouts = [h for h in holdouts if h['target'] == config['targets'][0]['id']]
    return config, records, holdouts


def build_jobs(c, inputs):
    features = '; '.join(c['must_preserve'])
    identity = f"Product: {c['product_description']}. Preserve these exact product features: {features}."
    references = 'These images show the same physical product. ' + ' '.join(
        f'Image {i + 1}: {description}.' for i, description in enumerate(c['input_descriptions']))
    original_paths = [r['path'] for r in inputs]
    arms = ['B0', 'B1', 'L0', 'P1'] if c.get('include_lora_off_control') else ['B0', 'B1', 'P1']
    jobs = []
    for target in c['targets']:
        camera = target['camera']
        for seed in c['seeds']:
            for arm in arms:
                prefix = f"{target['id']}_{seed}_{arm}"
                common = {'arm': arm, 'target': target['id'], 'pair_seed': seed}
                if arm == 'B0':
                    jobs.append(dict(common, id=prefix + '_final', stage='final', lora=False,
                        seed=seed + 1000000, inputs=original_paths, dependency=None,
                        prompt=f'{references}\n{identity}\nCreate a new advertising photograph. Camera: {camera}. {c["scene"]}'))
                    continue
                camera_instruction = (f'<sks> {camera}' if arm in ['L0', 'P1'] else f'Rephotograph this product. Camera: {camera}.')
                product_id = prefix + '_product'
                jobs.append(dict(common, id=product_id, stage='product', lora=(arm == 'P1'),
                    seed=seed, inputs=original_paths, dependency=None,
                    prompt=f'{camera_instruction}\n{references}\n{identity}\nShow one complete product centered on a plain neutral light-gray background, with soft even lighting. No props, people or additional text.'))
                jobs.append(dict(common, id=prefix + '_final', stage='final', lora=False,
                    seed=seed + 1000000, inputs=[f'images/{product_id}.png'], dependency=product_id,
                    prompt=f'{identity}\nUse the product in the reference image, preserving its camera angle, shape, parts, labels and proportions. Change only the setting and lighting for this advertising photograph: {c["scene"]}'))
    if c.get('diagnostic_only'):
        # A single independent product call also exercises the angle adapter.
        jobs = [j for j in jobs if j['arm'] == 'P1' and j['stage'] == 'product']
    if c.get('diagnostic_only') and c.get('diagnostic_lora_off'):
        for job in jobs:
            job['lora'] = False
    return jobs


def prepare(config_path, out, smoke=False):
    c, inputs, holdouts = load_config(config_path, smoke)
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    jobs = build_jobs(c, inputs)
    # Source changes and different image contents cannot silently share a run directory.
    payload = {'config': c, 'inputs': inputs, 'holdouts': holdouts, 'models': lock,
               'runner_sha256': digest(__file__), 'jobs': jobs}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    plan_path = out / 'plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise ValueError('Config, input, code or model lock changed. Choose a new --out directory.')
        return old
    plan = dict(payload, fingerprint=fingerprint, created_at=datetime.now(timezone.utc).isoformat())
    write_json(plan_path, plan)
    review = out / 'review.csv'
    with review.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'arm', 'target', 'stage', 'pair_seed', 'image',
            'angle_pass', 'shape_parts_pass', 'label_pass_or_na', 'scene_pass_or_na', 'overall_pass', 'notes'])
        writer.writeheader()
        for job in jobs:
            row = {key: job[key] for key in ['id', 'arm', 'target', 'stage', 'pair_seed']}
            writer.writerow(dict(row, image=f"images/{job['id']}.png"))
    report(out)
    return plan


class QwenBackend:
    def __init__(self, c):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required. Use the supplied Colab notebook. Planning works on Mac.')
        if c['precision'] != 'nf4_fp32' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Use an Ampere-or-newer GPU (A100/L4 etc.). For T4 use the experimental nf4_fp32 profile.')
        lock = json.loads((ROOT / 'models.lock.json').read_text())
        self.runtime = {'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
            'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            'cuda': torch.version.cuda, 'precision': c['precision'], 'packages': {}}
        for pkg in ['torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'bitsandbytes', 'huggingface-hub']:
            self.runtime['packages'][pkg] = importlib.metadata.version(pkg)
        base_dir = ROOT / 'models' / 'base'
        # Partially downloaded config files are not mistaken for a full local model.
        base_specs = lock['base']['files']
        complete = all((base_dir / f['name']).exists() and (base_dir / f['name']).stat().st_size == f['size']
                       for f in base_specs if f['name'].endswith('.safetensors'))
        source = str(base_dir) if complete else lock['base']['repo']
        compute_dtype = torch.float32 if c['precision'] == 'nf4_fp32' else torch.bfloat16
        self.runtime['compute_dtype'] = str(compute_dtype)
        self.runtime['reference_size'] = c.get('reference_size', 1024)
        self.runtime['quantization_skip_modules'] = ['transformer_blocks.0.img_mod'] if c['precision'] in ['nf4', 'nf4_fp32'] else []
        # This pinned pipeline independently upsizes VAE references to 1024².
        # Lowering output dimensions alone does not lower that attention cost.
        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as pipeline_module
        pipeline_module.VAE_IMAGE_SIZE = c.get('reference_size', 1024) ** 2
        kwargs = {'torch_dtype': compute_dtype, 'low_cpu_mem_usage': True}
        if not complete:
            kwargs['revision'] = lock['base']['revision']
        if c['precision'] in ['nf4', 'nf4_fp32']:
            from diffusers import BitsAndBytesConfig as DiffusersQuantConfig
            from transformers import BitsAndBytesConfig as TransformersQuantConfig
            quant_kwargs = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                            'bnb_4bit_compute_dtype': compute_dtype}
            kwargs['quantization_config'] = PipelineQuantizationConfig(quant_mapping={
                'transformer': DiffusersQuantConfig(**quant_kwargs,
                    llm_int8_skip_modules=['transformer_blocks.0.img_mod']),
                'text_encoder': TransformersQuantConfig(**quant_kwargs)})
        started = time.monotonic()
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(source, **kwargs)
        lora_path = ROOT / 'models' / 'lora' / LORA_FILE
        expected = next(f['sha256'] for f in lock['lora']['files'] if f['name'] == LORA_FILE)
        if not lora_path.exists() or digest(lora_path) != expected:
            raise RuntimeError('LoRA file missing or invalid. Run download_models.py first.')
        self.pipe.load_lora_weights(str(lora_path.parent), weight_name=LORA_FILE, adapter_name='angles')
        self.pipe.set_adapters('angles', adapter_weights=c['lora_scale'])
        self.pipe.disable_lora()
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.runtime['load_seconds'] = round(time.monotonic() - started, 3)

    def generate(self, job, paths, c):
        from PIL import Image, ImageOps
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(ImageOps.exif_transpose(im).convert('RGB'))
        if job['lora']:
            self.pipe.enable_lora()
        else:
            self.pipe.disable_lora()
        self.torch.cuda.reset_peak_memory_stats()
        extra = {}
        if c['precision'] == 'nf4_fp32':
            def check_latents(pipe, step, timestep, callback_kwargs):
                if not self.torch.isfinite(callback_kwargs['latents']).all().item():
                    raise RuntimeError(f'Non-finite latents at step {step}; output is not valid.')
                return callback_kwargs
            extra['callback_on_step_end'] = check_latents
        try:
            with self.torch.inference_mode():
                image = self.pipe(image=images, prompt=job['prompt'], negative_prompt=' ',
                    width=c['width'], height=c['height'], num_inference_steps=c['steps'],
                    true_cfg_scale=c['true_cfg_scale'], guidance_scale=1.0,
                    generator=self.torch.Generator(device='cpu').manual_seed(job['seed']),
                    num_images_per_prompt=1, **extra).images[0]
            peak = self.torch.cuda.max_memory_allocated() / 2**30
            return image, {'peak_cuda_allocated_gib': round(peak, 3)}
        finally:
            self.pipe.disable_lora()


def execute(plan, out, backend, phase='all'):
    out = Path(out).resolve()
    runtime_path = out / 'runtime.json'
    if runtime_path.exists():
        previous = json.loads(runtime_path.read_text())
        # Loading duration can change on resume; hardware and software must not.
        stable = lambda x: {k: v for k, v in x.items() if k != 'load_seconds'}
        if stable(previous) != stable(backend.runtime):
            raise ValueError('Runtime changed. Use a new output directory to avoid mixing conditions.')
    else:
        write_json(runtime_path, backend.runtime)
    for job in plan['jobs']:
        if phase != 'all' and job['stage'] != phase:
            continue
        record_path = out / 'records' / (job['id'] + '.json')
        image_path = out / 'images' / (job['id'] + '.png')
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get('status') == 'complete':
                if not image_path.exists() or digest(image_path) != old['output_sha256']:
                    raise ValueError(f'Completed output changed: {job["id"]}. Use a new directory.')
                continue
        paths = [Path(p) if Path(p).is_absolute() else out / p for p in job['inputs']]
        if job['dependency']:
            dependency_record = out / 'records' / (job['dependency'] + '.json')
            if not dependency_record.exists():
                raise ValueError('Run --phase product before --phase final for staged arms.')
            dependency = json.loads(dependency_record.read_text())
            if dependency.get('status') != 'complete' or not paths[0].exists() or digest(paths[0]) != dependency['output_sha256']:
                raise ValueError('Intermediate image is missing, failed or modified.')
        else:
            for path, expected in zip(paths, plan['inputs']):
                if digest(path) != expected['sha256']:
                    raise ValueError('Input changed after planning.')
        record = dict(job, status='running', started_at=datetime.now(timezone.utc).isoformat(),
            input_files=[{'path': str(p), 'sha256': digest(p)} for p in paths])
        write_json(record_path, record)
        started = time.monotonic()
        print(f"Generating {job['id']} | LoRA={'on' if job['lora'] else 'off'}", flush=True)
        try:
            image, details = backend.generate(job, paths, plan['config'])
            temporary = image_path.with_suffix('.tmp.png')
            image.save(temporary)
            temporary.replace(image_path)
            record.update(status='complete', output_sha256=digest(image_path), **details)
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            record['seconds'] = round(time.monotonic() - started, 3)
            write_json(record_path, record)
            report(out)


def report(out):
    out = Path(out).resolve()
    plan = json.loads((out / 'plan.json').read_text())
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Product view comparison</title>',
        '<style>body{font-family:system-ui;max-width:1400px;margin:24px auto;background:#fafafa;color:#222} '
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px} '
        'article{background:white;padding:12px;border:1px solid #ddd}img{width:100%}pre{white-space:pre-wrap;font-size:12px}</style>',
        '<h1>Product view comparison</h1><p>B0: direct / B1: staged / L0: same LoRA prompt, adapter off / P1: adapter on for product only.</p>',
        '<p>Review every output using review.csv. No automated quality score or selection has been applied.</p>']
    for target in plan['config']['targets']:
        for seed in plan['config']['seeds']:
            body.append(f'<h2>{html.escape(target["id"])} · seed {seed}</h2><div class="grid">')
            for job in plan['jobs']:
                if job['target'] != target['id'] or job['pair_seed'] != seed:
                    continue
                record = out / 'records' / (job['id'] + '.json')
                status = json.loads(record.read_text()) if record.exists() else {'status': 'not run'}
                body.append(f'<article><h3>{html.escape(job["arm"])} / {job["stage"]}</h3><p>{html.escape(status["status"])}</p>')
                if status['status'] == 'complete':
                    body.append(f'<a href="images/{job["id"]}.png"><img src="images/{job["id"]}.png"></a>')
                    body.append(f'<p>{status["seconds"]} seconds</p>')
                body.append(f'<details><summary>Prompt / provenance</summary><pre>{html.escape(json.dumps(job, ensure_ascii=False, indent=2))}</pre></details></article>')
            body.append('</div>')
    body.append('</html>')
    (out / 'report.html').write_text('\n'.join(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run', 'report'])
    parser.add_argument('--config', default=str(ROOT / 'experiment.json'))
    parser.add_argument('--out', default=str(ROOT / 'runs' / 'first'))
    parser.add_argument('--smoke', action='store_true', help='One target, one seed; same quality settings')
    parser.add_argument('--phase', choices=['all', 'product', 'final'], default='all')
    args = parser.parse_args()
    if args.action == 'report':
        report(args.out)
        return
    plan = prepare(args.config, args.out, args.smoke)
    print(f"Plan: {len(plan['jobs'])} image calls; output: {Path(args.out).resolve()}")
    if args.action == 'run':
        backend = QwenBackend(plan['config'])
        execute(plan, args.out, backend, args.phase)


if __name__ == '__main__':
    main()

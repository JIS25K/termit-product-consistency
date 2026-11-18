"""CPU-only fal queue runner. Explicit jobs; no automatic paid retries."""
import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'product_view'))
from run_experiment import build_jobs, digest, write_json


def prepare(config_path, out):
    from PIL import Image
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    c = json.loads(config_path.read_text())
    inputs, holdouts = [], []
    for name in c['inputs']:
        p = (config_path.parent / name).resolve()
        with Image.open(p) as im:
            im.verify()
        inputs.append({'path': str(p), 'sha256': digest(p)})
    for t in c['targets']:
        if t.get('holdout'):
            p = (config_path.parent / t['holdout']).resolve()
            with Image.open(p) as im:
                im.verify()
            assert digest(p) not in [x['sha256'] for x in inputs]
            holdouts.append({'path': str(p), 'sha256': digest(p), 'target': t['id']})
    data = {'config': c, 'inputs': inputs, 'holdouts': holdouts,
            'jobs': build_jobs(c, inputs), 'runner_sha256': digest(__file__),
            'protocol_sha256': digest(HERE.parent/'product_view/run_experiment.py')}
    data['fingerprint'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    for name in ['records', 'images']:
        (out/name).mkdir(exist_ok=True)
    p = out/'plan.json'
    if p.exists() and json.loads(p.read_text())['fingerprint'] != data['fingerprint']:
        raise ValueError('Settings or files changed: use a new result directory.')
    write_json(p, data)
    return data


def request_body(plan, job, out, include_image=False):
    c = plan['config']
    paths = [Path(n) if Path(n).is_absolute() else out/n for n in job['inputs']]
    if job['dependency']:
        rec = json.loads((out/'records'/(job['dependency']+'.json')).read_text())
        if rec.get('status') != 'complete' or digest(paths[0]) != rec['sha256']:
            raise ValueError('Intermediate image is missing or changed.')
    elif [digest(p) for p in paths] != [x['sha256'] for x in plan['inputs']]:
        raise ValueError('Input image changed.')
    urls = []
    for p in paths:
        if include_image:
            mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
            urls.append('data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode())
        else:
            urls.append({'local_path': str(p), 'sha256': digest(p)})
    return dict(prompt=job['prompt'], negative_prompt=' ', image_urls=urls,
        image_size={'width': c['width'], 'height': c['height']},
        num_inference_steps=c['steps'], guidance_scale=c['true_cfg_scale'],
        seed=job['seed'], num_images=1, output_format='png', sync_mode=False,
        enable_safety_checker=True, acceleration=c['api']['acceleration'],
        loras=[{'path': c['api']['lora_url'], 'scale': c['lora_scale']}] if job['lora'] else [])


def estimate(plan, count):
    c = plan['config']
    # Reserve whole megapixel units conservatively. Billing is confirmed separately.
    return count * max(1, math.ceil(c['width']*c['height']/1e6)) * c['api']['price_usd_per_megapixel']


def api_json(url, key, body=None):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'queue.fal.run':
        raise ValueError('Refusing to send credentials outside the fal queue.')
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': 'Key '+key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def report(plan, out):
    rows = ['<!doctype html><meta charset="utf-8"><title>API experiment</title>',
            '<style>body{font-family:sans-serif;margin:24px}img{width:300px;max-height:360px;object-fit:contain}section{display:inline-block;vertical-align:top;margin:12px}</style>',
            '<h1>API product-view experiment</h1><p>Separate from Colab results. Holdout images are never sent to the model.</p>']
    for job in plan['jobs']:
        p = out/'records'/(job['id']+'.json')
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        rows.append('<section><h3>'+html.escape(job['id'])+'</h3><p>'+html.escape(rec['status'])+'</p>')
        if rec['status'] == 'complete':
            rows.append('<img src="images/'+job['id']+'.png">')
        rows.append('</section>')
    (out/'report.html').write_text('\n'.join(rows))


def submit(plan, jobs, out, key, budget):
    # All prospective jobs are checked before the first billable request.
    pending = []
    for job in jobs:
        record = out/'records'/(job['id']+'.json')
        if record.exists():
            old = json.loads(record.read_text())
            if old['status'] == 'submitting':
                raise RuntimeError('Ambiguous prior submission; inspect fal history, do not resubmit.')
            if old['status'] == 'complete' and digest(out/'images'/(job['id']+'.png')) != old['sha256']:
                raise ValueError('Completed output changed.')
            continue  # Existing request is collected, never submitted twice.
        pending.append((job, request_body(plan, job, out)))
    reserved = sum(json.loads(p.read_text()).get('reserved_usd', 0) for p in (out/'records').glob('*.json'))
    if reserved + estimate(plan, len(pending)) > budget + 1e-9:
        raise ValueError('Conservative run estimate exceeds --budget-usd.')
    endpoint = plan['config']['api']['endpoint']
    for job, preview in pending:
        record = out/'records'/(job['id']+'.json')
        rec = {'status': 'submitting', 'request': preview, 'endpoint': endpoint,
               'reserved_usd': estimate(plan, 1), 'estimate_not_actual_charge': True}
        write_json(record, rec)
        response = api_json('https://queue.fal.run/'+endpoint, key, request_body(plan, job, out, True))
        rec.update(status='submitted', queue=response)
        write_json(record, rec)
        print(job['id'], response['request_id'], flush=True)


def collect(plan, jobs, out, key):
    from PIL import Image
    for job in jobs:
        p = out/'records'/(job['id']+'.json')
        if not p.exists():
            print(job['id'], 'not submitted'); continue
        rec = json.loads(p.read_text())
        if rec['status'] == 'complete':
            if digest(out/'images'/(job['id']+'.png')) != rec['sha256']:
                raise ValueError('Completed output changed.')
            continue
        if 'queue' not in rec:
            raise RuntimeError('Submission uncertain; inspect fal request history.')
        status = api_json(rec['queue']['status_url'], key)
        rec['last_queue_status'] = status
        write_json(p, rec)
        print(job['id'], status['status'], flush=True)
        if status['status'] != 'COMPLETED':
            continue
        result = api_json(rec['queue']['response_url'], key)
        rec['result'] = result
        write_json(p, rec)  # Persist result before downloading; result URLs may expire.
        if result.get('has_nsfw_concepts') and any(result['has_nsfw_concepts']):
            raise RuntimeError('Output flagged by service; not a successful experiment image.')
        image_info = result['images'][0]
        if urlparse(image_info['url']).scheme != 'https':
            raise ValueError('Expected HTTPS result image.')
        image = out/'images'/(job['id']+'.png')
        with urllib.request.urlopen(image_info['url'], timeout=45) as response:
            data = response.read()
        temporary = image.with_suffix('.download')
        temporary.write_bytes(data)
        with Image.open(temporary) as im:
            im.load()
            if im.size != (plan['config']['width'], plan['config']['height']):
                raise ValueError('Unexpected output size; inspect before comparison.')
        temporary.replace(image)
        rec.update(status='complete', sha256=digest(image))
        write_json(p, rec)
    report(plan, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan','submit','collect'])
    parser.add_argument('--config', default=str(HERE/'experiment.figurine.json'))
    parser.add_argument('--out', default=str(HERE/'runs/figurine_api_01'))
    parser.add_argument('--jobs', nargs='+', default=['bag_side_101_L0_product','bag_side_101_P1_product'])
    parser.add_argument('--budget-usd', type=float, default=0.15)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    plan = prepare(args.config, out)
    by_id = {j['id']: j for j in plan['jobs']}
    if len(set(args.jobs)) != len(args.jobs):
        raise ValueError('Duplicate job IDs.')
    jobs = [by_id[j] for j in args.jobs]
    if args.action == 'plan':
        for job in jobs:
            write_json(out/(job['id']+'.request-preview.json'), request_body(plan,job,out))
        print('Jobs:',args.jobs,'Conservative estimate USD:',estimate(plan,len(jobs)))
        return
    key = os.environ.get('FAL_KEY')
    if not key:
        raise SystemExit('FAL_KEY is not configured. Never paste the key into chat.')
    if args.action == 'submit':
        submit(plan,jobs,out,key,args.budget_usd)
    else:
        collect(plan,jobs,out,key)


if __name__ == '__main__':
    main()

"""Protocol tests use synthetic files, not generated product results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as experiment
from PIL import Image


class FakeBackend:
    runtime = {'backend': 'TEST ONLY — no model inference'}

    def __init__(self):
        self.calls = []

    def generate(self, job, paths, c):
        self.calls.append((job, paths))
        return Image.new('RGB', (64, 64), (len(self.calls), 100, 100)), {'test_only': True}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = json.loads((experiment.ROOT / 'experiment.example.json').read_text())
        self.config['product_description'] = 'A test product'
        self.config['must_preserve'] = ['two handles', 'one lid', 'red label']
        names = self.config['inputs'] + [t['holdout'] for t in self.config['targets']]
        for i, name in enumerate(names):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (800, 800), (i * 30, 90, 150)).save(path)
        self.path = self.root / 'experiment.json'
        self.path.write_text(json.dumps(self.config))

    def test_pairing_and_no_holdout_leak(self):
        c, inputs, holdouts = experiment.load_config(self.path)
        jobs = experiment.build_jobs(c, inputs)
        self.assertEqual(len(jobs), 42)
        self.assertEqual(sum(j['stage'] == 'final' for j in jobs), 24)
        evaluation_paths = {h['path'] for h in holdouts}
        for job in jobs:
            self.assertFalse(evaluation_paths.intersection(job['inputs']))
            self.assertEqual(job['lora'], job['arm'] == 'P1' and job['stage'] == 'product')
        on = next(j for j in jobs if j['arm'] == 'P1' and j['stage'] == 'product')
        off = next(j for j in jobs if j['arm'] == 'L0' and j['stage'] == 'product')
        for field in ['prompt', 'seed', 'inputs']:
            self.assertEqual(on[field], off[field])
        finals = [j for j in jobs if j['stage'] == 'final' and j['arm'] != 'B0']
        self.assertEqual(len({j['prompt'] for j in finals}), 1)

    def test_t4_diagnostic_is_one_independent_lora_call(self):
        self.config.update(precision='nf4_fp32', width=384, height=384,
                           reference_size=384, steps=8, diagnostic_only=True)
        self.config['targets'] = [dict(self.config['targets'][0], holdout=None)]
        self.config['seeds'] = [101]
        self.config['inputs'] = self.config['inputs'][:1]
        self.config['input_descriptions'] = ['front view']
        self.path.write_text(json.dumps(self.config))
        plan = experiment.prepare(self.path, self.root / 't4')
        self.assertEqual(len(plan['jobs']), 1)
        job = plan['jobs'][0]
        self.assertEqual((job['arm'], job['stage'], job['lora'], job['dependency']),
                         ('P1', 'product', True, None))
        backend = FakeBackend()
        experiment.execute(plan, self.root / 't4', backend)
        self.assertEqual(len(backend.calls), 1)
        self.config['reference_size'] = 384.5
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'reference_size'):
            experiment.load_config(self.path)

    def test_duplicate_holdout_rejected(self):
        self.config['targets'][0]['holdout'] = self.config['inputs'][0]
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'duplicates an input'):
            experiment.load_config(self.path)

    def test_resume_and_modified_output_rejected(self):
        out = self.root / 'run'
        plan = experiment.prepare(self.path, out, smoke=True)
        self.assertEqual(len(plan['jobs']), 7)
        backend = FakeBackend()
        experiment.execute(plan, out, backend)
        self.assertEqual(len(backend.calls), 7)
        experiment.execute(plan, out, backend)
        self.assertEqual(len(backend.calls), 7)
        for job, paths in backend.calls:
            if job['dependency']:
                self.assertEqual(paths, [out / 'images' / (job['dependency'] + '.png')])
        self.assertTrue((out / 'report.html').exists())
        review = out / 'review.csv'
        review.write_text(review.read_text() + 'human review retained\n')
        experiment.prepare(self.path, out, smoke=True)
        self.assertIn('human review retained', review.read_text())
        (out / 'images' / (plan['jobs'][0]['id'] + '.png')).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            experiment.execute(plan, out, backend)

    def test_changed_config_requires_new_run(self):
        out = self.root / 'run'
        experiment.prepare(self.path, out)
        self.config['lora_scale'] = 0.8
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'changed'):
            experiment.prepare(self.path, out)

    def test_failure_logged_and_no_success_claimed(self):
        out = self.root / 'run'
        plan = experiment.prepare(self.path, out, smoke=True)
        backend = FakeBackend()
        backend.generate = lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic failure'))
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            experiment.execute(plan, out, backend)
        record = json.loads((out / 'records' / (plan['jobs'][0]['id'] + '.json')).read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertNotIn('output_sha256', record)

    def test_lora_disabled_after_generation_and_exception(self):
        class Pipeline:
            enabled = False
            seen = []
            fail = False
            def enable_lora(self): self.enabled = True
            def disable_lora(self): self.enabled = False
            def __call__(self, **kwargs):
                self.seen.append(self.enabled)
                if self.fail:
                    raise RuntimeError('inference failed')
                return SimpleNamespace(images=[Image.new('RGB', (64, 64))])
        from contextlib import nullcontext
        backend = object.__new__(experiment.QwenBackend)
        backend.pipe = Pipeline()
        backend.torch = SimpleNamespace(
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0),
            inference_mode=nullcontext,
            Generator=lambda **kw: SimpleNamespace(manual_seed=lambda x: x))
        c, inputs, _ = experiment.load_config(self.path)
        paths = [Path(i['path']) for i in inputs]
        job = {'lora': True, 'prompt': 'test', 'seed': 1}
        backend.generate(job, paths, c)
        self.assertFalse(backend.pipe.enabled)
        job['lora'] = False
        backend.generate(job, paths, c)
        self.assertEqual(backend.pipe.seen, [True, False])
        backend.pipe.fail = True
        job['lora'] = True
        with self.assertRaises(RuntimeError):
            backend.generate(job, paths, c)
        self.assertFalse(backend.pipe.enabled)


if __name__ == '__main__':
    unittest.main()

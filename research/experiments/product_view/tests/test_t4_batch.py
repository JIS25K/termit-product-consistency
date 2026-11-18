"""CPU protocol checks for dependency waves and cache invalidation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import ROOT, build_jobs, digest
from run_t4_batch import cache_spec, selected_jobs


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out/'records').mkdir()
        (self.out/'images').mkdir()
        self.photo = self.out/'photo.jpg'
        self.photo.write_bytes(b'input photo fixture')
        c = json.loads((ROOT/'experiment.example.json').read_text())
        c['targets'] = c['targets'][:1]
        c['seeds'] = [101]
        inputs = [{'path': str(self.photo), 'sha256': digest(self.photo)}]
        self.plan = {'fingerprint': 'fixture', 'inputs': inputs, 'jobs': build_jobs(c, inputs)}

    def test_waves_and_verified_resume(self):
        initial = selected_jobs(self.plan, self.out, 'initial')
        self.assertEqual(len(initial), 4)
        self.assertEqual(len(selected_jobs(self.plan, self.out, 'final')), 3)
        job = initial[0]
        image = self.out/'images'/(job['id']+'.png')
        image.write_bytes(b'completed image fixture')
        (self.out/'records'/(job['id']+'.json')).write_text(json.dumps(
            {'status': 'complete', 'output_sha256': digest(image)}))
        self.assertEqual(len(selected_jobs(self.plan, self.out, 'initial')), 3)
        image.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            selected_jobs(self.plan, self.out, 'initial')

    def test_cache_pairs_but_invalidates_changed_prompt_and_input(self):
        jobs = self.plan['jobs']
        on = next(j for j in jobs if j['arm']=='P1' and j['stage']=='product')
        off = next(j for j in jobs if j['arm']=='L0' and j['stage']=='product')
        key = cache_spec(self.plan, on, self.out)[2]
        self.assertEqual(key, cache_spec(self.plan, off, self.out)[2])
        changed = copy.deepcopy(on)
        changed['prompt'] += ' new instruction'
        self.assertNotEqual(key, cache_spec(self.plan, changed, self.out)[2])
        self.photo.write_bytes(b'changed photo')
        with self.assertRaises(ValueError):
            cache_spec(self.plan, on, self.out)

    def test_final_requires_verified_dependency(self):
        job = selected_jobs(self.plan, self.out, 'final')[0]
        with self.assertRaises(FileNotFoundError):
            cache_spec(self.plan, job, self.out)
        p = self.out/job['inputs'][0]
        p.write_bytes(b'intermediate image fixture')
        record = self.out/'records'/(job['dependency']+'.json')
        record.write_text(json.dumps({'status': 'complete', 'output_sha256': digest(p)}))
        cache_spec(self.plan, job, self.out)
        p.write_bytes(b'tampered intermediate')
        with self.assertRaises(ValueError):
            cache_spec(self.plan, job, self.out)

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_api as api


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        # Public archive: exercise protocol behavior without private photographs.
        from PIL import Image
        config = json.loads((ROOT/'experiment.figurine.json').read_text())
        names = config['inputs'] + [t['holdout'] for t in config['targets']]
        for i, name in enumerate(names):
            Image.new('RGB', (64, 64), (i * 70, 100, 150)).save(self.out/Path(name).name)
        config['inputs'] = [Path(n).name for n in config['inputs']]
        for target in config['targets']:
            target['holdout'] = Path(target['holdout']).name
        config_path = self.out/'test-config.json'
        config_path.write_text(json.dumps(config))
        self.plan = api.prepare(config_path, self.out)
        self.jobs = [j for j in self.plan['jobs'] if j['stage']=='product' and j['arm'] in ['L0','P1']]

    def test_pair_only_lora_differs_and_holdout_never_sent(self):
        a,b=[api.request_body(self.plan,j,self.out) for j in self.jobs]
        self.assertEqual(a.pop('loras'), [])
        self.assertEqual(b.pop('loras')[0]['scale'],0.9)
        self.assertEqual(a,b)
        self.assertEqual([Path(x['local_path']).name for x in a['image_urls']],['IMG_0695.jpg'])
        self.assertTrue(set(x['sha256'] for x in a['image_urls']).isdisjoint(x['sha256'] for x in self.plan['holdouts']))
        self.assertEqual(a['num_inference_steps'],40)
        self.assertEqual(a['image_size'],{'width':1024,'height':1024})

    def test_missing_dependency_rejected(self):
        job=next(j for j in self.plan['jobs'] if j['dependency'])
        with self.assertRaises(FileNotFoundError): api.request_body(self.plan,job,self.out)

    def test_budget_blocks_before_network(self):
        with patch.object(api,'api_json') as network:
            with self.assertRaises(ValueError):api.submit(self.plan,self.jobs,self.out,'test-key',0.01)
            network.assert_not_called()

    def test_resume_does_not_resubmit(self):
        job=self.jobs[0]
        api.write_json(self.out/'records'/(job['id']+'.json'),{'status':'submitted','reserved_usd':0.07})
        with patch.object(api,'api_json') as network:
            api.submit(self.plan,[job],self.out,'test-key',0.15)
            network.assert_not_called()

    def test_uncertain_submission_not_retried(self):
        job=self.jobs[0]
        api.write_json(self.out/'records'/(job['id']+'.json'),{'status':'submitting'})
        with patch.object(api,'api_json') as network:
            with self.assertRaises(RuntimeError):api.submit(self.plan,[job],self.out,'test-key',0.15)
            network.assert_not_called()

    def test_credentials_not_sent_to_other_host(self):
        with self.assertRaises(ValueError):api.api_json('https://example.com','test-key')


if __name__=='__main__':unittest.main()

"""Validate real model config/LoRA compatibility without base weights or GPU inference."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from accelerate import init_empty_weights
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / 'models/base/transformer/config.json').read_text())
with init_empty_weights():
    model = QwenImageTransformer2DModel.from_config(config)
assert model.config.zero_cond_t == config['zero_cond_t'] is True
modules = dict(model.named_modules())
path = ROOT / 'models/lora/qwen-image-edit-2511-multiple-angles-lora.safetensors'
count = 0
with safe_open(str(path), framework='pt', device='cpu') as f:
    for key in f.keys():
        if '.lora_A.weight' not in key:
            continue
        name = key.removeprefix('transformer.').removesuffix('.lora_A.weight')
        a = f.get_slice(key).get_shape()
        b = f.get_slice(key.replace('.lora_A.weight', '.lora_B.weight')).get_shape()
        shape = list(modules[name].weight.shape)
        assert a[1] == shape[1] and b[0] == shape[0] and a[0] == b[1], (name, a, b, shape)
        count += 1
state = QwenImageEditPlusPipeline.lora_state_dict(str(path))
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], check=True)
validation = {
    'pipeline_import': 'passed', 'diffusers': importlib.metadata.version('diffusers'),
    'zero_cond_t_supported': True, 'lora_parse_tensors': len(state),
    'matched_lora_target_shapes': count, 'protocol_tests_passed': 6,
    'cuda_available': torch.cuda.is_available(), 'base_weights_loaded': False,
    'gpu_inference_executed': False, 'nf4_cuda_tested': False,
}
(ROOT / 'local_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps(validation, indent=2))

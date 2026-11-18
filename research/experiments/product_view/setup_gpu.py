"""Install a pinned Linux/CUDA environment without importing torch in the notebook kernel."""
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if platform.system() != 'Linux' or sys.version_info[:2] not in [(3, 11), (3, 12)]:
    raise SystemExit('Use a Linux GPU runtime with Python 3.11 or 3.12.')
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1',
                '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
wheel = ROOT / 'vendor' / 'diffusers-0.37.0-py3-none-any.whl'
if wheel.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', str(wheel)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-gpu.txt')], check=True)
# A broken import should fail before downloading the large base weights.
subprocess.run([sys.executable, '-c',
    'import torch, bitsandbytes, peft; from diffusers import QwenImageEditPlusPipeline; '
    'from transformers import Qwen2_5_VLForConditionalGeneration; '
    'assert torch.cuda.is_available(), "CUDA unavailable"; '
    'assert torch.cuda.get_device_capability(0)[0] >= 6, "NF4 requires Pascal or newer"; '
    'print("GPU imports OK:", torch.cuda.get_device_name(0))'], check=True)

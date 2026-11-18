from pathlib import Path
import json,hashlib
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers
r=Path(__file__).resolve().parent;s=json.loads((r/'outputs/train/summary.json').read_text());assert s['steps']==200 and s['nonfinite_adapter_tensors']==0
state=load_file(r/'outputs/train/product_adapter.safetensors');o=r/'outputs/adapter';o.mkdir(exist_ok=True)
StableDiffusionXLPipeline.save_lora_weights(save_directory=o,unet_lora_layers=convert_state_dict_to_diffusers(state),safe_serialization=True)
p=o/'pytorch_lora_weights.safetensors';meta={'base_model':'stabilityai/stable-diffusion-xl-base-1.0','revision':'462165984030d82259a11f4367a4eed129e94a7b','trigger':'sksdrill','steps':200,'rank':4,'resolution':768,'real_training_images':12,'trained_modules':'UNet attention LoRA only; text encoders frozen','format':'Diffusers pipeline LoRA safetensors','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'quality_status':'See quality_evaluation.json; existence of weights is not evidence of product-fidelity advantage'};(o/'metadata.json').write_text(json.dumps(meta,indent=2))
(o/'README.md').write_text('''# Run 31 전동드릴 실험용 LoRA

기존 SDXL에 덧붙이는 상품별 추가 학습 가중치다. 독립 실행 가능한 새 생성 모델이나 범용 제품 보존 엔진이 아니다.

- 기반 모델: stabilityai/stable-diffusion-xl-base-1.0
- 식별 문구: `sksdrill red and black cordless power drill`
- 학습: 실제 사진12장, 768px, rank4, 200단계, UNet attention만 추가 학습.
- 데이터 출처: YCB / Calli et al., CC BY 4.0. 원본 URL·사진 해시는 상위 `data/manifest.json`에 보관.
- 실행 환경: Apple M5,24GB,MPS. 클라우드 학습 없음.
- 파일: `pytorch_lora_weights.safetensors`는 Diffusers 형식. 기반 모델을 로딩한 뒤 `pipe.load_lora_weights(이 폴더 경로)`로 추가한다.
- 검증: 상품 정확성·광고 품질의 판정은 상위 결과 보고서를 따른다. 가중치가 저장됐다는 이유로 성공 판정하지 않는다.

검사 한 장으로 만든 10단계 파일은 별개이며, 이 가중치에는 이어 학습하지 않았다.
''')
print(json.dumps(meta,indent=2))

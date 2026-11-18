# 외부 코드·모델·자료

- `research/experiments/product_lora/run31/{train,pilot}.py`와 `source/train_dreambooth_lora_sdxl.py`는 Hugging Face Diffusers의 SDXL DreamBooth LoRA 학습 예제에서 파생됐다. 원저작권: Copyright 2025 The HuggingFace Inc. team. Apache-2.0. [라이선스 전문](LICENSES/Apache-2.0.txt). 측정/중단/저장 및 MPS 실험 관련 수정이 포함된다.
- SDXL, Qwen Image Edit, 외부 Multiple Angles LoRA, MV-Adapter, Hunyuan, LaMa 등은 외부 모델이다. 본 과제에서 기반 모델 자체를 개발하지 않았다. 가중치는 저장소에 포함하지 않으며 각 배포자의 라이선스·이용 조건을 따른다.
- OpenCV/SIFT/RANSAC, PyTorch, Diffusers, Pyodide, FastAPI 등은 외부 구현을 사용했다. 자체 기여는 적용·연결 코드, 원본 등록, 비교 설계, 검사/보류 규칙, 실험과 시제품이다.
- YCB 자료는 Calli et al.의 공개 데이터에서 얻었다. 원본 데이터와 3D 자산은 이 공개본에 재배포하지 않는다. 다운로드 도구를 사용할 때 원 출처·라이선스를 확인한다.
- 팀원 코드의 출처는 `docs/publication.md`와 매니페스트에 기록했다. 별도 제공되지 않은 라이선스를 추정해서 새로 부여하지 않았다.

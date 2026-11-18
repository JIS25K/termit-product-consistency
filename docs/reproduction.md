# 실행과 재현 범위

이 저장소는 원래 연구 환경의 전체 백업이 아니다. 코드 구조는 보존했지만 비공개 입력과 모델 캐시를 뺐다. `LOCAL_INPUT_REQUIRED`는 실제 파일로 교체해야 하는 자리다. 경로만 바꾼다고 원래 상품 전용 좌표·마스크·임계값이 다른 상품에 그대로 적용되는 것은 아니다.

## 1. API·GPU가 필요 없는 검사

README의 Python 3.12 + Pillow 환경에서 두 `unittest` 명령을 실행한다. 생성/요청 backend는 가짜 구현이다. 입력/평가 사진 분리, 조건 페어링, 재개, 불확실한 제출 재시도 방지, 예산 검사, 키의 다른 호스트 전송 방지를 확인한다. 실제 API 요금이나 품질을 검증하지 않는다.

## 2. 우리 측 실험 모듈

| 경로 | 준비와 실행 | 공개본에서의 제한 |
|---|---|---|
| `research/experiments/product_view/` | `requirements-gpu.txt`, `models.lock.json`, Colab 노트북과 `run_experiment.py --help` 참고 | 모델·입력 사진을 별도 준비. GPU 실행을 이번 정리에서 재수행하지 않음 |
| `research/experiments/product_view_api/` | Pillow 설치, 입력 경로 수정 후 `python run_api.py plan`; 실제 전송은 별도 `submit` 명령 | FAL_KEY를 환경변수로 설정. 원래 가격은 실험 당시 추정치이며 현재 청구 제한이 아님 |
| `product_view_builtin/`, `edit_agent/` | 프로토콜과 평가·비교 스크립트 참고 | 내장 도구 호출은 코드만으로 자동 재현되지 않음. CS 원본·입력 매핑 제외 |
| `multiview/` | 모델·입력별 protocol 참고 | 외부 서비스의 당시 실행 상태는 재현 보장하지 않음 |
| `evidence_renderer/run18–21/` | numpy, OpenCV, scipy, h5py. 공개 YCB 자료 다운로드 도구 포함 | 촬영 보정·소스 주석·실사 입력이 필요. 로컬 자료 없는 상태에서 즉시 실행 불가 |
| `surface_restore/run23/` | numpy/OpenCV 설치, protocol의 reference/cases와 원본 라벨·인쇄 영역 등록 후 `python restore.py` | 상품별 수동 등록과 지원 범위 확인 필요 |
| `structural_restore/run27–29/` | numpy/OpenCV/Pillow, 등록 마스크·원본/광고 준비 | 특정 상품·컷을 위한 탐색 코드. 범용 자동 부품 검출 아님 |
| `asset_renderer/run30/` | Blender 및 원래 제품과 일치하는 자산·실사 준비 | 원래 자산 게이트에서 중단. 자산/Blender 바이너리 미포함 |
| `product_lora/run31/` | `requirements.lock.txt`, `train.py --help`, `infer.py`, 다운로드·사진 준비 도구 참고 | MPS/가중치·학습 사진 필요. `run_train.sh`는 원래 환경용이며 경로와 설정을 먼저 점검 |

원래 엔진/모델에 맞는 의존성을 사용한다. 하나의 requirements로 모든 환경을 합쳐 호환을 보장하지 않는다. `requirements.lock.txt`는 당시 환경 기록이며 다른 운영체제에서 그대로 설치된다는 의미가 아니다.

## 3. JamMate 웹 시제품 소스

`research/sites/jammate-print-lab/dist/`에 HTML/JS/CSS와 `assets/engine.py`가 있다. Pyodide Worker에서 원본 등록과 인쇄 후처리를 수행하며 이미지 생성 API는 호출하지 않는다.

**공개본에는 기본 실사·광고·비교 이미지가 없다.** 현재 `index.html`의 기본 샘플과 구조 비교 화면은 원본 자료가 없으면 표시/초기화되지 않는다. 완성된 공개 웹 데모로 배포한 상태가 아니라, 제출용으로 기존 시제품 소스를 보존한 것이다. 원래 입력을 합법적으로 확보한 경우 해당 `assets/`를 복원하고 HTTP 서버로 제공할 수 있다. 일반적인 이미지 파일 업로드 기능이 있다는 사실만으로 기본 초기화에 필요한 자료가 대체되지는 않는다.

```sh
python -m http.server 8000 --directory research/sites/jammate-print-lab/dist
```

`worker.js`는 Pyodide CDN 접근이 필요하다. `identity.js`와 `structural.js`는 보관된 비교 결과를 보여주는 코드이며 범용 구조 복원 엔진이 아니다.

## 4. 팀원 코드

```sh
cd teammate/creazy-consistency
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m demo.app
```

기본 서버는 `http://127.0.0.1:8024`. `.env` 예시는 참고용이며 코드가 읽는 방식에 맞춰 필요한 키를 실행 환경에 설정한다. `.env` 파일을 만들기만 하면 모든 실행 경로에 자동 주입된다고 가정하지 않는다. VLM/API 기능을 켜면 외부 전송·요금이 발생할 수 있다.

원본 ZIP에는 머그·펌프 기본 자료가 있었지만 이 공개본에서는 제외했다. 기본 샘플 버튼과 CS 라벨링 화면에는 대응하는 `data/`, `out/`가 필요하다. 본인 자료의 CLI 흐름은 다음과 같다.

```sh
.venv/bin/python -m refset.build --uploads data/local/uploads --query 'a mug' --out out/local --main main.jpg
.venv/bin/python -m spec.make --refset out/local/refset.json --request '정면 광고' --out out/local/spec
.venv/bin/python -m check.run --refset out/local/refset.json --outputs data/local/outputs --out out/local/check --spec out/local/spec
```

`spikes/`는 사례별 탐색 스크립트다. `data/cs/<번호>`와 특정 파일명은 비공개 입력을 찾는 원래 인터페이스이며 자료 자체가 배포된다는 뜻이 아니다. 생성 실행을 전부 무심코 일괄 호출하지 않는다.

## 5. 검증과 재현의 구분

이번 패키징 검사는 구문·JSON·문서 링크·일부 합성 입력 단위 테스트다. 과거 보고서의 제품 품질 평가는 다시 돌리지 않았다. 정확한 입력/모델 버전/seed가 없는 생성 결과는 동일 픽셀 재현을 약속하지 않는다. 학습·3D·API·브라우저 전체 동작은 각 실험의 당시 검증과 현재 공개본 검사로 구분한다.

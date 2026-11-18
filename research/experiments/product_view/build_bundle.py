"""Build a portable Colab notebook and a bundle with the already downloaded LoRA."""
import ast
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})


def code(text):
    ast.parse(text)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)})


md('''# 제품 각도 보존: B0 / B1 / LoRA 실험

GPU 추론은 아직 사전 검증하지 않은 실행 노트북입니다. 먼저 1각도 × 1seed로 확인합니다.

**런타임 → 런타임 유형 변경 → GPU. A100급 권장, T4는 지원하지 않습니다.**
NF4에서도 실제 메모리 사용량은 입력과 실행 환경에 따라 다릅니다. 기반 모델 다운로드는 약 58GB입니다.

준비물: `product_view_bundle.zip`, 같은 제품의 front/right/back 사진, 가능하면 평가용 front_right/back_right 사진.
''')
code('''from google.colab import files
from pathlib import Path
import zipfile, subprocess, sys, json, shutil

uploaded = files.upload()  # product_view_bundle.zip 선택
bundle = next((n for n in uploaded if n.endswith('.zip')), None)
assert bundle, '실험 패키지 ZIP을 업로드하세요.'
destination = Path('/content/creazy_experiment')
destination.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for name in archive.namelist():
        assert (destination / name).resolve().is_relative_to(destination.resolve()), 'Invalid ZIP path'
    archive.extractall(destination)
ROOT = destination / 'product_view'
assert (ROOT / 'run_experiment.py').exists()
print('패키지:', ROOT)
del uploaded
''')
md('## GPU 확인 및 고정 라이브러리 설치\n설치와 실행은 별도 Python 프로세스로 수행합니다. 아직 모델 추론을 시작하지 않습니다.')
code('''subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, str(ROOT / 'setup_gpu.py')], check=True)
subprocess.run([sys.executable, str(ROOT / 'download_models.py')], check=True)
print('남은 디스크 GB:', round(shutil.disk_usage('/content').free / 1e9, 1))
assert shutil.disk_usage('/content').free > 70e9, '기반 모델용 디스크 여유가 부족합니다. 약 90GB 이상 여유 권장.'
''')
md('''## 제품 사진 업로드

생성용 `front.jpg`, `right.jpg`, `back.jpg`와 평가용 `front_right.jpg`, `back_right.jpg`를 선택합니다. PNG도 됩니다. 파일의 기본 이름은 유지하세요.
평가용 두 장은 생성에 전달되지 않습니다. 이미지는 자동으로 자르거나 보정하지 않습니다.
''')
code('''photos = files.upload()
photo_paths = {}
for name, data in photos.items():
    file = Path(name)
    stem = file.stem
    assert stem in ['front', 'right', 'back', 'front_right', 'back_right'], f'사진 이름 확인: {name}'
    assert file.suffix.lower() in ['.jpg', '.jpeg', '.png'], 'JPG 또는 PNG를 사용하세요.'
    assert stem not in photo_paths, f'중복 시점: {stem}'
    folder = 'input' if stem in ['front', 'right', 'back'] else 'holdout'
    relative = Path('data') / folder / file.name
    (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / relative).write_bytes(data)
    photo_paths[stem] = str(relative)
assert all(s in photo_paths for s in ['front', 'right', 'back']), '생성용 앞/오른쪽/뒤 사진 3장이 필요합니다.'
del photos
print(photo_paths)
''')
md('''## 제품 설명 설정

아래 두 항목을 실제 제품에 맞게 채우세요. 권장하는 영문 설정 문구는 준비 과정에서 함께 작성할 수 있습니다.
정면은 `front` 사진을 기준으로 합니다. 기본 광고 장면은 단순한 베이지색 받침대입니다.
''')
code('''PRODUCT_DESCRIPTION = ""  # 예: A red insulated travel mug with a black lid and one side handle.
MUST_PRESERVE = []  # 예: ["one handle attached at two points", "black lid with one button", "white ABC logo"]
assert PRODUCT_DESCRIPTION.strip() and 3 <= len(MUST_PRESERVE) <= 5, '제품 설명과 핵심 특징 3~5개를 채우세요.'
config = json.loads((ROOT / 'experiment.example.json').read_text())
config['product_description'] = PRODUCT_DESCRIPTION
config['must_preserve'] = MUST_PRESERVE
config['inputs'] = [photo_paths[s] for s in ['front', 'right', 'back']]
for target in config['targets']:
    target['holdout'] = photo_paths.get(target['id'])
(ROOT / 'experiment.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
print(json.dumps(config, ensure_ascii=False, indent=2))
''')
md('''## 결과 저장 위치

Drive 저장을 선택하면 인증 후 그 위치에 실행 결과를 보관합니다. 입력 사진·설정도 재개를 위해 함께 저장합니다.
같은 이름으로 다른 사진/설정을 실행할 때는 EXPERIMENT_NAME을 바꾸세요.
''')
code('''SAVE_TO_DRIVE = True
EXPERIMENT_NAME = 'product01_v1'
assert EXPERIMENT_NAME.replace('_', '').replace('-', '').isalnum()
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS = Path('/content/drive/MyDrive/creazy_experiments') / EXPERIMENT_NAME
else:
    RESULTS = ROOT / 'runs' / EXPERIMENT_NAME
RESULTS.mkdir(parents=True, exist_ok=True)
snapshot = RESULTS / 'input_snapshot'
snapshot.mkdir(exist_ok=True)
for relative in list(photo_paths.values()) + ['experiment.json', 'models.lock.json']:
    source = ROOT / relative
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), '기존 사진/설정과 다릅니다. EXPERIMENT_NAME을 바꾸세요.'
    else:
        target.write_bytes(source.read_bytes())
print('결과:', RESULTS)
''')
md('''## 실행 계획 검사

이 단계는 이미지 생성을 하지 않습니다. 첫 점검은 7회 생성(최종 4장 + 중간 3장)입니다.
''')
code('''def run_cli(action, folder, smoke=False):
    command = [sys.executable, str(ROOT / 'run_experiment.py'), action,
               '--config', str(ROOT / 'experiment.json'), '--out', str(RESULTS / folder)]
    if smoke:
        command.append('--smoke')
    subprocess.run(command, check=True, cwd=ROOT)

run_cli('plan', 'smoke', smoke=True)
''')
md('''## 첫 실제 생성: 1각도 × 1seed

**이 셀부터 기반 모델 다운로드·GPU 추론이 시작됩니다.** 다운로드가 완료된 뒤 7개의 결과를 생성합니다.
오류가 나면 메시지와 `records`를 보관하세요. 설정을 임의로 바꿔 같은 결과 폴더에 섞지 않습니다.
''')
code("run_cli('run', 'smoke', smoke=True)\n")
md('## 점검 결과 미리 보기 및 다운로드\nZIP을 풀고 report.html을 열면 조건별 결과와 지시문을 볼 수 있습니다. 실제 품질은 준비한 제품 사진과 대조하세요.')
code('''from IPython.display import display, Markdown
from PIL import Image
smoke_plan = json.loads((RESULTS / 'smoke' / 'plan.json').read_text())
for job in smoke_plan['jobs']:
    image_path = RESULTS / 'smoke' / 'images' / (job['id'] + '.png')
    if image_path.exists():
        display(Markdown('**' + job['arm'] + ' / ' + job['stage'] + '**'))
        with Image.open(image_path) as im:
            display(im.copy().resize((512, 512)))
archive = shutil.make_archive('/content/product_view_smoke_results', 'zip', RESULTS)
files.download(archive)
''')
md('''## 본 실험: 2각도 × 3seed

점검 결과를 확인한 뒤에만 RUN_FULL을 True로 바꾸세요. 총 42회 생성입니다. 점검에서 모델 품질을 판단하기 어려우면 불필요하게 전체 실행을 시작하지 않아도 됩니다.
''')
code('''RUN_FULL = False
if RUN_FULL:
    run_cli('plan', 'full')
    run_cli('run', 'full')
    archive = shutil.make_archive('/content/product_view_full_results', 'zip', RESULTS)
    files.download(archive)
else:
    print('본 실험은 실행하지 않았습니다.')
''')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': 'product_view_colab.ipynb'}},
    'nbformat': 4, 'nbformat_minor': 5}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
(ROOT / 'product_view_colab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')

bundle_path = ROOT / 'product_view_bundle.zip'
with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_STORED) as archive:
    archive.writestr('product_view/data/input/.gitkeep', '')
    archive.writestr('product_view/data/holdout/.gitkeep', '')
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(x in relative.parts for x in ['.venv', '__pycache__', 'runs', '.git']):
            continue
        if path.suffix in ['.zip', '.gif', '.png'] or relative.parts[0] == 'data':
            continue
        if relative.parts[:2] == ('models', 'base') and path.suffix == '.safetensors':
            continue
        archive.write(path, Path('product_view') / relative)
print('Notebook:', ROOT / 'product_view_colab.ipynb')
print('Bundle:', bundle_path, f'({bundle_path.stat().st_size / 1e6:.1f} MB)')

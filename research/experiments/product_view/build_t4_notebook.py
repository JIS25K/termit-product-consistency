"""Generate the experimental T4 one-image notebook (no GPU execution here)."""
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
cells = []
def cell(kind, source):
    if kind == 'code':
        ast.parse(source)
    item = dict(cell_type=kind, id=f't4-{len(cells):02}', metadata={}, source=source.splitlines(True))
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)
cell('markdown', '''# T4: 제품 사진 한 장 → 각도 LoRA 결과 한 장
Colab T4용 40스텝 진단 노트북입니다. 첫 이미지 조절 층을 4비트 양자화에서 제외합니다. 실제 품질은 아래 결과와 최신 실행 기록을 확인하세요.
**런타임 → 런타임 유형 변경 → T4 GPU → 저장**, 연결 후 위에서부터 한 셀씩 실행하세요. 모두 실행은 누르지 마세요.
384×384 / 40스텝 / NF4 가중치 + FP32 계산 / 인코딩·생성 프로세스 분리 / 참고 사진 VAE 처리 384급.
FP16 오버플로를 피하기 위한 경로라 느릴 수 있습니다. 일반 RAM 부족으로 중단될 수도 있습니다.
준비물: 최신 product_view_bundle.zip, 제품 사진 한 장(front.jpg 또는 PNG, 짧은 변 512px 이상).
이 결과는 작동 확인용이며 원래 1024/40스텝 결과와 품질을 비교하지 않습니다.
''')
cell('markdown', '## 1. 최신 실험 ZIP 업로드\n아래 실행 버튼을 누르고 product_view_bundle.zip을 선택하세요.')
cell('code', '''from google.colab import files
from pathlib import Path
import subprocess, sys, json, shutil, zipfile, os
uploaded = files.upload()
bundle = next((n for n in uploaded if n.endswith('.zip')), None)
assert bundle, 'product_view_bundle.zip을 선택하세요.'
destination = Path('/content/creazy_t4')
destination.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for name in archive.namelist():
        assert (destination / name).resolve().is_relative_to(destination.resolve())
    archive.extractall(destination)
ROOT = destination / 'product_view'
assert 'nf4_fp32' in (ROOT / 'run_experiment.py').read_text(), '최신 ZIP을 업로드하세요.'
del uploaded
os.environ['PYTHONUNBUFFERED'] = '1'
''')
cell('markdown', '## 2. GPU·RAM·디스크 확인 및 설치\n약 58GB 모델 다운로드 공간이 필요합니다. 이 셀은 라이브러리만 설치합니다.')
cell('code', '''subprocess.run(['nvidia-smi'], check=True)
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
ram_gib = int(mem['MemTotal'].split()[0]) / 1024**2
free_gb = shutil.disk_usage('/content').free / 1e9
print(f'시스템 RAM: {ram_gib:.1f} GiB / 남은 디스크: {free_gb:.1f} GB')
assert free_gb > 75, '디스크가 부족합니다. 새 런타임에서 다시 시작하세요.'
if ram_gib < 20:
    print('주의: 일반 RAM이 작아 모델 로딩 중 런타임이 종료될 수 있습니다. 입력과 환경에 따라 메모리 부족이 발생할 수 있습니다.')
PYTHON = sys.executable
if sys.version_info[:2] not in [(3, 11), (3, 12)]:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.12', '--seed', '/content/creazy_py312'], check=True)
    PYTHON = '/content/creazy_py312/bin/python'
subprocess.run([PYTHON, str(ROOT / 'setup_gpu.py')], check=True)
''')
cell('markdown', '## 3. 제품 사진 한 장 업로드\n제품 전체가 선명하게 보이는 사진을 고르세요. 배경이 단순한 정면 사진을 권장합니다. 자동으로 front 이름으로 복사합니다.')
cell('code', '''photos = files.upload()
assert len(photos) == 1, '사진 한 장만 선택하세요.'
name, data = next(iter(photos.items()))
ext = Path(name).suffix.lower()
assert ext in ['.jpg', '.jpeg', '.png']
photo = ROOT / 'data/input' / ('front' + ext)
photo.parent.mkdir(parents=True, exist_ok=True)
photo.write_bytes(data)
del photos, data
from PIL import Image, ImageOps
from IPython.display import display
with Image.open(photo) as im:
    im = ImageOps.exif_transpose(im).convert('RGB')
    assert min(im.size) >= 512, '짧은 변이 512px 이상인 사진을 사용하세요.'
    preview = im.copy()
    preview.thumbnail((512, 512))
    display(preview)
''')
cell('markdown', '''## 4. 제품 설명 입력
따옴표 안을 실제 제품에 맞게 바꾸세요. 영어를 권장합니다. 아래 예시는 텀블러입니다.
이 셀은 GPU를 사용하지 않습니다. 첫 결과는 오른쪽 앞 45도에 가까운 제품 사진 한 장입니다.''')
cell('code', '''PRODUCT_DESCRIPTION = "REPLACE: A red travel mug with a black lid and a side handle."
MUST_PRESERVE = ["REPLACE: red cylindrical body", "black lid", "one handle"]
assert 'REPLACE' not in PRODUCT_DESCRIPTION + str(MUST_PRESERVE), 'REPLACE 표시를 실제 제품 설명으로 바꾸세요.'
config = json.loads((ROOT / 'experiment.example.json').read_text())
config.update(product_description=PRODUCT_DESCRIPTION, must_preserve=MUST_PRESERVE,
    inputs=[str(photo.relative_to(ROOT))], input_descriptions=['front view'],
    width=384, height=384, reference_size=384, steps=40, precision='nf4_fp32',
    seeds=[101], diagnostic_only=True)
config['targets'] = [dict(config['targets'][0], holdout=None)]
config_path = ROOT / 'experiment.t4.json'
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
from datetime import datetime
RESULTS = ROOT / 'runs' / ('t4_check_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
command = [PYTHON, str(ROOT / 'run_experiment.py'), '--config', str(config_path), '--out', str(RESULTS)]
subprocess.run(command + ['plan'], check=True, cwd=ROOT)
print('결과 폴더:', RESULTS)
''')
cell('markdown', '''## 5. 모델 다운로드
약 58GB입니다. Colab 연결이 끊기거나 런타임을 삭제하면 다시 다운로드해야 합니다.
진행 중에는 다음 셀을 실행하지 마세요. 이 셀 자체는 이미지 생성을 하지 않습니다.''')
cell('code', '''subprocess.run([PYTHON, str(ROOT / 'download_models.py'), '--base'], check=True, cwd=ROOT)
print('모델 다운로드 완료. 다음 셀에서 한 장을 생성합니다.')
''')
cell('markdown', '''## 6. 한 장 생성
**각도 LoRA를 켠 P1 product 한 장만 생성**합니다. 7장이나 본 실험은 자동 실행하지 않습니다.
시간은 아직 실측되지 않았습니다. encode 프로세스가 사진·문장을 해석한 뒤 종료되어 메모리를 반환합니다. 이어 denoise 프로세스가 이미지 모델을 로딩한 후 40스텝 진행률을 표시합니다.
실패하면 마지막 오류와 다음 셀에서 내려받는 로그 ZIP을 보내주세요. RAM 부족으로 런타임 자체가 꺼지면 Colab 알림 문구를 알려주세요.''')
cell('code', '''log_path = RESULTS / 'console.log'
returncode = 0
with log_path.open('w') as log:
    for phase in ['encode', 'denoise']:
        args = [PYTHON, '-u', str(ROOT / 'run_t4_staged.py'), phase,
                '--config', str(config_path), '--out', str(RESULTS)]
        print('PHASE:', phase, flush=True)
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
        if returncode != 0:
            break
print('실행 종료 코드:', returncode)
if returncode != 0:
    print('실패 로그를 다음 셀에서 다운로드하세요.')
''')
cell('markdown', '''## 7. 결과 확인 및 다운로드
이미지가 있어도 정상 작동을 자동 판정하지 않습니다. 검은 화면·단색·심한 노이즈가 아닌지, 제품과 요청 각도가 보이는지 확인하세요.
40스텝 결과의 제품 형태와 노이즈를 확인한 뒤 동일한 설정으로 7장 비교를 준비합니다.
**다운로드 후 런타임 → 연결 해제 및 런타임 삭제.** 아래 ZIP과 원본 사진은 보관하세요.''')
cell('code', '''from PIL import ImageStat
images = list((RESULTS / 'images').glob('*.png'))
for path in images:
    with Image.open(path) as im:
        display(im.copy())
        print('출력:', im.size, '채널 표준편차:', ImageStat.Stat(im).stddev)
        if max(ImageStat.Stat(im).stddev) < 1:
            print('단색에 가까운 출력입니다. 정상 생성으로 판단하지 마세요.')
for path in (RESULTS / 'records').glob('*.json'):
    record = json.loads(path.read_text())
    print(record.get('status'), '생성 소요 초:', record.get('seconds'), record.get('error', ''))
if not images:
    print('생성된 이미지가 없습니다. console.log를 확인하세요.')
shutil.copy2(config_path, RESULTS / config_path.name)
archive = shutil.make_archive('/content/t4_check_results', 'zip', RESULTS)
files.download(archive)
''')
notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'colab': {'name': 'product_view_t4_colab.ipynb'}}, nbformat=4, nbformat_minor=5)
(ROOT / 'product_view_t4_colab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
print('Built T4 notebook.')

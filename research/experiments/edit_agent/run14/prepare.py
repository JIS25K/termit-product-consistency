from pathlib import Path
import json,shutil,random,hashlib
r=Path(__file__).resolve().parent;p=r/'public';(p/'assets').mkdir(exist_ok=True)
def save(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=2))
rows=[]
def add(product,candidates,refs,request,phase):
 for f in candidates:rows.append(dict(product=product,source=str(r/'source'/product/f),refs=[str(r/'source'/product/a) for a in refs],request=request,phase=phase))
add('almond',['2ea38e327005205c.png','AI-Generated-c95923eb-c447-4327-b6cb-e1baffd7ccf8.png','AI-Generated-037990a2-86cb-41be-bb45-c4b7411f83ab.png','AI-Generated-3e48b963-ff49-410c-87f3-0a6a44e45565.png','AI-Generated-b0c28de9-f7d5-4554-8dfe-982ec36d4801.png'],['2ec507662821f9ad.webp','a6f7e3b3d527f40f.webp'],'제품 사진에 근거한 아몬드 스프레드 상세페이지 이미지. 제품 형태와 로고, 제품명을 보존한다. 장면과 뚜껑 개폐는 연출에 맞게 달라도 된다.','initial_first5')
add('almond',['2c9f6f47bb93baf0.png','a7847a808c9fc34c.png','84c44d92e432d195.png','dd0c3683f98d49c6.png','df7cae6fe62b24ed.png'],['2ec507662821f9ad.webp','a6f7e3b3d527f40f.webp'],'제품 사진에 근거한 아몬드 스프레드 상세페이지 이미지. 제품 형태와 로고, 제품명을 보존한다. 장면과 뚜껑 개폐는 연출에 맞게 달라도 된다.','repair_all5')
add('pot',['a0ba5f10d1bcdf77.jpg','5705a0ffdad28f60.jpg','774ada6c22b226fb.jpg','0dc8e83721877579.jpg'],['47dc5481fa695be7.webp','818fac7e9b94e149.webp'],'17cm 스테인리스 우드 손잡이 편수냄비 상세페이지 이미지. 원본 제품 형태와 손잡이 구조 보존. 조리 장면, 내용물, 뚜껑 개폐, 시점은 달라도 된다. 외부 중국어 광고 문구 보존은 요구하지 않는다.','initial_all4')
add('pot',['90973ac300dff193.jpg','de2fe893238c815f.jpg','a98ea4de882baa9b.jpg'],['469f2865ad185276.webp','47dc5481fa695be7.webp','818fac7e9b94e149.webp'],'17cm 스테인리스 우드 손잡이 편수냄비 상세페이지 이미지. 첫 번째 참조 사진을 제품 형태의 기준으로 한다. 원본 몸통과 손잡이 구조 보존. 조리 장면, 내용물, 뚜껑 개폐, 시점은 달라도 된다.','later_batch3')
add('pen',['AI-Generated-25e59a17-4ddf-4de2-af32-5f6000e4bbd1.png','AI-Generated-0a6d68b2-36a8-4744-9fb4-00cb3255d571.png','e4ab70859d4a4708.png','ce5d8584356c28df.png'],['2e19c3e260783f84.webp','7e2576523a8ba645.webp','539c7870498f1a1e.webp','7e07a52b9b78d187.webp'],'45mm 목공용 긴 금속 촉 마킹펜 상세페이지 이미지. 제품의 주요 구조와 긴 직선 촉을 보존한다. 시점, 손, 작업 배경은 달라도 된다. 여러 제품/비교 사진이 있으므로 45mm 긴 촉 제품을 기준으로 한다. 작은 인쇄 글자는 채점하지 않는다.','initial2_and_repair2')
random.Random(1417).shuffle(rows);mapping=[];cases=[];assets={}
def mask(src):
 if src not in assets:
  f=Path(src);dst=p/'assets'/f'asset_{len(assets)+1:02d}{f.suffix}';shutil.copy2(f,dst);assets[src]=str(dst)
 return assets[src]
for i,row in enumerate(rows,1):
 id=f'case_{i:02d}';mapping.append(dict(id=id,**row));cases.append(dict(id=id,candidate=mask(row['source']),references=[mask(a) for a in row['refs']],request=row['request']))
save(p/'cases.json',cases);save(r/'private/mapping.json',mapping)
rules='''제품 보존 검수. 모든 후보와 참조를 view_image로 확인한다. 제품의 큰 형태, 핵심 부품, 주요 로고/제품명만 판정한다. 작은 법정 표시, 미세 글자, 픽셀 차이, 허용된 조명·시점·자연스러운 가림은 실패로 잡지 않는다. 원본의 핵심 한글 제품명이 작게 렌더링되어 확인 불가능하면 uncertain이며 글자가 있다고 추측하지 않는다. 비교 이미지나 사용 예시에 다른 제품이 있을 수 있어 기준이 불명확하면 uncertain. 요청에서 허용하지 않은 크고 명확한 구조 변경은 fail. 후보마다 pass/fail/uncertain과 근거를 제시한다. 실패 개수를 추정하거나 분포를 가정하지 않는다. 제공된 참조만으로 확인 불가능한 면을 상상하여 감점하지 않는다. 배경 미관은 평가하지 않는다.'''
(p/'rules.txt').write_text(rules)
save(r/'frozen_sha256.json',{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()})
(r/'실험설계.md').write_text('''# run14 — 실제 고객 프로젝트 검수 예비시험\n\n목표: 캐릭터·식혜병 밖에서도 검수기가 핵심 제품 변형을 찾는지 확인한다. 독자: 크리에이지 기업과제 팀. 기준 자료: 사용자 CSV 우측 실제 프로젝트 링크, 화면의 원본 첨부·생성 결과·고객 수정 요청. 서비스 변경과 새 이미지 생성은 하지 않는다. 완료 증거: 원본 파일·출처, 고정된 21개 사례, 독립된 두 평가, 비교 결과.\n\n## 고정 범위\n아몬드 스프레드 10장, 냄비 7장, 마킹펜 4장. 수집 가능성과 요청/제품 대응을 기준으로 선정한 편의 표본이다. 전체 39건의 대표 표본 또는 무작위 표본이 아니다. 루트는 준비 중 미리보기를 보았으므로 루트의 평가는 맹검이 아니다. 새 두 평가자는 문의와 수정 이력·조건명을 보지 않는다. 순서를 seed 1417로 섞고 자산 이름을 가렸다.\n\n아몬드는 첫 생성 5장+수정 5장, 냄비는 최초 4장+후속 생성 묶음 3장, 펜은 최초 제품 컷 2장+수정 2장이다. 펜 최초1(문제 제기 장면), 최초4(일반 짧은 마커)는 대상 제품 표현인지 분명하지 않아 제외했다. 냄비 중간 단일 수정본은 후속 3장과 중복된 수정 흐름을 줄이기 위해 제외했다. 날짜별 로그를 통한 최초 투입 사진 전체 복원은 하지 않았다. 냄비 후속은 화면에서 확보한 새 원본 1장을 우선 기준으로 한다; 다른 추가 사진을 모두 확보하지 못한 한계가 있다.\n\n## 유지·변경 요소\n두 평가자에게 동일 후보·동일 참조·동일 핵심 판정 기준을 준다. 문의 문구·CSV 판정·수정 전후 구분은 숨긴다. 한 번씩 평가하고 평가 피드백을 이용한 재판정은 하지 않는다. 생성기나 프롬프트 개선 A/B는 아니다.\n\n## 판정 및 해석\npass/fail/uncertain. 작은 문구는 제외하되 요청에서 중요하다고 지정한 제품명은 평가한다. 판독 불가능은 통과로 간주하지 않는다. 독립 검토도 AI이므로 두 평가의 일치율은 인간 정답 대비 정확도가 아니다. CSV의 확실/경계는 문의 분류이며 모든 결과의 실패/정상 라벨이 아니다. 고객의 수정 요청은 그 시점의 섹션·묶음에 대한 증거로만 사용한다. 서비스의 수정 완료 안내는 성공 라벨이 아니다. 정상 사람 라벨이 없으므로 오탐률/정밀도/재현율을 확정하지 않는다.\n\n## 결과에 따른 결정\n명확한 큰 변형을 반복 탐지하면 검수 후보를 유지한다. 제품명 판독이나 참조 선택에서 불일치가 크면 그 항목은 사람 확인으로 보낸다. 자동 재생성은 아직 연결하지 않는다. 다음 단계는 불일치 및 uncertain을 우선 사람이 확인한 뒤 정상 사례를 추가해 오탐/누락을 측정하는 것이다.\n''')
print('Prepared',len(cases),'cases',len(assets),'assets')

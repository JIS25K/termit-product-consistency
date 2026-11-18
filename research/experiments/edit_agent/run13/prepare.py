from pathlib import Path
import json, hashlib, random, shutil

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'run12'
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2) if not isinstance(x,str) else x)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'frozen_sha256.json').exists(), 'Already frozen; do not overwrite'
proto=read(PREV/'frozen/protocol.json')
old=read(PREV/'outputs/evaluation_c.json')['rows']
paths={}
mapping={}
def copy_asset(source):
    source=Path(source)
    if str(source) not in paths:
        name=f'asset_{len(paths)+1:02d}{source.suffix}'
        target=ROOT/'public'/'assets'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        paths[str(source)]=str(target)
        mapping[str(target.relative_to(ROOT))]={'source':str(source),'sha256':sha(source)}
    return paths[str(source)]
ordered=list(old)
random.Random(713).shuffle(ordered)
cases=[]; oracle=[]
for n,r in enumerate(ordered,1):
    product=r['product'];arm=r['arm'];step=r['step'];meta=proto['products'][product]
    before=PREV/'images'/product/('shared' if step==1 else arm)/f'v{step-1}.png'
    current=PREV/'images'/product/arm/f'v{step}.png'
    case={'id':f'case_{n:02d}','product':meta['label'],'request_history':proto['track_C']['requests'][:step],
          'current_request':proto['track_C']['requests'][step-1], 'step':step,
          'target_side_description':meta['target'],'product_invariants':meta['invariants'],
          'images':{'before':copy_asset(before),'candidate':copy_asset(current),'initial_warm_light':copy_asset(PREV/'images'/product/'shared'/'v0.png'),'actual_side':copy_asset(meta['side']),'actual_front':copy_asset(meta['front'])}}
    cases.append(case)
    oracle.append({'id':case['id'],'product':product,'arm':arm,'step':step,'expected':r['conservative_overall'],'source':r['image']})
save(ROOT/'public/cases.json',cases)
save(ROOT/'private/oracle.json',oracle)
save(ROOT/'private/assets.json',mapping)
rules=read(PREV/'frozen/C_evaluation_rules.json')
save(ROOT/'public/evaluation_rules.json',rules)
instruction='''You are an independent visual quality-control model for product editing. Read ONLY this instruction, public/cases.json, public/evaluation_rules.json and the image paths listed there. Do not read any parent directories, previous runs, private files, other agents, reports, scripts, or prior conversation. Inspect every candidate and relevant before/initial/actual product images with view_image. Images may be reused across cases; inspect each unique relevant image once and reason for each case independently. You are not told how many defects there are; do not assume a distribution. Evaluate the requested change and all unrevoked preservation requirements separately. Use pass/fail/uncertain; uncertain is not pass. Do not penalize small letters, pixels, normal view occlusion or allowed lighting. Detect concrete visible deviations, not speculative issues. A change in view can violate the request while the product itself remains intact. Produce exactly one first response for each case without retries: {"rows":[{"id":"case_XX","decision":"pass|fail|uncertain","checks":{"requested_change":"pass|fail|uncertain","preservation":"pass|fail|uncertain","product_identity":"pass|fail|uncertain"},"violations":[{"attribute":"view|scene|lighting|identity|position|framing","observation":"Korean evidence"}],"evidence":"concise Korean visual explanation","repair_feedback":"If fail, concise Korean targeted correction preserving other requested attributes; empty when pass or uncertain"}]}. Save outputs/detection.json under this run13 directory. Do not generate images, delegate, browse, or open other files. Final answer only output path. The evaluation will not send you grading feedback. Inspect images before deciding.'''
save(ROOT/'public/detector_instruction.txt',instruction)
repair_prompt='''현재 광고를 마지막 사용자 요청에 맞게 한 번 더 편집해줘.
전체 사용자 요청 이력:
초기: 실제 정면 제품을 밝은 카페, 따뜻하고 부드러운 제품 조명으로 중앙에 전체가 보이게 광고화.
1: 제품을 실제 목표 측면으로 회전. 밝은 카페와 따뜻한 제품 조명, 중앙 전체 구도 유지.
2: 카페를 어두운 저녁으로, 제품 조명을 차가운 푸른빛으로 변경. 현재 측면·제품 형태·디자인·위치 유지.
3 (현재 최종 목표): 제품 조명만 첫 광고처럼 따뜻하고 부드럽게 복원. 측면 각도와 어두운 카페 유지.
이미지 1은 현재 광고이자 편집할 이미지다. 이미지 2는 직전 광고로, 측면 각도와 어두운 카페 배경의 기준이다. 이미지 3은 첫 광고로, 따뜻하고 부드러운 제품 조명만 참고한다. 이미지 4는 실제 제품 측면으로 형태와 라벨 배치의 기준이다.
위 이력과 자료를 사용해 최종 목표를 충족하는 광고 한 장을 완성해줘. 병 전체의 중앙 배치와 크기, 노란 뚜껑·목·어깨·몸통 형태와 주요 라벨을 유지한다. 작은 성분표 철자는 평가하지 않는다. 새로운 문구·로고·소품을 추가하지 않는다. 정사각형 한 장.'''
save(ROOT/'frozen/repair_common_prompt.txt',repair_prompt)
refs=[PREV/'images/bottle/baseline/v3.png',PREV/'images/bottle/baseline/v2.png',PREV/'images/bottle/shared/v0.png',Path(proto['products']['bottle']['side'])]
save(ROOT/'frozen/repair_references.json',[str(p) for p in refs])
protocol={'run':'run13','date':'2026-11-17','objective':'검수 모델이 기존 오류를 감지하며 같은 생성 예산에서 구체적 피드백이 복구에 도움이 되는가',
 'scope':'기존 편집 12장 검수; 엄격한 진입 조건 충족 시 단일 실패의 두 조건 각 2회 재시도, 총 새 이미지 4장',
 'gate':{'misses':0,'false_alarms':0,'uncertain':0,'positive_cases':1,'negative_cases':11,'failure_action':'복구 생성 중지하고 검수 결과와 기준 개선 필요 보고. 같은 사례로 검수 프롬프트 튜닝하지 않음.'},
 'detector':'Fresh isolated Codex agent; masked shuffled inputs; no answer labels, failure count, arm names or parent chat; first response.',
 'oracle':'Prior run12 root and independently masked visual judgments; model-defined reference labels, not human ground truth.',
 'repair':{'base':'Same failed bottle baseline v3 from run12 for all 4 calls','arms':['ordinary_retry','feedback_retry'],'repeats_per_arm':2,'same_refs_and_order':True,'same_common_prompt':True,'only_difference':'Feedback arm appends verbatim repair_feedback from the detector. No manual rewrite or extra planning.','feedback_template':'\n\n검수에서 확인된 수정 피드백:\n{repair_feedback}','order':['ordinary_retry_1','feedback_retry_1','feedback_retry_2','ordinary_retry_2'],'independence':'Each starts from same failed image; no chaining or best-of selection.','same_image_generation_tool':'built-in image_gen','seed':'not exposed; no seed matching claimed','quality_rerolls':0},
 'success':['target_side','warm_soft_product_light','dark_cafe','major_product_identity','center_full_product'],
 'final_evaluation':'Another fresh masked reviewer receives output + goals/reference photos, no prompts/arm mapping; root checks separately.',
 'decision':'Detecting this one known failure is feasibility, not accuracy generalization. Repair successes per arm reported separately; 2 repeats cannot establish statistical superiority. Any further detection or repair trials require a new task.'}
save(ROOT/'frozen/protocol.json',protocol)
save(ROOT/'실험설계.md','''# run13 — 검수와 복구의 효과

사용자와 과제팀이 검수·복구 기능의 도입 가치를 판단하기 위한 실험이다. 기존 run12의 편집 12장을 다시 생성하지 않고 검수한다. 판단은 별도 새 모델 세션에 맡기고 정답·오류 개수·조건명은 숨긴다.

기준 자료: run12의 요청·실제 사진·생성 이미지, 기존 두 시각 평가가 일치한 판정. 이는 인간 확정 정답이 아니라 이번 실험의 비교 기준이다.

진입 조건: 알려진 실패 1건 탐지, 정상 11건 오탐 없음, 불확실 판정 없음. 통과하면 동일한 실패 식혜병에서 일반 재시도 2장과 검수 피드백 재시도 2장을 생성한다. 같은 요청 이력·입력 4장·공통 프롬프트를 쓰며 검수기의 실제 피드백 추가 여부만 바꾼다. 각 출력은 동일 실패에서 독립적으로 시작하고 재추첨하지 않는다.

성공 기준: 목표 측면 복구, 따뜻한 제품 조명, 어두운 카페, 주요 병 형태·라벨, 중앙 전체 구도. 작은 성분표·픽셀 차이는 제외한다. 잘못 바뀐 각도와 제품 자체 손상을 구분한다.

완료 증거: 사전 고정 해시, 검수 원본과 혼동행렬, 조건 진입 여부, 최대 4개 실제 생성 결과와 정확한 프롬프트·입력 기록, 조건명을 가린 평가, 통합 보고. 단일 실패의 제한된 탐색 결과로 보고하며 서비스 배포나 일반적 정확도를 주장하지 않는다.
''')
files=['public/cases.json','public/evaluation_rules.json','public/detector_instruction.txt','private/oracle.json','private/assets.json','frozen/protocol.json','frozen/repair_common_prompt.txt','frozen/repair_references.json']
save(ROOT/'frozen_sha256.json',{p:sha(ROOT/p) for p in files})
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'images').mkdir(exist_ok=True)
print(json.dumps({'cases':len(cases),'assets':len(paths),'frozen':len(files)},ensure_ascii=False))

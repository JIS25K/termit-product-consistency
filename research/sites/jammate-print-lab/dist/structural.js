(() => {
  const $ = id => document.getElementById(id);
  const data = {
    '29': {
      verdict: '중단 · 제품을 먼저 고정해도 받침과 공받침에 이중 윤곽이 남았습니다. 결과 위치 재조정 0회였지만 합성 흔적 게이트에 실패했습니다. 이 경로를 더 튜닝하지 않습니다.',
      caveat: 'Run 28의 원본 마스크·위치·배율을 그대로 재사용했습니다. 일반 편집 1회와 주변 생성 1회, 재시도 0회입니다. 실험 경로에는 같은 원본에서 만든 고정 배치도가 추가됩니다. 전체 출력은 모두 1448×1086이며 비교용으로 1024×768에 정규화했습니다. 수동 등록을 포함한 제작 절차 비교이며 새 시점 3D 복원은 아닙니다.',
      baseline: 'baseline_normalized.png',
      extra: [['fixed_product_guide.png','생성 전 고정한 제품 배치도'],['scene_generated_normalized.png','주변 생성 원출력 · 고정 레이어 복원 전']],
      candidateLabel: '제품 고정 후 복원 · 접합 게이트 실패'
    },
    '27': {
      verdict: '조건부 유망 · 일반 호출은 금속 촉을 너무 짧게 줄였고, 원본 배치는 원래 비율에 가까웠습니다. 휘어짐 해결의 우위는 입증하지 못했습니다.',
      caveat: '동일 좌표 크롭 비교입니다. 원본 부품 영역·연결점·지름은 수동 등록했습니다. 원본 마스크 정리 1회, 결과 위치 재조정 0회. 저해상도 부품의 부드러운 경계가 남습니다. 새 각도 3D 재구성 결과가 아닙니다.',
      baseline: 'baseline.png'
    },
    '28': {
      verdict: '전체 품질 보류 · 일반 호출도 초록 받침을 복원했습니다. 원본 배치는 실제 받침 폭을 유지했지만 잔디 번짐 때문에 광고 완성도 게이트에 실패했습니다.',
      caveat: 'CSV에서 찾은 골프티 프로젝트의 현재 출력으로 실행했습니다. 과거 문의의 바로 그 이미지인지는 미확정입니다. 일반 출력은 1448×1086, 비교에는 1024×768 축소본을 사용합니다. 몸통/받침 비율 게이트는 둘 다 통과했습니다. 받침 폭/높이 차이는 사후 보조 관찰입니다.',
      baseline: 'baseline_normalized.png'
    }
  };
  function render() {
    const run = $('structuralCase').value, entry = data[run], prefix = `assets/structural/${run}-`;
    $('structuralVerdict').textContent = entry.verdict;
    $('structuralCaveat').textContent = entry.caveat;
    $('structuralBoard').src = prefix + 'comparison.png';
    $('structuralBoardLink').href = prefix + 'comparison.png';
    $('structuralReport').href = prefix + 'README.md';
    $('structuralPrompt').href = prefix + 'baseline_prompt.txt';
    $('structuralMetrics').href = prefix + 'evaluation.json';
    const whole = $('structuralView').value === 'whole';
    $('structuralBoard').hidden = whole;
    $('structuralWhole').hidden = !whole;
    $('structuralWhole').replaceChildren();
    for (const [file, label] of [['reference.webp','원본 제품 사진'],['target.png','공통 장면 입력'],[entry.baseline,'일반 호출 1회'],['candidate.png',entry.candidateLabel || '원본 부품 배치 · 수동 등록'],...(entry.extra || [])]) {
      const figure = document.createElement('figure'), a = document.createElement('a'), img = document.createElement('img'), cap = document.createElement('figcaption');
      a.href = prefix + file; a.target = '_blank';
      img.src = a.href; img.alt = label; cap.textContent = label;
      a.append(img); figure.append(a,cap); $('structuralWhole').append(figure);
    }
  }
  $('structuralCase').addEventListener('change', render);
  $('structuralView').addEventListener('change', render);
  render();
})();

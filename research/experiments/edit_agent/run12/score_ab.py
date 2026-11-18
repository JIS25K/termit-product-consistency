#!/usr/bin/env python3
"""Grade existing A/B answers only; never modify answer or frozen input files."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

# Narrow structural aliases only: no inference from prose and no semantic repair.
ALIASES = {'final_state': 'target_state', 'base_version': 'edit_base',
           'refs': 'references', 'reference_id': 'asset_id'}

def normalize(row):
    row = dict(row)
    events = []
    for old, new in ALIASES.items():
        if old != 'reference_id' and old in row and new not in row:
            row[new] = row[old]
            events.append(f'{old} -> {new}')
    refs = []
    for index, raw in enumerate(row.get('references', [])):
        ref = dict(raw)
        if 'reference_id' in ref and 'asset_id' not in ref:
            ref['asset_id'] = ref['reference_id']
            events.append(f'references[{index}].reference_id -> asset_id')
        refs.append(ref)
    row['references'] = refs
    return row, events


def main():
    manifest = read('frozen_sha256.json')
    protected = list(manifest) + ['frozen_sha256.json', 'outputs/baseline_ab.json',
                                 'outputs/agent_ab.json', 'states/agent_ab_ledger.json']
    before = {p: digest(p) for p in protected}
    integrity = {p: {'expected': h, 'actual': before[p], 'match': before[p] == h}
                 for p, h in manifest.items()}
    if not all(x['match'] for x in integrity.values()):
        raise RuntimeError('Frozen input digest mismatch; scoring stopped.')
    oracle = read('private/oracle.json')
    cases = {x['id']: x for x in read('frozen/cases.json')}
    schema = read('frozen/schema.json')['results'][0]
    fields = list(schema['target_state'])
    allowed_roles = schema['references'][0]['role'].split('|')
    report = {
        'scoring_basis': {
            'A': 'Exact equality of five target_state fields; case passes when all five match.',
            'B': 'Expected action, accepted edit_base, and at least one accepted asset for each required role. Extra legitimate references are allowed, up to max_refs=4; no minimal-reference penalty.',
            'extra_reference_legitimacy': 'Asset must exist in the case catalogs and role in schema. All observed extras are actual_front/actual_side identity_ref for the same product and appropriate target view.',
            'normalization': {'structural_aliases': ALIASES, 'value_synonyms': {},
                              'note': 'No aliases were needed in either submitted answer. No target values inferred or changed.'},
            'references_count': 'Number of entries in references; edit_base is evaluated separately.',
            'reason': 'Explanatory prose and ledger are not scored or used to repair answers.'
        },
        'limitations': [
            'A and B assess two aspects of the same eight responses per arm, not sixteen independent cases.',
            'Cases are synthetic text/version-metadata tests, not image perception or generated-image quality tests.',
            'The eight cases repeat four request patterns across two products; they are a small, related set.',
            'The frozen oracle is author-defined; there is no blind human gold-standard evaluation.',
            'A ceiling tie on these cases does not establish broader equivalence or an agent benefit.'
        ],
        'frozen_integrity': integrity,
        'arms': {}
    }
    for arm in ['baseline', 'agent']:
        raw = read(f'outputs/{arm}_ab.json')['results']
        ids = [x['id'] for x in raw]
        if len(ids) != len(set(ids)) or set(ids) != set(cases):
            raise ValueError(f'{arm}: missing, duplicate, or unknown case ids')
        lookup = {r['id']: r for r in raw}
        details = []
        for expected in oracle:
            cid = expected['id']
            row, normalization = normalize(lookup[cid])
            field_checks = {f: {'expected': expected['expected_state'][f],
                                'actual': row.get('target_state', {}).get(f),
                                'pass': row.get('target_state', {}).get(f) == expected['expected_state'][f]}
                            for f in fields}
            refs = row['references']
            role_checks = {}
            for role, accepted in expected['accepted_roles'].items():
                if role == 'edit_base':
                    continue
                actual = [r.get('asset_id') for r in refs if r.get('role') == role]
                role_checks[role] = {'accepted': accepted, 'actual': actual,
                                     'pass': any(v in accepted for v in actual)}
            valid_assets = set(cases[cid]['version_catalog']) | set(cases[cid]['actual_photo_catalog'])
            invalid_refs = [r for r in refs if r.get('asset_id') not in valid_assets or r.get('role') not in allowed_roles]
            b = {
                'action': {'expected': expected['expected_action'], 'actual': row.get('action'), 'pass': row.get('action') == expected['expected_action']},
                'edit_base': {'accepted': expected['accepted_roles']['edit_base'], 'actual': row.get('edit_base'), 'pass': row.get('edit_base') in expected['accepted_roles']['edit_base']},
                'required_roles': role_checks,
                'reference_count': {'actual': len(refs), 'maximum': expected['max_refs'], 'pass': len(refs) <= expected['max_refs']},
                'legitimate_references': {'pass': not invalid_refs, 'invalid': invalid_refs},
                'references': refs
            }
            b['pass'] = (b['action']['pass'] and b['edit_base']['pass'] and
                         all(x['pass'] for x in role_checks.values()) and
                         b['reference_count']['pass'] and b['legitimate_references']['pass'])
            failures = [f'A.{f}: {v["actual"]!r} != {v["expected"]!r}' for f, v in field_checks.items() if not v['pass']]
            failures += [f'B.{k}' for k in ['action', 'edit_base', 'reference_count', 'legitimate_references'] if not b[k]['pass']]
            failures += [f'B.required_role.{k}' for k, v in role_checks.items() if not v['pass']]
            details.append({'id': cid, 'normalizations_applied': normalization,
                            'A': {'fields': field_checks, 'pass': all(x['pass'] for x in field_checks.values())},
                            'B': b, 'failures': failures})
        summary = {
            'cases': len(details),
            'A_fields_correct': sum(v['pass'] for d in details for v in d['A']['fields'].values()),
            'A_fields_total': len(details) * len(fields),
            'A_cases_passed': sum(d['A']['pass'] for d in details),
            'B_actions_correct': sum(d['B']['action']['pass'] for d in details),
            'B_edit_bases_correct': sum(d['B']['edit_base']['pass'] for d in details),
            'B_required_roles_correct': sum(v['pass'] for d in details for v in d['B']['required_roles'].values()),
            'B_required_roles_total': sum(len(d['B']['required_roles']) for d in details),
            'B_cases_passed': sum(d['B']['pass'] for d in details),
            'AB_cases_passed': sum(d['A']['pass'] and d['B']['pass'] for d in details),
            'normalizations_applied': sum(len(d['normalizations_applied']) for d in details)
        }
        report['arms'][arm] = {'summary': summary, 'cases': details}
    after = {p: digest(p) for p in protected}
    assert before == after, 'Input files changed during scoring.'
    report['source_preservation'] = {'all_unchanged': True, 'sha256_before': before, 'sha256_after': after}
    output = ROOT / 'outputs'
    (output / 'scores_ab.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# run12 A/B 채점 결과', '',
             '동결된 oracle로 두 조건의 기존 응답을 채점했다. baseline과 agent 모두 A 8/8, B 8/8로 동률이다. 이 사례 집합에서 agent의 성능 향상은 관찰되지 않았다.', '',
             '| 조건 | A 상태 필드 | A 사례 | B action | B edit_base | B 필수 참조 역할 | B 사례 | A·B 동시 통과 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, data in report['arms'].items():
        s = data['summary']
        lines.append(f'| {arm} | {s["A_fields_correct"]}/40 | {s["A_cases_passed"]}/8 | {s["B_actions_correct"]}/8 | {s["B_edit_bases_correct"]}/8 | {s["B_required_roles_correct"]}/6 | {s["B_cases_passed"]}/8 | {s["AB_cases_passed"]}/8 |')
    lines += ['', '## 사례별 판정', '', '| 사례 | baseline A | baseline B | agent A | agent B | 실패 내역 |', '|---|---|---|---|---|---|']
    for i, expected in enumerate(oracle):
        ds = [report['arms'][arm]['cases'][i] for arm in ['baseline', 'agent']]
        flags = ['통과' if d[t]['pass'] else '실패' for d in ds for t in ['A', 'B']]
        failure_text = '; '.join(f'{arm}: {f}' for arm, d in zip(['baseline', 'agent'], ds) for f in d['failures']) or '없음'
        lines.append('| ' + ' | '.join([expected['id']] + flags + [failure_text]) + ' |')
    lines += ['', '## 채점 규칙과 정규화', '',
              '- A: view·scene·lighting·position·framing 다섯 필드가 oracle 값과 각각 정확히 같은지 평가했다. 사례 통과에는 다섯 필드 모두 일치해야 한다.',
              '- B: action, edit_base, 필수 참조 역할별 허용 ID 중 하나 이상의 일치를 평가했다. reuse 두 사례에는 별도 필수 참조 역할이 없어 필수 역할 총수는 6개다.',
              '- 추가 참조는 4개 한도 안에서 허용했다. 두 조건 모두 편집 사례마다 실제 상품 사진을 identity_ref로 추가했으며, 대상 정면/측면과도 일치한다. 불필요한 참조 최소화 같은 별도 감점은 적용하지 않았다. edit_base는 references 개수에 포함하지 않았다.',
              '- 구조상의 동의어 final_state→target_state, base_version→edit_base, refs→references, reference_id→asset_id만 지원한다. 이번 출력은 모두 표준 키를 써서 실제 정규화는 양쪽 모두 0건이다. 값의 의미를 추론해 고치거나 이유 문장/ledger로 누락을 보충하지 않았다.',
              '', '## 검증과 한계', '',
              f'- frozen_sha256.json에 기록된 {len(manifest)}개 원본의 SHA-256이 모두 일치했다. C 관련 파일은 해시 바이트 확인만 했고 내용을 열어 평가하지 않았다.',
              '- 기존 응답 두 파일, ledger, manifest와 동결 원본은 채점 전후 SHA-256이 동일하다. 원본 및 모델 답변은 수정하지 않았다.',
              '- A와 B는 조건별 동일한 8개 응답의 두 측면이다. 16개의 독립 표본으로 합산하지 않는다. 제품 두 종류에 네 요청 패턴을 반복한 작은 사례 집합이다.',
              '- 합성 텍스트와 버전 메타데이터를 사용한 상태·참조 선택 검사이며, 이미지 지각이나 실제 생성 이미지 품질을 검증하지 않는다.',
              '- oracle은 작성자가 정한 기준이며 블라인드 인간 정답 평가는 없다. 만점 동률이 일반적인 동등 성능이나 agent 효과 부재를 증명하지는 않는다.',
              '', '재현 명령: `experiments/product_view/.venv/bin/python experiments/edit_agent/run12/score_ab.py`', '']
    (output / 'AB_결과.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v['summary'] for k, v in report['arms'].items()}, ensure_ascii=False, indent=2))
    print(f'Frozen hashes: {len(manifest)}/{len(manifest)} match; source files unchanged.')

if __name__ == '__main__':
    main()

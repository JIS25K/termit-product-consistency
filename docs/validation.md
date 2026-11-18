# 공개본 검증 기록

검증일: 2026-11-18. 이번 검증은 소스 정리와 실행 경계 확인이다. README의 과거 제품 품질 결과를 다시 측정한 것이 아니다.

| 검사 | 결과 | 확인한 범위 |
|---|---|---|
| 우리 측 product_view unittest | 10개 통과 | 프로토콜·조건 구성·실행/재개, 가짜 backend |
| 우리 측 product_view_api unittest | 6개 통과 | 요청 계획·예산·재개·요청 주소 검사, 임시 합성 이미지 |
| 팀원 자료 독립 테스트 | 43개 통과, 자료 의존 테스트 1개 제외 | count/score/suggest/strong의 선택된 테스트, 실제 모델 품질 아님 |
| Python 구문 | 158개 통과 | AST 파싱, import/실행 보장은 아님 |
| JSON·노트북 | JSON 35개, 노트북 2개 통과 | 파싱 및 노트북 출력 제거 |
| JavaScript·Shell | JS 4개, Shell 2개 통과 | `node --check`, `bash -n` |
| 문서 링크·공개본 검사 | 통과 | 저장소 내부 파일 링크, 개인 절대 경로·비공개 스토리지 URL·일부 토큰 패턴 |

자격증명 탐지는 패턴 검사이며 모든 형태의 비밀값이 없음을 수학적으로 보장하는 검사는 아니다. 공개 대상 파일과 URL 호스트도 함께 검토했고 원본 데이터는 포함하지 않았다.

## 실행 명령

루트에서 Python 3.12와 Pillow를 준비한 뒤:

```sh
python scripts/check_archive.py
python -m unittest discover -s research/experiments/product_view/tests -v
python -m unittest discover -s research/experiments/product_view_api/tests -v
```

팀원 코드 디렉터리에서 해당 의존성을 설치한 뒤:

```sh
python -m pytest tests/test_count.py tests/test_score.py tests/test_suggest.py tests/test_strong.py \
  -k 'not test_after_fix_on_real_creazy_edits' -q
```

실제 검증은 macOS arm64/Python 3.12에서 수행했다. 우리 측 두 suite는 Pillow가 있는 런타임, 팀원 suite는 기존 OpenCV/numpy/torch 환경에 pytest·FastAPI·httpx·anthropic·python-multipart를 추가해 실행했다. 모든 의존성을 새로 설치한 깨끗한 환경 전체를 검증한 것은 아니다.

## 자료 의존 테스트의 실패도 남긴다

처음에는 팀원 `test_count`, `test_revision`, `test_restore`, `test_score`, `test_suggest`, `test_strong`를 함께 실행했다. **43개 통과, 16개 실패**였다. 16개는 제외한 `data/mug/outputs/run10b/` 이미지 또는 `run10b_edit/edits.json`에 의존했다. 파일 미존재나 `cv2.imread`의 빈 반환값으로 실패했으며, 알고리즘 검증을 통과한 것으로 계산하지 않았다.

최종 독립 테스트 명령에서는 사진에 의존하는 revision/restore 파일을 실행 대상에서 빼고, suggest의 실사 편집 테스트 1개를 제외했다. 원래 테스트 파일과 assertions는 보존했다. 원본 자료를 합법적으로 복원한 뒤 해당 테스트를 별도로 실행해야 한다. 그 외 팀원 전체 suite는 재실행하지 않았다.

## 이번에 실행하지 않은 검증

- GPU 학습·생성, 모델 다운로드, 실제 유료 API 호출
- 원본 상품 이미지의 품질 평가·시점 재투영 비교·3D 렌더링
- 기본 이미지가 제외된 두 웹 시제품의 브라우저 전체 흐름
- 고객 승인·사용 가능성·서비스 내 비용/수정 횟수 비교

테스트 통과 수를 제품 정확도나 광고 품질의 성과로 해석하지 않는다.

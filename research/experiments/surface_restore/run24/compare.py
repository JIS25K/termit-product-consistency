"""Run 24: fixed Run 23 transform, one 4x area-sampling alternative.

No feature matching, alignment changes, ink dilation, sharpening, or new glyphs.
The candidate scene is never resized; only transferred ink is supersampled.
"""
import datetime
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bbox(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None


def project(points, H):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)


def composite(candidate, color_times_alpha, alpha):
    return np.rint(candidate*(1-alpha[:, :, None])+color_times_alpha).clip(0, 255).astype(np.uint8)


def main():
    cfg = json.loads((ROOT/'protocol.json').read_text())
    for entry in cfg['frozen_files']:
        assert sha(ROOT/entry['path']) == entry['sha256'], entry['path']
    t0 = time.perf_counter()
    reference = cv2.imread(str(ROOT/cfg['reference']))
    candidate = cv2.imread(str(ROOT/cfg['candidate']))
    H = np.array(cfg['H_source_to_target'], dtype=np.float64)
    h, w = candidate.shape[:2]
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = cfg['source_text_roi']
    alpha_source = np.zeros(gray.shape, np.float32)
    alpha_source[y0:y1, x0:x1] = np.clip((170-gray[y0:y1, x0:x1].astype(float))/100, 0, 1)
    name_quad = project([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], H)

    # Baseline must reproduce Run 23 exactly before interpreting the alternative.
    baseline_start = time.perf_counter()
    alpha_base = cv2.warpPerspective(alpha_source, H, (w,h), flags=cv2.INTER_LINEAR)
    color_base = cv2.warpPerspective(reference, H, (w,h), flags=cv2.INTER_LINEAR)
    baseline = composite(candidate, color_base*alpha_base[:, :, None], alpha_base)
    baseline_seconds = time.perf_counter()-baseline_start
    baseline_saved = cv2.imread(str(ROOT/cfg['baseline']))
    assert np.array_equal(baseline, baseline_saved), 'Run 23 baseline reproduction failed'

    # Crop only for computation. Bounds follow the fixed projection, not annotations.
    left, top = np.floor(name_quad.min(axis=0)).astype(int)-3
    right, bottom = np.ceil(name_quad.max(axis=0)).astype(int)+4
    assert 0 <= left < right <= w and 0 <= top < bottom <= h
    cw, ch = int(right-left), int(bottom-top)
    T = np.array([[1,0,-left],[0,1,-top],[0,0,1]], dtype=float)
    s = cfg['alternative']['scale_factor']
    # Center the s*s subsamples around each original pixel center. This compensates
    # for resize grid conventions; it is not a change to the source-target mapping.
    S = np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1]], dtype=float)
    hi_H = S @ T @ H
    alt_start = time.perf_counter()
    alpha_hi = cv2.warpPerspective(alpha_source, hi_H, (s*cw,s*ch), flags=cv2.INTER_LINEAR)
    color_hi = cv2.warpPerspective(reference, hi_H, (s*cw,s*ch), flags=cv2.INTER_LINEAR)
    alpha_lo = cv2.resize(alpha_hi, (cw,ch), interpolation=cv2.INTER_AREA)
    premult_lo = cv2.resize(color_hi*alpha_hi[:, :, None], (cw,ch), interpolation=cv2.INTER_AREA)
    alternative = candidate.copy()
    alternative[top:bottom, left:right] = composite(candidate[top:bottom, left:right], premult_lo, alpha_lo)
    alpha_alt = np.zeros((h,w), np.float32)
    alpha_alt[top:bottom, left:right] = alpha_lo
    alternative_seconds = time.perf_counter()-alt_start

    # Guard is fixed from geometry for both methods, before either output is assessed.
    guard = np.zeros((h,w), bool)
    guard[top:bottom, left:right] = True
    metrics = {}
    for name, output, alpha, seconds in [
        ('baseline', baseline, alpha_base, baseline_seconds),
        ('area4x', alternative, alpha_alt, alternative_seconds)
    ]:
        support = alpha > 0
        changed = np.any(candidate != output, axis=2)
        assert not np.any(changed & ~support)
        assert not np.any(changed & ~guard)
        path = OUT/(name+'.png')
        cv2.imwrite(str(path), output)
        cv2.imwrite(str(OUT/(name+'_support.png')), support.astype(np.uint8)*255)
        assert np.array_equal(cv2.imread(str(path)), output)
        np.save(OUT/(name+'_alpha.npy'), alpha)
        b = bbox(changed)
        metrics[name] = {
            'changed_pixels':int(changed.sum()), 'support_pixels':int(support.sum()),
            'changed_bbox':b, 'ink_bbox_height_px':b[3]-b[1],
            'changed_outside_support':0, 'changed_outside_fixed_guard':0,
            'sum_alpha':float(alpha.sum()), 'output_sha256':sha(path),
            'render_seconds':seconds
        }
    cv2.imwrite(str(OUT/'method_difference.png'), np.any(baseline!=alternative, axis=2).astype(np.uint8)*255)
    # Prior pipeline & inputs are read-only throughout this experiment.
    for entry in cfg['frozen_files']:
        assert sha(ROOT/entry['path']) == entry['sha256'], entry['path']
    results = {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256':sha(ROOT/'protocol.json'), 'code_sha256':sha(__file__),
        'baseline_pixel_identical_to_run23':True, 'frozen_files_unchanged':True,
        'H_reused_exactly':cfg['H_source_to_target'], 'alignment_reruns':0,
        'alternative_configs_tested':1, 'scale_factor':s,
        'fixed_guard_xyxy':[int(left),int(top),int(right),int(bottom)],
        'different_pixels_between_methods':int(np.any(baseline!=alternative, axis=2).sum()),
        'generative_calls':0, 'metrics':metrics,
        'elapsed_seconds':time.perf_counter()-t0
    }
    (OUT/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    make_comparison(cfg, candidate, baseline, alternative, name_quad)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def make_comparison(cfg, candidate, baseline, alternative, name_quad):
    font = lambda n:ImageFont.truetype('/System/Library/Fonts/AppleSDGothicNeo.ttc', n)
    panel = Image.new('RGB', (1260,650), '#f4f5f7')
    d = ImageDraw.Draw(panel)
    d.text((20,15), 'Run 24 | 위치 고정, 축소 방식만 비교', font=font(28), fill='#17212b')
    d.text((20,58), '상단: 라벨 영역을 원본 픽셀 크기로 표시 · 하단: 같은 제품명 영역 5배 확대 (최근접)', font=font(18), fill='#4b5563')
    label_quad = np.array(cfg['projected_label_quad'])
    low = np.floor(label_quad.min(axis=0)).astype(int)-8
    high = np.ceil(label_quad.max(axis=0)).astype(int)+8
    label_box = tuple(map(int, [*low, *high]))
    nlow = np.floor(name_quad.min(axis=0)).astype(int)-5
    nhigh = np.ceil(name_quad.max(axis=0)).astype(int)+5
    text_box = tuple(map(int, [*nlow,*nhigh]))
    for i,(arr,title) in enumerate([(candidate,'입력: 제품명 누락'),(baseline,'기존: 한 번에 축소 투영'),(alternative,'비교: 4배 투영 → 면적 평균 축소')]):
        x = 20+420*i
        im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        d.text((x,104), title, font=font(21), fill='#17212b')
        native = im.crop(label_box)
        panel.paste(native, (x+(390-native.width)//2,150))
        d.text((x,357), '제품명 5배 확대',font=font(18),fill='#4b5563')
        crop = im.crop(text_box)
        crop = crop.resize((crop.width*5,crop.height*5),Image.Resampling.NEAREST)
        panel.paste(crop,(x,390))
    d.text((20,557),'변환 행렬·글자 원본·크기·색·위치 고정 · 배경은 확대/축소하지 않음',font=font(19),fill='#4b5563')
    d.text((20,590),'확대본의 매끈함과 최종 크기의 가독성을 구분해 판정 · 샤프닝/글자 굵게 만들기/재생성 없음',font=font(19),fill='#4b5563')
    panel.save(OUT/'comparison.png')


if __name__ == '__main__':
    main()

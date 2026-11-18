"""Evaluation fixture preparation only; not imported by the repair pipeline."""
import hashlib, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'edit_agent/run14/source/almond'
original_path = DATA / '84c44d92e432d195.png'
original = cv2.imread(str(original_path))
mask = np.zeros(original.shape[:2], np.uint8)
# Manually chosen ONLY for constructing the controlled missing-name fixture.
# Neither this rectangle, mask nor original image is loaded by restore.py.
mask[338:364, 753:863] = 255
omitted = cv2.inpaint(original, mask, 5, cv2.INPAINT_TELEA)
assert np.array_equal(omitted[mask == 0], original[mask == 0])
cv2.imwrite(str(ROOT / 'inputs/controlled_missing.png'), omitted)
cv2.imwrite(str(ROOT / 'evaluation/deletion_mask.png'), mask)
(ROOT / 'evaluation/preparation.json').write_text(json.dumps({
    'type': 'controlled omission, not a second observed customer failure',
    'original': str(original_path), 'deletion_rectangle': [753, 338, 863, 364],
    'method': 'OpenCV TELEA inpainting, radius 5, name rectangle only',
    'changed_pixels': int(np.any(original != omitted, axis=2).sum()),
    'outside_mask_changed_pixels': 0,
    'original_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest()
}, ensure_ascii=False, indent=2))

import cv2
import numpy as np
from pathlib import Path
from check.sift import PrintMatcher, MultiPrintMatcher

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/mug/outputs/run10b/AI-Generated-3271d5a2-bf88-4931-9461-8e2473ee318b.png"


def test_locate_returns_the_print_box_in_original_pixels(tmp_path):
    img = cv2.imread(str(IMG))
    cv2.imwrite(str(tmp_path / "p.png"), img[300:560, 560:800])
    loc = PrintMatcher(tmp_path / "p.png").locate(img)
    assert loc["pass"]
    x0, y0, x1, y1 = loc["box"]
    assert abs(x0 - 560) < 25 and abs(y0 - 300) < 25 and abs(x1 - 800) < 25 and abs(y1 - 560) < 25


def test_multi_locate_names_the_print(tmp_path):
    img = cv2.imread(str(IMG))
    cv2.imwrite(str(tmp_path / "p.png"), img[300:560, 560:800])
    m = MultiPrintMatcher([("print.png", tmp_path / "p.png")])
    assert m.locate(img)["print"] == "print.png"


def test_locate_without_a_match_has_no_box(tmp_path):
    blank = (255 * np.ones((400, 400, 3))).astype("uint8")
    img = cv2.imread(str(IMG))
    cv2.imwrite(str(tmp_path / "p.png"), img[300:560, 560:800])
    loc = PrintMatcher(tmp_path / "p.png").locate(blank)
    assert not loc["pass"] and loc["box"] is None

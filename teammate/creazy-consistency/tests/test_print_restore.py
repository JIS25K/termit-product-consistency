import cv2
import numpy as np
import pytest
from pathlib import Path

from check.print_restore import full_res_homography, matcher_for, region_mask, restore

ROOT = Path(__file__).resolve().parent.parent
REFSET = ROOT / "out/mug/refset.json"  # built by refset.build; out/ is not in git
pytestmark = pytest.mark.skipif(not REFSET.exists(), reason="build the mug reference set first (out/mug/refset.json)")
MATCH = ROOT / "data/mug/outputs/run9/AI-Generated-a9c75624-32f5-4b55-a8b6-836bd937d64d.png"  # real print, score 33
INVENTED = ROOT / "data/mug/outputs/run8/c01_2bd77713.png"  # invented print, score 0


def test_restore_raises_the_score_and_leaves_the_outside_alone():
    m, pr = matcher_for(REFSET)
    img = cv2.imread(str(MATCH))
    out, info = restore(img, pr, m)
    assert info["status"] == "restored"
    assert info["inliers_after"] > info["inliers_before"]
    assert info["outside_max_diff"] == 0


def test_gate_sends_an_invented_print_elsewhere_and_returns_the_image_unchanged():
    m, pr = matcher_for(REFSET)
    img = cv2.imread(str(INVENTED))
    out, info = restore(img, pr, m)
    assert info["status"] == "no_print"
    assert np.array_equal(out, img)


def test_full_res_homography_lands_on_the_located_box():
    m, pr = matcher_for(REFSET)
    img = cv2.imread(str(MATCH))
    r = m.match(img)
    Hf = full_res_homography(r["H"], pr.shape, img.shape)
    _, outline, _ = region_mask(Hf, pr.shape, img.shape, shrink=0.0)
    ys, xs = np.nonzero(outline)
    x0, y0, x1, y1 = m.locate(img)["box"]
    assert abs(xs.min() - x0) < 4 and abs(ys.min() - y0) < 4 and abs(xs.max() - x1) < 4 and abs(ys.max() - y1) < 4


def test_blurred_print_comes_back():
    m, pr = matcher_for(REFSET)
    img = cv2.imread(str(MATCH))
    r = m.match(img)
    _, outline, width = region_mask(full_res_homography(r["H"], pr.shape, img.shape), pr.shape, img.shape)
    blurred = np.where(outline[..., None], cv2.GaussianBlur(img, (0, 0), max(2.0, 0.006 * width)), img)
    before = m.match(blurred)["inliers"]
    out, info = restore(blurred, pr, m)
    assert info["status"] == "restored"
    assert info["inliers_after"] > max(before, r["inliers"])


def test_restore_best_picks_a_print_and_matches_the_single_print_result():
    from check.print_restore import prints_for, restore_best
    multi, prints = prints_for(REFSET)
    img = cv2.imread(str(MATCH))
    out, info = restore_best(img, multi, prints)
    assert info["print"] in prints and info["status"] == "restored" and info["outside_max_diff"] == 0


def test_version3_options_still_restore_and_leave_the_outside_alone():
    # clean_print, refine, support, edge_safe: the settings of C version 3 (spikes/validation/run.py)
    m, pr = matcher_for(REFSET)
    img = cv2.imread(str(MATCH))
    out, info = restore(img, pr, m, clean_print=True, refine=True, support=True, edge_safe=True)
    assert info["status"] == "restored"
    assert info["inliers_after"] > info["inliers_before"]
    assert info["outside_max_diff"] == 0
    assert 0 < info["supported_share"] <= 1


def test_supported_surface_leaves_out_what_the_match_does_not_reach():
    # a print with an extra panel glued on (like the retail box behind a bottle): the match lands
    # on the real print only, so the glued panel is not supported
    from check.print_restore import supported_surface
    from check.sift import PrintMatcher
    m0, pr = matcher_for(REFSET)
    img = cv2.imread(str(MATCH))
    h, w = pr.shape[:2]
    panel = np.full((h, w // 2, 3), (40, 90, 30), np.uint8)
    cv2.putText(panel, "BOX", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    wide = np.hstack([pr, panel])
    p = ROOT / "out" / "mug" / "_wide_print_test.png"
    cv2.imwrite(str(p), wide)
    try:
        m = PrintMatcher(p)
        sup = supported_surface(wide, m, m.match(img))
    finally:
        p.unlink()
    assert sup[:, w + 10:].mean() < 0.2
    assert sup[:, :w].mean() > 0.3

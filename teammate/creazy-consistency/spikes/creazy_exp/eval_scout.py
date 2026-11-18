"""Scouting sheet: where does plain generation (A) break the product?

For each image: the full frame, a zoom on the product, and numbers:
  label   SIFT matches that agree with the real label on a cylinder (overlay_bottle fit); low means
          the label is redrawn beyond recognition, hidden, or too small to match
  scale   size of the product relative to the seller's photo (from the same fit)
  turn    how far the bottle is turned
  bottles OWLv2 boxes for "a bottle" (a shelf scene has several by design)

  python spikes/creazy_exp/eval_scout.py out/creazy_exp3
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import overlay_bottle as O  # noqa: E402
from refset.models import detect  # noqa: E402


def zoom_box(g, par, R, yb0, yb1, cx, dets):
    H, W = g.shape[:2]
    if par is not None:
        phi, s, cxg, cyg = par
        x0, x1 = cxg - s * R * 1.3, cxg + s * R * 1.3
        y0, y1 = cyg + s * (yb0 - 120), cyg + s * (yb1 + 15)
    elif dets:
        x0, y0, x1, y1 = dets[0]["box"]
        pw, ph = 0.2 * (x1 - x0), 0.08 * (y1 - y0)
        x0, x1, y0, y1 = x0 - pw, x1 + pw, y0 - ph, y1 + ph
    else:
        return g
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), W), min(int(y1), H)
    return g[y0:y1, x0:x1] if x1 > x0 + 10 and y1 > y0 + 10 else g


def main():
    exp = Path(sys.argv[1])
    rb, ra, cx, R, yb0, yb1 = O.real()
    rows, info = [], {}
    for f in sorted((exp / "outputs" / "A").glob("*.png")):
        g = cv2.imread(str(f))
        p, q = O.matches(rb, ra, g)
        n, par = O.fit_cylinder(p, q, cx, R) if len(p) >= O.MIN_INLIERS else (0, None)
        if n < O.MIN_INLIERS:
            par = None
        dets = detect(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)), ["a bottle"], threshold=0.2)
        z = zoom_box(g, par, R, yb0, yb1, cx, dets)
        rec = {"label_matches": int(n), "scale": round(float(par[1]), 2) if par else None,
               "turn": par[0] if par else None, "bottles": len(dets)}
        info[f.name] = rec
        full = cv2.resize(g, (360, 360))
        zz = cv2.resize(z, (int(360 * z.shape[1] / z.shape[0]), 360))[:, :300]
        zz = cv2.copyMakeBorder(zz, 0, 0, 0, 300 - zz.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        txt = f"{f.stem.replace('P3_', '')}  label {n}  scale {rec['scale']}  bottles {len(dets)}"
        row = np.hstack([full, zz]); cv2.putText(row, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        rows.append(row)
        print(f.name, rec, flush=True)
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.full_like(rows[i], 255)]) for i in range(0, len(rows), 2)]
    cv2.imwrite(str(exp / "scout_sheet.jpg"), np.vstack(pairs), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (exp / "scout.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

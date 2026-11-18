"""First-pass check: does the output contain the real print?

SIFT keypoints of the print region (from the reference set) are matched against the output
image. Ratio test removes ambiguous matches, then a RANSAC homography keeps only matches that
agree on one geometric mapping. The number of those inliers is the score.

A redrawn or invented print can share a few local features with the real one, but they do
not line up under one mapping, so it stays near zero.
"""
import cv2
import numpy as np

REF_SIDE = 740
OUT_SIDE = 1280
PASS = 20


class PrintMatcher:
    def __init__(self, print_path):
        ref = cv2.imread(str(print_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(print_path)
        s = REF_SIDE / max(ref.shape)
        self.ref = cv2.resize(ref, None, fx=s, fy=s)
        self.clahe = cv2.createCLAHE(2.0, (8, 8))
        self.sift = cv2.SIFT_create(nfeatures=4000)
        self.bf = cv2.BFMatcher()
        self.kr, self.dr = self.sift.detectAndCompute(self.clahe.apply(self.ref), None)

    def _prep(self, gray):
        s = OUT_SIDE / max(gray.shape)
        return cv2.resize(gray, None, fx=s, fy=s) if s < 1 else gray

    def features(self, image_bgr):
        gray = self._prep(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        ki, di = self.sift.detectAndCompute(self.clahe.apply(gray), None)
        return gray, ki, di

    def match(self, image_bgr):
        return self.match_features(*self.features(image_bgr))

    def match_features(self, gray, ki, di):
        res = {"good": 0, "inliers": 0, "H": None, "kp": ki, "matches": [], "gray": gray}
        if di is None or len(ki) < 2:
            return res
        pairs = self.bf.knnMatch(self.dr, di, k=2)
        good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
        res["good"] = len(good)
        if len(good) < 6:
            return res
        src = np.float32([self.kr[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        dst = np.float32([ki[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
        if mask is None:
            return res
        inl = [g for g, m in zip(good, mask.ravel()) if m]
        res.update(inliers=len(inl), H=H, matches=inl)
        return res

    def score(self, image_bgr):
        r = self.match(image_bgr)
        return {"good": r["good"], "inliers": r["inliers"], "pass": r["inliers"] >= PASS}

    def _locate(self, image_bgr, r):
        res = {"inliers": r["inliers"], "pass": r["inliers"] >= PASS, "box": None}
        if r["H"] is None or not res["pass"]:
            return res
        h, w = self.ref.shape
        quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(quad, r["H"]).reshape(-1, 2)
        H0, W0 = image_bgr.shape[:2]
        gh, gw = r["gray"].shape
        proj = proj * np.float32([W0 / gw, H0 / gh])  # undo _prep downscale
        x0, y0 = np.floor(proj.min(axis=0)).astype(int)
        x1, y1 = np.ceil(proj.max(axis=0)).astype(int)
        x0, x1 = max(0, int(x0)), min(W0, int(x1))
        y0, y1 = max(0, int(y0)), min(H0, int(y1))
        if x1 > x0 and y1 > y0:
            res["box"] = (x0, y0, x1, y1)
        return res

    def locate(self, image_bgr):
        """Bounding box of the projected print outline in ORIGINAL image pixels, or None."""
        return self._locate(image_bgr, self.match(image_bgr))

    def draw(self, image_bgr, path):
        """Side-by-side picture: reference print, output, inlier lines, and the projected
        outline of the print when a homography was found."""
        r = self.match(image_bgr)
        out = cv2.cvtColor(r["gray"], cv2.COLOR_GRAY2BGR)
        if r["H"] is not None and r["inliers"] >= PASS:
            h, w = self.ref.shape
            quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(quad, r["H"])
            cv2.polylines(out, [np.int32(proj)], True, (70, 140, 22), 4)
        ref = cv2.cvtColor(self.ref, cv2.COLOR_GRAY2BGR)
        vis = cv2.drawMatches(
            ref, self.kr, out, r["kp"] or [], r["matches"], None,
            matchColor=(70, 140, 22), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(path), vis)
        return {"good": r["good"], "inliers": r["inliers"], "pass": r["inliers"] >= PASS}


class MultiPrintMatcher:
    """Several print regions (front label, back label, ...). An output passes when any of
    them is found; the score is the best one and `print` says which region it was."""

    def __init__(self, prints):
        # prints: list of (name, path)
        self.names = [n for n, _ in prints]
        self.matchers = [PrintMatcher(p) for _, p in prints]

    def _best(self, image_bgr):
        # Output keypoints are the same for every region, so they are computed once.
        feats = self.matchers[0].features(image_bgr)
        best = None
        for name, m in zip(self.names, self.matchers):
            r = m.match_features(*feats)
            if best is None or r["inliers"] > best[2]["inliers"]:
                best = (name, m, r)
        return best

    def score(self, image_bgr):
        name, _, r = self._best(image_bgr)
        return {"good": r["good"], "inliers": r["inliers"], "pass": r["inliers"] >= PASS, "print": name}

    def draw(self, image_bgr, path):
        name, m, _ = self._best(image_bgr)
        return {**m.draw(image_bgr, path), "print": name}

    def locate(self, image_bgr):
        name, m, r = self._best(image_bgr)
        return {**m._locate(image_bgr, r), "print": name}

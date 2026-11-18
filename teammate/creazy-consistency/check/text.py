"""Read text with EasyOCR and compare it with anchor words from the real photos."""
import difflib
import re
import unicodedata
from functools import lru_cache

import numpy as np
from PIL import Image

MIN_CONF = 0.6
MIN_LEN = 2
MATCH_SIM = 0.8
GARBLED_SIM = 0.5
MIN_PX = 14

_KEEP = re.compile(r"[^0-9a-z가-힣ㄱ-ㆎ]")


def normalize(s: str) -> str:
    return _KEEP.sub("", unicodedata.normalize("NFKC", s).lower())


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


@lru_cache(maxsize=1)
def reader():
    import easyocr
    return easyocr.Reader(["ko", "en"], gpu=False)


def read(image: Image.Image, box: tuple | None = None, upscale_to: int = 1000) -> list[dict]:
    ox, oy = (int(box[0]), int(box[1])) if box else (0, 0)
    crop = image.convert("RGB")
    if box:
        crop = crop.crop(tuple(int(v) for v in box))
    scale = 1.0
    longest = max(crop.size)
    if box and 0 < longest < upscale_to:
        scale = upscale_to / longest
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    out = []
    for pts, txt, conf in reader().readtext(np.array(crop)):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append({
            "text": txt,
            "conf": float(conf),
            "box": (ox + min(xs) / scale, oy + min(ys) / scale,
                    ox + max(xs) / scale, oy + max(ys) / scale),
        })
    return out


def words(readings: list[dict], min_conf: float = MIN_CONF) -> list[dict]:
    out = []
    for r in readings:
        if r["conf"] < min_conf:
            continue
        h = r["box"][3] - r["box"][1]
        for piece in r["text"].split():
            w = normalize(piece)
            if len(w) >= MIN_LEN:
                out.append({"word": w, "conf": r["conf"], "h": h})
    return out


def compare(anchors: list[dict], readings: list[dict], box_h: float | None = None) -> dict:
    joined = "".join(normalize(r["text"]) for r in readings)
    loose = [w["word"] for w in words(readings, min_conf=0)]
    expected = [a["word"] for a in anchors if box_h is None or a["rel_h"] * box_h >= MIN_PX]
    found = [a["word"] for a in anchors
             if a["word"] in joined or any(similar(a["word"], w) >= MATCH_SIM for w in loose)]
    missing = [w for w in expected if w not in found]
    garbled, foreign = [], []
    for cw in words(readings):
        w = cw["word"]
        best = max(anchors, key=lambda a: similar(w, a["word"]), default=None)
        s = similar(w, best["word"]) if best else 0.0
        if (best and GARBLED_SIM <= s < MATCH_SIM and abs(len(w) - len(best["word"])) <= 1
                and best["word"] not in found):
            garbled.append(w)
        elif s < GARBLED_SIM and len(w) >= 3:
            foreign.append(w)
    return {"read": len(readings), "expected": expected, "found": found,
            "missing": missing, "garbled": garbled, "foreign": foreign}

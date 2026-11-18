"""Combine the text and count checks into one verdict for a generated image."""
from PIL import Image

from check import count, text


def combine(first: str, text_res: dict | None, count_res: dict | None, where: str) -> dict:
    why = []
    if count_res and count_res["over"]:
        why.append("count")
    t = text_res
    if t:
        if t["garbled"]:
            why.append("text")
        elif first == "match" and t["missing"] and t["read"] > 0:
            why.append("text")
        elif first != "match" and t["foreign"] and not t["found"]:
            why.append("invented_text")
    if why:
        return {"verdict": "different", "why": why}
    if first == "match":
        return {"verdict": "match", "why": []}
    if t and t["found"]:
        return {"verdict": "match", "why": ["text"]}
    return {"verdict": "unverified", "why": []}


def _inside(readings, box):
    out = []
    for r in readings:
        b = r["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            out.append(r)
    return out


def inspect(pil, bgr, matcher, expect, ocr=None, detect=None) -> dict:
    if ocr is None:
        ocr = text.read
    loc = matcher.locate(bgr) if matcher is not None else None
    first = "match" if loc and loc["pass"] else ("no_print" if matcher is None else "unverified")
    anchors = expect.get("anchors") or []
    query = expect.get("query")
    rng = (expect.get("count") or {}).get("range")

    boxes = []
    if query:
        if detect is None:
            from refset.models import detect
        boxes = count.count_boxes(detect(pil, [query]))["boxes"]
    product_box = count.union(boxes)
    count_res = None
    if query and rng is not None:
        n = len(boxes)
        count_res = {"n": n, "range": rng, "over": n > rng[1]}

    print_box = loc["box"] if loc else None
    if loc and loc["pass"] and print_box is not None:
        where = "print"
        read_box = print_box
        key = loc.get("print")
        use = [a for a in anchors if (a.get("print") == key if "print" in loc else a.get("print") is not None)]
        box_h = print_box[3] - print_box[1]
    else:
        where = "product"
        read_box = product_box
        use = anchors
        box_h = None

    text_res = None
    if use:
        readings = _inside(ocr(pil, read_box), read_box) if read_box is not None else []
        text_res = text.compare(use, readings, box_h=box_h)
    res = combine(first, text_res, count_res, where)
    return {**res, "where": where, "print_box": print_box, "product_box": product_box,
            "text": text_res, "count": count_res}


def zoom_crops(pil, print_box, product_box, side=768) -> list[tuple[str, Image.Image]]:
    out = []
    for label, box in (("product close-up", product_box), ("print close-up", print_box)):
        if box is None:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        x0, y0 = max(0, box[0] - 0.08 * w), max(0, box[1] - 0.08 * h)
        x1, y1 = min(pil.width, box[2] + 0.08 * w), min(pil.height, box[3] + 0.08 * h)
        crop = pil.crop((round(x0), round(y0), round(x1), round(y1)))
        s = side / max(crop.size)
        out.append((label, crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))),
                                       Image.LANCZOS)))
    return out


def checks(expect) -> list[str]:
    """Reference lines for the VLM. The text and count checks have false positives (props
    counted as the product, OCR misreads), so these are hints, not rules: the VLM judges by
    the product it sees. Only words read on a print go in."""
    out = []
    words = []
    for a in expect.get("anchors") or []:
        if a.get("print") is not None and a["word"] not in words:
            words.append(a["word"])
    if words:
        out.append(f"참고: 실물 사진의 프린트에서 읽힌 글자: {', '.join(words[:12])}. "
                   "보이는 면에서 이 글자가 빠지거나 바뀌었는지 실제 제품 기준으로 판단하라.")
    rng = (expect.get("count") or {}).get("range")
    if rng:
        out.append(f"참고: 실물 사진 속 제품 개수는 {rng[0]}~{rng[1]}개로 보인다. "
                   "장면 속 소품이나 다른 물건은 세지 말고, 실제로 보이는 제품 기준으로 판단하라.")
    return out

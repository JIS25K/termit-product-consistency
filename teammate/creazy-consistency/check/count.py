"""OWLv2 detections -> product instance count (pure Python)."""

MIN_SCORE = 0.2
REL = 0.6
IOU = 0.5
GROUP_COVER = 0.8


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def count_boxes(dets):
    if not dets:
        return {"n": 0, "boxes": []}
    top = max(x["score"] for x in dets)
    floor = max(MIN_SCORE, REL * top)
    kept = []
    for x in sorted((x for x in dets if x["score"] >= floor), key=lambda x: -x["score"]):
        box = tuple(x["box"])
        if all(_iou(box, k) <= IOU for k in kept):
            kept.append(box)
    final = []
    for i, b in enumerate(kept):
        covered = 0
        for j, o in enumerate(kept):
            if i == j:
                continue
            a = _area(o)
            if a > 0 and _inter(b, o) / a >= GROUP_COVER:
                covered += 1
        if covered < 2:
            final.append(b)
    return {"n": len(final), "boxes": final}


def expected_range(counts):
    pos = [c for c in counts if c >= 1]
    return [min(pos), max(pos)] if pos else None


def union(boxes):
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))

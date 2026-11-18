"""Expected words and product count, taken from the seller's real photos."""
import json
from pathlib import Path

from PIL import Image

from check import count, text


def _folder(refset_path) -> tuple[Path, Path]:
    p = Path(refset_path)
    if p.is_dir():
        return p, p / "refset.json"
    return p.parent, p


def _open(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def build_expect(refset_path, ocr=None, detect=None) -> dict:
    folder, rs_path = _folder(refset_path)
    out = folder / "expect.json"
    if out.exists() and out.stat().st_mtime > rs_path.stat().st_mtime:
        return json.loads(out.read_text())
    if ocr is None:
        ocr = text.read
    if detect is None:
        from refset.models import detect
    rs = json.loads(rs_path.read_text())

    plist = [p for p in (rs.get("prints") or []) if p.get("path")]
    if not plist and (rs.get("print") or {}).get("path"):
        plist = [rs["print"]]
    items = rs.get("items", [])
    # One photo is one upload: the frames of a guided-capture video ("v.mp4@3.0s") count once.
    photo_of = {it["crop"]: it["file"].split("@")[0] for it in items if it.get("crop")}
    # (file, print name, source photo key): a print counts as the photo it was cut from
    sources = [(p["path"], p["path"], photo_of.get(p.get("from"), p.get("from") or p["path"])) for p in plist]
    # Real photos only (seller's product photos and capture frames); detail-page (info) crops
    # carry ad copy and tables, not the product's own print.
    sources += [(it["crop"], None, it["file"].split("@")[0]) for it in items
                if it.get("crop") and it.get("role") in ("product", "capture")]

    seen: dict[str, dict] = {}  # word -> {"images": set, "print", "rel_h"}
    for rel, print_name, key in sources:
        path = folder / rel
        if not path.exists():
            continue
        im = _open(path)
        for w in text.words(ocr(im, (0, 0, im.width, im.height))):
            e = seen.setdefault(w["word"], {"images": set(), "print": print_name,
                                            "rel_h": w["h"] / im.height})
            e["images"].add(key)
    need = 2 if len({key for rel, _, key in sources if (folder / rel).exists()}) >= 2 else 1
    anchors = [{"word": w, "rel_h": e["rel_h"], "print": e["print"]}
               for w, e in seen.items() if len(e["images"]) >= need]

    query = rs.get("query")
    per_photo: dict[str, int] = {}
    if query:
        uploads = Path(rs.get("uploads") or "")
        for it in rs.get("items", []):
            if it.get("role") != "product":
                continue
            f = uploads / it["file"]
            if not f.is_file():
                continue
            per_photo[it["file"]] = count.count_boxes(detect(_open(f), [query]))["n"]
    result = {"query": query, "anchors": anchors,
              "count": {"per_photo": per_photo,
                        "range": count.expected_range(list(per_photo.values()))}}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result

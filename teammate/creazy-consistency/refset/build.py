"""Build a reference set from a seller's uploads.

Steps
  1. Find the product in every upload (including info images such as detail-page captures)
     with OWLv2 and crop it out, so a product that is only visible inside an info image is
     not lost.
  2. Embed the crops with DINOv2 and flag crops that do not look like the rest
     (another product mixed in).
  3. Score how much printed detail each crop carries (SIFT keypoint density inside the body)
     and pick the representative face: the crop with the most detail.
  4. Cut the print region out of the representative face, and out of other detailed faces
     whose print is a different one (a bottle has a front label and a back label). The check
     step matches outputs against every region.

  5. With an API key (refset/vlm.py): name each crop's face, drop crops that show another
     product, and list the faces that are missing. --name lets the VLM write --query and
     --print-query from the seller's Korean product name. Without a key these steps are
     skipped and recorded as such in refset.json.
  0. Roles: product photos and guided capture videos (a slow turn around the product; frames
     are sampled, blurry ones dropped, and the views that add a side kept) are product
     evidence. --main names the seller's main photo, the product being sold. Reference photos
     are scene references only: kept aside, never cropped into the set, and flagged when
     they show a product of the same kind (CS 11). Info images are mined for the product.
  6. Put the representative face and the views that add a face into one image
     (composite.jpg). Creazy attaches one product photo per cut, so this single upload
     brings the print to whichever cut it lands on.

Usage
  python -m refset.build --uploads data/mug/uploads --query "a mug" --out out/mug
  python -m refset.build ... --print-query "a cat line drawing"   # optional, locates the print with OWLv2
  python -m refset.build --uploads data/pump/uploads --name "부케가르니 베이비파우더 바디로션 1000ml" --out out/pump
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import models, vlm

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".mov", ".webm", ".m4v"}


def role_of(name: str) -> str:
    if name.startswith("info-"):
        return "info"
    if name.startswith("reference-"):
        return "reference"
    return "product"


def load_uploads(folder):
    """Photos keep the role their file name gives. A video without a role prefix is a guided
    capture (a slow turn around the product) and is sampled into frames."""
    items = []
    for p in sorted(Path(folder).iterdir()):
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            items.append({"file": p.name, "path": str(p), "role": role_of(p.name)})
        elif ext in VID_EXT:
            role = role_of(p.name)
            items.append({"file": p.name, "path": str(p), "role": "capture" if role == "product" else role, "video": True})
    return items


def open_rgb(path) -> Image.Image:
    """Open an image upright. Phone photos carry the rotation in EXIF; PIL ignores it unless
    asked, while OpenCV (used by the check step) applies it, so both sides must agree."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def find_product(img: Image.Image, query: str, min_score: float):
    dets = models.detect(img, [query], threshold=0.1)
    dets = [d for d in dets if d["score"] >= min_score]
    return dets[0] if dets else None


def inspect_reference(img: Image.Image, query: str, min_score: float):
    """Every product of the seller's kind in a scene reference, not only the best one: a
    reference is often another shop's detail page with the same item shown several times.
    product_fill is the share of the photo the boxes cover (union), which decides whether
    hiding them leaves anything worth borrowing."""
    dets = [d for d in models.detect(img, [query], threshold=0.1) if d["score"] >= min_score]
    boxes = [[int(v) for v in d["box"]] for d in dets]
    fill = 0.0
    if boxes:
        mask = np.zeros((img.height, img.width), bool)
        for x0, y0, x1, y1 in boxes:
            mask[max(0, y0) : y1, max(0, x0) : x1] = True
        fill = float(mask.mean())
    return {
        "has_product": bool(boxes),
        "product_score": round(dets[0]["score"], 3) if dets else None,
        "product_boxes": boxes,
        "product_fill": round(fill, 3),
    }


def capture_frames(path, query, min_score, fps=1.0, keep=6):
    """Frames from a guided capture video: sample, drop blurry frames, find the product, and
    keep the frames that add a view (DINOv2 farthest point), so the set gets the sides a
    seller's still photos often miss. Returns [(t, sharpness, PIL frame, detection)], log."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src / fps)))
    raw, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            sharp = float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            raw.append((round(i / src, 1), sharp, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
        i += 1
    cap.release()
    log = {"sampled": len(raw)}
    if not raw:
        return [], log
    med = float(np.median([r[1] for r in raw]))
    sharp = [r for r in raw if r[1] >= 0.5 * med]
    found = []
    for t, sh, im in sharp:
        det = find_product(im, query, min_score)
        if det is not None:
            found.append((t, sh, im, det))
    log.update(sharp=len(sharp), with_product=len(found))
    if not found:
        return [], log
    embs = models.embed([im.crop(pad_box(d["box"], im.size, 0.04)) for _, _, im, d in found])
    picked = [int(np.argmax([f[1] for f in found]))]
    while len(picked) < min(keep, len(found)):
        sims = (embs @ embs[picked].T).max(axis=1)
        k = int(np.argmin(sims))
        if sims[k] >= 0.97:
            break
        picked.append(k)
    frames = [found[k] for k in sorted(picked, key=lambda k: found[k][0])]
    log["kept"] = [f[0] for f in frames]
    return frames, log


def pad_box(box, size, ratio):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    W, H = size
    return (
        max(0, int(x0 - w * ratio)),
        max(0, int(y0 - h * ratio)),
        min(W, int(x1 + w * ratio)),
        min(H, int(y1 + h * ratio)),
    )


def _keypoints(gray):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    sift = cv2.SIFT_create(nfeatures=4000)
    return sift.detect(clahe.apply(gray), None)


def detail_map(crop: Image.Image, inner=0.10):
    """Keypoint density inside the body of the product (outer margin ignored, since the
    silhouette edge produces keypoints on every face)."""
    gray = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mx, my = int(w * inner), int(h * inner)
    dens = np.zeros((h, w), np.float32)
    n = 0
    for kp in _keypoints(gray):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if mx <= x < w - mx and my <= y < h - my:
            dens[y, x] += 1.0
            n += 1
    area = max(1, (w - 2 * mx) * (h - 2 * my))
    return dens, n / area * 1e4


def print_box_from_density(dens, pad=0.06):
    h, w = dens.shape
    sigma = 0.04 * min(h, w)
    blur = cv2.GaussianBlur(dens, (0, 0), sigma)
    if blur.max() <= 0:
        return None
    # Thin line art (e.g. a cat outline) yields fewer keypoints than lettering, so keep every
    # component that carries a fair share of the densest one and take their union.
    mask = (blur >= 0.10 * blur.max()).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    # Components centred near the crop border are usually the handle, rim or base edge.
    sums = [
        (i, float(blur[lab == i].sum()))
        for i in range(1, n)
        if 0.15 * w <= cents[i][0] <= 0.85 * w and 0.10 * h <= cents[i][1] <= 0.90 * h
    ]
    if not sums:
        return None
    best_sum = max(s for _, s in sums)
    keep = [i for i, s in sums if s >= 0.2 * best_sum]
    x0 = min(stats[i][0] for i in keep)
    y0 = min(stats[i][1] for i in keep)
    x1 = max(stats[i][0] + stats[i][2] for i in keep)
    y1 = max(stats[i][1] + stats[i][3] for i in keep)
    return pad_box((x0, y0, x1, y1), (w, h), pad)


def extra_prints(items, rep, out, keep=0.35, limit=5):
    """Print regions of other detailed crops, skipping any whose print is already covered
    (it matches a kept region), so two photos of the same label count once."""
    from check.sift import PASS, PrintMatcher

    top = rep["detail"]
    kept = [PrintMatcher(out / "print.png")]
    extra = []
    for it in sorted(items, key=lambda i: -i["detail"]):
        if it is rep or it["detail"] < keep * top or len(kept) >= limit:
            continue
        box = print_box_from_density(it["_dens"])
        if box is None:
            continue
        region = cv2.cvtColor(np.asarray(it["_img"].crop(box)), cv2.COLOR_RGB2BGR)
        if any(m.match(region)["inliers"] >= PASS for m in kept):
            continue
        name = f"print_{len(kept) + 1}.png"
        it["_img"].crop(box).save(out / name)
        kept.append(PrintMatcher(out / name))
        extra.append({"path": name, "from": it["crop"], "box": box, "face": it.get("face"), "method": "density"})
    return extra


FACE_ORDER = ("front", "back", "side", "handle", "top", "bottom", "detail", "other")


def close_up(item, fill=0.6):
    """A photo where the product fills most of the frame: kept whole (no detection), or a
    detection box that covers most of the photo (a pump head shot from up close)."""
    return item["full_frame"] or item.get("frame_fill", 0) >= fill


TINY = 200  # px: a crop smaller than this on its short side is a blurry fragment


def tiny(item):
    return min(item["crop_size"]) < TINY


def whole_view(item):
    """A view that can stand for the product: not a close-up, not a tiny fragment. On real
    seller uploads the densest crop is often a texture close-up (a clasp, a leather grain,
    a hook tip), which made a poor representative face."""
    return not close_up(item) and not tiny(item)


def pick_views(items, rep, limit=4, dup=0.92, prefer=None):
    """Views for the composite, the main face first, then views that add a face.

    The main face is the front when the VLM named one (on the pump the densest print is the
    back label, which is not what a buyer sees first), otherwise the representative face.
    With face names, one view per face; without them, the views least like the ones already
    picked (DINOv2), skipping near-duplicates. Whole-product views come before close-ups,
    which show the product at another scale."""
    fronts = [i for i in items if i.get("face") == "front" and not close_up(i)]
    main = max(fronts, key=lambda i: i["detail"]) if fronts else rep
    if prefer is not None and not tiny(prefer):
        # the seller's main photo names the product being sold; a product shot that fills the
        # frame is still a whole view, so the close-up test does not apply to it
        main = prefer
    rest = [i for i in items if i is not main]
    picked = [main]
    if any(i.get("face") for i in items):
        seen = {main.get("face")}
        order = {f: n for n, f in enumerate(FACE_ORDER)}
        for it in sorted(rest, key=lambda i: (close_up(i), order.get(i.get("face"), len(order)), -i["detail"])):
            if len(picked) >= limit:
                break
            if it.get("face") in seen or tiny(it):
                continue
            seen.add(it.get("face"))
            picked.append(it)
        return picked
    for near in (False, True):
        pool = [i for i in rest if close_up(i) == near and not tiny(i)]
        while pool and len(picked) < limit:
            sims = [max(float(i["_emb"] @ p["_emb"]) for p in picked) for i in pool]
            k = int(np.argmin(sims))
            if sims[k] >= dup:
                break
            picked.append(pool.pop(k))
    return picked


def composite(views, path, H=1600, margin=40, gap=24):
    """One image that carries every picked face, for upload as a single product photo.

    Creazy attaches one product photo to each cut, and the generator, not the planner,
    decides the angle. A cut that got a plain-face photo and is drawn from the front gets an
    invented print (Run 6, Run 9 desk cut). If the one photo shows all faces, any cut it
    lands on carries the print. The main face (views[0]) is drawn largest. No text is drawn,
    since a generator may copy lettering into its output."""
    if len(views) < 2:
        return None
    inner = H - 2 * margin
    main = views[0]["_img"]
    if main.width / main.height < 0.75:
        # tall products (bottles, tubes): all views side by side, bottoms aligned
        layout, boxes = "row", []
        for n, v in enumerate(views):
            im = v["_img"]
            h = inner if n == 0 else min(int(inner * 0.78), int(inner * 0.6 * im.height / im.width))
            boxes.append((im, int(im.width * h / im.height), h))
        W = 2 * margin + sum(w for _, w, _ in boxes) + gap * (len(boxes) - 1)
        canvas = Image.new("RGB", (W, H), "white")
        x = margin
        for im, w, h in boxes:
            canvas.paste(im.resize((w, h), Image.LANCZOS), (x, margin + inner - h))
            x += w + gap
    else:
        # the main face on the left, the other views stacked on the right
        layout = "main+stack"
        canvas = Image.new("RGB", (H, H), "white")
        mw = int((inner - gap) * 0.62)
        sw = inner - gap - mw
        others = views[1:]
        k = max(2, len(others))
        th = (inner - gap * (k - 1)) // k
        y0 = margin + (inner - (len(others) * th + gap * (len(others) - 1))) // 2

        def put(im, x, y, w, h):
            sc = min(w / im.width, h / im.height)
            r = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
            canvas.paste(r, (x + (w - r.width) // 2, y + (h - r.height) // 2))

        put(main, margin, margin, mw, inner)
        for j, v in enumerate(others):
            put(v["_img"], margin + mw + gap, y0 + j * (th + gap), sw, th)
    canvas.save(path, quality=92)
    return {"layout": layout, "size": canvas.size}


def same_product_flags(embs, drop=0.15):
    """Median cosine similarity of each crop to the others. A crop is flagged when it sits
    well below the group median (another product, or a wrong photo)."""
    n = len(embs)
    if n < 3:
        return [None] * n, [False] * n
    sims = embs @ embs.T
    med = []
    for i in range(n):
        others = np.delete(sims[i], i)
        med.append(float(np.median(others)))
    group = float(np.median(med))
    flags = [m < group - drop for m in med]
    return med, flags


def build(uploads, query, out, print_query=None, min_score=0.3, crop_pad=0.04, name=None, use_vlm=None, main=None, exclude=()):
    """main: file name of the seller's main photo (the product being sold). Photos with the
    reference role are scene references, never product evidence: they are kept aside in
    "references" and flagged when they show a product of the same kind.
    exclude: file names the seller took out of the set at confirmation (a detail-page card
    with text, another color, a wrong crop). They are listed in "excluded"."""
    out = Path(out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    items, dropped, excluded, references, capture_log = [], [], [], [], []
    use_vlm = vlm.available() if use_vlm is None else (use_vlm and vlm.available())
    vlm_log = {"used": use_vlm, "model": vlm.MODEL if use_vlm else None, "steps": {}}
    product = None

    uploads_list = load_uploads(uploads)
    if query is None:
        if not (use_vlm and name):
            raise SystemExit("--query is required unless --name is given and an API key is set")
        sample = [open_rgb(u["path"]) for u in uploads_list if u["role"] == "product"][:3]
        q = vlm.make_queries(name, sample)
        query, product = q.query, q.product
        print_query = print_query or q.print_query
        vlm_log["steps"]["queries"] = q.model_dump()

    def add(file, role, img, det, full, crop_name):
        box = det["box"] if full else pad_box(det["box"], img.size, crop_pad)
        crop = img.crop(box)
        crop.save(out / "crops" / crop_name)
        dens, detail = detail_map(crop)
        items.append(
            {
                "file": file,
                "role": role,
                "product_box": box,
                "product_score": round(det["score"], 3),
                "full_frame": full,
                "frame_fill": round((box[2] - box[0]) * (box[3] - box[1]) / (img.width * img.height), 3),
                "crop": f"crops/{crop_name}",
                "crop_size": crop.size,
                "detail": round(detail, 2),
                "face": None,
                "face_ko": None,
                "_dens": dens,
                "_img": crop,
            }
        )

    for it in uploads_list:
        stem = Path(it["file"]).stem
        if it.get("video"):
            if it["role"] != "capture":
                dropped.append({**it, "reason": "video with an info or reference role is not a product capture"})
                continue
            frames, log = capture_frames(it["path"], query, min_score)
            capture_log.append({"file": it["file"], **log})
            for t, _, im, det in frames:
                add(f"{it['file']}@{t}s", "capture", im, det, False, f"{stem}_t{t}.png")
            continue
        img = open_rgb(it["path"])
        if it["role"] == "reference":
            (out / "references").mkdir(exist_ok=True)
            img.save(out / "references" / f"{stem}.jpg", quality=90)
            references.append({"file": it["file"], "path": f"references/{stem}.jpg", **inspect_reference(img, query, min_score)})
            continue
        det = find_product(img, query, min_score)
        full = False
        if det is None:
            # A close-up (label, pump head, base) fills the frame and has no silhouette for
            # OWLv2 to find. A photo the seller filed as a product photo is kept whole;
            # info images without a detection are usually text-only and are dropped.
            if it["role"] != "product":
                dropped.append({**it, "reason": "product not found (text-only info, or wrong query)"})
                continue
            det, full = {"box": (0, 0, img.width, img.height), "score": 0.0}, True
        add(it["file"], it["role"], img, det, full, f"{stem}.png")

    if not items:
        raise SystemExit("no product found in any upload")
    main_item = next((i for i in items if main and i["file"] == main), None)
    main_at = next((k for k, i in enumerate(items) if i is main_item), None)

    embs = models.embed([i["_img"] for i in items])
    med, flags = same_product_flags(embs)
    for it, e, m, f in zip(items, embs, med, flags):
        it["_emb"] = e
        it["same_product"] = {"median_sim": None if m is None else round(m, 3), "flag": bool(f)}
        if main_item is not None:
            it["same_product"]["sim_to_main"] = round(float(e @ embs[main_at]), 3)

    # The similarity flag is advisory only. On the mug test the print face itself sat lowest
    # (it looks unlike the plain faces by design), so flagged crops are not excluded on it.
    # The VLM sees every crop at once; a crop it calls another product is excluded.
    missing = []
    if use_vlm:
        try:
            main_no = main_at + 1 if main_at is not None else None
            review = vlm.review_set([i["_img"] for i in items], name, main=main_no)
            product = product or review.product
            views = {v.index: v for v in review.views}
            for n, it in enumerate(items, 1):
                v = views.get(n)
                if v is None:
                    continue
                it["face"], it["face_ko"] = v.face, vlm.FACE_KO[v.face]
                it["same_product"]["vlm"] = v.same_product
                it["note"] = v.note
            missing = [{**m.model_dump(), "face_ko": vlm.FACE_KO[m.face]} for m in review.missing]
            keep = [i for i in items if i["same_product"].get("vlm", True) or i is main_item]
            if keep:
                excluded = [i for i in items if not any(i is k for k in keep)]
                items = keep
            vlm_log["steps"]["review"] = "ok"
        except Exception as e:  # a failed VLM call must not stop the reference set
            vlm_log["steps"]["review"] = f"failed: {type(e).__name__}: {e}"
    else:
        vlm_log["skipped"] = "no API key (ANTHROPIC_API_KEY)" if not vlm.available() else "disabled (--no-vlm)"

    seller_out = [i for i in items if i["file"] in set(exclude) and i is not main_item]
    for i in seller_out:
        i["excluded_by"] = "seller"
    items = [i for i in items if not any(i is o for o in seller_out)]
    excluded = list(excluded) + seller_out
    rep = max([i for i in items if whole_view(i)] or items, key=lambda i: i["detail"])

    method, pbox = "density", print_box_from_density(rep["_dens"])
    if print_query:
        dets = models.detect(rep["_img"], [print_query], threshold=0.1)
        if dets and dets[0]["score"] >= 0.25:
            ob = pad_box(dets[0]["box"], rep["_img"].size, 0.08)
            pbox = ob if pbox is None else (min(pbox[0], ob[0]), min(pbox[1], ob[1]), max(pbox[2], ob[2]), max(pbox[3], ob[3]))
            method = "density+owl"
    print_path = None
    prints = []
    if pbox is not None:
        rep["_img"].crop(pbox).save(out / "print.png")
        print_path = "print.png"
        prints.append({"path": print_path, "from": rep["crop"], "box": pbox, "face": rep.get("face"), "method": method})
        prints += extra_prints(items, rep, out)

    views = pick_views(items, rep, prefer=main_item)
    made = composite(views, out / "composite.jpg")
    comp = {
        "path": "composite.jpg" if made else None,
        "views": [v["crop"] for v in views],
        "faces": [v.get("face") for v in views],
        **(made or {"reason": "only one view: upload the representative crop itself"}),
    }

    def clean(i):
        return {k: v for k, v in i.items() if not k.startswith("_")}

    refset = {
        "name": name,
        "uploads": str(Path(uploads).resolve()),
        "product": product,
        "query": query,
        "print_query": print_query,
        "items": [clean(i) for i in items],
        "excluded": [clean(i) for i in excluded],
        "dropped": dropped,
        "main": main_item["crop"] if main_item is not None else None,
        "representative": rep["crop"],
        "print": {"from": rep["crop"], "box": pbox, "path": print_path, "method": method},
        "prints": prints,
        "composite": comp,
        "references": references,
        "capture": capture_log,
        "missing_faces": missing,
        "vlm": vlm_log,
    }
    (out / "refset.json").write_text(json.dumps(refset, ensure_ascii=False, indent=2))
    sheet(items, dropped, rep, pbox, out / "sheet.jpg", excluded=excluded, missing=missing, prints=prints)
    return refset


def _font(size):
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def sheet(items, dropped, rep, pbox, path, T=260, excluded=(), missing=(), prints=()):
    f = _font(18)
    shown = list(items) + list(excluded)
    cols = max(1, len(shown))
    extra = 28 * (1 + len(missing)) if missing else 0
    s = Image.new("RGB", (max(cols * (T + 16) + 16, 900), T + 110 + extra), "white")
    d = ImageDraw.Draw(s)
    for i, it in enumerate(shown):
        im = it["_img"].copy()
        sc = T / max(im.size)
        im = im.resize((int(im.width * sc), int(im.height * sc)))
        x, y = 16 + i * (T + 16), 16
        s.paste(im, (x, y))
        out_ = any(it is e for e in excluded)
        color = (210, 40, 40) if out_ or (it["same_product"]["flag"] and "vlm" not in it["same_product"]) else (22, 140, 70) if it is rep else (120, 120, 120)
        d.rectangle([x - 3, y - 3, x + im.width + 3, y + im.height + 3], outline=color, width=4)
        for pr in prints or ([{"from": rep["crop"], "box": pbox}] if pbox is not None else []):
            if pr["from"] == it["crop"]:
                b = [int(v * sc) for v in pr["box"]]
                d.rectangle([x + b[0], y + b[1], x + b[2], y + b[3]], outline=(220, 120, 0), width=3)
        tag = ("제외(셀러)" if it.get("excluded_by") == "seller" else "제외(다른 제품)") if out_ else "REP" if it is rep else ("OTHER?" if it["same_product"]["flag"] and "vlm" not in it["same_product"] else "")
        face = f" {it['face_ko']}" if it.get("face_ko") else ""
        d.text((x, y + T + 6), f"{it['file'][:18]} [{it['role']}]{face} {tag}", fill="black", font=f)
        d.text((x, y + T + 30), f"detail {it['detail']}  sim {it['same_product']['median_sim']}", fill=color, font=f)
    if dropped:
        d.text((16, T + 80), "dropped: " + ", ".join(x["file"] for x in dropped), fill=(120, 120, 120), font=f)
    if missing:
        y = T + 110
        d.text((16, y), "빠진 면 (촬영 필요)", fill=(200, 90, 0), font=f)
        for m in missing:
            y += 28
            d.text((16, y), f"- {m['face_ko']}: {m['why']} / {m['shot']}", fill=(60, 60, 60), font=f)
    s.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", required=True)
    ap.add_argument("--query", default=None, help='English product noun for OWLv2, e.g. "a mug". Written by the VLM from --name when omitted')
    ap.add_argument("--name", default=None, help="Seller's product name (Korean is fine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-query", default=None)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM steps even when an API key is set")
    ap.add_argument("--main", default=None, help="file name of the seller's main photo (the product being sold)")
    ap.add_argument("--exclude", nargs="*", default=[], help="file names the seller takes out of the set")
    a = ap.parse_args()
    r = build(a.uploads, a.query, a.out, a.print_query, a.min_score, name=a.name, use_vlm=not a.no_vlm, main=a.main, exclude=a.exclude)
    print(json.dumps({k: r[k] for k in ("name", "query", "print_query", "main", "representative", "print", "composite", "references", "capture", "dropped", "missing_faces", "vlm")}, ensure_ascii=False, indent=2))
    for it in r["items"] + r["excluded"]:
        print(it["file"], it["role"], it.get("face_ko"), "detail", it["detail"], "sim", it["same_product"])


if __name__ == "__main__":
    main()

"""VLM steps (Claude vision) for the reference set and for the second check pass.

What the VLM does here, and why it is not done with OWLv2/DINOv2/SIFT
  make_queries   Korean product name -> English OWLv2 queries (--query, --print-query).
  review_set     All crops in one request: face name per crop (front, back, side, handle,
                 top, bottom, detail), same-product check (the model sees every crop at once
                 and picks the one that is another product), and the faces that are missing
                 for a detail page and short videos. DINOv2 similarity could not do the
                 same-product check: on the mug the print face itself scored lowest.
  judge          Second check pass for outputs SIFT could not verify. SIFT gives 0 both when
                 the print was invented and when the print side is simply not in view; the
                 VLM sees the reference crops and the output together and says which.

The API key is read from ANTHROPIC_API_KEY (environment or .env at the repo root). Without
a key every step reports itself as skipped and the rest of the pipeline still runs.
"""
import base64
import io
import os
from pathlib import Path
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

MODEL = os.environ.get("CC_VLM_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CC_VLM_EFFORT", "medium")
SIDE = 1024  # longest side sent to the model; enough for labels, keeps image tokens low
BETAS = ["server-side-fallback-2026-07-01"]

FACES = ("front", "back", "side", "handle", "top", "bottom", "detail", "other")
FACE_KO = {
    "front": "정면",
    "back": "뒷면",
    "side": "옆면",
    "handle": "손잡이",
    "top": "윗면",
    "bottom": "바닥",
    "detail": "디테일",
    "other": "기타",
}
Face = Literal["front", "back", "side", "handle", "top", "bottom", "detail", "other"]

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- setup


def _load_env():
    """Read KEY=VALUE lines from .env at the repo root into os.environ (existing values win)."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def available() -> bool:
    _load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


_client = None


def client():
    global _client
    if _client is None:
        import anthropic

        _load_env()
        _client = anthropic.Anthropic()
    return _client


def set_client(c):
    """Swap the client (tests use a client with a mock transport)."""
    global _client
    _client = c


def image_block(im: Image.Image, side=SIDE, cache=False):
    im = im.convert("RGB")
    s = side / max(im.size)
    if s < 1:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.getvalue()).decode()},
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def text_block(t):
    return {"type": "text", "text": t}


class VLMRefused(RuntimeError):
    pass


def ask(content, schema, max_tokens=8000):
    """One request, parsed into `schema`. Raises VLMRefused when the model declines."""
    resp = client().beta.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        betas=BETAS,
        fallbacks="default",
        output_config={"effort": EFFORT},
        output_format=schema,
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        raise VLMRefused(str(getattr(resp, "stop_details", "")))
    if resp.parsed_output is None:
        raise RuntimeError(f"no parsed output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


# ---------------------------------------------------------------- queries


class Queries(BaseModel):
    product: str = Field(description="What the product is, in Korean, a short noun (e.g. 머그컵, 펌프형 바디로션)")
    query: str = Field(description='OWLv2 query for the whole product: "a/an" + short English noun, e.g. "a mug", "a pump bottle"')
    print_query: Optional[str] = Field(
        description='OWLv2 query for the most distinctive printed element (illustration, logo, label), e.g. "a cat line drawing", "a printed label". null when the product has no print or label.'
    )


QUERIES_PROMPT = """A seller uploaded photos for an online product detail page. We locate the product in every photo with OWLv2, an open-vocabulary detector that takes short English noun phrases.

Product name as the seller wrote it: {name}

Write:
- product: what the product is, in Korean, one short noun.
- query: one English noun phrase for the whole physical product as it sits in a photo ("a mug", "a pump bottle", "a sweatshirt"). Use the container, not the contents (a lotion is "a pump bottle" or "a bottle").
- print_query: one English noun phrase for the most distinctive printed element on the product, if you can see one in the photos (or if the name implies one); null otherwise.
Keep noun phrases plain and visual; OWLv2 does not understand brand names."""


def make_queries(name, images=()):
    content = [image_block(im, side=768) for im in list(images)[:3]]
    content.append(text_block(QUERIES_PROMPT.format(name=name)))
    return ask(content, Queries, max_tokens=4000)


# ---------------------------------------------------------------- reference set review


class CropView(BaseModel):
    index: int = Field(description="Photo number as given (1-based)")
    face: Face
    same_product: bool = Field(description="False only when this photo shows a different product from the rest (another model, color, brand, or unrelated object)")
    note: str = Field(description="Korean, one short sentence: what this photo shows that matters for keeping the product unchanged")


class MissingFace(BaseModel):
    face: Face
    why: str = Field(description="Korean: which page cut or video motion needs this face (e.g. 회전 영상, 뒷면 성분표 컷)")
    shot: str = Field(description="Korean: how the seller should shoot it, one sentence")


class SetReview(BaseModel):
    product: str = Field(description="Korean, what the product is")
    views: List[CropView]
    missing: List[MissingFace] = Field(description="Faces that exist on this product, carry information, and no photo shows")


REVIEW_PROMPT = """These {n} photos are product crops taken from one seller's uploads for an online detail page (some come from info images such as old detail-page captures). An image generator will later draw this product in new scenes and short videos, and it can only copy what these photos show; anything missing it will invent.
{name_line}
For every photo, give:
- face: which side of the product it mainly shows.
  front = the side with the main print, label or logo; back = the opposite side; side = a side view (for a mug, a side view without the handle in the middle); handle = the handle side; top = from above (cap, pump head, opening); bottom = the base; detail = a close-up of one part; other = none of these.
- same_product: false only if the photo shows a different product from the rest (different model, color, design, brand, or an unrelated object). The same product seen from another angle, or a photo where its print is not visible, is still the same product.
- note: one short Korean sentence.

Then list missing faces: sides that this kind of product physically has, that carry information a generator would otherwise invent (print, label text, structure, handle shape, buttons, base), and that none of the photos shows. Think of what a detail page and short videos (rotation, zoom-in, in-use) would need. Do not list sides that are plain and fully implied by other photos. Write why and shot in Korean."""


def review_set(crops, name=None, main=None):
    """main: 1-based number of the seller's main photo (the product being sold), if chosen."""
    content = []
    for i, im in enumerate(crops, 1):
        content.append(text_block(f"Photo {i}:"))
        content.append(image_block(im))
    name_line = f"\nProduct name as the seller wrote it: {name}\n" if name else ""
    if main:
        name_line += f"Photo {main} is the seller's main photo: it shows the product being sold. Judge same_product against it.\n"
    content.append(text_block(REVIEW_PROMPT.format(n=len(crops), name_line=name_line)))
    return ask(content, SetReview)


# ---------------------------------------------------------------- second check pass


class Verdict(BaseModel):
    product_present: bool = Field(description="Is the product in the output image at all")
    verdict: Literal["match", "not_visible", "different"]
    differences: List[str] = Field(description="Korean, each a concrete visible difference from the real product; empty unless verdict is different")
    reason: str = Field(description="Korean, one or two sentences")


JUDGE_PROMPT = """The images before this text are the seller's real product, cropped from their own photos{faces}. The image after this text is an AI-generated marketing image (or a video frame) that is supposed to show exactly this product.

A keypoint matcher could not find the product's print in the generated image. That happens in two different situations, and you decide which one this is:

- not_visible: the side carrying the print or label is simply not in view (the back or handle side faces the camera, it is cropped, covered by a hand, too small or blurred to see), and nothing that is visible contradicts the real product. Normal, not an error.
- different: something visible contradicts the real product. Examples: the print or illustration is redrawn, replaced or invented; the print is missing from the side where it should be (use the reference photos to work out where the print sits relative to the handle, cap or seams, and check whether that side faces the camera); small text is gone or changed; the shape, structure, proportions, material, color, or number of parts differ.
- match: the print and the product are clearly visible and the same as the real one (the matcher can miss stylized or strongly angled views).

If the product is not in the image at all, set product_present to false and verdict to not_visible.
List each visible difference in Korean, concretely (what is different, where). Write reason in Korean."""


FOUND_PROMPT = """The images before this text are the seller's real product, cropped from their own photos{faces}. The image after this text is an AI-generated marketing image (or a video frame) that is supposed to show exactly this product.

A keypoint matcher found the product's main print in the generated image, so the large print is there. Generators still drop or garble the small things next to it, so look closely:
- small text (sub-titles, ingredient lines, sizes, brand lines under a logo): missing, misspelled, or turned into fake letters;
- parts of the print or label missing or added; extra logos;
- shape, structure, proportions, material, color, number of parts.

- match: nothing visible contradicts the real product (small rendering differences, lighting and angle are fine; text too small to read at this size is fine).
- different: something visible contradicts the real product.
- not_visible: only if the product is in fact not visible (the matcher was fooled).

List each visible difference in Korean, concretely (what is different, where). Write reason in Korean."""


PLAIN_PROMPT = """The images before this text are the seller's real product, cropped from their own photos{faces}. The image after this text is an AI-generated marketing image (or a video frame) that is supposed to show exactly this product.

This product has no print or label for a keypoint matcher to find, so judge it by eye. Compare what is visible: shape, structure, proportions, joints and moving parts, material and transparency, color, number of parts, logos and text.

- match: the visible product is the same as the real one (lighting, angle and small rendering differences are fine).
- not_visible: the product is not in the image, or too small, cropped or covered to tell.
- different: something visible contradicts the real product. A side that no reference photo shows is not by itself a difference; judge what can be compared.

List each visible difference in Korean, concretely (what is different, where). Write reason in Korean."""


def judge(refs, output, product=None, found=False, plain=False, keep=None, zoom=None, checks=None):
    """refs: list of (label, PIL image). output: PIL image. Returns a Verdict.

    keep: the conditions of the confirmed production spec (spec/make.py), in Korean. A
    visible violation of any of them is a difference.

    zoom: (label, PIL image) close-ups of the generated image, sent after it. checks: extra
    lines (Korean) to verify against the real product, sent before the generated image.

    found=True is the strict review of an output whose print SIFT already found (small text,
    parts, shape); found=False decides between not_visible and different.
    The reference block is identical across calls in one check run, so it is cached.
    """
    content = []
    for i, (label, im) in enumerate(refs):
        content.append(text_block(f"Real product, {label}:"))
        content.append(image_block(im, cache=(i == len(refs) - 1)))
    faces = f" ({product})" if product else ""
    prompt = PLAIN_PROMPT if plain else FOUND_PROMPT if found else JUDGE_PROMPT
    content.append(text_block(prompt.format(faces=faces)))
    if keep:
        content.append(text_block("The seller confirmed these conditions before generation; a visible violation of any of them is a difference:\n" + "\n".join(f"- {k}" for k in keep)))
    if checks:
        content.append(text_block("Also check these against the real product:\n" + "\n".join(f"- {c}" for c in checks)))
    content.append(text_block("Generated image:"))
    content.append(image_block(output))
    for label, im in zoom or []:
        content.append(text_block(f"Close-up of the generated image, {label}:"))
        content.append(image_block(im))
    return ask(content, Verdict, max_tokens=6000)

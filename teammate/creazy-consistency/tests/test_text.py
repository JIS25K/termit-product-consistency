from PIL import Image, ImageDraw, ImageFont
from check import text

# Anchors are single words: build_expect splits readings on spaces (text.words).
A = [{"word": "pause", "rel_h": 0.2}, {"word": "cat", "rel_h": 0.02}]

def r(t, conf=0.9, box=(0, 0, 100, 20)):
    return {"text": t, "conf": conf, "box": box}

def test_normalize_keeps_hangul_latin_digits():
    assert text.normalize("Faire une Pause!") == "faireunepause"
    assert text.normalize("두부 100g") == "두부100g"

def test_anchor_found_even_when_ocr_splits_it():
    c = text.compare(A[:1], [r("Faire une pau"), r("se")])
    assert c["found"] == ["pause"] and c["missing"] == []

def test_garbled_word_is_reported():
    assert text.compare(A[:1], [r("Faire une Pause")])["found"] == ["pause"]
    c = text.compare(A[:1], [r("Faire une Pqvse")])   # "pqvse" vs "pause": 0.6
    assert c["garbled"] == ["pqvse"] and c["found"] == []

def test_small_anchor_is_not_expected_in_a_small_box():
    # rel_h 0.02 * 300 px = 6 px < MIN_PX: the word "cat" is not expected
    c = text.compare(A, [r("faire une pause")], box_h=300)
    assert c["expected"] == ["pause"] and c["missing"] == []

def test_foreign_text_is_confident_and_unlike_any_anchor():
    c = text.compare(A[:1], [r("BIG SALE", conf=0.95), r("xq", conf=0.3)])
    assert c["foreign"] == ["big", "sale"] and c["found"] == []

def test_read_finds_large_printed_word():
    im = Image.new("RGB", (900, 300), "white")
    ImageDraw.Draw(im).text((40, 80), "PAUSE", fill="black", font=ImageFont.load_default(size=120))
    got = " ".join(x["text"] for x in text.read(im))
    assert "PAUSE" in got.upper()

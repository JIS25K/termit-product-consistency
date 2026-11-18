"""Model loading and thin wrappers (OWLv2 for open-vocabulary detection, DINOv2 for image embeddings).

Both models run on CPU. On Apple Silicon the device is switched to MPS when available.
"""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

OWL_ID = "google/owlv2-base-patch16-ensemble"
DINO_ID = "facebook/dinov2-small"


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=None)
def owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(OWL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(_device()).eval()
    return proc, model


@lru_cache(maxsize=None)
def dino():
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID).to(_device()).eval()
    return proc, model


def detect(image: Image.Image, queries, threshold=0.1):
    """Return detections as a list of dicts sorted by score: {query, score, box(x0,y0,x1,y1)}.

    OWLv2 pads the image to a square on the bottom/right, so boxes are scaled with the
    longer side and stay in the original pixel coordinates.
    """
    proc, model = owl()
    inputs = proc(text=[list(queries)], images=image, return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model(**inputs)
    side = max(image.size)
    res = proc.post_process_grounded_object_detection(
        outputs=out, threshold=threshold, target_sizes=torch.tensor([[side, side]])
    )[0]
    w, h = image.size
    dets = []
    for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        x0, y0, x1, y1 = b
        box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        dets.append({"query": queries[l], "score": float(s), "box": tuple(int(round(v)) for v in box)})
    return sorted(dets, key=lambda d: -d["score"])


def embed(images):
    """L2-normalised DINOv2 CLS embeddings, shape (n, d)."""
    proc, model = dino()
    inputs = proc(images=list(images), return_tensors="pt").to(_device())
    with torch.no_grad():
        out = model(**inputs)
    v = out.last_hidden_state[:, 0].float().cpu().numpy()
    return v / np.linalg.norm(v, axis=1, keepdims=True)

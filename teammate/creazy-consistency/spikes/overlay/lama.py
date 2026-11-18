"""LaMa (big-lama TorchScript from IOPaint) wrapper: erase a masked region without inventing objects."""
import os, numpy as np, cv2, torch

_M = {}
PATH = os.path.expanduser("~/.cache/lama/big-lama.pt")

def _model(dev):
    if dev not in _M:
        m = torch.jit.load(PATH, map_location="cpu")
        m.eval()
        _M[dev] = m.to(dev)
    return _M[dev]

def inpaint(bgr, mask, dev=None):
    """bgr uint8 HxWx3, mask HxW (nonzero = erase). Returns bgr uint8, original pixels kept outside mask."""
    H, W = mask.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    m = np.pad((mask > 0).astype(np.float32), ((0, ph), (0, pw)), mode="symmetric")
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
    mk = torch.from_numpy(m)[None, None]
    devs = [dev] if dev else (["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"])
    err = None
    for d in devs:
        try:
            with torch.no_grad():
                y = _model(d)(x.to(d), mk.to(d))
            out = y[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W]
            out = cv2.cvtColor(np.clip(out * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            keep = (mask > 0)[..., None]
            return np.where(keep, out, bgr), d
        except Exception as e:  # MPS lacks some FFT ops in older torch; fall back
            err = e
            print("lama on", d, "failed:", repr(e)[:200], flush=True)
    raise err

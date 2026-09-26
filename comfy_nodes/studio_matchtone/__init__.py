"""
studio_matchtone - a ComfyUI node that gives a redrawn patch the colours and
tone curves of the picture it came from, for Studio Assist's Fix a spot.

Why: a redraw at 0.65 or more comes back in the model's own grade - a hand
a little pinker, the shadows a little lifted - and blended into the photo it
shows as a patch. So over the part redrawn (`mask`), each RGB channel of the
redraw is mapped through the curve that takes its distribution onto the
original's there (quantile matching: Photoshop's Curves, fitted, per
channel), then mixed back by `amount`. The shapes the redraw drew are kept;
only their colour and contrast move.

numpy only. The source of truth is comfy_nodes/studio_matchtone in the
Studio Assist repo; copy it into ComfyUI's custom_nodes and restart ComfyUI.
"""

import numpy as np

POINTS = 33                # points on each channel's curve
MIN_PIXELS = 256           # fewer under the mask and the redraw is left as it is


def curve(src, ref, points=POINTS):
    """(xs, ys): the monotone curve taking `src`'s values onto `ref`'s."""
    q = np.linspace(0.0, 1.0, points)
    xs, ys = np.quantile(src, q), np.quantile(ref, q)
    xs = np.maximum.accumulate(xs + np.arange(points) * 1e-6)   # strictly rising
    return xs, ys


def match(image, reference, mask, amount):
    """`image` (H, W, 3, 0-1) with its colour curves moved `amount` of the
    way to `reference`'s (the same size), fitted where `mask` (H, W) > 0.5."""
    sel = mask > 0.5
    if int(sel.sum()) < MIN_PIXELS or amount <= 0:
        return image
    out = image.copy()
    for c in range(3):
        xs, ys = curve(image[..., c][sel], reference[..., c][sel])
        mapped = np.interp(image[..., c], xs, ys)
        out[..., c] = image[..., c] + amount * (mapped - image[..., c])
    return np.clip(out, 0.0, 1.0)


def _fit(a, h, w):
    """A mask or picture resized to (h, w) by nearest neighbour."""
    if a.shape[0] == h and a.shape[1] == w:
        return a
    ys = (np.arange(h) * a.shape[0] / float(h)).astype(int)
    xs = (np.arange(w) * a.shape[1] / float(w)).astype(int)
    return a[ys][:, xs]


class StudioMatchTone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "reference": ("IMAGE",),
                             "mask": ("MASK",),
                             "amount": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0,
                                                  "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Studio Assist"

    def run(self, image, reference, mask, amount=0.85):
        import torch
        img = image[0].cpu().numpy().astype(np.float32)[..., :3]
        h, w = img.shape[:2]
        ref = _fit(reference[0].cpu().numpy().astype(np.float32)[..., :3], h, w)
        m = mask.cpu().numpy().astype(np.float32)
        m = _fit(m[0] if m.ndim == 3 else m, h, w)
        out = match(img, ref, m, float(amount))
        return (torch.from_numpy(out.astype(np.float32))[None],)


NODE_CLASS_MAPPINGS = {"StudioMatchTone": StudioMatchTone}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioMatchTone": "Match colour curves (Studio Assist)"}
